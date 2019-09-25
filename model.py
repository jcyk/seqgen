import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import LSTMencoder
from decoder import LSTMdecoder
from module import Embedding, Linear
from search import search_by_batch, get_init_beam


class ResponseGenerator(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, embed_dim, hidden_size, num_layers, dropout, input_feed):
        super(ResponseGenerator, self).__init__()
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.encoder_src = LSTMencoder(vocab_src, embed_dim=embed_dim, hidden_size = hidden_size//2, 
            num_layers= num_layers, dropout_in= dropout, dropout_out= dropout,
            bidirectional = True, pretrained_embed = None)
        self.embed_tgt = Embedding(vocab_tgt.size, embed_dim, vocab_tgt.padding_idx)
        self.encoder_ske = LSTMencoder(vocab_tgt, embed_dim=embed_dim, hidden_size = hidden_size//2, 
            num_layers= num_layers, dropout_in= dropout, dropout_out= dropout,
            bidirectional = True, pretrained_embed = self.embed_tgt)
        self.decoder = LSTMdecoder(vocab_tgt, embed_dim=embed_dim, hidden_size = hidden_size,
            num_layers= num_layers, dropout_in= dropout, dropout_out= dropout,
            encoder_hidden_size = hidden_size, pretrained_embed = self.embed_tgt, input_feed = input_feed)
        self.copy = Linear(hidden_size, 1)
        self.generate = Linear(hidden_size, vocab_tgt.size)

    def forward(self, batch_dict):
        (init_h, init_c), mem_src, mem_src_mask = self.encoder_src(batch_dict['src_tokens'], batch_dict['src_lengths'])
        _, mem_ske, mem_ske_mask = self.encoder_ske(batch_dict['ske_tokens'], batch_dict['ske_lengths'])
        mem_dict = {'mem_src':mem_src,
                    'mem_src_mask':mem_src_mask,
                    'mem_ske':mem_ske,
                    'mem_ske_mask':mem_ske_mask}
        state_dict = {'h':init_h, 'c': init_c}
        _, outs, attn_scores = self.decoder(batch_dict['input_tgt_tokens'], prev_state_dict = state_dict, mem_dict = mem_dict)
        seq_len, bsz, hidden_size = outs.size()

        generation_probs, copy_probs = self.compute_log_likelihood(outs, attn_scores, batch_dict['copy']['idx2token'], batch_dict['copy']['batch_pos_idx_map'], batch_dict['copy']['idx2idx_mapping'], speed_train = True)

        generation_target = batch_dict['generation_tgt_tokens'].view(-1, 1)
        copy_target = batch_dict['copy_tgt_tokens'].view(-1, 1)
        target = batch_dict['output_tgt_tokens'].view(-1, 1)

        # deal with padding (-1)
        generation_padding = generation_probs.data.new(seq_len, bsz, 1).zero_()
        generation_probs = torch.cat([generation_padding, generation_probs], 2)
        g_likelihood = generation_probs.view(seq_len*bsz, -1).gather(dim=1 , index=1+generation_target)
        copy_padding = copy_probs.data.new(seq_len, bsz, 1).zero_()
        copy_probs = torch.cat([copy_padding, copy_probs], 2)
        c_likelihood = copy_probs.view(seq_len*bsz, -1).gather(dim=1, index=1+copy_target)

        NLL = -torch.log(g_likelihood + c_likelihood + 1e-12).masked_fill_(torch.eq(target, self.vocab_tgt.padding_idx), 0.)
        loss = torch.sum(NLL) / batch_dict['num_tokens']
        return loss

    def work(self, batch_dict, beam_size, max_time_step):
        (init_h, init_c), mem_src, mem_src_mask = self.encoder_src(batch_dict['src_tokens'], batch_dict['src_lengths'])
        _, mem_ske, mem_ske_mask = self.encoder_ske(batch_dict['ske_tokens'], batch_dict['ske_lengths'])
        mem_dict = {'mem_src':mem_src,
                    'mem_src_mask':mem_src_mask,
                    'mem_ske':mem_ske,
                    'mem_ske_mask':mem_ske_mask}

        _, bsz, _ = init_h.size()
        beams = []
        for i in range(bsz):
            state_dict_i = {'h':init_h[:,i:i+1,:], 'c': init_c[:,i:i+1,:]}
            beam = get_init_beam(self.vocab_tgt, state_dict_i, beam_size, max_time_step)
            beams.append(beam)

        search_by_batch(self, beams, batch_dict, mem_dict, self.vocab_tgt)
        return [beam.completed_hypotheses if beam.completed_hypotheses else beam.hypotheses for beam in beams]

    def compute_log_likelihood(self, outs, attn_scores, idx2token, batch_pos_idx_map, idx2idx_mapping, speed_train= False):

        seq_len, bsz, hidden_size = outs.size()

        generate_gate = torch.sigmoid(self.copy(outs)) #seq_len x bsz x1
        generation_probs = generate_gate * F.softmax(self.generate(outs), -1) # seq_len x bsz x vocab_size

        # batch_pos_idx_map: bsz x ske_len x indices
        # attn_score seq_len x bsz x ske_len

        copy_probs = (1- generate_gate) * torch.bmm(attn_scores.transpose(0, 1), batch_pos_idx_map).transpose(0, 1) # seq_len x bsz x indices

        if not speed_train:
            if len(idx2token) > 0:
                extended_probs = copy_probs[:,:,-len(idx2token):]
                probs = torch.cat([generation_probs, extended_probs], 2)
            else:
                probs = generation_probs

            known_words = len(idx2idx_mapping) - len(idx2token)
            if known_words > 0:
                known_words = sorted(idx2idx_mapping.keys())[:known_words]
                index = torch.LongTensor(known_words).cuda().unsqueeze(0).unsqueeze(0).expand(seq_len, bsz, -1)
                probs.scatter_add_(-1, index, copy_probs[:,:,:len(known_words)])

            LL = torch.log(probs + 1e-12)
            return LL
        return generation_probs, copy_probs
