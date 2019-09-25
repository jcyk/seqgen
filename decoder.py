import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Embedding, LSTM, GeneralAttention, Linear

class LSTMdecoder(nn.Module):
    def __init__(self, vocab, embed_dim=512, hidden_size=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
        encoder_hidden_size = 512, pretrained_embed = None, input_feed = False):
        super(LSTMdecoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.input_feed = input_feed
        if pretrained_embed is not None:
            self.embed_tokens = pretrained_embed
        else:
            self.embed_tokens = Embedding(vocab.size, embed_dim, vocab.padding_idx)

        self.lstm = LSTM(
            input_size = embed_dim + (hidden_size if self.input_feed else 0),
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = self.dropout_out if num_layers > 1 else 0.,
        )
        self.attention_src = GeneralAttention(hidden_size, encoder_hidden_size, encoder_hidden_size)
        self.proj_with_src = Linear(encoder_hidden_size + hidden_size, hidden_size)
        self.attention_ske = GeneralAttention(hidden_size, hidden_size, hidden_size)
        self.gate_with_ske = Linear(2*hidden_size, hidden_size)

    def forward(self, tgt_tokens, prev_state_dict, mem_dict):
        seq_len, bsz = tgt_tokens.size()
        x = self.embed_tokens(tgt_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        state_size = self.num_layers, bsz, self.hidden_size
        if 'h' in prev_state_dict:
            h_t = prev_state_dict['h']
        else:
            h_t = x.data.new(*state_size).zero_()
        if 'c' in prev_state_dict:
            c_t = prev_state_dict['c']
        else:
            c_t = x.data.new(*state_size).zero_()
        if self.input_feed:
            if 'input_feed' in prev_state_dict:
                input_feed = prev_state_dict['input_feed']
            else:
                input_feed = x.data.new(bsz, self.hidden_size).zero_()

            final_outs, attn_scores = [], []

        for t in range(seq_len):
            if not self.input_feed and t>0:
                break
            input = x if (not self.input_feed) else torch.cat([x[t,:,:], input_feed], 1).unsqueeze(0)

            outs, (h_t, c_t) = self.lstm(input, (h_t, c_t))
            outs = outs.view(-1, bsz, 1, self.hidden_size).squeeze(2)

            #### 
            # This is what we do
            attn_src, _ = self.attention_src(outs, mem_dict['mem_src'], mem_dict['mem_src'], mem_dict['mem_src_mask'])
            with_src = torch.tanh(self.proj_with_src(torch.cat((outs, attn_src), dim=2)))
            attn_ske, attn_score = self.attention_ske(with_src, mem_dict['mem_ske'], mem_dict['mem_ske'], mem_dict['mem_ske_mask'])
            gate_ske = torch.sigmoid(self.gate_with_ske(torch.cat((with_src, attn_ske), dim=2)))
            input_feed = gate_ske * attn_ske + (1 - gate_ske) * with_src
            if self.input_feed:
                input_feed = input_feed.squeeze(0)
                final_outs.append(input_feed)
                attn_scores.append(attn_score)
            ####

        if self.input_feed:    
            final_outs = torch.stack(final_outs)   # seq_len x bsz x hidden_size
            attn_scores = torch.cat(attn_scores, 0) # seq_len x bsz x ske_len
            state_dict = {'h':h_t, 'c':c_t, 'input_feed': input_feed}
        else:
            final_outs = input_feed
            attn_scores = attn_score
            state_dict = {'h':h_t, 'c':c_t}

        return state_dict, final_outs, attn_scores
