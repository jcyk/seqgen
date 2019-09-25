import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Embedding, LSTM


class LSTMencoder(nn.Module):
    def __init__(self, vocab, embed_dim=512, hidden_size=512, 
        num_layers=1, dropout_in=0.1, dropout_out=0.1, 
        bidirectional = True, pretrained_embed = None):
        super(LSTMencoder, self).__init__()
        self.vocab  = vocab
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        if pretrained_embed is not None:
            self.embed_tokens = pretrained_embed
        else:
            self.embed_tokens = Embedding(vocab.size, embed_dim, vocab.padding_idx)
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )

    def forward(self, src_tokens, src_lengths):
        seq_len, bsz = src_tokens.size()
        ###
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        sorted_src_tokens = src_tokens.index_select(1, indices)
        ###
        x = self.embed_tokens(sorted_src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.data.tolist())

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_h, final_c) = self.lstm(packed_x, (h0, c0))

        mem, _ = nn.utils.rnn.pad_packed_sequence(packed_outs)
        mem = F.dropout(mem, p=self.dropout_out, training=self.training)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            final_h = combine_bidir(final_h)
            final_c = combine_bidir(final_c)

        ###
        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size
        final_c = final_c.index_select(1, positions) 
        mem = mem.index_select(1, positions) # seq_len x bsz x hidden_size
        ###
        mem_mask = src_tokens.eq(self.vocab.padding_idx) # seq_len x bsz

        return (final_h, final_c), mem, mem_mask
