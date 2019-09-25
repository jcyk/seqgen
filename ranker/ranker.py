import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

import math

from .transformer import TransformerLayer, SinusoidalPositionalEmbedding, Embedding


def label_smoothed_nll_loss(log_probs, target, eps):
    #log_probs: N x C
    #target: N
    nll_loss =  -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    if eps == 0.:
        return nll_loss
    smooth_loss = -log_probs.sum(dim=-1)
    eps_i = eps / log_probs.size(-1)
    loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    return loss

class Ranker(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, embed_dim, ff_embed_dim, num_heads, dropout, num_layers):
        super(Ranker, self).__init__()
        self.transformer_src = nn.ModuleList()
        self.transformer_tgt = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_src.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
            self.transformer_tgt.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.embed_src = Embedding(vocab_src.size, embed_dim, vocab_src.padding_idx)
        self.embed_tgt = Embedding(vocab_tgt.size, embed_dim, vocab_tgt.padding_idx)
        self.absorber_src = Parameter(torch.Tensor(embed_dim))
        self.absorber_tgt = Parameter(torch.Tensor(embed_dim))
        self.scorer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.absorber_src, mean=0, std=self.embed_dim ** -0.5)
        nn.init.normal_(self.absorber_tgt, mean=0, std=self.embed_dim ** -0.5)
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, 0.)

    def work(self, src_input, tgt_input):
        src, tgt = self.forward(src_input, tgt_input, work = True)
        src = self.scorer(src)
        return torch.sum(src * tgt, -1).tolist()

    def forward(self, src_input, tgt_input, work = False):
        _, bsz = src_input.size()
        src = self.embed_src(src_input) * self.embed_scale + self.embed_positions(src_input)
        tgt = self.embed_tgt(tgt_input) * self.embed_scale + self.embed_positions(tgt_input)

        src = F.dropout(src, p=self.dropout, training=self.training)
        tgt = F.dropout(tgt, p=self.dropout, training=self.training)

        # seq_len x bsz x embed_dim
        absorber = self.embed_scale * self.absorber_src.unsqueeze(0).unsqueeze(0).expand(1, bsz, self.embed_dim)
        src = torch.cat([absorber, src], 0)

        absorber = self.embed_scale * self.absorber_tgt.unsqueeze(0).unsqueeze(0).expand(1, bsz, self.embed_dim)
        tgt = torch.cat([absorber, tgt], 0)

        src_padding_mask = src_input.eq(self.vocab_src.padding_idx)
        tgt_padding_mask = tgt_input.eq(self.vocab_tgt.padding_idx)


        absorber = src_padding_mask.data.new(1, bsz).zero_()
        src_padding_mask = torch.cat([absorber, src_padding_mask], 0)
        tgt_padding_mask = torch.cat([absorber, tgt_padding_mask], 0)

        for layer in self.transformer_src:
            src, _, _ = layer(src, self_padding_mask=src_padding_mask)
        for layer in self.transformer_tgt:
            tgt, _, _ = layer(tgt, self_padding_mask=tgt_padding_mask)

        src = src[0]
        tgt = tgt[0] #bsz x embed_dim

        if work:
            return src, tgt

        scores = torch.mm( self.scorer(src), tgt.transpose(0, 1)) # bsz x bsz
        log_probs = F.log_softmax(scores, -1)

        gold = torch.arange(bsz).cuda()

        _, pred = torch.max(log_probs, -1)

        acc = torch.sum(torch.eq(gold, pred).float()) / bsz
        #print pred

        loss = label_smoothed_nll_loss(log_probs, gold, 0.1)
        #F.nll_loss(log_probs, torch.arange(bsz).cuda(), reduction = "elementwise_mean")
        loss = loss.mean()
        return loss, acc
