import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

import math

from transformer import TransformerLayer, SinusoidalPositionalEmbedding, Embedding

def compute_metrics(pred, target, mask):
    pred = pred.float()
    target = target.float()
    mask = 1. - mask.float()

    tot = mask.sum().item()
    acc = (torch.eq(pred, target).float() * mask ).sum().item()
    good = (pred * target * mask).sum().item()
    p = (pred * mask).sum().item()
    r = (target * mask).sum().item()
    return acc, good, p, r, tot

class Masker(nn.Module):
    def __init__(self, vocab_src, vocab_tgt, embed_dim, ff_embed_dim, num_heads, dropout, num_layers):
        super(Masker, self).__init__()
        self.transformer_src = nn.ModuleList()
        self.transformer_tgt = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_src.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
            self.transformer_tgt.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external=True))
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.embed_src = Embedding(vocab_src.size, embed_dim, vocab_src.padding_idx)
        self.embed_tgt = Embedding(vocab_tgt.size, embed_dim, vocab_tgt.padding_idx)
        self.masker = nn.Linear(embed_dim, 1)
        self.dropout = dropout
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.masker.weight)
        nn.init.constant_(self.masker.bias, 0.)

    def work(self, src_input, ref_src_input, ref_tgt_input, keep_mask_target):
        res = self.forward(src_input, ref_src_input, ref_tgt_input, keep_mask_target, work = True)
        return res.t().tolist()

    def forward(self, src_input, ref_src_input, ref_tgt_input, keep_mask_target, work=False):
        _, bsz = src_input.size()
        src = self.embed_src(src_input) * self.embed_scale + self.embed_positions(src_input)
        ref_src = self.embed_src(ref_src_input) * self.embed_scale + self.embed_positions(ref_src_input)
        ref_tgt = self.embed_tgt(ref_tgt_input) * self.embed_scale + self.embed_positions(ref_tgt_input)

        src = F.dropout(src, p=self.dropout, training=self.training)
        ref_src = F.dropout(ref_src, p=self.dropout, training=self.training)
        ref_tgt = F.dropout(ref_tgt, p=self.dropout, training=self.training)

        # seq_len x bsz x embed_dim
        src_padding_mask = src_input.eq(self.vocab_src.padding_idx)
        ref_src_padding_mask = ref_src_input.eq(self.vocab_src.padding_idx)
        ref_tgt_padding_mask = ref_tgt_input.eq(self.vocab_tgt.padding_idx)

        for layer in self.transformer_src:
            src, _, _ = layer(src, self_padding_mask=src_padding_mask)
            ref_src, _, _ = layer(ref_src, self_padding_mask=ref_src_padding_mask)

        for idx, layer in enumerate(self.transformer_tgt):
            if idx%2 ==0:
                ref_tgt, _, _ = layer(ref_tgt, self_padding_mask=ref_tgt_padding_mask, 
                                  external_memories = src, external_padding_mask=src_padding_mask)
            else:
                ref_tgt, _, _ = layer(ref_tgt, self_padding_mask=ref_tgt_padding_mask, 
                                  external_memories = ref_src, external_padding_mask=ref_src_padding_mask)

        mask_probs = torch.sigmoid(self.masker(ref_tgt)).squeeze(-1)
        loss = F.binary_cross_entropy(mask_probs, keep_mask_target.float(), reduction='none').masked_fill_(ref_tgt_padding_mask, 0.).sum()

        pred = torch.gt(mask_probs, 0.5)
        if work:
            return pred
        acc, good, p, r, tot = compute_metrics(pred, keep_mask_target, ref_tgt_padding_mask)
        loss = loss / tot

        return loss, acc, good, p, r, tot
