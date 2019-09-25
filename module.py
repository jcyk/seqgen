import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dot = False):
        super(GeneralAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dot = dot
        if not dot:
            self.input_proj = Linear(self.query_dim, self.key_dim, bias=False)

    def forward(self, queries, keys, values, mask):
        # queries query_len x bsz x hidden1
        # keys mem_len x bsz x hidden2
        # values mem_len x bsz x hidden3
        # mask mem_len x bsz
        if not self.dot:
            x = self.input_proj(queries)

        queries = x.transpose(0, 1) # bsz x query_len x hidden
        keys = keys.transpose(0, 1).transpose(1, 2) # bsz x hidden x mem_size
        values = values.transpose(0, 1) # bsz x mem_len x value_dim
        mask = mask.transpose(0, 1) # bsz x mem_len


        attn_scores = torch.bmm(queries, keys) # bsz x query_len x mem_size

        if mask is not None:
            attn_scores = attn_scores.masked_fill_(
                mask.unsqueeze(1),
                float('-inf')
            )
        attn_scores = F.softmax(attn_scores, dim=-1) # bsz x query_len x mem_len
        x = torch.bmm(attn_scores, values).transpose(0, 1) # query_len x bsz x value_dim
        return x, attn_scores.transpose(0, 1)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs): 
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

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
