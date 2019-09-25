import torch 
BLANK = '<BLANK>'

def build_copy_mapping(raw_tokens_batch, vocab):

    new_tokens = set()
    for raw_tokens in raw_tokens_batch:
        for raw_token in raw_tokens:
            if raw_token == BLANK:
                continue
            if vocab.token2idx(raw_token) == vocab.unk_idx:
                new_tokens.add(raw_token)
    nxt_idx = vocab.size
    token2idx = dict()
    idx2token = dict()
    for new_token in new_tokens:
        token2idx[new_token] = nxt_idx
        idx2token[nxt_idx] = new_token
        nxt_idx += 1

    def toIdx(w):
        if w in token2idx:
            return token2idx[w]
        return vocab.token2idx(w)
    
    indices = set()
    max_len = 0
    grouped_pos_idx_batch = []
    for raw_tokens in raw_tokens_batch:
        max_len = max(max_len, len(raw_tokens))
        grouped_pos_idx = []
        idx2pos = dict()
        for pos, raw_token in enumerate(raw_tokens):
            idx = toIdx(raw_token)
            idx2pos[idx] = idx2pos.get(idx, [])  + [pos]
            indices.add(idx)
        for idx, pos in idx2pos.items():
            grouped_pos_idx.append((pos, idx))
        grouped_pos_idx_batch.append(grouped_pos_idx)

    indices = list(indices)
    indices.sort()
    idx2idx_mapping = dict(zip(indices, range(len(indices))))
    batch_pos_idx_map = torch.zeros(len(grouped_pos_idx_batch), max_len, len(indices))

    for bidx, grouped_pos_idx in enumerate(grouped_pos_idx_batch):
        for pos, idx in grouped_pos_idx:
            idx = idx2idx_mapping[idx]
            for p in pos:
                batch_pos_idx_map[bidx, p, idx] = 1.
    batch_pos_idx_map = batch_pos_idx_map.cuda()
    
    return token2idx, idx2token, batch_pos_idx_map, idx2idx_mapping

from collections import Counter
def eval_dist(sents):
    unigrams = Counter()
    bigrams = Counter()
    tot_tokens = 0
    for sent in sents:
        words = sent.strip().split()
        unigrams.update(words)
        bigrams.update(zip(words[:-1], words[1:]))
        tot_tokens += len(words)
    return len(unigrams)/float(tot_tokens), len(bigrams)/float(tot_tokens)
