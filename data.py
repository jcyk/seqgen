import torch
import random
from utils import build_copy_mapping, BLANK

import numpy as np
PAD, BOS, EOS, UNK = '<_>', '<bos>', '<eos>', '<unk>'

class Vocab(object):
    def __init__(self, filename, with_SE):
        with open(filename) as f:
            if with_SE:
                self.itos = [PAD, BOS, EOS, UNK] + [ token.strip() for token in f.readlines() ]
            else:
                self.itos = [PAD, UNK] + [ token.strip() for token in f.readlines() ]
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self._size = len(self.stoi)
        self._padding_idx = self.stoi[PAD]
        self._unk_idx = self.stoi[UNK]
        self._start_idx = self.stoi.get(BOS, -1)
        self._end_idx = self.stoi.get(EOS, -1)

    def random_sample(self):
        return self.idx2token(1 + np.random.randint(self._size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self.itos[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self.stoi.get(x, self.unk_idx)

    @property
    def size(self):
        return self._size

    @property
    def padding_idx(self):
        return self._padding_idx

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx


def ListsToTensor(xs, vocab, token2idx, with_S= False, with_E = False):

    def toIdx(w):
        if (token2idx is not None) and (w in token2idx):
            return token2idx[w]
        return vocab.token2idx(w)

    batch_size = len(xs)
    lens = [ len(x) + (1 if with_S else 0) + (1 if with_E else 0) for x in xs]
    mx_len = max(max(lens),1)
    ys = []
    for i, x in enumerate(xs):
        y = ([vocab.start_idx] if with_S else [] )+ [toIdx(w) for w in x] + ([vocab.end_idx] if with_E else []) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    lens = torch.LongTensor([ max(1, x) for x in lens])
    data = torch.LongTensor(ys).t_().contiguous()
    return data.cuda(), lens.cuda()

def batchify(data, vocab_src, vocab_tgt, stop_words, random_mask):
    blk = BLANK

    src_, tgt_, ref_tgt_ = [], [], []
    for src, tgt, ref_src, ref_tgt in data:
        src_.append(src)
        tgt_.append(tgt)
        if random_mask is not None:
            masked_ref_tgt = []
            # the lengths of 10% skeleton are random
            if random.random() < 0.1:
                random_mask = random.random()
            for x in ref_tgt:
                if ((x not in stop_words) and (random.random() < random_mask)):
                    masked_ref_tgt.append(x)
                else:
                    if random.random() < 0.8:
                        masked_ref_tgt.append(blk)
                    else:
                        masked_ref_tgt.append(vocab_tgt.random_sample())
            # the words in 10% skeleton are in random order
            if random.random() < 0.1:
                random.shuffle(masked_ref_tgt)
        else:
            masked_ref_tgt = ref_tgt
        clean_ske = []
        prev_blk = False
        for x in masked_ref_tgt:
            if prev_blk and x == blk:
                pass
            else:
                clean_ske.append(x)
            prev_blk = (x==blk)
        ref_tgt_.append(clean_ske)

    token2idx, idx2token, batch_pos_idx_map, idx2idx_mapping = build_copy_mapping(ref_tgt_, vocab_tgt)

    src_tokens, src_lengths = ListsToTensor(src_, vocab_src, None)
    input_tgt_tokens, tgt_lengths = ListsToTensor(tgt_, vocab_tgt, None, with_S = True)
    output_tgt_tokens, _ = ListsToTensor(tgt_, vocab_tgt, token2idx, with_E = True)
    ske_tokens, ske_lengths = ListsToTensor(ref_tgt_ , vocab_tgt, None)


    copy_dict = {'idx2token': idx2token, 'batch_pos_idx_map':batch_pos_idx_map,
                 'token2idx':token2idx, 'idx2idx_mapping':idx2idx_mapping}

    def split_generation_copy(tensor):
        def transform(x, g = False, c = False):
            assert g^c, 'either g or c'
            if isinstance(x, list):
                return [ transform( i, g, c) for i in x]

            return (x if x<vocab_tgt.size else -1) if g else (idx2idx_mapping.get(x, -1))
        gc = tensor.data.tolist()
        g_ = transform(gc, g = True)
        c_ = transform(gc, c = True)
        return torch.LongTensor(g_).cuda(), torch.LongTensor(c_).cuda()

    generation_tgt_tokens, copy_tgt_tokens = split_generation_copy(output_tgt_tokens)

    batch_dict = {'src_tokens':src_tokens, 'src_lengths':src_lengths,
                'input_tgt_tokens':input_tgt_tokens, 'tgt_lengths':tgt_lengths,
                'output_tgt_tokens': output_tgt_tokens,
                'generation_tgt_tokens': generation_tgt_tokens,
                'copy_tgt_tokens': copy_tgt_tokens,
                'ske_tokens':ske_tokens, 'ske_lengths':ske_lengths,
                'num_tokens': torch.sum(tgt_lengths).item(),
                'copy': copy_dict}
    return batch_dict

class DataLoader(object):
    def __init__(self, filename, vocab_src, vocab_tgt, batch_size, for_train, stop_words_file = None, random_mask = None):
        all_data = [[ x.split() for x in line.strip().split('|') ] for line in open(filename).readlines()]
        self.data = []
        for d in all_data:
            skip = not (len(d)==4)
            for j, i in enumerate(d):
                if not for_train:
                    d[j] = i[:30]
                    if len(d[j]) == 0:
                        d[j] = [UNK]
                if len(i) ==0 or len(i) > 30:
                    skip = True
            if not (skip and for_train):
                self.data.append(d)

        self.batch_size = batch_size
        self.train = for_train
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        if stop_words_file is not None:
            self.stop_words = set([line.strip() for line in open(stop_words_file).readlines()])
        else:
            self.stop_words = set([])
        self.random_mask = random_mask

    def __iter__(self):
        idx = list(range(len(self.data)))
        if self.train:
            random.shuffle(idx)
        cur = 0
        while cur < len(idx):
            data = [self.data[i] for i in idx[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab_src, self.vocab_tgt, self.stop_words, self.random_mask)
        raise StopIteration
