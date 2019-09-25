import torch
import random

PAD, BOS, EOS, UNK, BLANK = '<_>', '<bos>', '<eos>', '<unk>', '<BLANK>'

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


def ListsToTensor(xs, vocab, with_S= False, with_E = False):

    batch_size = len(xs)
    lens = [ len(x) + (1 if with_S else 0) + (1 if with_E else 0) for x in xs]
    mx_len = max(max(lens),1)
    ys = []
    for i, x in enumerate(xs):
        y = ([vocab.start_idx] if with_S else [] )+ [vocab.token2idx(w) for w in x] + ([vocab.end_idx] if with_E else []) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    #lens = torch.LongTensor([ max(1, x) for x in lens])
    data = torch.LongTensor(ys).t_().contiguous()
    return data.cuda()

def LCS_mask(src, tgt, stop_words):
    m = len(src)
    n = len(tgt)
    if stop_words is None:
        stop_words = set()
    mat = [[0] * (n+1) for row in range(m+1)]
    for row in range(1, m+1):
        for col in range(1, n+1):
            if src[row - 1] == tgt[col - 1] and (src[row-1] not in stop_words):
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])
    x,y = m,n
    mask = []
    while y >0 and x >0:
        if mat[x][y] == mat[x-1][y-1] + 1:
            x -=1
            y -=1
            mask.append(1)
        elif mat[x][y] == mat[x][y-1]:
            y -= 1
            mask.append(0)
        else:
            x -= 1
    while y>0:
        y -= 1
        mask.append(0)
    return mask[::-1]

def batchify(data, vocab_src, vocab_tgt, stop_words, include_raw = False):
    src = ListsToTensor([x[0] for x in data], vocab_src)
    ref_src = ListsToTensor([x[2] for x in data], vocab_src)
    ref_tgt = ListsToTensor([x[3] for x in data], vocab_tgt)
    seq_len, _ = ref_tgt.size()
    keep_mask = []
    for x in data:
        mask = LCS_mask(x[1], x[3], stop_words)
        mask = mask + [0] * (seq_len - len(mask))
        keep_mask.append(mask)
    keep_mask = torch.LongTensor(keep_mask).t_().to(torch.uint8).contiguous()
    keep_mask = keep_mask.cuda()
    if include_raw:
        raw = data
        return src, ref_src, ref_tgt, keep_mask, raw
    return src, ref_src, ref_tgt, keep_mask

class DataLoader(object):
    def __init__(self, filename, vocab_src, vocab_tgt, batch_size,for_train, stop_words_file = None):
        all_data = [[ x.split() for x in line.strip().split('|') ] for line in open(filename).readlines()]
        self.data = []
        for d in all_data:
            skip = not (len(d) == 4)
            for j, i in enumerate(d):
                if len(i) ==0 or len(i) > 30:
                    skip = True
            if not skip:
                self.data.append(d)

        self.batch_size = batch_size
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        if stop_words_file is not None:
            self.stop_words = set([line.strip() for line in open(stop_words_file).readlines()])
        else:
            self.stop_words = set([])
        self.train = for_train

    def __iter__(self):
        idx = list(range(len(self.data)))
        if self.train:
            random.shuffle(idx)
        cur = 0
        while cur < len(idx):
            data = [self.data[i] for i in idx[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab_src, self.vocab_tgt, self.stop_words, include_raw = not self.train)
        raise StopIteration
