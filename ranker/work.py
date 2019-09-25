import torch

from data import Vocab, DataLoader
from ranker import Ranker

import argparse

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--output_file', type = str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_config()
    ckpt = torch.load(args.load_path)
    model_args = ckpt['args']

    vocab_src = Vocab(model_args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(model_args.vocab_tgt, with_SE = False)
    model = Ranker(vocab_src, vocab_tgt,
                   model_args.embed_dim, model_args.ff_embed_dim,
                   model_args.num_heads, model_args.dropout, model_args.num_layers)

    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    test_data = DataLoader(args.test_data, vocab_src, vocab_tgt, args.test_batch_size, False)

    model.eval()

    lines = open(args.test_data).readlines()
    idx = 0
    with open(args.output_file, 'w') as fo:
        for src_input, tgt_input in test_data:
            scores = model.work(src_input, tgt_input)
            for score in scores:
                fo.write(lines[idx].rstrip()+"|"+str(score)+'\n')
                idx += 1
