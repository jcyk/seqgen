import torch

from data import Vocab, DataLoader
from model import ResponseGenerator

import argparse

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--beam_size', type= int)
    parser.add_argument('--max_time_step', type=int)
    parser.add_argument('--output_file', type = str)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_config()
    ckpt = torch.load(args.load_path)
    model_args = ckpt['args']

    vocab_src = Vocab(model_args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(model_args.vocab_tgt, with_SE = True)

    model = ResponseGenerator(vocab_src, vocab_tgt, model_args.embed_dim, model_args.hidden_size, model_args.num_layers, model_args.dropout, model_args.input_feed)

    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    test_data = DataLoader(args.test_data, vocab_src, vocab_tgt, args.test_batch_size, False)

    model.eval()

    if args.verbose:
        queries = [ x.strip().split('|')[0] for x in open(args.test_data).readlines()]
        qid = 0
    with open(args.output_file, 'w') as fo:
        for batch_dict in test_data:
            hyps_batch = model.work(batch_dict, args.beam_size, args.max_time_step)

            for hyps in hyps_batch:
                hyps.sort(key = lambda x:x.score/((1+len(x.seq))**0.6), reverse = True)
                best_hyp = hyps[0]
                predicted_tgt = [token.raw for token in best_hyp.seq]
                predicted_tgt = predicted_tgt[1:-1]
                if args.verbose:
                    fo.write(queries[qid]+'|'+' '.join(predicted_tgt)+'\n')
                    qid += 1
                else:
                    fo.write(' '.join(predicted_tgt)+'\n')
