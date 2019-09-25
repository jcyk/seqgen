import torch
import numpy as np
from data import Vocab, DataLoader, BLANK
from masker import Masker

import argparse

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, nargs='+')
    parser.add_argument('--min_vote', type =int)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--output_file', type = str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_config()

    models = []
    for load_path in args.load_path:
        ckpt = torch.load(load_path)
        model_args = ckpt['args']

        vocab_src = Vocab(model_args.vocab_src, with_SE = False)
        vocab_tgt = Vocab(model_args.vocab_tgt, with_SE = False)

        model = Masker(vocab_src, vocab_tgt, model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout, model_args.num_layers)

        model.load_state_dict(ckpt['model'])
        model = model.cuda()
        model.eval()
        models.append(model)

    test_data = DataLoader(args.test_data, vocab_src, vocab_tgt, args.test_batch_size, False, model_args.stop_words_file)

    with open(args.output_file, 'w') as fo:
        for src_input, ref_src_input, ref_tgt_input, keep_mask_target, raw in test_data:
            preds = []
            for model in models:
                pred = model.work(src_input, ref_src_input, ref_tgt_input, keep_mask_target)
                preds.append(np.array(pred))
            pred = (sum(preds) >= args.min_vote).astype(np.int)
            for mask, raw_sents in zip(pred, raw):
                prefix = '|'.join([ ' '.join(x) for x in raw_sents[:3]])
                pred_ref_tgt = []
                for _m, _w in zip(mask, raw_sents[3]):
                    if _m == 1:
                        pred_ref_tgt.append(_w)
                    else:
                        pred_ref_tgt.append(BLANK)
                fo.write(prefix+'|'+' '.join(pred_ref_tgt)+'\n')


