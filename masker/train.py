from masker import Masker
from data import Vocab, DataLoader
import torch

import random
import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_src', type=str)
    parser.add_argument('--vocab_tgt', type=str)
    parser.add_argument('--stop_words_file', type=str)

    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_batch_size', type=int)

    parser.add_argument('--print_every', type = int)
    parser.add_argument('--eval_every', type = int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    return parser.parse_args()

def update_lr(optimizer, coefficient):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * coefficient

if __name__ == "__main__":
    #random.seed(19940117)
    #torch.manual_seed(19940117)
    args = parse_config()
    vocab_src = Vocab(args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(args.vocab_tgt, with_SE = False)

    model = Masker(vocab_src, vocab_tgt,
            args.embed_dim, args.ff_embed_dim,
            args.num_heads, args.dropout, args.num_layers)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_data = DataLoader(args.train_data, vocab_src, vocab_tgt, args.train_batch_size, True, stop_words_file = args.stop_words_file)
    dev_data = DataLoader(args.dev_data, vocab_src, vocab_tgt, args.train_batch_size, True, stop_words_file = args.stop_words_file)

    model.train()
    loss_accumulated = 0.
    acc_accumulated, good_accumulated, p_accumulated, r_accumulated, tot_accumulated = 0., 0., 0., 0., 0.
    batches_processed = 0
    best_ff = 0.
    for epoch in range(args.epochs):
        for src_input, ref_src_input, ref_tgt_input, keep_mask_target in train_data:
            optimizer.zero_grad()
            loss, acc, good, p, r, tot = model(src_input, ref_src_input, ref_tgt_input, keep_mask_target)

            loss_accumulated += loss.item()
            acc_accumulated, good_accumulated, p_accumulated, r_accumulated, tot_accumulated = acc_accumulated+acc, good_accumulated+good, p_accumulated+p, r_accumulated+r, tot_accumulated+tot
            batches_processed += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batches_processed % args.print_every == -1 % args.print_every:
                pp, rr = good_accumulated / p_accumulated, good_accumulated/r_accumulated
                ff = 2*pp*rr/(pp+rr)
                print "Train Epoch %d Batch %d, acc %.3f, p %.3f, r %.3f, f %.3f, leave rates: gold %.3f, pred %.3f"%(epoch, batches_processed, acc_accumulated / tot_accumulated, pp, rr, ff, r_accumulated/tot_accumulated, p_accumulated / tot_accumulated)

            if batches_processed % args.eval_every == -1 % args.eval_every:
                model.eval()
                acc_accumulated, good_accumulated, p_accumulated, r_accumulated, tot_accumulated = 0., 0., 0., 0., 0.
                for src_input, ref_src_input, ref_tgt_input, keep_mask_target in dev_data:
                    loss, acc, good, p, r, tot = model(src_input, ref_src_input, ref_tgt_input, keep_mask_target)
                    acc_accumulated, good_accumulated, p_accumulated, r_accumulated, tot_accumulated = acc_accumulated+acc, good_accumulated+good, p_accumulated+p, r_accumulated+r, tot_accumulated+tot
                pp, rr = good_accumulated / p_accumulated, good_accumulated/r_accumulated
                ff = 2*pp*rr/(pp+rr)
                print "Dev Batch %d, acc %.3f, p %.3f, r %.3f, f %.3f, leave rates: gold %.3f, pred %.3f"%(batches_processed, acc_accumulated / tot_accumulated, pp, rr, ff, r_accumulated/tot_accumulated, p_accumulated / tot_accumulated)
                if ff > best_ff:
                    best_ff = ff
                    torch.save({'args':args, 'model':model.state_dict()}, 'ckpt/epoch%d_batch%d_best_ff%.3f'%(epoch, batches_processed, ff))
                acc_accumulated, good_accumulated, p_accumulated, r_accumulated, tot_accumulated = 0., 0., 0., 0., 0.
