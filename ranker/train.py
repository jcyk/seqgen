from data import Vocab, DataLoader
import torch

import random
import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_src', type=str)
    parser.add_argument('--vocab_tgt', type=str)

    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--dev_batch_size', type=int)
    parser.add_argument('--print_every', type = int)
    parser.add_argument('--eval_every', type = int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--which_ranker', type=str)
    return parser.parse_args()

def update_lr(optimizer, coefficient):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * coefficient

if __name__ == "__main__":
    random.seed(19940117)
    torch.manual_seed(19940117)
    args = parse_config()
    vocab_src = Vocab(args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(args.vocab_tgt, with_SE = False)

    if args.which_ranker == 'ranker':
        from ranker import Ranker
    elif args.which_ranker == 'masker_ranker':
        from masker_ranker import Ranker
    elif args.which_ranker == 'masker_ranker_interaction':
        from masker_ranker_interaction import Ranker
    model = Ranker(vocab_src, vocab_tgt,
            args.embed_dim, args.ff_embed_dim,
            args.num_heads, args.dropout, args.num_layers)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_data = DataLoader(args.train_data, vocab_src, vocab_tgt, args.train_batch_size, True)
    dev_data = DataLoader(args.dev_data, vocab_src, vocab_tgt, args.dev_batch_size, True)

    model.train()
    loss_accumulated = 0.
    acc_accumulated = 0.
    batches_processed = 0
    best_dev_acc = 0
    for epoch in range(args.epochs):
        for src_input, tgt_input in train_data:
            optimizer.zero_grad()
            loss, acc = model(src_input, tgt_input)

            loss_accumulated += loss.item()
            acc_accumulated += acc
            batches_processed += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batches_processed % args.print_every == -1 % args.print_every:
                print "Batch %d, loss %.5f, acc %.5f"%(batches_processed, loss_accumulated / batches_processed, acc_accumulated / batches_processed)
            if batches_processed % args.eval_every == -1 % args.eval_every:
                model.eval()
                dev_acc = 0.
                dev_batches = 0
                for src_input, tgt_input in dev_data:
                    _, acc = model(src_input, tgt_input)
                    dev_acc += acc
                    dev_batches += 1
                dev_acc = dev_acc / dev_batches
                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc
                    torch.save({'args':args, 'model':model.state_dict()}, 'ckpt/epoch%d_batch%d_acc_%.3f'%(epoch, batches_processed, dev_acc))

                print "Dev Batch %d, acc %.5f"%(batches_processed, dev_acc)
                model.train()
