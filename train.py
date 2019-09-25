import torch

from data import Vocab, DataLoader
from model import ResponseGenerator
from utils import eval_dist

import argparse
import time, random

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_src', type=str)
    parser.add_argument('--vocab_tgt', type=str)

    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--input_feed', action = 'store_true')

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_batch_size', type=int)

    parser.add_argument('--print_every', type = int)
    parser.add_argument('--eval_every', type = int)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str, nargs = '+')
    parser.add_argument('--beam_size', default=5)
    parser.add_argument('--max_time_step', default=20)

    parser.add_argument('--stop_words_file', type = str)
    parser.add_argument('--random_mask', default = None, type = float)
    return parser.parse_args()


def evaluate(args, model, dev_data, output_file):
    model.eval()
    sents = []
    dev_PPL = 0.
    dev_tokens = 0
    for batch_dict in dev_data:
        loss = model(batch_dict)
        dev_PPL += loss.item() * batch_dict['num_tokens']
        dev_tokens += batch_dict['num_tokens']

        hyps_batch = model.work(batch_dict, args.beam_size, args.max_time_step)
        for hyps in hyps_batch:
            hyps.sort(key = lambda x:x.score/len(x.seq), reverse = True)
            best_hyp = hyps[0]
            predicted_tgt = [token.raw for token in best_hyp.seq]
            sents.append(' '.join(predicted_tgt))
    dist1, dist2 = eval_dist(sents)
    PPL = dev_PPL / dev_tokens
    res = {'dist1':dist1, 'dist2':dist2, 'PPL':PPL}
    print res
    with open(output_file, 'w') as fo:
        for sent in sents:
            fo.write(sent+'\n')
    return res

def update_lr(optimizer, coefficient):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * coefficient

if __name__ == "__main__":
    random.seed(19940117)
    torch.manual_seed(19940117)
    args = parse_config()
    vocab_src = Vocab(args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(args.vocab_tgt, with_SE = True)

    model = ResponseGenerator(vocab_src, vocab_tgt, args.embed_dim, args.hidden_size, args.num_layers, args.dropout, args.input_feed)
    #model.load_state_dict(torch.load('./ckpt/epoch17')['model'])
    model = model.cuda()

    #args.lr = args.lr * 0.2
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    train_data = DataLoader(args.train_data, vocab_src, vocab_tgt, args.train_batch_size, True, args.stop_words_file, args.random_mask)
    dev_data_list = [DataLoader(dev_data, vocab_src, vocab_tgt, args.train_batch_size, True, args.stop_words_file, args.random_mask) for dev_data in args.dev_data]

    PPL_accumulated = 0
    tokens_processed = 0
    batches_processed = 0


    model.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        for batch_dict in train_data:
            optimizer.zero_grad()
            loss = model(batch_dict)

            num_tokens = batch_dict['num_tokens']
            PPL_accumulated += loss.item() * num_tokens
            tokens_processed += num_tokens
            batches_processed += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if batches_processed % args.print_every == -1 % args.print_every:
                print "Epoch %d, Batch %d, PPL %.5f, Speed %.0f tokens/s"%(epoch, batches_processed, PPL_accumulated / tokens_processed, tokens_processed/(time.time() - start_time))

            if batches_processed % args.eval_every == -1 % args.eval_every:
                for data_idx, data in enumerate(dev_data_list):
                    evaluate(args, model, data, 'ckpt/data%d_epoch%d_batch%d'%(data_idx,epoch, batches_processed))
                model.train()
                torch.save({'args':args, 'model':model.state_dict()}, 'ckpt/epoch%d_batch%d'%(epoch, batches_processed))
        update_lr(optimizer, 0.8)
    while True:
        never_stop = True
