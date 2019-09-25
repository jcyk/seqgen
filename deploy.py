# encoding = utf8

from waitress import serve
from flask import Flask, request

from collections import Counter
import json
import numpy as np

import jieba
from preprocess_zh import segment_line

import torch
beam_size = 5
max_time_step = 20

#### Response Generator
from data import Vocab, batchify
from model import ResponseGenerator
def load_ResponseGenerator(load_path):
    ckpt = torch.load(load_path, map_location='cpu')
    model_args = ckpt['args']
    v = set([ x.strip() for x in open(model_args.vocab_src).readlines()])

    vocab_src = Vocab(model_args.vocab_src, with_SE = False)
    vocab_tgt = Vocab(model_args.vocab_tgt, with_SE = True)


    model = ResponseGenerator(vocab_src, vocab_tgt, model_args.embed_dim, model_args.hidden_size, model_args.num_layers, model_args.dropout, model_args.input_feed)

    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()
    return model, v, vocab_src, vocab_tgt
model, v, vocab_src, vocab_tgt = load_ResponseGenerator('ckpt/epoch24.48')

#### Masker
from ranker.data import batchify as masker_batchify
from ranker.data import Vocab as masker_Vocab
def load_Masker(load_path):
    ckpt = torch.load(load_path, map_location='cpu')
    model_args = ckpt['args']
    vocab_src = masker_Vocab("ranker/"+model_args.vocab_src, with_SE = False)
    vocab_tgt = masker_Vocab("ranker/"+model_args.vocab_tgt, with_SE = False)
    if False:
        from ranker.masker_ranker_interaction import Ranker
    else:
        from ranker.masker_ranker import Ranker
    model = Ranker(vocab_src, vocab_tgt,
                   model_args.embed_dim, model_args.ff_embed_dim,
                   model_args.num_heads, model_args.dropout, model_args.num_layers)

    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()
    return model, vocab_src, vocab_tgt
masker, masker_vocab_src, masker_vocab_tgt = load_Masker('ranker/ckpt/epoch18_batch860999_acc_0.721')

#### Ranker
ranker_batchify = masker_batchify
ranker_Vocab = masker_Vocab
def load_Ranker(load_path):
    ckpt = torch.load(load_path, map_location='cpu')
    model_args = ckpt['args']

    vocab_src = ranker_Vocab("ranker/"+model_args.vocab_src, with_SE = False)
    vocab_tgt = ranker_Vocab("ranker/"+model_args.vocab_tgt, with_SE = False)
    from ranker.ranker import Ranker
    model = Ranker(vocab_src, vocab_tgt,
                   model_args.embed_dim, model_args.ff_embed_dim,
                   model_args.num_heads, model_args.dropout, model_args.num_layers)

    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()
    return model, vocab_src, vocab_tgt
ranker, ranker_vocab_src, ranker_vocab_tgt = load_Ranker('ranker/ckpt/epoch13_batch643999_acc_0.771')

def _query_skeletons_to_responses(query, skeletons):
    all_d = []
    for skeleton in skeletons:
        all_d.append([query, query, skeleton, skeleton])

    batch_dict = batchify(all_d, vocab_src, vocab_tgt, set([]), None)
    hyps_batch = model.work(batch_dict, beam_size, max_time_step)

    responses = []
    for hyps in hyps_batch:
        hyps.sort(key = lambda x:x.score/((1+len(x.seq))**0.6), reverse = True)
        best_hyp = hyps[0]
        predicted_tgt = [token.raw for token in best_hyp.seq]
        predicted_tgt = predicted_tgt[1:-1]
        response =  ''.join(predicted_tgt)
        responses.append(response)
    return responses

def _query_responses_to_skeletons(query, responses):
    all_d = []
    for response in responses:
        all_d.append([query, response])
    src_input, tgt_input = masker_batchify(all_d, masker_vocab_src, masker_vocab_tgt)
    beta, s, m = masker.work(src_input, tgt_input)
    skeletons = []
    for _beta, _s, _m, response in zip(beta, s, m, responses):
        assert _m == len(response)
        _beta = _beta[:_m]
        _s =_s[:_m]
        positive_scores = [ x for x in _s if x >0 ]
        if len(positive_scores) > 0:
            avg_pos_s = sum(positive_scores) / len(positive_scores)
        else:
            avg_pos_s = 0.
        skeleton = []
        for w, s in zip(response, _s):
            if s > avg_pos_s:
                skeleton.append(w)
            else:
                skeleton.append('<BLANK>')
        skeleton = ' '.join(skeleton)
        skeletons.append(skeleton)
    return skeletons

def _query_responses_to_rank(query, responses):
    all_d = []
    for response in responses:
        all_d.append([query, response])
    src_input, tgt_input = ranker_batchify(all_d, ranker_vocab_src, ranker_vocab_tgt)
    scores = ranker.work(src_input, tgt_input)

    assert len(scores) == len(responses)
    rank = [0 for i in range(len(scores))]
    for th, pos in enumerate(list(np.argsort(-np.array(scores)))):
        rank[pos] = th
    return rank


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/query_skeleton', methods=('GET', 'POST'))
    def query_and_skeleton_to_response():
        print ('!!!!!!!!')
        if request.method == "POST":
            form = request.form
        if request.method == "GET":
            form = request.args
        query = form['query']
        skeleton = form['skeleton']
        print (query, skeleton)
        cnt = Counter()
        query = segment_line(' '.join(jieba.cut(query.strip())), v , cnt)
        _skeleton = []
        for piece in skeleton.strip().split(';;;'):
            piece  = ' '.join(jieba.cut(piece))
            _skeleton.append(segment_line(piece, v, cnt))
        skeleton = ' <BLANK> '.join(_skeleton)
        query = query.encode('utf8').split()
        skeleton = skeleton.encode('utf8').split()
        responses = _query_skeletons_to_responses(query, [skeleton])
        res = json.dumps({"response": responses[0]})
        return res

    @app.route('/query_retrievals', methods=('GET', 'POST'))
    def query_and_retrievals_to_skeletons_and_responses():
        if request.method == "POST":
            form = request.form
        if request.method == "GET":
            form = request.args
        query = form['query']
        retrievals = form['retrievals']
        cnt = Counter()
        query = segment_line(' '.join(jieba.cut(query.strip())), v , cnt)
        responses = []
        for piece in retrievals.strip().split(';;;'):
            piece = segment_line(' '.join(jieba.cut(piece)), v, cnt)
            response = piece.encode('utf8').split()
            responses.append(response)
        query = query.encode('utf8').split()
        skeletons = _query_responses_to_skeletons(query, responses)
        responses = _query_skeletons_to_responses(query, [ x.split() for x in skeletons])
        res = json.dumps({"responses": responses, "skeletons": skeletons})
        return res

    @app.route('/query_responses', methods=('GET', 'POST'))
    def query_and_responses_to_rank():
        if request.method == "POST":
            form = request.form
        if request.method == "GET":
            form = request.args
        query = form['query']
        _responses = form['responses']
        cnt = Counter()
        query = segment_line(' '.join(jieba.cut(query.strip())), v , cnt)
        responses = []
        for piece in _responses.strip().split(';;;'):
            piece = segment_line(' '.join(jieba.cut(piece)), v, cnt)
            response = piece.encode('utf8').split()
            responses.append(response)
        query = query.encode('utf8').split()
        rank = _query_responses_to_rank(query, responses)
        res = json.dumps({"rank": rank})
        return res
    return app
if __name__ == "__main__":
    serve(create_app(), listen='*:8088')
