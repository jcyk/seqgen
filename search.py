import torch

from data import UNK

class Beam(object):
    def __init__(self, beam_size, max_time_step, hypotheses):
        self.beam_size = beam_size
        self.max_time_step = max_time_step
        self.completed_hypotheses = []
        self.steps = 0
        self.hypotheses = hypotheses

    def merge_score(self, prev_hyp, step):
        token, score = step
        prefix = prev_hyp.seq
        if token.raw == UNK:
            return float('-inf')
        new_score = prev_hyp.score + score
        for w in prefix:
            if token.raw == w.raw:
                new_score += score
        return new_score


    def update(self, new_states, last_steps, end_idx):
        # last_steps: lists of list of (token:Token, step_score:float)  num_hypotheses x beam_size
        live_nyp_num = self.beam_size - len(self.completed_hypotheses)
        candidates = []
        for prev_hyp_idx, steps in enumerate(last_steps):
            for pos, step in enumerate(steps):
                candidates.append((prev_hyp_idx, pos, self.merge_score(self.hypotheses[prev_hyp_idx], step)))
        candidates.sort(key = lambda x:-x[-1])
        # candidates: list of triples (prev_hyp_idx, last_step_pos, new_score)
        new_hyps = []
        for prev_hyp_idx, pos, score in candidates[:live_nyp_num]:
            state = dict()
            for k, v in new_states.items():
                if len(v.size()) == 3:
                    state[k] = v[:, prev_hyp_idx:prev_hyp_idx+1, :]
                else:
                    state[k] = v[prev_hyp_idx:prev_hyp_idx+1, :]
            seq = self.hypotheses[prev_hyp_idx].seq + [last_steps[prev_hyp_idx][pos][0]]
            new_hyps.append(Hypothesis(state, seq, score))
        
        self.hypotheses = []
        for hyp in new_hyps:
            if hyp.seq[-1].idx == end_idx:
                self.completed_hypotheses.append(hyp)
            else:
                self.hypotheses.append(hyp)
        self.steps += 1

    @property    
    def completed(self):
        if len(self.completed_hypotheses) < self.beam_size and self.steps < self.max_time_step:
            return False
        return True

class Token(object):
    def __init__ (self, idx, raw):
        self.idx = idx
        self.raw = raw

class Hypothesis(object):
    def __init__(self, state_dict, seq, score):
        #state: hidden state of the last step (have not yet consider seq[-1])
        #seq: current generated sequence
        #score: accumlated score so far
        self.state_dict = state_dict
        self.seq = seq
        self.score = score

def get_init_beam(vocab, init_state_dict, beam_size, max_time_step):
    start_token = Token(vocab.start_idx, vocab.idx2token(vocab.start_idx))
    init_hyp = Hypothesis(init_state_dict, [start_token], 0.)
    return Beam(beam_size, max_time_step, [init_hyp])

def search_by_batch(model, beams, batch_dict, mem_dict, vocab):

    def ready_to_submit(hypotheses):
        y_tm1 = torch.tensor([hyp.seq[-1].idx for hyp in hypotheses]).unsqueeze(0)
        concat_hyps= dict() 
        for hyp in hypotheses:
            for k, v in hyp.state_dict.items():
                concat_hyps[k] = concat_hyps.get(k, []) + [v]
        for k, v in concat_hyps.items():
            if len(v[0].size()) == 3:
                concat_hyps[k] = torch.cat(v, 1)
            else:
                concat_hyps[k] = torch.cat(v, 0)
        return concat_hyps, y_tm1.cuda()

    while True:
        hypotheses = []
        indices = []
        for idx, beam in enumerate(beams):
            if not beam.completed:
                for hyp in beam.hypotheses:
                    hypotheses.append(hyp)
                    indices.append(idx)
        if not hypotheses:
            break 

        indices = torch.tensor(indices).cuda()
        state_dict, y_tm1 = ready_to_submit(hypotheses)
        cur_mem_dict = dict()
        for k, v in mem_dict.items():
            cur_mem_dict[k] = v.index_select(1, indices)
        batch_pos_idx_map = batch_dict['copy']['batch_pos_idx_map'].index_select(0, indices)
        state_dict, outs, attn_scores = model.decoder(y_tm1, state_dict, cur_mem_dict)
        LL = model.compute_log_likelihood(outs, attn_scores, batch_dict['copy']['idx2token'], batch_pos_idx_map, batch_dict['copy']['idx2idx_mapping']).squeeze(0)
        batch_score, batch_idx =  torch.topk(LL, beams[0].beam_size, 1) # bsz x k

        batch_score = batch_score.data.tolist()
        batch_idx = batch_idx.data.tolist()

        pos = 0

        def toToken(i):
            if i >= vocab.size:
                token = batch_dict['copy']['idx2token'][i]
                i = vocab.unk_idx
            else:
                token = vocab.idx2token(i)
            return Token(i, token)

        for idx, beam in enumerate(beams):
            if not beam.completed:
                _len = len(beam.hypotheses)
                last_steps = [[(toToken(ti), si) for ti, si in zip(t, s)] for t, s in zip(batch_idx[pos : pos+_len], batch_score[pos : pos+_len])]
                new_states = dict()
                for k ,v in state_dict.items():
                    if len(v.size()) == 3:
                        new_states[k] = v[ :, pos:pos+_len,:]
                    else:
                        new_states[k] = v[pos:pos+_len, :]
                beam.update(new_states, last_steps, vocab.end_idx)
                pos += _len
