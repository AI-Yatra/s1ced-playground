#!/usr/bin/env python3
"""S1-CED CPU inference: load a release and score typed decisions. No GPU, no conversion.

--model takes a Hugging Face repo id (downloaded once; default syedazeez/s1ced-v3-deep) or a local release folder.

Usage:
  s1ced --input request.json                                          # one request, prints JSON
  s1ced --serve --port 8000                                           # HTTP: POST /decide
  s1ced --bench                                                       # CPU latency (add --device cuda on a GPU)
  s1ced --model ./release --weights step_6250.pt --head deep_fusion   # other weights

Request JSON:
  {"state": "Policy: ... Case: ...",
   "questions": [{"question": "What should be done?", "options": ["approve", "deny", "escalate"], "type": "choice"},
                 {"question": "Is it inside the window?", "type": "noul"}]}
`type` is choice | noul (options default to ["no", "yes"]) | score. The state is encoded once for all questions.
"""
import argparse, json, math, sys, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hub import resolve_release

SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[STATE]', '[QUESTION]', '[OPTION]']
PAD_ID = 0

# ---- model definition: copied verbatim from the training notebook (section 6) ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        y = x.float(); y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)

_ROPE_CACHE = {}
def rope_cos_sin(L, D, base, device, dtype):
    key = (L, D, base, str(device), dtype)
    if key not in _ROPE_CACHE:
        inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
        f = torch.outer(torch.arange(L, device=device, dtype=torch.float32), inv)
        f = torch.cat([f, f], -1)
        _ROPE_CACHE[key] = (f.cos()[None, None].to(dtype), f.sin()[None, None].to(dtype))
    return _ROPE_CACHE[key]

def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin

class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__(); self.gate = nn.Linear(d, ff, bias=False); self.up = nn.Linear(d, ff, bias=False); self.down = nn.Linear(ff, d, bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class Attention(nn.Module):
    """Multi-head attention with learnable QK-RMSNorm. rope_base=None -> no positions (set / cross attention)."""
    def __init__(self, d, n_heads, rope_base=None, cross=False):
        super().__init__(); self.h = n_heads; self.hd = d // n_heads; self.rope_base = rope_base; self.cross = cross
        if cross: self.q = nn.Linear(d, d, bias=False); self.kv = nn.Linear(d, 2 * d, bias=False)
        else: self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.hd); self.k_norm = RMSNorm(self.hd); self.out = nn.Linear(d, d, bias=False)
    def forward(self, x, key_mask, ctx=None):
        B, Lq, D = x.shape
        if self.cross:
            Lk = ctx.shape[1]
            q = self.q(x).view(B, Lq, self.h, self.hd).transpose(1, 2)
            k, v = self.kv(ctx).view(B, Lk, 2, self.h, self.hd).unbind(2); k, v = k.transpose(1, 2), v.transpose(1, 2)
        else:
            q, k, v = self.qkv(x).view(B, Lq, 3, self.h, self.hd).unbind(2); q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_base is not None:
            cos, sin = rope_cos_sin(Lq, self.hd, self.rope_base, x.device, q.dtype); q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(B, Lq, D))

class EncoderBlock(nn.Module):
    def __init__(self, d, h, ff, rope_base):
        super().__init__(); self.n1 = RMSNorm(d); self.attn = Attention(d, h, rope_base); self.n2 = RMSNorm(d); self.ff = SwiGLU(d, ff)
    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask); return x + self.ff(self.n2(x))

class Backbone(nn.Module):
    def __init__(self, vocab, d, n_layers, h, ff, rope_base):
        super().__init__(); self.d_model = d
        self.embedding = nn.Embedding(vocab, d, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList([EncoderBlock(d, h, ff, rope_base) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d); self.mlm_norm = RMSNorm(d); self.mlm_bias = nn.Parameter(torch.zeros(vocab))
        self.apply(init_weights)
        with torch.no_grad():
            for b in self.blocks:
                b.attn.out.weight.mul_(1 / math.sqrt(2 * n_layers)); b.ff.down.weight.mul_(1 / math.sqrt(2 * n_layers))
    def encode(self, ids, mask, depth=None):
        x = self.embedding(ids)
        for b in self.blocks[:depth]: x = b(x, mask)
        return x if depth is not None else self.final_norm(x)
    def mlm_logits(self, ids, labels):
        h = self.encode(ids, ids.ne(PAD_ID)); sel = labels.ne(-100)
        return F.linear(self.mlm_norm(h[sel]), self.embedding.weight, self.mlm_bias), labels[sel]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.padding_idx is not None:
            with torch.no_grad(): m.weight[m.padding_idx].zero_()

class CrossBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__(); self.nq = RMSNorm(d); self.attn = Attention(d, h, None, cross=True); self.n2 = RMSNorm(d); self.ff = SwiGLU(d, ff)
    def forward(self, x, state_h, state_mask):
        x = x + self.attn(self.nq(x), state_mask, ctx=state_h); return x + self.ff(self.n2(x))

class SetBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__(); self.n1 = RMSNorm(d); self.attn = Attention(d, h, None); self.n2 = RMSNorm(d); self.ff = SwiGLU(d, ff)
    def forward(self, x, opt_mask):
        x = x + self.attn(self.n1(x), opt_mask); return x + self.ff(self.n2(x))

TASK_TO_ID = {'choice': 0, 'noul': 1, 'score': 2}
MASKED_LOGIT = -1e4   # applied in fp32 only

class DecisionModel(nn.Module):
    def __init__(self, backbone, branch_layers, cross_layers, set_layers, h, ff):
        super().__init__(); d = backbone.d_model; self.backbone = backbone; self.branch_layers = branch_layers
        self.branch_norm = RMSNorm(d); self.task_embedding = nn.Embedding(len(TASK_TO_ID), d)
        self.cross = nn.ModuleList([CrossBlock(d, h, ff) for _ in range(cross_layers)])
        self.sets = nn.ModuleList([SetBlock(d, h, ff) for _ in range(set_layers)])
        self.out_norm = RMSNorm(d); self.scorer = nn.Linear(d, 1, bias=False)
        for mod in [self.task_embedding, self.cross, self.sets, self.scorer]: mod.apply(init_weights)
    def encode_state(self, ids, mask): return self.backbone.encode(ids, mask)
    def encode_branches(self, branch_ids, branch_mask, opt_mask):
        Q, O, L = branch_ids.shape; flat = opt_mask.reshape(-1)
        ids = branch_ids.reshape(Q * O, L)[flat]; m = branch_mask.reshape(Q * O, L)[flat]   # encode real options only
        h = self.backbone.encode(ids, m, depth=self.branch_layers)
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        out = torch.zeros(Q * O, h.shape[-1], device=h.device, dtype=pooled.dtype); out[flat] = pooled
        return self.branch_norm(out.view(Q, O, -1))
    def score(self, state_h, state_mask, q2s, branch_ids, branch_mask, opt_mask, task_ids):
        """state_h: [S,L,D] cached states; q2s: [Q] index of each question's state."""
        x = self.encode_branches(branch_ids, branch_mask, opt_mask) + self.task_embedding(task_ids)[:, None, :]
        sh, sm = state_h[q2s], state_mask[q2s]
        for blk in self.cross: x = blk(x, sh, sm)
        for blk in self.sets: x = blk(x, opt_mask)
        logits = self.scorer(self.out_norm(x)).squeeze(-1).float()
        return logits.masked_fill(~opt_mask, MASKED_LOGIT)
    def forward(self, b):
        return self.score(self.encode_state(b['state_ids'], b['state_mask']), b['state_mask'], b['q2s'],
                          b['branch_ids'], b['branch_mask'], b['opt_mask'], b['task_ids'])

class DeepFusionModel(nn.Module):
    """State encoded once: per-layer state keys/values are cached. Each (question, option) branch runs all
    backbone layers attending to [cached state ; own tokens]; RoPE positions continue after the state.
    State tokens never attend to branches, so one state encoding serves every question and option."""
    def __init__(self, backbone, set_layers, h, ff):
        super().__init__(); d = backbone.d_model; self.backbone = backbone
        self.branch_norm = RMSNorm(d); self.task_embedding = nn.Embedding(len(TASK_TO_ID), d)
        self.sets = nn.ModuleList([SetBlock(d, h, ff) for _ in range(set_layers)])
        self.out_norm = RMSNorm(d); self.scorer = nn.Linear(d, 1, bias=False)
        for mod in [self.task_embedding, self.sets, self.scorer]: mod.apply(init_weights)
    @staticmethod
    def _qkv(att, x):
        B, L, _ = x.shape
        q, k, v = att.qkv(x).view(B, L, 3, att.h, att.hd).unbind(2); q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return att.q_norm(q), att.k_norm(k), v
    def encode_state_kv(self, ids, mask):
        x = self.backbone.embedding(ids); kvs = []; L = ids.shape[1]
        for blk in self.backbone.blocks:
            att = blk.attn; q, k, v = self._qkv(att, blk.n1(x))
            cos, sin = rope_cos_sin(L, att.hd, att.rope_base, x.device, q.dtype); q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            kvs.append((k, v))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
            x = x + att.out(y.transpose(1, 2).reshape(x.shape[0], L, -1)); x = x + blk.ff(blk.n2(x))
        return kvs
    def score(self, kvs, state_mask, b):
        Q, O, L = b['branch_ids'].shape; flat = b['opt_mask'].reshape(-1)
        ids = b['branch_ids'].reshape(Q * O, L)[flat]; m = b['branch_mask'].reshape(Q * O, L)[flat]
        srow = b['q2s'].repeat_interleave(O)[flat]
        pos = state_mask.sum(1)[srow][:, None] + torch.arange(L, device=ids.device)[None]
        x = self.backbone.embedding(ids); km = torch.cat([state_mask[srow], m], 1)[:, None, None, :]
        for blk, (ks, vs) in zip(self.backbone.blocks, kvs):
            att = blk.attn; q, k, v = self._qkv(att, blk.n1(x))
            cos, sin = rope_cos_sin(int(pos.max()) + 1, att.hd, att.rope_base, x.device, q.dtype)
            cos, sin = cos[0, 0][pos][:, None], sin[0, 0][pos][:, None]; q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            y = F.scaled_dot_product_attention(q, torch.cat([ks[srow], k], 2), torch.cat([vs[srow], v], 2), attn_mask=km)
            x = x + att.out(y.transpose(1, 2).reshape(x.shape[0], L, -1)); x = x + blk.ff(blk.n2(x))
        h = self.backbone.final_norm(x); pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        out = torch.zeros(Q * O, h.shape[-1], device=h.device, dtype=pooled.dtype); out[flat] = pooled
        z = self.branch_norm(out.view(Q, O, -1)) + self.task_embedding(b['task_ids'])[:, None, :]
        for blk in self.sets: z = blk(z, b['opt_mask'])
        return self.scorer(self.out_norm(z)).squeeze(-1).float().masked_fill(~b['opt_mask'], MASKED_LOGIT)
    def forward(self, b):
        return self.score(self.encode_state_kv(b['state_ids'], b['state_mask']), b['state_mask'], b)

# ---- inference wrapper ----
def head_tail(ids, n):
    if len(ids) <= n: return ids
    h = (n * 3) // 4; return ids[:h] + ids[len(ids) - (n - h):]


class S1CED:
    def __init__(self, release=None, weights=None, head=None, calibration=None, threads=None, int8=False, device='cpu'):
        from tokenizers import Tokenizer
        rel = resolve_release(release); meta = json.loads((rel / 'config.json').read_text(encoding='utf-8')); cfg = meta['config']
        if threads: torch.set_num_threads(threads)
        self.cfg = cfg; self.head = head or cfg.get('head', 'late_fusion')
        self.tok = Tokenizer.from_file(str(rel / 'tokenizer.json'))
        sid = [self.tok.token_to_id(t) for t in SPECIAL_TOKENS]
        assert sid[:5] == [0, 1, 2, 3, 4], 'unexpected tokenizer special ids'
        self.CLS, self.SEP, self.Q, self.OPT = sid[2], sid[3], sid[6], sid[7]
        bb = Backbone(meta['vocab_size'], cfg['d_model'], cfg['n_layers'], cfg['n_heads'], cfg['d_ff'], cfg['rope_base'])
        if self.head == 'deep_fusion': m = DeepFusionModel(bb, cfg['set_layers'], cfg['n_heads'], cfg['d_ff'])
        else: m = DecisionModel(bb, cfg['branch_layers'], cfg['cross_layers'], cfg['set_layers'], cfg['n_heads'], cfg['d_ff'])
        wpath = Path(weights) if weights else rel / 'model.safetensors'
        if wpath.suffix == '.safetensors':
            from safetensors.torch import load_file; sd = load_file(str(wpath))
        else:
            ck = torch.load(wpath, map_location='cpu', weights_only=False); sd = ck['model'] if 'model' in ck else ck
        m.load_state_dict(sd, strict=True); m.eval()
        self.device = torch.device(device)
        if int8 and self.device.type != 'cpu': raise ValueError('--int8 is CPU-only')
        if int8:   # dynamic int8 for Linear layers; this torch API is deprecated upstream (torchao replaces it) but still works
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); m = torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        self.model = m.to(self.device)
        # calibration belongs to the weights it was fitted on: use the release file only with the release weights
        cpath = Path(calibration) if calibration else (rel / 'calibration.json' if not weights else None)
        if cpath and cpath.exists():
            cal = json.loads(cpath.read_text(encoding='utf-8')); self.T = float(cal['temperature']); self.threshold = float(cal['prob_threshold'])
        else:
            self.T, self.threshold = 1.0, None
            print('note: no calibration for these weights; temperature 1.0, no prediction sets', file=sys.stderr)
        self.lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model=None, **kwargs):
        """S1CED.from_pretrained('syedazeez/s1ced-v3-deep') or a local release folder; kwargs as for __init__."""
        return cls(model, **kwargs)

    def _batch(self, state, questions):
        n_st = self.cfg['max_state_len'] - 2; n_br = self.cfg['max_branch_len'] - 5
        st = [self.CLS] + head_tail(self.tok.encode(state).ids, n_st) + [self.SEP]
        qids = [e.ids for e in self.tok.encode_batch([q['question'] for q in questions])]
        branches = []
        for q, qi in zip(questions, qids):
            row = []
            for oi in (e.ids for e in self.tok.encode_batch(q['options'])):
                ob = min(len(oi), max(16, n_br - len(qi))); qb = n_br - ob
                row.append([self.CLS, self.Q] + head_tail(qi, qb) + [self.SEP, self.OPT] + oi[:ob] + [self.SEP])
            branches.append(row)
        Qn = len(questions); O = max(len(r) for r in branches); Lb = max(len(x) for r in branches for x in r)
        bids = torch.full((Qn, O, Lb), PAD_ID, dtype=torch.long); om = torch.zeros(Qn, O, dtype=torch.bool)
        for i, r in enumerate(branches):
            om[i, :len(r)] = True
            for j, x in enumerate(r): bids[i, j, :len(x)] = torch.tensor(x)
        sids = torch.tensor([st])
        b = {'state_ids': sids, 'state_mask': sids != PAD_ID, 'q2s': torch.zeros(Qn, dtype=torch.long),
             'branch_ids': bids, 'branch_mask': bids != PAD_ID, 'opt_mask': om,
             'task_ids': torch.tensor([TASK_TO_ID.get(q.get('type', 'choice'), 0) for q in questions])}
        return {k: v.to(self.device) for k, v in b.items()}

    def decide(self, state, questions):
        qs = []
        for q in questions:
            q = dict(q); q.setdefault('type', 'choice')
            if not q.get('options'):
                if q['type'] != 'noul': raise ValueError(f'question needs options: {q}')
                q['options'] = ['no', 'yes']
            qs.append(q)
        b = self._batch(state, qs)
        with self.lock, torch.inference_mode(): z = self.model(b)
        p = (z.float().cpu() / self.T).softmax(-1); out = []
        for i, q in enumerate(qs):
            pi = p[i, :len(q['options'])].tolist(); top = int(np.argmax(pi))
            r = {'type': q['type'], 'selected': q['options'][top], 'confidence': round(pi[top], 4),
                 'probabilities': {o: round(x, 4) for o, x in zip(q['options'], pi)}}
            if self.threshold is not None:
                ps = [o for o, x in zip(q['options'], pi) if x >= self.threshold]
                r['prediction_set'] = ps; r['should_escalate'] = len(ps) != 1
            low = [o.lower() for o in q['options']]
            if q['type'] == 'noul' and 'yes' in low: r['p_yes'] = round(pi[low.index('yes')], 4)
            out.append(r)
        return out


def serve(engine, host, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode(); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            self._send(200, {'ok': True, 'head': engine.head}) if self.path == '/health' else self._send(404, {'error': 'not found'})
        def do_POST(self):
            if self.path != '/decide': return self._send(404, {'error': 'not found'})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                t = time.perf_counter(); res = engine.decide(req['state'], req['questions'])
                self._send(200, {'results': res, 'latency_ms': round((time.perf_counter() - t) * 1000, 2)})
            except Exception as e:
                self._send(400, {'error': repr(e)})
        def log_message(self, *a): pass
    print(f'serving S1-CED ({engine.head}) on http://{host}:{port}  POST /decide  GET /health', file=sys.stderr)
    ThreadingHTTPServer((host, port), H).serve_forever()


def bench(engine, reps=20):
    words = ('the policy says the customer requested a refund for the item delivered days ago and the amount was '
             'recorded in the log with several unrelated entries ').split()
    for L in (128, 256, 512):
        state = ' '.join(words[i % len(words)] for i in range(int(L * 0.8)))
        for nq in (1, 8):
            qs = [{'question': f'Which action applies to check {i}?', 'options': ['allow', 'deny', 'escalate', 'review']} for i in range(nq)]
            engine.decide(state, qs); ts = []
            for _ in range(reps):
                t = time.perf_counter(); engine.decide(state, qs); ts.append((time.perf_counter() - t) * 1000)
            print(f'~{L} state tokens, {nq} question(s) x 4 options: p50 {np.percentile(ts, 50):.1f} ms, p95 {np.percentile(ts, 95):.1f} ms')


def main():
    ap = argparse.ArgumentParser(prog='s1ced', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', '--release', dest='model',
                    help='Hugging Face repo id or release folder (default: $S1CED_MODEL, else syedazeez/s1ced-v3-deep)')
    ap.add_argument('--weights', help='alternative weights (.safetensors or training checkpoint .pt)')
    ap.add_argument('--head', choices=['deep_fusion', 'late_fusion'], help='override head type (needed for v2 checkpoints)')
    ap.add_argument('--calibration', help='calibration.json fitted for these weights')
    ap.add_argument('--threads', type=int, help='torch CPU threads (default: torch decides)')
    ap.add_argument('--int8', action='store_true', help='dynamic int8 quantization of linear layers (CPU only)')
    ap.add_argument('--device', default='cpu', help='cpu or cuda')
    ap.add_argument('--input', help='request JSON file ("-" for stdin)')
    ap.add_argument('--serve', action='store_true'); ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    if not (a.input or a.serve or a.bench): ap.error('choose --input, --serve or --bench (or run s1ced-playground for the web UI)')
    eng = S1CED(a.model, a.weights, a.head, a.calibration, a.threads, a.int8, a.device)
    if a.bench: bench(eng)
    if a.input:
        req = json.load(sys.stdin if a.input == '-' else open(a.input, encoding='utf-8-sig'))
        print(json.dumps(eng.decide(req['state'], req['questions']), indent=2))
    if a.serve: serve(eng, a.host, a.port)


if __name__ == '__main__':
    main()
