"""Vega (vegaml) as a second local model: loaded in the background on first use, answers reshaped to the playground's
TypeSafe-style response shape.

Vega is a frozen Qwen3.5 reader plus a small trained decision engine (https://github.com/NandhaKishorM/vegaml).
The 0.8B download is about 1.8 GB and loading takes a minute or more, so nothing happens until the page asks for it.
It is fast on a GPU (about 0.2 s per decision on a laptop RTX 3050) and slow on CPU (several seconds per decision).
"""
import math, sys, threading, time


def to_vega(questions):
    """Playground questions -> vegaml questions: the yes/no type is called "boolean" there."""
    return {qid: {**q, 'type': 'boolean' if q.get('type') == 'noul' else q.get('type', 'choice')} for qid, q in questions.items()}


def _num(x):
    if type(x) not in (int, float) or not math.isfinite(x): raise ValueError('Vega returned a non-finite number')
    return float(x)


def reshape(questions, out):
    """vegaml.decide output -> {model, answers, usage, latency_ms}, matching run_local / run_remote."""
    answers = {}
    for qid, q in questions.items():
        a, t = out['answers'][qid], q.get('type', 'choice')
        probs = {str(k): _num(v) for k, v in a['probs'].items()}
        pset = a.get('prediction_set')
        if t == 'noul':
            if pset is not None: pset = ['yes' if p is True or p == 'True' else 'no' for p in pset]
            ans = {'type': 'noul', 'noul': probs['True']}
        elif t == 'choice':
            ans = {'type': 'choice', 'choice': a['choice'], 'probabilities': probs, 'confidence': _num(a['confidence'])}
        else:
            ans = {'type': 'score', 'score': round(_num(a['expected_score']), 4),
                   'legend': {str(i): c if isinstance(c, str) else str(c) for i, c in enumerate(q['criteria'])},
                   'probabilities': probs, 'confidence': _num(a['confidence'])}
        ans['vega'] = {'top_prob': _num(a['confidence']), 'prediction_set': pset, 'abstain': bool(a.get('abstain')),
                       'unbound': bool(a.get('unbound')), 'readout': a.get('readout', 'engine')}
        answers[qid] = ans
    return {'answers': answers, 'usage': {'input_tokens': out.get('n_tokens'), 'output_tokens': 0},
            'latency_ms': out.get('latency_ms')}


class VegaRunner:
    """Holds one vegaml model. States: off (disabled), unavailable (vegaml not importable), idle, loading, ready, error."""

    def __init__(self, model=None, device=None, enabled=True):
        self.model, self.device = model, device
        self.v, self.error, self.load_s = None, None, None
        self.state = 'idle' if enabled else 'off'
        self.lock, self.run_lock, self.ready = threading.Lock(), threading.Lock(), threading.Event()
        if enabled:
            try:
                import vegaml  # noqa: F401  (import cost is small; the weights load later)
            except Exception as e:
                self.state, self.error = 'unavailable', f'vegaml is not installed ({type(e).__name__}); run: pip install vegaml'

    def status(self):
        dev = getattr(getattr(self.v, 'engine', None), 'device', None)   # where vegaml actually put it
        dev = str(dev) if dev is not None else None
        return {'state': self.state, 'model': f'vega-{self.model or "0.8b"}', 'device': dev or self.device or 'auto',
                'error': self.error, 'load_s': self.load_s}

    def start(self):
        """Begin loading in a background thread; a no-op unless idle (or retrying after an error)."""
        with self.lock:
            if self.state not in ('idle', 'error'): return
            self.state, self.error = 'loading', None
            self.ready.clear()
        threading.Thread(target=self._load, name='vega-load', daemon=True).start()

    def _load(self):
        t0 = time.perf_counter()
        try:
            import vegaml
            kw = {'device': self.device} if self.device else {}
            v = vegaml.load(self.model, **kw)
            with self.lock: self.v, self.state, self.load_s = v, 'ready', round(time.perf_counter() - t0, 1)
            print(f'Vega {self.status()["model"]} ready on {self.status()["device"]} in {self.load_s} s', file=sys.stderr, flush=True)
        except Exception as e:
            with self.lock: self.state, self.error = 'error', f'Vega failed to load: {e}'
            print(self.error, file=sys.stderr, flush=True)
        finally:
            self.ready.set()

    def decide(self, req, timeout=900):
        if self.state == 'off': return {'error': 'vega_off', 'message': 'Vega is turned off for this server (--no-vega).'}
        if self.state == 'unavailable': return {'error': 'vega_unavailable', 'message': self.error}
        self.start()
        if not self.ready.wait(timeout): return {'error': 'vega_loading', 'message': 'Vega is still loading; try again shortly.'}
        if self.state != 'ready': return {'error': 'vega_load', 'message': self.error}
        questions = req.get('questions') or {}
        try:
            with self.run_lock:   # one decision at a time on the shared model
                out = self.v.decide(req.get('state') if req.get('state') is not None else '', to_vega(questions))
            res = reshape(questions, out)
        except Exception as e:
            return {'error': 'vega', 'message': str(e) or type(e).__name__}
        return {'model': f'{self.status()["model"]} ({self.status()["device"]})', **res}
