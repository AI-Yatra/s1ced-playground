"""S1-CED, Vega and Jev behind one contract (contract.py): the same request, answers, errors and limits rules.

    import s1ced
    m = s1ced.load('vega-0.8b')            # or 's1ced', 'jev-latest'
    out = m.decide(state, questions)       # or s1ced.decide({'model': 'vega-0.8b', 'state': ..., 'questions': ...})

Every adapter only translates; contract.check_request runs before it and contract.check_response after it, so no
model can accept a request another would reject or return an answer in a different shape.
"""
import json, os, sys, threading, time
from urllib import request as urlreq, error as urlerr

from .contract import DecisionError, check_request, check_response, option_labels, text_of, uncertainty

JEV_URL = 'https://api.typesafe.ai/v1/systemone'
JEV_PRICE_IN = 0.042 / 1e6          # USD per input token (output is free), docs.typesafe.ai/models
VEGA_SIZES = {'vega-0.8b': None, 'vega-4b': '4b'}
ALIASES = {'vega': 'vega-0.8b', 's1ced-v3-deep': 's1ced'}


class DecisionModel:
    """One model under the contract. Subclasses implement _run(req, allow_truncation) -> response without
    latency_ms; run() validates both sides and times the call."""
    name = description = release_date = None
    local = True

    def info(self):
        return {'name': self.name, 'description': self.description, 'release_date': self.release_date}

    def run(self, req, allow_truncation=False):
        """Answer a contract request. req['model'] may be omitted; if present it must name this model.
        allow_truncation lets a model with a shorter context shorten the input, with a warning, instead of refusing."""
        check_request(req, model_required=False)
        if req.get('model') and canonical(req['model']) != self.name and not (self.name == 'jev' or self.name.startswith('jev-')):
            raise DecisionError(422, 'invalid_request', f'This is {self.name}; the request asks for {req["model"]}.', ('model',))
        t0 = time.perf_counter()
        resp = self._run(req, allow_truncation)
        resp.setdefault('warnings', [])
        resp['latency_ms'] = round((time.perf_counter() - t0) * 1000, 1)
        return check_response(resp, req)

    def decide(self, state, questions, allow_truncation=False):
        return self.run({'state': state, 'questions': questions}, allow_truncation)


# ---------------- S1-CED ----------------
class S1CEDModel(DecisionModel):
    name = 's1ced'
    release_date = '2026-10-10'

    def __init__(self, engine=None, **kwargs):
        if engine is None:
            from .infer import S1CED
            engine = S1CED(kwargs.pop('release', None), **kwargs)
        self.eng = engine
        self.version = engine.cfg['run_name'].replace('_', '-')
        self.description = (f'S1-CED ({self.version}), a 38M-parameter decision encoder on {engine.device}. Reads up to '
                            f'{engine.cfg["max_state_len"] - 2} state tokens and {engine.cfg["max_options"]} options per question.')

    def _run(self, req, allow_truncation):
        eng = self.eng
        state = text_of(req['state'])
        max_opt, n_state, n_branch = eng.cfg['max_options'], eng.cfg['max_state_len'] - 2, eng.cfg['max_branch_len'] - 5
        warnings, asks, meta = [], [], []
        st_len = len(eng.tok.encode(state).ids)
        if st_len > n_state:
            msg = f'state is {st_len} tokens; {self.name} reads at most {n_state}.'
            if not allow_truncation: raise DecisionError(400, 'max_tokens_exceeded', msg + ' Shorten it, or allow truncation.', ('state',))
            warnings.append(msg + ' It kept the first 3/4 and the last 1/4.')
        for qid, q in req['questions'].items():
            t, text, crit = q['type'], text_of(q['instructions']), q.get('criteria')
            # Criteria descriptions are part of the question for every model, so they are always read.
            if t == 'noul':
                options = ['no', 'yes']
                if isinstance(crit, dict) and any(crit.values()):
                    text += f" (yes = {text_of(crit.get('true'))}; no = {text_of(crit.get('false'))})"
            elif t == 'choice':
                options = list(crit)
                if any(v for v in crit.values()):
                    text += ' Options: ' + '; '.join(f'{k} = {text_of(v)}' for k, v in crit.items() if v) + '.'
            else:
                options = [text_of(c) for c in crit]
            if len(options) > max_opt:
                raise DecisionError(400, 'too_many_options', f'{qid} has {len(options)} options; {self.name} handles at most {max_opt}.',
                                    ('questions', qid, 'criteria'))
            q_len = len(eng.tok.encode(text).ids)
            if any(q_len + len(eng.tok.encode(o).ids) > n_branch for o in options):
                msg = f'{qid}: the question plus its longest option is over {n_branch} tokens.'
                if not allow_truncation: raise DecisionError(400, 'max_tokens_exceeded', msg + ' Shorten it, or allow truncation.', ('questions', qid))
                warnings.append(msg + ' It was trimmed.')
            asks.append({'question': text, 'options': options, 'type': t}); meta.append((qid, t, options))
        res = eng.decide(state, asks)
        answers = {}
        for (qid, t, options), r in zip(meta, res):
            p, labels = r['probabilities'], option_labels(req['questions'][qid])
            pset = None if r.get('prediction_set') is None else [labels[options.index(o)] for o in r['prediction_set']]
            if t == 'noul':
                a = {'type': 'noul', 'noul': p['yes']}
            elif t == 'choice':
                a = {'type': 'choice', 'choice': r['selected'], 'probabilities': p, 'confidence': r['confidence']}
            else:
                probs = {str(i): p[o] for i, o in enumerate(options)}
                a = {'type': 'score', 'score': round(sum(i * v for i, v in enumerate(probs.values())), 4),
                     'legend': {str(i): o for i, o in enumerate(options)}, 'probabilities': probs, 'confidence': r['confidence']}
            answers[qid] = {**a, 'uncertainty': uncertainty(prediction_set=pset)}
        return {'model': self.version, 'answers': answers, 'warnings': warnings,
                'usage': {'input_tokens': min(st_len, n_state) + 2, 'output_tokens': 0}}


# ---------------- Vega ----------------
class VegaModel(DecisionModel):
    """Vega (vegaml, Apache 2.0) loads in a background thread on first use: the 0.8B is a 1.8 GB download."""
    release_date = '2026-10-10'

    def __init__(self, name='vega-0.8b', device=None):
        self.name, self.device = name, device
        self.description = (f'Vega {name[5:].upper()} by Nandakishor M: a frozen Qwen3.5 reader and a small decision engine. '
                            'Answers carry abstain and out-of-scope flags. Fast on a GPU, slow on CPU.')
        self.v, self.error, self.load_s = None, None, None
        self.state = 'idle'
        self.lock, self.run_lock, self.ready = threading.Lock(), threading.Lock(), threading.Event()
        try:
            import vegaml  # noqa: F401  (cheap; the weights load later)
        except Exception as e:
            self.state, self.error = 'unavailable', f'vegaml is not installed ({type(e).__name__}); run: pip install vegaml'

    def status(self):
        dev = getattr(getattr(self.v, 'engine', None), 'device', None)
        return {'state': self.state, 'model': self.name, 'device': str(dev) if dev is not None else self.device or 'auto',
                'error': self.error, 'load_s': self.load_s}

    def start(self):
        """Begin loading in the background; does nothing unless idle or failed."""
        with self.lock:
            if self.state not in ('idle', 'error'): return
            self.state, self.error = 'loading', None
            self.ready.clear()
        threading.Thread(target=self._load, name=f'{self.name}-load', daemon=True).start()

    def _load(self):
        t0 = time.perf_counter()
        try:
            import vegaml
            v = vegaml.load(VEGA_SIZES.get(self.name), **({'device': self.device} if self.device else {}))
            with self.lock: self.v, self.state, self.load_s = v, 'ready', round(time.perf_counter() - t0, 1)
            print(f'{self.name} ready on {self.status()["device"]} in {self.load_s} s', file=sys.stderr, flush=True)
        except Exception as e:
            with self.lock: self.state, self.error = 'error', f'{self.name} failed to load: {e}'
            print(self.error, file=sys.stderr, flush=True)
        finally:
            self.ready.set()

    def _run(self, req, allow_truncation, timeout=900):
        if self.state == 'unavailable': raise DecisionError(503, 'model_unavailable', self.error)
        self.start()
        if not self.ready.wait(timeout): raise DecisionError(503, 'model_loading', f'{self.name} is still loading; try again shortly.')
        if self.state != 'ready': raise DecisionError(503, 'model_unavailable', self.error)
        qs = req['questions']
        vq = {qid: {**q, 'type': 'boolean' if q['type'] == 'noul' else q['type']} for qid, q in qs.items()}
        try:
            with self.run_lock:   # one decision at a time on the shared model
                out = self.v.decide(req['state'], vq)
        except ValueError as e:   # vegaml refuses rather than truncates, which is the contract's default anyway
            raise DecisionError(400, 'max_tokens_exceeded' if 'token' in str(e).lower() else 'invalid_request', str(e))
        answers = {}
        for qid, q in qs.items():
            a, t, labels = out['answers'][qid], q['type'], option_labels(q)
            probs = {str(k): v for k, v in a['probs'].items()}
            pset = a.get('prediction_set')
            if pset is not None: pset = [('yes' if p in (True, 'True') else 'no') if t == 'noul' else str(p) for p in pset]
            if t == 'noul':
                ans = {'type': 'noul', 'noul': probs['True']}
            elif t == 'choice':
                ans = {'type': 'choice', 'choice': a['choice'], 'probabilities': probs, 'confidence': a['confidence']}
            else:
                ans = {'type': 'score', 'score': round(a['expected_score'], 4), 'legend': {k: text_of(c) for k, c in zip(labels, q['criteria'])},
                       'probabilities': probs, 'confidence': a['confidence']}
            ans['uncertainty'] = uncertainty(pset, bool(a.get('abstain')), bool(a.get('unbound')))
            answers[qid] = ans
        return {'model': self.name, 'answers': answers, 'usage': {'input_tokens': int(out.get('n_tokens') or 0), 'output_tokens': 0}}


# ---------------- Jev ----------------
def valid_key(key):
    return isinstance(key, str) and 0 < len(key) <= 400 and all(33 <= ord(c) <= 126 for c in key)


class JevModel(DecisionModel):
    """TypeSafe's hosted Jev. key is a string or a function returning one (so a key added later is picked up)."""
    local = False
    release_date = ''

    def __init__(self, name='jev-latest', key=None, url=JEV_URL, timeout=60):
        self.name, self.url, self.timeout = name, url, timeout
        self._key = key if callable(key) else (lambda: key or os.environ.get('TYPESAFE_API_KEY', '').strip() or None)
        self.description = f'TypeSafe Jev ({name}), hosted. Needs a TypeSafe API key; about $0.042 per million input tokens.'

    def _run(self, req, allow_truncation):
        key = self._key()
        if not key: raise DecisionError(401, 'authentication_error', 'No TypeSafe key set. Set TYPESAFE_API_KEY or add one in the playground.')
        if not valid_key(key): raise DecisionError(401, 'authentication_error', 'The TypeSafe key contains invalid characters.')
        body = {'model': self.name, 'state': req['state'], 'questions': req['questions']}
        r = urlreq.Request(self.url, data=json.dumps(body).encode(), method='POST',
                           headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        try:
            with urlreq.urlopen(r, timeout=self.timeout) as resp: out = json.loads(resp.read())
        except urlerr.HTTPError as e:
            raw = e.read().decode(errors='replace')[:2000]
            try:
                d = json.loads(raw).get('detail')
                msg, etype = (d.get('message') or raw, d.get('error_type')) if isinstance(d, dict) else (str(d) if d else raw, None)
            except (ValueError, AttributeError):
                msg, etype = raw or e.reason, None
            raise DecisionError(e.code if 400 <= e.code < 600 else 502, etype or ('authentication_error' if e.code == 401 else 'upstream_error'), msg)
        except (urlerr.URLError, TimeoutError, OSError) as e:
            raise DecisionError(502, 'network_error', f'Could not reach TypeSafe: {e.reason if isinstance(e, urlerr.URLError) else type(e).__name__}')
        except ValueError:
            raise DecisionError(502, 'invalid_model_response', 'TypeSafe returned a body that is not JSON.')
        if not isinstance(out, dict) or not isinstance(out.get('answers'), dict):
            raise DecisionError(502, 'invalid_model_response', 'TypeSafe returned no answers.')
        for a in out['answers'].values():
            if isinstance(a, dict): a['uncertainty'] = uncertainty()   # Jev reports confidence only
        u = out.get('usage') or {}
        if type(u.get('input_tokens')) is int: out['cost_usd'] = round(u['input_tokens'] * JEV_PRICE_IN, 8)
        return out

    def check_key(self, key):
        """GET /v1/models needs a valid key but runs no model, so checking a key costs nothing."""
        if not valid_key(key): return False, 'API keys must contain only printable ASCII characters, without spaces.'
        r = urlreq.Request(self.url.rsplit('/v1/', 1)[0] + '/v1/models', headers={'Authorization': f'Bearer {key}'})
        try:
            with urlreq.urlopen(r, timeout=20) as resp: d = json.loads(resp.read())
            return True, {'models': [m.get('name') for m in d.get('models', []) if isinstance(m, dict)]}
        except urlerr.HTTPError as e:
            return False, f'TypeSafe rejected the key (HTTP {e.code}).'
        except (urlerr.URLError, TimeoutError, OSError) as e:
            return False, f'Could not reach TypeSafe to check the key: {e!r}'
        except Exception:
            return False, 'TypeSafe returned an invalid or incomplete key-check response.'


# ---------------- registry ----------------
def canonical(name):
    name = (name or 's1ced').strip().lower()
    return ALIASES.get(name, name)


class Registry:
    """Builds each model on first use and keeps it. The playground server and s1ced.load() share this."""

    def __init__(self, s1ced=None, vega_device=None, vega=True, jev_key=None, jev_url=JEV_URL):
        self.s1ced_kwargs = s1ced or {}
        self.vega_device, self.vega_enabled = vega_device, vega
        self.jev_key, self.jev_url = jev_key, jev_url
        self.models, self.lock = {}, threading.Lock()

    def names(self):
        out = ['s1ced']
        if self.vega_enabled: out += list(VEGA_SIZES)
        return out + ['jev-latest', 'jev-preview']

    def get(self, name):
        name = canonical(name)
        with self.lock:
            if name in self.models: return self.models[name]
            if name == 's1ced':
                m = S1CEDModel(**self.s1ced_kwargs) if 'engine' not in self.s1ced_kwargs else S1CEDModel(self.s1ced_kwargs['engine'])
            elif name in VEGA_SIZES and self.vega_enabled:
                m = VegaModel(name, self.vega_device)
            elif name.startswith('jev-'):
                m = JevModel(name, self.jev_key, self.jev_url)
            else:
                raise DecisionError(404, 'model_not_found', f'Unknown model {name!r}. Available: {", ".join(self.names())}.', ('model',))
            self.models[name] = m
            return m

    def run(self, req, allow_truncation=False):
        """A full contract request, routed by its model field."""
        check_request(req)
        return self.get(req['model']).run(req, allow_truncation)

    def list(self):
        """GET /v1/models: every model this registry can serve, in TypeSafe's ModelMetadata shape. Listing never
        loads a model: unloaded local models are described from static text."""
        out = []
        for n in self.names():
            if n in self.models: out.append(self.models[n].info())
            elif n.startswith('jev-'): out.append(JevModel(n, self.jev_key, self.jev_url).info())
            elif n == 's1ced': out.append({'name': n, 'description': _S1CED_TEXT, 'release_date': S1CEDModel.release_date})
            else: out.append({'name': n, 'description': _VEGA_TEXT.format(n=n), 'release_date': VegaModel.release_date})
        return {'models': out}


_S1CED_TEXT = 'S1-CED, a 38M-parameter decision encoder. Fast on CPU; reads up to 510 state tokens.'
_VEGA_TEXT = ('Vega ({n}) by Nandakishor M: a frozen Qwen3.5 reader plus a small decision engine, with abstain and '
              'out-of-scope flags. Loads on first use; fast on a GPU.')

_default = None


def load(name='s1ced', **kwargs):
    """A model under the contract: load('s1ced'), load('vega-0.8b'), load('vega-4b') or load('jev-latest').
    kwargs: for s1ced, those of S1CED (release, device, ...); for vega, device; for jev, key."""
    name = canonical(name)
    if name == 's1ced': return S1CEDModel(**kwargs)
    if name in VEGA_SIZES:
        m = VegaModel(name, kwargs.get('device')); m.start(); return m
    if name.startswith('jev-'): return JevModel(name, kwargs.get('key'))
    raise DecisionError(404, 'model_not_found', f'Unknown model {name!r}. Use s1ced, vega-0.8b, vega-4b or a jev- model.')


def decide(request, allow_truncation=False):
    """s1ced.decide({'model': ..., 'state': ..., 'questions': ...}): one contract request, any model."""
    global _default
    if _default is None: _default = Registry()
    return _default.run(request, allow_truncation)
