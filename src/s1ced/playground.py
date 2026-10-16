#!/usr/bin/env python3
"""S1-CED decision playground: the local release, Vega (a second local model) and TypeSafe's Jev side by side.

  s1ced-playground                          # downloads the model once, opens http://127.0.0.1:8770
  s1ced-playground --model ./release        # a local release folder instead of Hugging Face
  s1ced-playground --port 9000 --no-browser
  s1ced-playground --examples my_cases.json # your own scenarios in the sidebar
  s1ced-playground --vega-preload           # load Vega at startup (otherwise it loads when first picked)

Requests use the TypeSafe System One shape (POST /v1/systemone) for every model:
  {"state": "..." | {...} | [...],
   "questions": {"<id>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
                 "<id>": {"type": "choice", "instructions": "...", "criteria": {"option": "description or null", ...}},
                 "<id>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}}}
and the local models' answers are reshaped to the same response shape (answers / probabilities / confidence).
POST /api/decide takes target local, vega, remote, both (local + remote) or all (every model; remote only with a key).

TypeSafe key: TYPESAFE_API_KEY in the environment wins. Otherwise the page asks for it the first time the
remote model is used. The key stays in this server process and is never sent back to the browser. If you tick
"remember", it is stored encrypted with Windows DPAPI (readable only by your Windows user) under
%APPDATA%\\s1ced_playground\\. On macOS and Linux it is a file readable only by your user under
~/.config/s1ced_playground/.
"""
import argparse, errno, json, os, sys, time, math, socket, threading, uuid, webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlreq, error as urlerr
from urllib.parse import urlparse

from .infer import S1CED
from .vega import VegaRunner

PAGE = Path(__file__).resolve().parent / 'playground.html'
EXAMPLES = Path(__file__).resolve().parent / 'examples.json'   # extra scenarios with expected answers
DEFAULT_PORT = 8770
MAX_BODY = 2_000_000                   # bytes; Jev's 64K-token request context is far below this
REMOTE_PRICE_IN = 0.042 / 1e6          # USD per input token for Jev (output is free)


# ---------------- key storage ----------------
class KeyStore:
    """Holds the TypeSafe key in memory; optionally persists it encrypted for the current OS user."""

    def __init__(self):
        base = Path(os.environ.get('APPDATA') or Path.home() / '.config') / 's1ced_playground'
        self.path = base / ('typesafe_key.dpapi' if os.name == 'nt' else 'typesafe_key')
        self.key, self.source = None, None
        self.lock = threading.RLock()
        self.env_key = os.environ.get('TYPESAFE_API_KEY', '').strip()
        if os.environ.get('TYPESAFE_API_KEY'):
            self.key, self.source = os.environ['TYPESAFE_API_KEY'].strip(), 'environment variable'
        elif self.path.exists():
            try:
                raw = self.path.read_bytes()
                self.key = (_dpapi(raw, protect=False) if os.name == 'nt' else raw).decode().strip()
                self.source = 'saved (encrypted)' if os.name == 'nt' else 'saved (user-only file)'
            except Exception as e:
                print(f'note: could not read saved key ({e!r}); the page will ask again', file=sys.stderr)

    def status(self):
        k = self.key
        return {'set': bool(k), 'source': self.source, 'masked': f'{k[:6]}...{k[-4:]}' if k and len(k) > 12 else ('set' if k else None),
                'remembered': self.path.exists(), 'store': str(self.path), 'from_env': bool(self.env_key)}

    def set(self, key, remember):
        with self.lock:
            if self.env_key: raise ValueError('TYPESAFE_API_KEY is set; manage that environment variable to change the key.')
            if not valid_key(key): raise ValueError('API keys must contain only printable ASCII characters, without spaces.')
            source = 'this session'
            if remember:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                data = _dpapi(key.encode(), protect=True) if os.name == 'nt' else key.encode()
                temp = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
                try:
                    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'wb') as f: f.write(data)
                    os.replace(temp, self.path)
                finally:
                    if temp.exists(): temp.unlink()
                source = 'saved (encrypted)' if os.name == 'nt' else 'saved (user-only file)'
            elif self.path.exists(): self.path.unlink()
            self.key, self.source = key, source

    def forget(self):
        with self.lock:
            if self.path.exists(): self.path.unlink()
            self.key = self.env_key or None
            self.source = 'environment variable' if self.env_key else None


def valid_key(key):
    return isinstance(key, str) and 0 < len(key) <= 400 and all(33 <= ord(c) <= 126 for c in key)


def _dpapi(data, protect):
    """Windows CryptProtectData / CryptUnprotectData, scoped to the current user."""
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.argtypes = [ctypes.POINTER(BLOB), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(BLOB)]
    fn.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]; kernel32.LocalFree.restype = ctypes.c_void_p
    buf = ctypes.create_string_buffer(data, len(data))
    src, out = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), BLOB()
    desc = ctypes.c_wchar_p('S1-CED playground TypeSafe key') if protect else None
    if not fn(ctypes.byref(src), desc, None, None, None, 0x1, ctypes.byref(out)):   # 0x1 = CRYPTPROTECT_UI_FORBIDDEN
        raise ctypes.WinError()
    try: return ctypes.string_at(out.pbData, out.cbData)
    finally: kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


# ---------------- local model: decisions shape <-> S1CED.decide ----------------
def as_text(x):
    """state / instructions / criteria may be a string, object or array; the local model only reads text."""
    if x is None: return ''
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def run_local(eng, req, include_criteria=False):
    state = as_text(req.get('state'))
    qs = req.get('questions') or {}
    if not isinstance(qs, dict) or not qs: raise ValueError('questions must be a non-empty object keyed by question id')
    max_opt = eng.cfg['max_options']; n_state = eng.cfg['max_state_len'] - 2; n_branch = eng.cfg['max_branch_len'] - 5
    warnings, asks, meta = [], [], []
    st_len = len(eng.tok.encode(state).ids)
    if st_len > n_state:
        warnings.append(f'State is {st_len} tokens; the local model keeps the first 3/4 and last 1/4 of {n_state} tokens.')
    for qid, q in qs.items():
        if not isinstance(q, dict): raise ValueError(f'{qid}: each question must be an object with type and instructions')
        t = q.get('type', 'choice'); text = as_text(q.get('instructions')); crit = q.get('criteria')
        if t == 'noul':
            options = ['no', 'yes']
            if include_criteria and isinstance(crit, dict):
                text += f" (yes = {as_text(crit.get('true'))}; no = {as_text(crit.get('false'))})"
        elif t == 'choice':
            if not isinstance(crit, dict) or len(crit) < 2: raise ValueError(f'{qid}: choice criteria must map at least 2 options to descriptions')
            options = list(crit)
            if include_criteria and any(v for v in crit.values()):
                text += ' Options: ' + '; '.join(f'{k} = {as_text(v)}' for k, v in crit.items() if v) + '.'
        elif t == 'score':
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10: raise ValueError(f'{qid}: score criteria must be a list of 2 to 10 levels')
            options = [as_text(c) for c in crit]
            if len(set(options)) != len(options): raise ValueError(f'{qid}: score levels must be distinct')
        else:
            raise ValueError(f'{qid}: unknown type {t!r} (noul, choice or score)')
        if len(options) > max_opt:
            raise ValueError(f'{qid}: {len(options)} options; the local model handles at most {max_opt}')
        q_len = len(eng.tok.encode(text).ids)
        for o in options:
            o_len = len(eng.tok.encode(o).ids)
            if q_len + o_len > n_branch:          # _batch trims the question or the option whenever they do not both fit
                warnings.append(f'{qid}: question plus option "{o[:40]}" exceeds {n_branch} tokens; the local model trims it.'); break
        asks.append({'question': text, 'options': options, 'type': t}); meta.append((qid, t, options))
    t0 = time.perf_counter(); res = eng.decide(state, asks); ms = (time.perf_counter() - t0) * 1000
    answers = {}
    for (qid, t, options), r in zip(meta, res):
        p = r['probabilities']
        extra = {'top_prob': r['confidence'], 'prediction_set': r.get('prediction_set'), 'should_escalate': r.get('should_escalate')}
        if t == 'noul':
            a = {'type': 'noul', 'noul': p['yes']}
        elif t == 'choice':
            a = {'type': 'choice', 'choice': r['selected'], 'probabilities': p, 'confidence': r['confidence']}
        else:
            probs = {str(i): p[o] for i, o in enumerate(options)}
            a = {'type': 'score', 'score': round(sum(i * v for i, v in enumerate(probs.values())), 4),
                 'legend': {str(i): o for i, o in enumerate(options)}, 'probabilities': probs, 'confidence': r['confidence']}
        answers[qid] = {**a, 'local': extra}
    note = ('confidence = top probability; no calibration for these weights, so no prediction sets' if eng.threshold is None else
            f'confidence = top probability; escalate when the conformal set (p >= {eng.threshold:.3f}) is not a single option')
    return {'model': f's1ced/{eng.cfg["run_name"]} ({eng.head}, {eng.device})', 'answers': answers,
            'usage': {'input_tokens': min(st_len, n_state) + 2, 'output_tokens': 0},
            'latency_ms': round(ms, 1), 'warnings': warnings, 'confidence_note': note}


# ---------------- remote model ----------------
def run_remote(keys, url, model, req):
    key = keys.key
    if not key: return {'error': 'no_key', 'message': 'No TypeSafe key set.'}
    if not valid_key(key): return {'error': 'invalid_key', 'message': 'The configured API key contains invalid characters.'}
    body = {'model': model, 'state': req.get('state'), 'questions': req.get('questions')}
    r = urlreq.Request(url, data=json.dumps(body).encode(), method='POST', headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urlreq.urlopen(r, timeout=60) as resp: out = json.loads(resp.read())
        if not isinstance(out, dict): raise ValueError('remote response must be a JSON object')
        if out.get('error'):
            err = out['error']
            code = err.get('code') if isinstance(err, dict) else None
            return {'error': 'unauthorized' if str(code) == '401' else 'remote', 'status': code,
                    'message': str(err.get('message', err)) if isinstance(err, dict) else str(err),
                    'latency_ms': round((time.perf_counter() - t0) * 1000, 1)}
        if not isinstance(out.get('answers'), dict): raise ValueError('remote response is missing an answers object')
        for answer in out['answers'].values():
            if not isinstance(answer, dict): raise ValueError('each remote answer must be an object')
            # Local conformal metadata is produced only by our local adapter.
            answer.pop('local', None)
            for field in ('noul', 'score', 'confidence'):
                value = answer.get(field)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f'remote {field} must be a finite number')
                if field != 'score' and value is not None and not 0 <= value <= 1:
                    raise ValueError(f'remote {field} must be between 0 and 1')
            if answer.get('choice') is not None and not isinstance(answer['choice'], str):
                raise ValueError('remote choice must be a string')
            probs = answer.get('probabilities')
            if probs is not None and (not isinstance(probs, dict) or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())):
                raise ValueError('remote probabilities must be finite numbers between 0 and 1')
        if out.get('usage') is not None and not isinstance(out['usage'], dict): raise ValueError('remote usage must be an object')
    except urlerr.HTTPError as e:
        detail = e.read().decode(errors='replace')[:2000]
        try:   # TypeSafe errors: {"detail": {"error_type": ..., "message": ...}}
            d = json.loads(detail).get('detail')
            detail = d.get('message') or detail if isinstance(d, dict) else str(d) if d else detail
        except (ValueError, AttributeError): pass
        return {'error': 'unauthorized' if e.code == 401 else f'http_{e.code}', 'status': e.code, 'message': detail,
                'latency_ms': round((time.perf_counter() - t0) * 1000, 1)}
    except (urlerr.URLError, TimeoutError, OSError) as e:
        return {'error': 'network', 'message': str(e.reason) if isinstance(e, urlerr.URLError) else type(e).__name__}
    except Exception as e:
        return {'error': 'remote_response', 'message': f'Invalid or incomplete remote response ({type(e).__name__}).'}
    out['latency_ms'] = round((time.perf_counter() - t0) * 1000, 1)
    u = out.get('usage') or {}
    if isinstance(u.get('input_tokens'), (int, float)) and math.isfinite(u['input_tokens']):
        out['cost_usd_estimate'] = round(u['input_tokens'] * REMOTE_PRICE_IN, 8)
    return out


def check_key(key, remote_url):
    """TypeSafe's model list (GET /v1/models) needs a valid key but runs no inference, so the check costs nothing.
    Other providers' keys are accepted as given and checked on first use."""
    if not valid_key(key): return False, 'API keys must contain only printable ASCII characters, without spaces.'
    u = urlparse(remote_url)
    if u.hostname != 'api.typesafe.ai': return True, None
    r = urlreq.Request(f'{u.scheme}://{u.netloc}/v1/models', headers={'Authorization': f'Bearer {key}'})
    try:
        with urlreq.urlopen(r, timeout=20) as resp: d = json.loads(resp.read())
        models = d.get('models', d) if isinstance(d, dict) else d
        return True, {'models': [m.get('name') for m in models if isinstance(m, dict)] if isinstance(models, list) else None}
    except urlerr.HTTPError as e:
        return False, f'TypeSafe rejected the key (HTTP {e.code}).'
    except (urlerr.URLError, TimeoutError, OSError) as e:
        return False, f'Could not reach TypeSafe to check the key: {e!r}'
    except Exception:
        return False, 'TypeSafe returned an invalid or incomplete key-check response.'


# ---------------- web server ----------------
TARGETS = {'local': ('local',), 'vega': ('vega',), 'remote': ('remote',), 'both': ('local', 'remote'),
           'all': ('local', 'vega', 'remote')}


def make_handler(eng, keys, a, vega=None):
    vega = vega or VegaRunner(enabled=False)
    allowed_hosts = {f'127.0.0.1:{a.port}', f'localhost:{a.port}', f'[::1]:{a.port}'}
    pool = ThreadPoolExecutor(4)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode(); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def _trusted(self):
            # only this page may call the API: blocks DNS rebinding (Host) and cross-site posts (Origin, JSON content type)
            if self.headers.get('Host') not in allowed_hosts: return False
            origin = self.headers.get('Origin')
            if origin and origin.split('://', 1)[-1] not in allowed_hosts: return False
            return self.command == 'GET' or self.headers.get('Content-Type', '').startswith('application/json')

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if not 0 <= n <= MAX_BODY: raise ValueError(f'body must be at most {MAX_BODY} bytes')
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            if not self._trusted(): return self._json(403, {'error': 'forbidden'})
            if self.path in ('/', '/index.html'):
                body = PAGE.read_bytes(); self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)
            elif self.path == '/api/examples':
                try: self._json(200, json.loads(EXAMPLES.read_text(encoding='utf-8-sig')) if EXAMPLES.exists() else {'groups': []})
                except (ValueError, OSError) as e: self._json(500, {'error': f'Could not load {EXAMPLES.name}: {e}'})
            elif self.path == '/api/health':
                self._json(200, {'local': {'model': f's1ced/{eng.cfg["run_name"]}', 'head': eng.head, 'device': str(eng.device),
                                           'max_state_tokens': eng.cfg['max_state_len'] - 2, 'max_options': eng.cfg['max_options'],
                                           'threshold': eng.threshold, 'temperature': eng.T},
                                 'remote': {'url': a.remote_url, 'model': a.remote_model, 'price_in_per_m': REMOTE_PRICE_IN * 1e6},
                                 'vega': vega.status(),
                                 'key': keys.status()})
            else:
                self._json(404, {'error': 'not found'})

        def do_POST(self):
            if not self._trusted(): return self._json(403, {'error': 'forbidden'})
            try: body = self._body()
            except (ValueError, RecursionError) as e: return self._json(400, {'error': f'bad request body: {e}'})
            if not isinstance(body, dict): return self._json(400, {'error': 'bad request body: expected a JSON object'})
            if self.path == '/api/decide':
                req, target = body.get('request') or {}, body.get('target', 'local')
                if not isinstance(req, dict) or not isinstance(req.get('questions'), dict) or not req['questions']:
                    return self._json(400, {'error': 'request must be an object with a non-empty "questions" object'})
                if target not in TARGETS: return self._json(400, {'error': f'unknown target {target!r}'})
                names = [n for n in TARGETS[target] if not (target == 'all' and n == 'remote' and not keys.key)]
                runners = {'local': lambda: self._local(req, bool(body.get('include_criteria'))),
                           'vega': lambda: vega.decide(req),
                           'remote': lambda: run_remote(keys, a.remote_url, a.remote_model, req)}
                jobs = {n: pool.submit(runners[n]) for n in names}
                results = {}
                for k, f in jobs.items():
                    try: results[k] = f.result()
                    except Exception: results[k] = {'error': k, 'message': f'{k} decision failed unexpectedly.'}
                self._json(200, results)
            elif self.path == '/api/key':
                key = str(body.get('key', '')).strip()
                if keys.env_key: return self._json(400, {'error': 'TYPESAFE_API_KEY is set; manage that environment variable to change the key.'})
                if not valid_key(key): return self._json(400, {'error': 'API keys must contain only printable ASCII characters, without spaces.'})
                ok, info = check_key(key, a.remote_url)
                if not ok: return self._json(400, {'error': info})
                try: keys.set(key, bool(body.get('remember')))
                except Exception as e: return self._json(500, {'error': f'Key works but could not be saved: {e!r}'})
                self._json(200, {'key': keys.status(), 'account': info})
            elif self.path == '/api/vega/load':
                vega.start(); self._json(200, {'vega': vega.status()})
            elif self.path == '/api/key/forget':
                try: keys.forget()
                except OSError: return self._json(500, {'error': 'Could not remove the saved key; the current key is unchanged.'})
                self._json(200, {'key': keys.status()})
            else:
                self._json(404, {'error': 'not found'})

        def _local(self, req, include_criteria):
            try: return run_local(eng, req, include_criteria)
            except Exception as e: return {'error': 'local', 'message': str(e)}
    return H


class PlaygroundServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can permit two live servers on the same port.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt': self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind(eng, keys, a, vega=None):
    """Bind --port, or without one the first free port from 8770; the handler's Host check needs the real port."""
    ports = [a.port] if a.port else range(DEFAULT_PORT, DEFAULT_PORT + 20)
    for port in ports:
        a.port = port
        try: return PlaygroundServer((a.host, port), make_handler(eng, keys, a, vega))
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES) and getattr(e, 'winerror', None) not in (10048, 10013): raise
    sys.exit(f'error: port {ports[0]} is in use; pick another with --port' if len(ports) == 1 else
             f'error: ports {ports[0]}-{ports[-1]} are all in use; pick one with --port')


def main():
    global EXAMPLES
    ap = argparse.ArgumentParser(prog='s1ced-playground', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', '--release', dest='model',
                    help='Hugging Face repo id or release folder (default: $S1CED_MODEL, else syedazeez/s1ced-v3-deep)')
    ap.add_argument('--weights'); ap.add_argument('--calibration')
    ap.add_argument('--head', choices=['deep_fusion', 'late_fusion'], help='override head type (needed for v2 checkpoints)')
    ap.add_argument('--threads', type=int); ap.add_argument('--device', default='cpu')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, help=f'default: first free port from {DEFAULT_PORT}')
    ap.add_argument('--examples', type=Path, help='scenario file to show instead of the bundled examples')
    ap.add_argument('--no-browser', action='store_true', help='do not open the page in a browser')
    ap.add_argument('--remote-url', default='https://api.typesafe.ai/v1/systemone')
    ap.add_argument('--remote-model', default='jev-latest', help='jev-latest, jev-preview, or a versioned id such as jev-1.14.0')
    ap.add_argument('--vega-model', help='Vega checkpoint: 0.8b (default), 4b, a Hub repo id or a local folder')
    ap.add_argument('--vega-device', help='cuda, mps or cpu (default: CUDA, then MPS, then CPU)')
    ap.add_argument('--vega-preload', action='store_true', help='load Vega at startup instead of on first use')
    ap.add_argument('--no-vega', action='store_true', help='hide Vega')
    a = ap.parse_args()
    if a.examples:
        if not a.examples.is_file(): ap.error(f'--examples: {a.examples} not found')
        EXAMPLES = a.examples.resolve()
    try:
        eng = S1CED(a.model, a.weights, a.head, a.calibration, a.threads, False, a.device)
    except FileNotFoundError as e:
        sys.exit(f'error: {e}')
    keys = KeyStore()
    vega = VegaRunner(a.vega_model, a.vega_device, enabled=not a.no_vega)
    if a.vega_preload: vega.start()
    srv = bind(eng, keys, a, vega); srv.daemon_threads = True
    url = f'http://{"127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host}:{a.port}'
    print(f'S1-CED playground ({eng.head}, {eng.device}) ready at {url}\n'
          f'vega: {vega.state}  remote: {a.remote_model}  key: {keys.status()["source"] or "not set"}\nPress Ctrl+C to stop.', file=sys.stderr, flush=True)
    if not a.no_browser: threading.Timer(0.5, webbrowser.open, (url,)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print('stopped', file=sys.stderr)
    finally: srv.server_close()


if __name__ == '__main__':
    main()
