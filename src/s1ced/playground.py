#!/usr/bin/env python3
"""S1-CED decision playground and local API: S1-CED, Vega and TypeSafe's Jev under one contract (contract.py).

  s1ced-playground                          # downloads S1-CED once, opens http://127.0.0.1:8770
  s1ced-playground --model ./release        # a local S1-CED release folder instead of Hugging Face
  s1ced-playground --port 9000 --no-browser
  s1ced-playground --examples my_cases.json # your own scenarios in the sidebar
  s1ced-playground --vega-preload           # load Vega at startup (otherwise it loads when first picked)

The same server is a local copy of TypeSafe's API, so any client written for Jev works against it:
  POST /v1/systemone   {"model": "s1ced" | "vega-0.8b" | "vega-4b" | "jev-latest", "state": ..., "questions": {...}}
  GET  /v1/models
For example TypeSafeClient(base_url="http://127.0.0.1:8770", api_key="local", model="vega-0.8b").

TypeSafe key: TYPESAFE_API_KEY in the environment wins. Otherwise the page asks for it the first time the
remote model is used. The key stays in this server process and is never sent back to the browser. If you tick
"remember", it is stored encrypted with Windows DPAPI (readable only by your Windows user) under
%APPDATA%\\s1ced_playground\\. On macOS and Linux it is a file readable only by your user under
~/.config/s1ced_playground/.
"""
import argparse, errno, gzip, json, os, sys, socket, threading, uuid, webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .contract import DecisionError, check_request, loads
from .models import JEV_PRICE_IN, JEV_URL, VEGA_SIZES, JevModel, Registry, canonical, valid_key

PAGE = Path(__file__).resolve().parent / 'playground.html'
EXAMPLES = Path(__file__).resolve().parent / 'examples.json'   # extra scenarios with expected answers
DEFAULT_PORT = 8770
MAX_BODY = 2_000_000                   # bytes; Jev's 64K-token request context is far below this
PLACEHOLDER_KEYS = {'', 'local', 'none', 'x'}   # what clients send as api_key when they only use local models


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




# ---------------- web server ----------------
def decode_body(raw, encoding):
    """Request bodies may be gzip or zstd compressed, as TypeSafe's SDKs send them."""
    encoding = (encoding or 'identity').strip().lower()
    if encoding == 'identity': return raw
    try:
        if encoding in ('gzip', 'x-gzip'): return gzip.decompress(raw)
        if encoding == 'zstd':
            try:
                from compression.zstd import decompress   # Python 3.14+
            except ImportError:
                try:
                    import zstandard
                    decompress = lambda b: zstandard.ZstdDecompressor().decompressobj().decompress(b)
                except ImportError: raise DecisionError(415, 'unsupported_encoding', 'zstd needs Python 3.14 or the zstandard package; '
                                                        'send the body uncompressed (TypeSafeClient(..., compression=False)).')
            return decompress(raw)
    except DecisionError: raise
    except Exception: raise DecisionError(400, 'invalid_encoding', f'The {encoding} body could not be decompressed.')
    raise DecisionError(415, 'unsupported_encoding', f'Content-Encoding {encoding!r} is not supported; use gzip, zstd or none.')


def make_handler(reg, keys, a):
    """reg: models.Registry. keys: KeyStore for the Jev key. a: port, remote_model (the Jev model the page shows),
    vega_model (the Vega size the page shows, or None when Vega is off)."""
    allowed_hosts = {f'127.0.0.1:{a.port}', f'localhost:{a.port}', f'[::1]:{a.port}'}
    pool = ThreadPoolExecutor(4)
    sides = {'local': 's1ced', 'vega': getattr(a, 'vega_model', None), 'remote': a.remote_model}
    targets = {'local': ('local',), 'vega': ('vega',), 'remote': ('remote',), 'both': ('local', 'remote'), 'all': ('local', 'vega', 'remote')}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode(); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

        def _error(self, e): self._json(e.status, e.body())

        def _trusted(self):
            # only local clients may call: blocks DNS rebinding (Host) and cross-site posts (Origin, JSON content type)
            if self.headers.get('Host') not in allowed_hosts: return False
            origin = self.headers.get('Origin')
            if origin and origin.split('://', 1)[-1] not in allowed_hosts: return False
            return self.command == 'GET' or self.headers.get('Content-Type', '').startswith('application/json')

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if not 0 <= n <= MAX_BODY: raise DecisionError(413, 'payload_too_large', f'The body must be at most {MAX_BODY} bytes.')
            raw = decode_body(self.rfile.read(n), self.headers.get('Content-Encoding'))
            if len(raw) > MAX_BODY: raise DecisionError(413, 'payload_too_large', f'The body must be at most {MAX_BODY} bytes.')
            return loads(raw or b'{}')

        def _model(self, name):
            """The model for a /v1 request. A Jev request uses the server's key, or else the caller's Bearer token,
            or else, with --jev-fallback, the named local model (the response then says which model answered)."""
            m = reg.get(name)
            if isinstance(m, JevModel) and not keys.key:
                bearer = (self.headers.get('Authorization') or '').removeprefix('Bearer ').strip()
                if bearer.lower() not in PLACEHOLDER_KEYS: return JevModel(m.name, bearer, m.url)
                if getattr(a, 'jev_fallback', None): return reg.get(a.jev_fallback)
            return m

        def do_GET(self):
            if not self._trusted(): return self._error(DecisionError(403, 'forbidden', 'Only local clients may call this server.'))
            if self.path in ('/', '/index.html'):
                body = PAGE.read_bytes(); self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)
            elif self.path == '/v1/models':
                self._json(200, reg.list())
            elif self.path == '/api/examples':
                try: self._json(200, json.loads(EXAMPLES.read_text(encoding='utf-8-sig')) if EXAMPLES.exists() else {'groups': []})
                except (ValueError, OSError) as e: self._json(500, {'error': f'Could not load {EXAMPLES.name}: {e}'})
            elif self.path == '/api/health':
                s1 = reg.get('s1ced'); eng = s1.eng
                vega = reg.get(sides['vega']).status() if sides['vega'] else {'state': 'off'}
                self._json(200, {'local': {'model': s1.version, 'device': str(eng.device), 'max_state_tokens': eng.cfg['max_state_len'] - 2,
                                           'max_options': eng.cfg['max_options'], 'threshold': eng.threshold},
                                 'remote': {'model': a.remote_model, 'price_in_per_m': JEV_PRICE_IN * 1e6},
                                 'vega': vega, 'key': keys.status()})
            else:
                self._json(404, {'error': 'not found'})

        def do_POST(self):
            if not self._trusted(): return self._error(DecisionError(403, 'forbidden', 'Only local clients may call this server.'))
            try: body = self._body()
            except DecisionError as e: return self._error(e)
            if self.path == '/v1/systemone':
                try:
                    req = check_request(body)
                    m = self._model(req['model'])
                    if m.name == canonical(req['model']) or isinstance(m, JevModel): return self._json(200, m.run(req))
                    out = m.run({**req, 'model': m.name})   # --jev-fallback: answered locally, and the response says so
                    out['warnings'].append(f'No TypeSafe key, so this {req["model"]} request was answered by {m.name} (--jev-fallback).')
                    self._json(200, out)
                except DecisionError as e: self._error(e)
                except Exception as e: self._error(DecisionError(500, 'internal_error', f'{type(e).__name__}: {e}'))
                return
            if not isinstance(body, dict): return self._json(400, {'error': 'expected a JSON object'})
            if self.path == '/api/decide':
                req, target = body.get('request'), body.get('target', 'local')
                if target not in targets: return self._json(400, {'error': f'unknown target {target!r}'})
                try: check_request(req, model_required=False)
                except DecisionError as e: return self._json(e.status, {'error': e.message, 'detail': e.body()['detail']})
                req = {k: v for k, v in req.items() if k != 'model'}
                names = [n for n in targets[target] if sides[n] and not (target == 'all' and n == 'remote' and not keys.key)]
                jobs = {n: pool.submit(self._side, sides[n], req, bool(body.get('allow_truncation'))) for n in names}
                self._json(200, {n: f.result() for n, f in jobs.items()})
            elif self.path == '/api/key':
                key = str(body.get('key', '')).strip()
                if keys.env_key: return self._json(400, {'error': 'TYPESAFE_API_KEY is set; manage that environment variable to change the key.'})
                if not valid_key(key): return self._json(400, {'error': 'API keys must contain only printable ASCII characters, without spaces.'})
                ok, info = reg.get(a.remote_model).check_key(key) if reg.jev_url == JEV_URL else (True, None)
                if not ok: return self._json(400, {'error': info})
                try: keys.set(key, bool(body.get('remember')))
                except Exception as e: return self._json(500, {'error': f'Key works but could not be saved: {e!r}'})
                self._json(200, {'key': keys.status(), 'account': info})
            elif self.path == '/api/vega/load':
                if not sides['vega']: return self._json(200, {'vega': {'state': 'off'}})
                m = reg.get(sides['vega']); m.start(); self._json(200, {'vega': m.status()})
            elif self.path == '/api/key/forget':
                try: keys.forget()
                except OSError: return self._json(500, {'error': 'Could not remove the saved key; the current key is unchanged.'})
                self._json(200, {'key': keys.status()})
            else:
                self._json(404, {'error': 'not found'})

        @staticmethod
        def _side(name, req, allow_truncation):
            """One model's answer for the page, or its contract error in the same place."""
            try: return reg.get(name).run(req, allow_truncation)
            except DecisionError as e: return {'error': e.error_type, 'status': e.status, 'message': e.message}
            except Exception as e: return {'error': 'internal_error', 'status': 500, 'message': f'{name} failed: {type(e).__name__}: {e}'}
    return H


class PlaygroundServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can permit two live servers on the same port.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt': self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind(reg, keys, a):
    """Bind --port, or without one the first free port from 8770; the handler's Host check needs the real port."""
    ports = [a.port] if a.port else range(DEFAULT_PORT, DEFAULT_PORT + 20)
    for port in ports:
        a.port = port
        try: return PlaygroundServer((a.host, port), make_handler(reg, keys, a))
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES) and getattr(e, 'winerror', None) not in (10048, 10013): raise
    sys.exit(f'error: port {ports[0]} is in use; pick another with --port' if len(ports) == 1 else
             f'error: ports {ports[0]}-{ports[-1]} are all in use; pick one with --port')


def add_server_args(ap):
    ap.add_argument('--model', '--release', dest='model',
                    help='S1-CED: Hugging Face repo id or release folder (default: $S1CED_MODEL, else syedazeez/s1ced-v3-deep)')
    ap.add_argument('--weights'); ap.add_argument('--calibration')
    ap.add_argument('--head', choices=['deep_fusion', 'late_fusion'], help='override head type (needed for v2 checkpoints)')
    ap.add_argument('--threads', type=int); ap.add_argument('--device', default='cpu', help='S1-CED device (default: cpu)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, help=f'default: first free port from {DEFAULT_PORT}')
    ap.add_argument('--remote-url', default=JEV_URL)
    ap.add_argument('--remote-model', default='jev-latest', help='jev-latest, jev-preview, or a versioned id such as jev-1.14.0')
    ap.add_argument('--vega-model', default='vega-0.8b', choices=list(VEGA_SIZES), help='the Vega size the page shows')
    ap.add_argument('--vega-device', help='cuda, mps or cpu (default: CUDA, then MPS, then CPU)')
    ap.add_argument('--vega-preload', action='store_true', help='load Vega at startup instead of on first use')
    ap.add_argument('--no-vega', action='store_true', help='turn Vega off')
    ap.add_argument('--jev-fallback', metavar='MODEL', help='with no TypeSafe key, answer jev-* requests with this local model '
                    '(s1ced or a vega- model); the response names the model that answered')


def serve(a, open_browser=True):
    """Start the playground and the /v1 API from parsed add_server_args arguments."""
    from .infer import S1CED
    try:
        eng = S1CED(a.model, a.weights, a.head, a.calibration, a.threads, False, a.device)
    except FileNotFoundError as e:
        sys.exit(f'error: {e}')
    keys = KeyStore()
    reg = Registry(s1ced={'engine': eng}, vega_device=a.vega_device, vega=not a.no_vega, jev_key=lambda: keys.key, jev_url=a.remote_url)
    if a.no_vega: a.vega_model = None
    elif a.vega_preload: reg.get(a.vega_model).start()
    if a.jev_fallback:
        try: fb = reg.get(a.jev_fallback)
        except DecisionError as e: sys.exit(f'error: --jev-fallback: {e.message}')
        if isinstance(fb, JevModel): sys.exit('error: --jev-fallback must name a local model: s1ced or a vega- model')
        a.jev_fallback = fb.name
    srv = bind(reg, keys, a); srv.daemon_threads = True
    url = f'http://{"127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host}:{a.port}'
    names = ['s1ced'] + (list(VEGA_SIZES) if a.vega_model else []) + [a.remote_model]
    print(f'S1-CED playground ({eng.head}, {eng.device}) ready at {url}\n'
          f'API: POST {url}/v1/systemone with model {", ".join(names)}\n'
          f'TypeSafe key: {keys.status()["source"] or "not set"}\nPress Ctrl+C to stop.', file=sys.stderr, flush=True)
    if open_browser: threading.Timer(0.5, webbrowser.open, (url,)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print('stopped', file=sys.stderr)
    finally: srv.server_close()


def main():
    global EXAMPLES
    ap = argparse.ArgumentParser(prog='s1ced-playground', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_server_args(ap)
    ap.add_argument('--examples', type=Path, help='scenario file to show instead of the bundled examples')
    ap.add_argument('--no-browser', action='store_true', help='do not open the page in a browser')
    a = ap.parse_args()
    if a.examples:
        if not a.examples.is_file(): ap.error(f'--examples: {a.examples} not found')
        EXAMPLES = a.examples.resolve()
    serve(a, not a.no_browser)


if __name__ == '__main__':
    main()
