import gzip
import io
import json
import os
from pathlib import Path
import shutil
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from s1ced import models, playground as app
from test_contract import REQUEST, FakeS1CED, copy, fake_jev_urlopen, ready_vega


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.reg = models.Registry(s1ced={'engine': FakeS1CED()}, jev_key=lambda: self.key)
        self.reg.models['vega-0.8b'] = ready_vega()
        self.key = 'test-key'

    def handler(self, body=b'', path='/api/decide', method='POST', headers=None, vega_model='vega-0.8b', jev_fallback=None):
        a = SimpleNamespace(port=8770, remote_model='jev-latest', vega_model=vega_model, jev_fallback=jev_fallback)
        keys = SimpleNamespace(key=self.key, status=lambda: {'set': bool(self.key)})
        h = object.__new__(app.make_handler(self.reg, keys, a))
        h.path, h.command = path, method
        h.headers = {'Host': '127.0.0.1:8770', 'Content-Type': 'application/json', 'Content-Length': str(len(body)), **(headers or {})}
        h.rfile = io.BytesIO(body); h._json = lambda code, obj, headers=None: setattr(h, 'response', (code, obj))
        return h

    def post(self, obj, path="/v1/systemone", **kw):
        h = self.handler(json.dumps(obj).encode(), path, **kw)
        with patch.object(models.urlreq, 'urlopen', side_effect=fake_jev_urlopen): h.do_POST()
        return h.response

    def test_v1_systemone_answers_every_model_in_one_shape(self):
        shapes = set()
        for name in ('s1ced', 'vega-0.8b', 'jev-latest'):
            code, out = self.post({**copy(REQUEST), 'model': name})
            self.assertEqual(code, 200, out)
            shapes.add(tuple(sorted(out['answers']['team'])))
        self.assertEqual(len(shapes), 1)

    def test_v1_errors_use_the_typesafe_error_body(self):
        req = {**copy(REQUEST), 'model': 's1ced'}; req['questions']['team']['type'] = 'box'
        code, out = self.post(req)
        self.assertEqual(code, 422)
        self.assertEqual(set(out['detail']), {'error_type', 'message', 'loc'})
        self.assertEqual(self.post({**copy(REQUEST), 'model': 'gpt-5'})[0], 404)

    def test_v1_accepts_gzip_bodies(self):
        body = gzip.compress(json.dumps({**copy(REQUEST), 'model': 's1ced'}).encode())
        h = self.handler(body, '/v1/systemone', headers={'Content-Encoding': 'gzip'}); h.do_POST()
        self.assertEqual(h.response[0], 200)

    def test_unknown_encoding_is_415(self):
        h = self.handler(b'xx', '/v1/systemone', headers={'Content-Encoding': 'br'}); h.do_POST()
        self.assertEqual(h.response[0], 415)

    def test_jev_without_server_key_uses_the_callers_bearer_token(self):
        self.key = None; seen = []
        def capture(req, timeout=None):
            seen.append(req.get_header('Authorization')); return fake_jev_urlopen(req)
        h = self.handler(json.dumps({**copy(REQUEST), 'model': 'jev-latest'}).encode(), '/v1/systemone', headers={'Authorization': 'Bearer team-key'})
        with patch.object(models.urlreq, 'urlopen', side_effect=capture): h.do_POST()
        self.assertEqual((h.response[0], seen), (200, ['Bearer team-key']))
        h = self.handler(json.dumps({**copy(REQUEST), 'model': 'jev-latest'}).encode(), '/v1/systemone', headers={'Authorization': 'Bearer local'})
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': ''}): h.do_POST()
        self.assertEqual(h.response[0], 401)

    def test_jev_fallback_answers_locally_and_says_so(self):
        self.key = None
        h = self.handler(json.dumps({**copy(REQUEST), 'model': 'jev-latest'}).encode(), '/v1/systemone',
                         headers={'Authorization': 'Bearer local'}, jev_fallback='vega-0.8b')
        with patch.object(models.urlreq, 'urlopen', side_effect=AssertionError('must not call TypeSafe')): h.do_POST()
        code, out = h.response
        self.assertEqual((code, out['model']), (200, 'vega-0.8b'))
        self.assertIn('--jev-fallback', out['warnings'][-1])

    def test_jev_fallback_is_unused_when_a_key_exists_and_aliases_get_no_warning(self):
        code, out = self.post({**copy(REQUEST), 'model': 'jev-latest'}, jev_fallback='vega-0.8b')
        self.assertEqual(out['model'], 'jev-1.13.0')
        code, out = self.post({**copy(REQUEST), 'model': 'vega'}, jev_fallback='s1ced')
        self.assertEqual((out['model'], out['warnings']), ('vega-0.8b', []))

    def test_v1_accepts_zstd_bodies(self):
        try:
            from compression.zstd import compress
        except ImportError:
            import zstandard; compress = zstandard.ZstdCompressor().compress
        body = compress(json.dumps({**copy(REQUEST), 'model': 's1ced'}).encode())
        h = self.handler(body, '/v1/systemone', headers={'Content-Encoding': 'zstd'}); h.do_POST()
        self.assertEqual(h.response[0], 200)

    def test_loading_model_answers_503_with_retry_after_at_once(self):
        loading = models.VegaModel('vega-0.8b'); loading.state = 'loading'; loading.ready.clear()
        self.reg.models['vega-0.8b'] = loading
        sent = {}
        h = self.handler(json.dumps({**copy(REQUEST), 'model': 'vega-0.8b'}).encode(), '/v1/systemone')
        h._json = lambda code, obj, headers=None: sent.update(code=code, obj=obj, headers=headers)
        h.do_POST()
        self.assertEqual((sent['code'], sent['obj']['detail']['error_type'], sent['headers']), (503, 'model_loading', {'Retry-After': '10'}))

    def test_v1_models_lists_every_model(self):
        h = self.handler(path='/v1/models', method='GET'); h.do_GET()
        self.assertEqual([m['name'] for m in h.response[1]['models']], ['s1ced', 'vega-0.8b', 'vega-4b', 'jev-latest', 'jev-preview'])

    def test_compare_all_skips_jev_without_a_key_and_keeps_errors_per_model(self):
        self.key = None
        code, out = self.post({'target': 'all', 'request': copy(REQUEST)}, '/api/decide')
        self.assertEqual((code, set(out)), (200, {'local', 'vega'}))
        self.key = 'test-key'
        err = models.urlerr.HTTPError(models.JEV_URL, 500, 'boom', {}, io.BytesIO(b'upstream failed'))
        h = self.handler(json.dumps({'target': 'both', 'request': copy(REQUEST)}).encode())
        with patch.object(models.urlreq, 'urlopen', side_effect=err): h.do_POST()
        self.assertIn('answers', h.response[1]['local'])
        self.assertEqual(h.response[1]['remote']['status'], 500)

    def test_page_requests_follow_the_same_rules(self):
        req = copy(REQUEST); req['questions']['mood']['criteria'] = ['Calm']
        code, out = self.post({'target': 'local', 'request': req}, '/api/decide')
        self.assertEqual(code, 422); self.assertIn('score criteria', out['error'])

    def test_vega_off_is_not_offered(self):
        code, out = self.post({'target': 'vega', 'request': copy(REQUEST)}, '/api/decide', vega_model=None)
        self.assertEqual((code, out), (200, {}))

    def test_unknown_target_is_a_400_response(self):
        self.assertEqual(self.post({'target': 'everything', 'request': copy(REQUEST)}, '/api/decide')[0], 400)

    def test_deeply_nested_json_is_a_400_response(self):
        h = self.handler(('[' * 2000 + '0' + ']' * 2000).encode()); h.do_POST()
        self.assertEqual(h.response[0], 400)

    def test_cross_site_requests_are_forbidden(self):
        h = self.handler(b'{}', '/v1/systemone', headers={'Origin': 'https://evil.example'}); h.do_POST()
        self.assertEqual(h.response[0], 403)

    def test_bom_prefixed_examples_load(self):
        h = self.handler(path='/api/examples', method='GET')
        testfile = Path(__file__).parent / 'bom-examples.json'
        testfile.write_bytes(b'\xef\xbb\xbf{"groups":[]}')
        try:
            with patch.object(app, 'EXAMPLES', testfile): h.do_GET()
            self.assertEqual(h.response, (200, {'groups': []}))
        finally: testfile.unlink()

    def test_two_servers_cannot_bind_the_same_port(self):
        with app.PlaygroundServer(('127.0.0.1', 0), app.BaseHTTPRequestHandler) as server:
            with self.assertRaises(OSError):
                app.PlaygroundServer(server.server_address, app.BaseHTTPRequestHandler)


class KeyTests(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).resolve().parent / ('key-test-' + uuid.uuid4().hex)
        folder.mkdir(mode=0o777)
        self.tmp = SimpleNamespace(name=str(folder))
        self.env = patch.dict(os.environ, {'APPDATA': self.tmp.name, 'TYPESAFE_API_KEY': ''})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        folder = Path(self.tmp.name).resolve()
        assert folder.parent == Path(__file__).resolve().parent
        shutil.rmtree(folder)

    def test_session_replacement_removes_old_saved_key(self):
        k = app.KeyStore(); k.set('test-old', True); k.set('test-new', False)
        self.assertFalse(k.path.exists())
        self.assertIsNone(app.KeyStore().key)

    def test_failed_save_does_not_activate_new_key(self):
        k = app.KeyStore(); k.set('test-old', False)
        k.path.parent.write_text('file blocks directory')
        with self.assertRaises(OSError): k.set('test-new', True)
        self.assertEqual(k.key, 'test-old')

    def test_environment_key_wins_over_page_key(self):
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-env'}):
            k = app.KeyStore()
            with self.assertRaises(ValueError): k.set('test-page', False)
            k.forget()
            self.assertEqual(k.key, 'test-env')

    def test_atomic_save_failure_preserves_disk_and_memory(self):
        k = app.KeyStore(); k.set('test-old',True)
        with patch.object(app.os, 'replace', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError): k.set('test-new',True)
        self.assertEqual(k.key,'test-old')
        self.assertEqual(app.KeyStore().key,'test-old')


if __name__ == '__main__':
    unittest.main()

