import io
import json
import os
from pathlib import Path
import tempfile
import shutil
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from s1ced import playground as app


class RemoteTests(unittest.TestCase):
    def invoke(self, payload):
        with patch.object(app.urlreq, 'urlopen', return_value=io.BytesIO(payload)):
            return app.run_remote(SimpleNamespace(key='fake-test-key'), 'https://example.test/decide', 'test', {})

    def test_malformed_success_is_a_remote_error(self):
        for body in (b'<html>proxy error</html>', b'[]', b'null', b'{"usage":[1]}'):
            with self.subTest(body=body):
                self.assertIn('error', self.invoke(body))

    def test_provider_error_envelope_has_readable_message(self):
        out = self.invoke(b'{"error":{"code":502,"message":"upstream timeout"}}')
        self.assertIsInstance(out['error'], str)
        self.assertEqual(out['message'], 'upstream timeout')

    def test_403_does_not_claim_key_is_invalid(self):
        err = app.urlerr.HTTPError('https://example.test', 403, 'Forbidden', {}, io.BytesIO(b'policy blocked'))
        with patch.object(app.urlreq, 'urlopen', side_effect=err):
            out = app.run_remote(SimpleNamespace(key='test'), 'https://example.test', 'test', {})
        self.assertNotEqual(out['error'], 'unauthorized')
        self.assertIn('policy blocked', out['message'])

    def test_invalid_unicode_key_is_rejected_before_network(self):
        with patch.object(app.urlreq, 'urlopen', side_effect=AssertionError('must not contact provider')):
            out = app.run_remote(SimpleNamespace(key='test\u200bkey'), 'https://example.test', 'test', {})
        self.assertEqual(out['error'], 'invalid_key')

    def test_malformed_answer_values_are_errors(self):
        for answer in ({'noul':{}}, {'confidence':'high'}, {'probabilities':{'yes':'0.8'}}, 'not an answer'):
            with self.subTest(answer=answer):
                out = self.invoke(json.dumps({'answers':{'q':answer}}).encode())
                self.assertIn('error',out)

    def test_remote_cannot_supply_local_calibration_metadata(self):
        out = self.invoke(b'{"answers":{"q":{"noul":0.8,"local":{"should_escalate":true}}}}')
        self.assertNotIn('local',out['answers']['q'])


class HandlerTests(unittest.TestCase):
    def handler(self, body=b'', path='/api/decide'):
        a = SimpleNamespace(port=8770, remote_url='https://example.test', remote_model='test')
        h = object.__new__(app.make_handler(SimpleNamespace(), SimpleNamespace(key='test'), a))
        h.path = path; h.command = 'POST'; h.headers = {'Host':'127.0.0.1:8770','Content-Type':'application/json','Content-Length':str(len(body))}
        h.rfile = io.BytesIO(body); h._json = lambda code, obj: setattr(h, 'response', (code,obj))
        return h

    def test_compare_preserves_local_answer_when_remote_throws(self):
        h = self.handler(json.dumps({'target':'both','request':{'questions':{'q':{'type':'noul'}}}}).encode())
        h._local = lambda *args: {'answers':{'q':{'noul':0.75}}}
        with patch.object(app, 'run_remote', side_effect=RuntimeError('upstream failed')): h.do_POST()
        self.assertEqual(h.response[0],200)
        self.assertEqual(h.response[1]['local']['answers']['q']['noul'],0.75)
        self.assertIn('error',h.response[1]['remote'])

    def test_deeply_nested_json_is_a_400_response(self):
        h = self.handler(('['*2000+'0'+']'*2000).encode()); h.do_POST()
        self.assertEqual(h.response[0],400)

    def test_bom_prefixed_examples_load(self):
        h = self.handler(path='/api/examples'); h.command='GET'
        testfile = Path(__file__).parent/'bom-examples.json'
        testfile.write_bytes(b'\xef\xbb\xbf{"groups":[]}')
        try:
            with patch.object(app, 'EXAMPLES', testfile): h.do_GET()
            self.assertEqual(h.response,(200,{'groups':[]}))
        finally: testfile.unlink()

    def test_two_servers_cannot_bind_the_same_port(self):
        with app.PlaygroundServer(('127.0.0.1',0),app.BaseHTTPRequestHandler) as server:
            with self.assertRaises(OSError):
                app.PlaygroundServer(server.server_address,app.BaseHTTPRequestHandler)


class KeyTests(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).resolve().parent / ('key-test-' + uuid.uuid4().hex)
        folder.mkdir(mode=0o777)
        self.tmp = SimpleNamespace(name=str(folder))
        self.env = patch.dict(os.environ, {'APPDATA': self.tmp.name, 'OPENROUTER_API_KEY': ''})
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
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-env'}):
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

