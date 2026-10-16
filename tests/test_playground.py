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
from s1ced import vega


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

    def test_typesafe_error_detail_is_unwrapped(self):
        body = b'{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server."}}'
        err = app.urlerr.HTTPError('https://api.typesafe.ai/v1/systemone', 401, 'Unauthorized', {}, io.BytesIO(body))
        with patch.object(app.urlreq, 'urlopen', side_effect=err):
            out = app.run_remote(SimpleNamespace(key='test'), 'https://api.typesafe.ai/v1/systemone', 'jev-latest', {})
        self.assertEqual(out['error'], 'unauthorized')
        self.assertEqual(out['message'], 'Cannot authenticate with the server.')

    def test_typesafe_key_check_uses_model_list(self):
        seen = []
        def fake(req, timeout):
            seen.append(req.full_url); return io.BytesIO(b'{"models":[{"name":"jev-latest"}]}')
        with patch.object(app.urlreq, 'urlopen', side_effect=fake):
            ok, info = app.check_key('test', 'https://api.typesafe.ai/v1/systemone')
        self.assertTrue(ok)
        self.assertEqual(seen, ['https://api.typesafe.ai/v1/models'])
        self.assertEqual(info['models'], ['jev-latest'])

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
    def handler(self, body=b'', path='/api/decide', key='test', vega=None):
        a = SimpleNamespace(port=8770, remote_url='https://example.test', remote_model='test')
        h = object.__new__(app.make_handler(SimpleNamespace(), SimpleNamespace(key=key), a, vega))
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

    def test_compare_all_runs_every_model_and_skips_remote_without_key(self):
        vega = SimpleNamespace(decide=lambda req: {'answers': {'q': {'noul': 0.6}}})
        h = self.handler(json.dumps({'target':'all','request':{'questions':{'q':{'type':'noul'}}}}).encode(), key=None, vega=vega)
        h._local = lambda *args: {'answers':{'q':{'noul':0.75}}}
        with patch.object(app, 'run_remote', side_effect=AssertionError('must not call remote without a key')): h.do_POST()
        self.assertEqual(h.response[0], 200)
        self.assertEqual(set(h.response[1]), {'local', 'vega'})

    def test_vega_target_without_vega_reports_it_is_off(self):
        h = self.handler(json.dumps({'target':'vega','request':{'questions':{'q':{'type':'noul'}}}}).encode()); h.do_POST()
        self.assertEqual(h.response[1]['vega']['error'], 'vega_off')

    def test_unknown_target_is_a_400_response(self):
        h = self.handler(json.dumps({'target':'everything','request':{'questions':{'q':{'type':'noul'}}}}).encode()); h.do_POST()
        self.assertEqual(h.response[0], 400)

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


class VegaTests(unittest.TestCase):
    QUESTIONS = {'ok': {'type': 'noul', 'instructions': 'Is it fine?'},
                 'team': {'type': 'choice', 'instructions': 'Which team?', 'criteria': {'billing': 'money', 'tech': None}},
                 'level': {'type': 'score', 'instructions': 'How bad?', 'criteria': ['Low', 'Mid', 'High']}}
    OUT = {'answers': {
        'ok': {'probs': {'False': 0.2, 'True': 0.8}, 'confidence': 0.8, 'prediction_set': [True], 'abstain': False, 'unbound': False, 'boolean': 0.8},
        'team': {'probs': {'billing': 0.7, 'tech': 0.3}, 'confidence': 0.7, 'prediction_set': ['billing', 'tech'], 'abstain': True, 'unbound': False, 'choice': 'billing'},
        'level': {'probs': {'0': 0.1, '1': 0.6, '2': 0.3}, 'confidence': 0.6, 'prediction_set': None, 'abstain': False, 'unbound': True,
                  'score': 1, 'expected_score': 1.2}},
        'latency_ms': 210.0, 'n_tokens': 300, 'views': 1}

    def test_questions_use_vegas_boolean_type(self):
        q = vega.to_vega(self.QUESTIONS)
        self.assertEqual([x['type'] for x in q.values()], ['boolean', 'choice', 'score'])
        self.assertEqual(self.QUESTIONS['ok']['type'], 'noul')

    def test_answers_are_reshaped_to_the_playground_shape(self):
        out = vega.reshape(self.QUESTIONS, self.OUT)
        a = out['answers']
        self.assertEqual(a['ok']['noul'], 0.8)
        self.assertEqual(a['ok']['vega']['prediction_set'], ['yes'])
        self.assertEqual((a['team']['choice'], a['team']['probabilities']['tech'], a['team']['vega']['abstain']), ('billing', 0.3, True))
        self.assertEqual((a['level']['score'], a['level']['probabilities']['1'], a['level']['legend']['2']), (1.2, 0.6, 'High'))
        self.assertTrue(a['level']['vega']['unbound'])
        self.assertEqual(out['usage']['input_tokens'], 300)

    def test_non_finite_probability_is_an_error(self):
        bad = json.loads(json.dumps(self.OUT)); bad['answers']['team']['probs']['tech'] = float('nan')
        with self.assertRaises(ValueError): vega.reshape(self.QUESTIONS, bad)

    def test_disabled_runner_never_loads(self):
        r = vega.VegaRunner(enabled=False); r.start()
        self.assertEqual(r.status()['state'], 'off')
        self.assertEqual(r.decide({'questions': self.QUESTIONS})['error'], 'vega_off')

    def test_load_failure_is_reported_and_can_be_retried(self):
        r = vega.VegaRunner()
        with patch('vegaml.load', side_effect=OSError('no network')):
            out = r.decide({'questions': self.QUESTIONS})
        self.assertEqual(out['error'], 'vega_load')
        self.assertIn('no network', out['message'])
        fake = SimpleNamespace(engine=SimpleNamespace(device='cpu'), decide=lambda state, qs: self.OUT)
        with patch('vegaml.load', return_value=fake):
            out = r.decide({'state': 'x', 'questions': self.QUESTIONS})
        self.assertEqual(out['answers']['ok']['noul'], 0.8)
        self.assertEqual(r.status()['device'], 'cpu')


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

