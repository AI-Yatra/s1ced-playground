"""The shared contract: request rules, and every model adapter held to the same response rules. No downloads:
S1-CED, Vega and Jev are replaced by stand-ins that behave like the real engines at their boundary."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from s1ced import contract, models

REQUEST = {'state': {'ticket': 'Our whole team is locked out and we demo in two hours.'},
           'questions': {
               'urgent': {'type': 'noul', 'instructions': 'Does this need a reply within the hour?',
                          'criteria': {'true': 'Blocking or time-sensitive', 'false': 'Can wait'}},
               'team': {'type': 'choice', 'instructions': 'Which team owns this?',
                        'criteria': {'technical': 'Outages and access', 'billing': None, 'sales': None}},
               'mood': {'type': 'score', 'instructions': 'How frustrated is the customer?',
                        'criteria': ['Calm', 'Annoyed', 'Angry']}}}


def copy(x): return json.loads(json.dumps(x))


def skewed(n, top=0):
    """A valid distribution over n options with its peak at index top."""
    p = [0.1 / (n - 1)] * n; p[top] = 0.9
    return p


class FakeS1CED:
    """The S1CED engine at its boundary: whitespace tokens, decide(state, [{question, options, type}])."""
    def __init__(self, max_state=50, threshold=0.2):
        self.cfg = {'max_options': 8, 'max_state_len': max_state + 2, 'max_branch_len': 69, 'run_name': 's1ced_v3_deep'}
        self.tok = SimpleNamespace(encode=lambda t: SimpleNamespace(ids=t.split()))
        self.device, self.threshold, self.head, self.asks = 'cpu', threshold, 'deep_fusion', None

    def decide(self, state, asks):
        self.asks = asks; out = []
        for q in asks:
            p = skewed(len(q['options']), len(q['options']) - 1)
            ps = [o for o, x in zip(q['options'], p) if x >= self.threshold]
            out.append({'selected': q['options'][-1], 'confidence': max(p), 'prediction_set': ps, 'should_escalate': len(ps) != 1,
                        'probabilities': dict(zip(q['options'], p))})
        return out


class FakeVega:
    """vegaml at its boundary: type "boolean", probs keyed by label, abstain / unbound flags."""
    engine = SimpleNamespace(device='cpu')

    def decide(self, state, questions):
        answers = {}
        for qid, q in questions.items():
            labels = [False, True] if q['type'] == 'boolean' else list(q['criteria']) if q['type'] == 'choice' else list(range(len(q['criteria'])))
            p = skewed(len(labels), 1)
            a = {'probs': {str(l): x for l, x in zip(labels, p)}, 'confidence': max(p), 'prediction_set': [labels[1]],
                 'abstain': qid == 'mood', 'unbound': False}
            if q['type'] == 'boolean': a['boolean'] = p[1]
            elif q['type'] == 'choice': a['choice'] = labels[1]
            else: a['score'], a['expected_score'] = 1, sum(i * x for i, x in enumerate(p))
            answers[qid] = a
        return {'answers': answers, 'latency_ms': 1.0, 'n_tokens': 42, 'views': 1}


def jev_reply(req):
    """What TypeSafe returns for a request (version id, answers without uncertainty, usage)."""
    answers = {}
    for qid, q in req['questions'].items():
        if q['type'] == 'noul': answers[qid] = {'type': 'noul', 'noul': 0.7}
        else:
            labels = contract.option_labels(q); p = skewed(len(labels))
            a = {'type': q['type'], 'probabilities': dict(zip(labels, p)), 'confidence': 0.81}
            if q['type'] == 'choice': a['choice'] = labels[0]
            else: a.update(score=0.2, legend={l: contract.text_of(c) for l, c in zip(labels, q['criteria'])})
            answers[qid] = a
    return {'model': 'jev-1.13.0', 'answers': answers, 'usage': {'input_tokens': 300, 'output_tokens': 20}}


def fake_jev_urlopen(req, timeout=None):
    return io.BytesIO(json.dumps(jev_reply(json.loads(req.data))).encode())


def ready_vega():
    m = models.VegaModel('vega-0.8b'); m.v, m.state = FakeVega(), 'ready'; m.ready.set()
    return m


def all_models():
    return {'s1ced': models.S1CEDModel(FakeS1CED()), 'vega-0.8b': ready_vega(), 'jev-latest': models.JevModel('jev-latest', 'test-key')}


class RequestRuleTests(unittest.TestCase):
    def rejects(self, mutate, status=422):
        req = copy(REQUEST); req['model'] = 's1ced'; mutate(req)
        with self.assertRaises(contract.DecisionError) as ctx: contract.check_request(req)
        self.assertEqual(ctx.exception.status, status)
        return ctx.exception

    def test_valid_request_passes(self):
        contract.check_request({**copy(REQUEST), 'model': 'vega-0.8b'})

    def test_every_rule_is_enforced(self):
        cases = {
            'missing model': lambda r: r.pop('model'),
            'missing state': lambda r: r.pop('state'),
            'number state': lambda r: r.__setitem__('state', 3),
            'no questions': lambda r: r.__setitem__('questions', {}),
            'unknown top-level field': lambda r: r.__setitem__('assets', {}),
            'unknown question field': lambda r: r['questions']['urgent'].__setitem__('gate', 0.8),
            'missing type': lambda r: r['questions']['team'].pop('type'),
            'box type': lambda r: r['questions']['team'].__setitem__('type', 'box'),
            'empty instructions': lambda r: r['questions']['team'].__setitem__('instructions', ' '),
            'noul criteria keys': lambda r: r['questions']['urgent'].__setitem__('criteria', {'yes': 'x'}),
            'one choice option': lambda r: r['questions']['team'].__setitem__('criteria', {'only': None}),
            'one score level': lambda r: r['questions']['mood'].__setitem__('criteria', ['Calm']),
            'eleven score levels': lambda r: r['questions']['mood'].__setitem__('criteria', [str(i) for i in range(11)]),
            'repeated score levels': lambda r: r['questions']['mood'].__setitem__('criteria', ['Calm', 'Calm']),
        }
        for name, mutate in cases.items():
            with self.subTest(name): self.rejects(mutate)

    def test_errors_name_the_field(self):
        e = self.rejects(lambda r: r['questions']['team'].__setitem__('criteria', {'only': None}))
        self.assertEqual(e.body()['detail']['loc'], ['questions', 'team', 'criteria'])

    def test_non_finite_numbers_and_deep_nesting_are_400(self):
        self.rejects(lambda r: r.__setitem__('state', {'x': float('inf')}), 400)
        deep = 'x'
        for _ in range(70): deep = [deep]
        self.rejects(lambda r: r.__setitem__('state', deep), 400)
        with self.assertRaises(contract.DecisionError) as ctx: contract.loads(b'{"state": NaN}')
        self.assertEqual(ctx.exception.status, 400)


class ConformanceTests(unittest.TestCase):
    """The same request through every adapter: all must pass the same response check."""

    def test_every_model_returns_the_same_shape(self):
        with patch.object(models.urlreq, 'urlopen', side_effect=fake_jev_urlopen):
            outs = {n: m.run(copy(REQUEST)) for n, m in all_models().items()}
        for name, out in outs.items():
            with self.subTest(name):
                self.assertEqual(set(out['answers']), set(REQUEST['questions']))
                self.assertEqual(set(out) >= {'model', 'answers', 'usage', 'latency_ms', 'warnings'}, True)
                for qid, a in out['answers'].items():
                    self.assertEqual(set(a['uncertainty']), set(contract.UNCERTAINTY_KEYS))
                self.assertEqual(set(outs['s1ced']['answers']['team']), set(out['answers']['team']))

    def test_one_review_rule_for_every_model(self):
        with patch.object(models.urlreq, 'urlopen', side_effect=fake_jev_urlopen):
            outs = {n: m.run(copy(REQUEST)) for n, m in all_models().items()}
        self.assertIs(outs['vega-0.8b']['answers']['mood']['uncertainty']['needs_review'], True)    # abstained
        self.assertIs(outs['vega-0.8b']['answers']['team']['uncertainty']['needs_review'], False)
        self.assertIsNone(outs['jev-latest']['answers']['team']['uncertainty']['needs_review'])     # no signals
        self.assertEqual(outs['s1ced']['answers']['urgent']['uncertainty']['prediction_set'], ['yes'])

    def test_a_broken_adapter_cannot_pass(self):
        bad = models.S1CEDModel(FakeS1CED())
        bad.eng.decide = lambda state, asks: [{'selected': 'nope', 'confidence': 0.5, 'probabilities': {o: 0.5 for o in q['options']}} for q in asks]
        with self.assertRaises(contract.DecisionError) as ctx: bad.run(copy(REQUEST))
        self.assertEqual(ctx.exception.status, 502)

    def test_model_field_must_match_the_model(self):
        with self.assertRaises(contract.DecisionError): models.S1CEDModel(FakeS1CED()).run({**copy(REQUEST), 'model': 'vega-0.8b'})


class S1CEDAdapterTests(unittest.TestCase):
    def test_reads_criteria_descriptions_like_every_model(self):
        m = models.S1CEDModel(FakeS1CED()); m.run(copy(REQUEST))
        texts = {a['type']: a['question'] for a in m.eng.asks}
        self.assertIn('Blocking or time-sensitive', texts['noul'])
        self.assertIn('technical = Outages and access', texts['choice'])

    def test_too_long_state_is_refused_unless_truncation_is_allowed(self):
        m = models.S1CEDModel(FakeS1CED(max_state=5))
        with self.assertRaises(contract.DecisionError) as ctx: m.run(copy(REQUEST))
        self.assertEqual((ctx.exception.status, ctx.exception.error_type), (400, 'max_tokens_exceeded'))
        out = m.run(copy(REQUEST), allow_truncation=True)
        self.assertTrue(out['warnings'])

    def test_too_many_options_is_a_400(self):
        req = copy(REQUEST); req['questions']['team']['criteria'] = {f'o{i}': None for i in range(9)}
        with self.assertRaises(contract.DecisionError) as ctx: models.S1CEDModel(FakeS1CED()).run(req)
        self.assertEqual(ctx.exception.error_type, 'too_many_options')


class VegaAdapterTests(unittest.TestCase):
    def test_load_failure_is_a_503_and_can_be_retried(self):
        m = models.VegaModel('vega-0.8b')
        with patch('vegaml.load', side_effect=OSError('no network')):
            with self.assertRaises(contract.DecisionError) as ctx: m.run(copy(REQUEST))
        self.assertEqual(ctx.exception.status, 503); self.assertIn('no network', ctx.exception.message)
        with patch('vegaml.load', return_value=FakeVega()):
            out = m.run(copy(REQUEST))
        self.assertEqual(out['model'], 'vega-0.8b'); self.assertEqual(m.status()['device'], 'cpu')

    def test_vega_follows_the_same_length_rule(self):
        m = ready_vega()
        with patch.object(m, '_length_problems', return_value=['team: 1 option(s) are over 32 tokens.']):
            with self.assertRaises(contract.DecisionError) as ctx: m.run(copy(REQUEST))
            self.assertEqual((ctx.exception.status, ctx.exception.error_type), (400, 'max_tokens_exceeded'))
            out = m.run(copy(REQUEST), allow_truncation=True)
        self.assertEqual(out['warnings'], ['team: 1 option(s) are over 32 tokens. It was shortened.'])

    def test_out_of_scope_comes_from_unbound(self):
        m = ready_vega(); fake = m.v.decide
        def unbound(state, qs):
            out = fake(state, qs); out['answers']['team']['unbound'] = True; return out
        m.v = SimpleNamespace(engine=m.v.engine, decide=unbound)
        u = m.run(copy(REQUEST))['answers']['team']['uncertainty']
        self.assertEqual((u['out_of_scope'], u['needs_review']), (True, True))


class JevAdapterTests(unittest.TestCase):
    def call(self, side_effect, key='test-key'):
        with patch.object(models.urlreq, 'urlopen', side_effect=side_effect):
            return models.JevModel('jev-latest', key).run(copy(REQUEST))

    def test_typesafe_errors_keep_status_and_message(self):
        body = b'{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server."}}'
        err = models.urlerr.HTTPError(models.JEV_URL, 401, 'Unauthorized', {}, io.BytesIO(body))
        with self.assertRaises(contract.DecisionError) as ctx: self.call(err)
        self.assertEqual((ctx.exception.status, ctx.exception.error_type, ctx.exception.message),
                         (401, 'authentication_error', 'Cannot authenticate with the server.'))

    def test_403_is_not_reported_as_a_bad_key(self):
        err = models.urlerr.HTTPError(models.JEV_URL, 403, 'Forbidden', {}, io.BytesIO(b'policy blocked'))
        with self.assertRaises(contract.DecisionError) as ctx: self.call(err)
        self.assertEqual(ctx.exception.status, 403); self.assertIn('policy blocked', ctx.exception.message)

    def test_missing_or_invalid_key_never_reaches_the_network(self):
        for key in (None, 'test​key'):
            with self.subTest(key=key), patch.dict('os.environ', {'TYPESAFE_API_KEY': ''}):
                with self.assertRaises(contract.DecisionError) as ctx: self.call(AssertionError('must not call TypeSafe'), key)
                self.assertEqual(ctx.exception.status, 401)

    def test_malformed_replies_are_502(self):
        for body in (b'<html>proxy error</html>', b'[]', b'{"answers":{"urgent":{"type":"noul","noul":"high"}}}'):
            with self.subTest(body=body), self.assertRaises(contract.DecisionError) as ctx:
                self.call(lambda req, timeout=None, b=body: io.BytesIO(b))
            self.assertEqual(ctx.exception.status, 502)

    def test_key_check_uses_the_free_model_list(self):
        seen = []
        def fake(req, timeout):
            seen.append(req.full_url); return io.BytesIO(b'{"models":[{"name":"jev-latest"}]}')
        with patch.object(models.urlreq, 'urlopen', side_effect=fake):
            ok, info = models.JevModel().check_key('test')
        self.assertEqual((ok, seen, info['models']), (True, ['https://api.typesafe.ai/v1/models'], ['jev-latest']))


class RegistryTests(unittest.TestCase):
    def test_unknown_model_is_404_and_aliases_resolve(self):
        reg = models.Registry(s1ced={'engine': FakeS1CED()})
        with self.assertRaises(contract.DecisionError) as ctx: reg.get('gpt-5')
        self.assertEqual(ctx.exception.status, 404)
        self.assertIs(reg.get('vega'), reg.get('vega-0.8b'))

    def test_model_list_uses_typesafe_shape_and_never_loads(self):
        reg = models.Registry(s1ced={'engine': FakeS1CED()})
        listed = reg.list()['models']
        self.assertEqual([m['name'] for m in listed], ['s1ced', 'vega-0.8b', 'vega-4b', 'jev-latest', 'jev-preview'])
        self.assertTrue(all(set(m) == {'name', 'description', 'release_date'} for m in listed))
        self.assertEqual(reg.models, {})


if __name__ == '__main__':
    unittest.main()
