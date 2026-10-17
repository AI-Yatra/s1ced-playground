"""The Terraform change risk scorer: plan scanning, the Vega call, the approval gate and the CLI.
No model download: the model is a stand-in that returns a valid contract response."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from s1ced import terraform_risk as tr

PRODUCTION_PLAN = {
    'format_version': '1.2',
    'terraform_version': '1.9.5',
    'workspace': 'production',
    'resource_changes': [
        {'address': 'module.network.aws_internet_gateway.main', 'mode': 'managed',
         'type': 'aws_internet_gateway', 'name': 'main', 'module_address': 'module.network',
         'change': {'actions': ['delete'], 'before': {'tags': {'Environment': 'production'}}, 'after': None}},
        {'address': 'module.data.aws_subnet.db', 'mode': 'managed', 'type': 'aws_subnet', 'name': 'db',
         'change': {'actions': ['delete', 'create'], 'before': {'cidr_block': '10.0.2.0/24'},
                    'after': {'cidr_block': '10.0.2.0/24', 'tags': {'Name': 'production-database-subnet'}}}},
        {'address': 'module.assets.aws_s3_bucket.logs', 'mode': 'managed', 'type': 'aws_s3_bucket',
         'name': 'logs', 'change': {'actions': ['no-op'], 'before': {}, 'after': {}}},
    ],
}

SAFE_PLAN = {
    'terraform_version': '1.9.5',
    'workspace': 'staging',
    'resource_changes': [
        {'address': 'module.assets.aws_s3_bucket.logs', 'mode': 'managed', 'type': 'aws_s3_bucket',
         'name': 'logs', 'change': {'actions': ['update'],
                                    'before': {'tags': {'Owner': 'platform'}},
                                    'after': {'tags': {'Owner': 'platform', 'Environment': 'staging'}}}},
    ],
}


def distribution(labels, top):
    rest = 0.1 / (len(labels) - 1)
    probabilities = {label: rest for label in labels}
    probabilities[labels[top]] = 0.9
    return probabilities


class FakeModel:
    """A ready Vega at its boundary: decide(state, questions) -> a contract response."""

    name = 'vega-0.8b'

    def __init__(self, choice='network', severity=4.0, destructive=0.95, needs_review=False):
        self.choice, self.severity, self.destructive, self.needs_review = choice, severity, destructive, needs_review
        self.calls = []

    def _uncertainty(self, labels, answer):
        prediction_set = answer if not self.needs_review else labels[:2]
        return {'prediction_set': prediction_set, 'abstain': True if self.needs_review else None,
                'out_of_scope': None, 'needs_review': self.needs_review}

    def decide(self, state, questions, allow_truncation=False):
        self.calls.append((state, questions, allow_truncation))
        class_labels = list(questions['change_class']['criteria'])
        score_labels = [str(i) for i in range(len(questions['severity']['criteria']))]
        level = str(min(int(round(self.severity)), len(score_labels) - 1))
        destructive_label = 'yes' if self.destructive >= 0.5 else 'no'
        answers = {
            'change_class': {'type': 'choice', 'choice': self.choice,
                             'probabilities': distribution(class_labels, class_labels.index(self.choice)),
                             'confidence': 0.9, 'uncertainty': self._uncertainty(class_labels, [self.choice])},
            'severity': {'type': 'score', 'score': self.severity,
                         'legend': {key: label for key, label in zip(score_labels, questions['severity']['criteria'])},
                         'probabilities': distribution(score_labels, score_labels.index(level)),
                         'confidence': 0.8, 'uncertainty': self._uncertainty(score_labels, [level])},
            'destructive_prod': {'type': 'noul', 'noul': self.destructive,
                                 'uncertainty': self._uncertainty(['no', 'yes'], [destructive_label])},
        }
        return {'model': self.name, 'answers': answers, 'usage': {'input_tokens': 42, 'output_tokens': 0},
                'latency_ms': 1.0, 'warnings': []}


class BuildStateTests(unittest.TestCase):
    def test_production_plan_facts(self):
        state, facts = tr.build_state(PRODUCTION_PLAN)
        self.assertTrue(facts['destructive_production'])
        self.assertTrue(facts['internet_gateway_removed'])
        self.assertTrue(facts['production_database_recreated'])
        self.assertTrue(facts['network_change'])
        self.assertEqual(facts['counts']['total_changes'], 2)
        self.assertEqual(facts['counts']['replace'], 1)
        self.assertEqual(facts['counts']['delete'], 1)
        self.assertEqual(facts['counts']['no-op'], 1)
        self.assertIn('aws_internet_gateway', state)
        self.assertIn('replace', state)
        self.assertIn('Destructive production actions: yes', state)

    def test_safe_plan_facts(self):
        state, facts = tr.build_state(SAFE_PLAN)
        self.assertFalse(facts['destructive_production'])
        self.assertFalse(facts['internet_gateway_removed'])
        self.assertFalse(facts['production_database_recreated'])
        self.assertEqual(facts['counts']['update'], 1)
        self.assertEqual([r['address'] for r in facts['changed_resources']], ['module.assets.aws_s3_bucket.logs'])

    def test_production_flag_overrides_inference(self):
        _, facts = tr.build_state(SAFE_PLAN, production=True)
        self.assertEqual(facts['production_mode'], 'all resources')
        _, facts = tr.build_state(SAFE_PLAN, production=False)
        self.assertEqual(facts['production_mode'], 'none')

    def test_plain_text_plan_is_scanned(self):
        text = 'Terraform will destroy the production internet gateway and must be replaced: db subnet.'
        state, facts = tr.build_state(text)
        self.assertEqual(facts['kind'], 'text')
        self.assertTrue(facts['destructive_production'])
        self.assertTrue(facts['internet_gateway_removed'])
        self.assertTrue(facts['production_database_recreated'])
        self.assertIs(state, text)

    def test_non_plan_json_is_sent_as_text(self):
        _, facts = tr.build_state({'note': 'staging database will be destroyed'})
        self.assertEqual(facts['kind'], 'text')
        self.assertFalse(facts['destructive_production'])   # staging, not production


class ScorePlanTests(unittest.TestCase):
    def test_risky_plan_requires_approval(self):
        model = FakeModel()
        report = tr.score_plan(PRODUCTION_PLAN, model, source='plan.json')
        self.assertTrue(report['gate']['approval_required'])
        self.assertEqual(report['gate']['action'], 'HUMAN_APPROVAL_REQUIRED')
        self.assertEqual(report['risk']['change_class'], 'network')
        self.assertEqual(report['risk']['severity'], 4.0)
        self.assertTrue(report['risk']['destructive_production'])
        reasons = ' '.join(report['gate']['reasons'])
        self.assertIn('internet gateway', reasons)
        self.assertIn('production database subnet', reasons)
        self.assertEqual(model.calls[0][1], tr.QUESTIONS)

    def test_safe_plan_is_auto_approved(self):
        report = tr.score_plan(SAFE_PLAN, FakeModel(choice='safe', severity=0.4, destructive=0.01))
        self.assertFalse(report['gate']['approval_required'])
        self.assertEqual(report['gate']['action'], 'AUTO_APPROVED')

    def test_uncertainty_and_high_severity_gate(self):
        uncertain = tr.score_plan(SAFE_PLAN, FakeModel(choice='safe', severity=0.2, destructive=0.01, needs_review=True))
        self.assertTrue(uncertain['gate']['approval_required'])
        self.assertIn('uncertain', ' '.join(uncertain['gate']['reasons']))
        severe = tr.score_plan(SAFE_PLAN, FakeModel(choice='safe', severity=3.4, destructive=0.01))
        self.assertTrue(severe['gate']['approval_required'])
        self.assertIn('severity', ' '.join(severe['gate']['reasons']))

    def test_markdown_report_names_the_gate_and_the_resource(self):
        text = tr.format_markdown(tr.score_plan(PRODUCTION_PLAN, FakeModel()))
        self.assertIn('HUMAN APPROVAL REQUIRED', text)
        self.assertIn('module.network.aws_internet_gateway.main', text)
        self.assertIn('critical', text)

    def test_json_can_hide_the_state(self):
        report = tr.score_plan(PRODUCTION_PLAN, FakeModel())
        self.assertIn('"state"', tr.format_json(report, include_state=True))
        self.assertNotIn('"state"', tr.format_json(report))


class CliTests(unittest.TestCase):
    def run_cli(self, payload, *args, model=None):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
            handle.write(payload)
            name = handle.name
        out, err = io.StringIO(), io.StringIO()
        try:
            with patch.object(tr, '_load_model', return_value=model or FakeModel()), \
                    redirect_stdout(out), redirect_stderr(err):
                code = tr.main([name, *args])
        finally:
            Path(name).unlink()
        return code, out.getvalue(), err.getvalue()

    def test_risky_plan_exits_2_and_prints_json(self):
        code, out, _ = self.run_cli(json.dumps(PRODUCTION_PLAN), '--format', 'json')
        self.assertEqual(code, 2)
        self.assertTrue(json.loads(out)['gate']['approval_required'])

    def test_safe_plan_exits_0(self):
        code, _, _ = self.run_cli(json.dumps(SAFE_PLAN), model=FakeModel(choice='safe', severity=0.3, destructive=0.01))
        self.assertEqual(code, 0)

    def test_text_plan_and_out_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.md'
            code, out, _ = self.run_cli('Terraform will destroy the production internet gateway.',
                                        '--out', str(target))
            self.assertEqual(code, 2)
            self.assertEqual(out, '')
            self.assertIn('HUMAN APPROVAL REQUIRED', target.read_text())

    def test_model_error_is_exit_1(self):
        from s1ced.contract import DecisionError
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
            handle.write(json.dumps(SAFE_PLAN))
            name = handle.name
        err = io.StringIO()
        try:
            with patch.object(tr, '_load_model',
                              side_effect=DecisionError(503, 'model_loading', 'Vega is loading.')), \
                    redirect_stderr(err):
                code = tr.main([name])
        finally:
            Path(name).unlink()
        self.assertEqual(code, 1)
        self.assertIn('model_loading', err.getvalue())


if __name__ == '__main__':
    unittest.main()
