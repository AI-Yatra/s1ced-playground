#!/usr/bin/env python3
"""Terraform change risk scorer: turn a plan into a review gate with Vega 0.8.

Input is the JSON from ``terraform show -json plan.out`` (or the human-readable text of
``terraform plan``). The plan is scanned deterministically, then Vega answers three contract
questions: how to classify the change, how severe it is, and whether it destroys production.

The gate is deliberately conservative: a production delete/replace, an internet gateway removal,
a production database subnet recreation, a high severity score, a destructive classification, or
model uncertainty all require a human approval.

Usage:
  terraform-risk plan.json                     # markdown report; exit 2 when approval is required
  terraform-risk plan.json --format json       # machine-readable report for CI
  terraform-risk plan.json --model s1ced       # faster local model instead of Vega
  terraform-risk - < plan.txt                  # raw plan text from stdin, or a JSON plan

Exit codes: 0 auto-approved, 2 human approval required, 1 invalid input or model error.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from .contract import DecisionError, check_response

QUESTIONS = {
    'change_class': {
        'type': 'choice',
        'instructions': 'Which option best describes this case: Terraform change risk scorer?',
        'criteria': {
            'safe': 'Non-destructive change',
            'recreate': 'Resource recreation',
            'network': 'Critical network change',
            'destroy': 'Destructive removal',
        },
    },
    'severity': {
        'type': 'score',
        'instructions': 'Rate the infrastructure change severity in the supplied state.',
        'criteria': ['none or negligible', 'low', 'moderate', 'high', 'critical'],
    },
    'destructive_prod': {
        'type': 'noul',
        'instructions': 'Does the plan contain destructive production actions?',
        'criteria': {
            'true': 'A production resource is deleted or replaced',
            'false': 'No production resource is deleted or replaced',
        },
    },
}

NETWORK_RE = re.compile(
    r'internet_gateway|nat_gateway|vpc|subnet|route|firewall|security_group|network_acl|'
    r'load_balancer|listener|dns|gateway', re.I)
DATABASE_RE = re.compile(
    r'database|(?<![a-z0-9])db(?![a-z0-9])|db_|rds|aurora|dynamodb|sql|postgres|mysql|maria|redis|elasticache|cosmos',
    re.I)
PROD_RE = re.compile(r'(?:^|[^a-z0-9])(?:prod|prd|production)(?:[^a-z0-9]|$)', re.I)
DESTRUCTIVE_TEXT_RE = re.compile(r'will be destroyed|must be replaced|forces replacement|is tainted', re.I)
INTERNET_GATEWAY_TEXT_RE = re.compile(
    r'internet[_ ]gateway[^\n]{0,120}(?:destroy|delete|remove)|(?:destroy|delete|remove)[^\n]{0,120}internet[_ ]gateway',
    re.I)
DATABASE_RECREATE_TEXT_RE = re.compile(
    r'(?:database|db)[^\n]{0,120}(?:replace|recreat|destroy|delete)|(?:replace|recreat|destroy|delete)[^\n]{0,120}(?:database|db)',
    re.I)
INTERESTING_ATTRIBUTES = (
    'name', 'cidr_block', 'vpc_id', 'subnet_id', 'availability_zone', 'engine', 'engine_version',
    'instance_class', 'allocated_storage', 'multi_az', 'publicly_accessible', 'deletion_protection',
    'backup_retention_period', 'environment', 'env', 'tags', 'from_port', 'to_port', 'cidr_blocks',
    'protocol', 'rule_action',
)
MAX_RESOURCES = 200
MAX_ATTRIBUTE_CHARS = 300


def _clip(value, limit=MAX_ATTRIBUTE_CHARS):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + '...'


def _strings(value, depth=0):
    """Strings inside a plan value, bounded so a huge state cannot slow the scan."""
    if depth > 6:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.append(str(key))
            out.extend(_strings(item, depth + 1))
            if len(out) > 200:
                break
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_strings(item, depth + 1))
            if len(out) > 200:
                break
        return out
    return []


def _is_production(resource, plan, production, prod_re):
    if production is True:
        return True
    if production is False:
        return False
    haystack = [resource.get('address') or '', resource.get('module_address') or '']
    change = resource.get('change') or {}
    haystack.extend(_strings(change.get('before')))
    haystack.extend(_strings(change.get('after')))
    haystack.extend(_strings(plan.get('variables')))
    if plan.get('workspace'):
        haystack.append(str(plan['workspace']))
    return bool(prod_re.search(' '.join(haystack)))


def _action(actions):
    if 'create' in actions and 'delete' in actions:
        return 'replace'
    for action in ('delete', 'update', 'create', 'read'):
        if action in actions:
            return action
    return 'no-op'


def _attributes(data):
    if not isinstance(data, dict):
        return {}
    out = {}
    for key in INTERESTING_ATTRIBUTES:
        if key in data and data[key] not in (None, {}, []):
            out[key] = _clip(data[key])
    return out


def _resource_facts(resource, plan, production, prod_re):
    change = resource.get('change') or {}
    actions = list(change.get('actions') or [])
    action = _action(actions)
    resource_type = resource.get('type') or ''
    address = resource.get('address') or resource.get('name') or resource_type
    before = _attributes(change.get('before'))
    after = _attributes(change.get('after'))
    return {
        'address': address,
        'type': resource_type,
        'actions': actions,
        'action': action,
        'production': _is_production(resource, plan, production, prod_re),
        'network': bool(NETWORK_RE.search(resource_type) or NETWORK_RE.search(address)),
        'database': bool(DATABASE_RE.search(resource_type) or DATABASE_RE.search(address)),
        'internet_gateway': 'internet_gateway' in resource_type,
        'before': before,
        'after': after,
    }


def _counts(resources):
    counts = {key: 0 for key in ('create', 'update', 'delete', 'replace', 'read', 'no-op')}
    for resource in resources:
        counts[resource['action']] = counts.get(resource['action'], 0) + 1
    counts['total_changes'] = sum(counts[key] for key in ('create', 'update', 'delete', 'replace'))
    return counts


def _summary(resources, counts, production, prod_re):
    return {
        'destructive_production': any(r['production'] and r['action'] in ('delete', 'replace') for r in resources),
        'network_change': any(r['network'] and r['action'] not in ('no-op', 'read') for r in resources),
        'internet_gateway_removed': any(r['internet_gateway'] and 'delete' in r['actions'] for r in resources),
        'production_database_recreated': any(
            r['production'] and r['database'] and r['action'] == 'replace' for r in resources),
        'counts': counts,
        'production_mode': 'all resources' if production is True else (
            'none' if production is False else f'inferred with /{prod_re.pattern}/'),
    }


def _text_facts(text, production, prod_re):
    """Fallback scan for human-readable `terraform plan` output."""
    destructive = bool(DESTRUCTIVE_TEXT_RE.search(text))
    production_hit = bool(prod_re.search(text))
    is_prod = production is True or (production is None and production_hit)
    facts = {
        'kind': 'text',
        'source': None,
        'terraform_version': None,
        'workspace': None,
        'resources': [],
        'changed_resources': [],
        'destructive_production': bool(destructive and is_prod),
        'network_change': bool(NETWORK_RE.search(text)),
        'internet_gateway_removed': bool(INTERNET_GATEWAY_TEXT_RE.search(text)),
        'production_database_recreated': bool(is_prod and DATABASE_RECREATE_TEXT_RE.search(text)),
        'counts': {key: 0 for key in ('create', 'update', 'delete', 'replace', 'read', 'no-op', 'total_changes')},
        'production_mode': 'all resources' if production is True else (
            'none' if production is False else 'inferred from plan text'),
    }
    return text, facts


def build_state(plan, production=None, production_pattern=PROD_RE):
    """Return (state text for the model, deterministic facts about the plan).

    plan may be the parsed JSON object from ``terraform show -json`` or a string. A JSON object
    without ``resource_changes`` (or a string that is not a plan) is sent as text and scanned
    with the text rules.
    """
    if isinstance(plan, str):
        return _text_facts(plan, production, production_pattern)
    if not isinstance(plan, dict) or 'resource_changes' not in plan:
        return _text_facts(json.dumps(plan, ensure_ascii=False, indent=1, default=str), production, production_pattern)

    resources = [_resource_facts(rc, plan, production, production_pattern)
                 for rc in plan.get('resource_changes') or []
                 if isinstance(rc, dict) and rc.get('mode') != 'data']
    counts = _counts(resources)
    facts = _summary(resources, counts, production, production_pattern)
    changed = [r for r in resources if r['action'] not in ('no-op', 'read')]
    facts.update({
        'kind': 'plan_json',
        'source': None,
        'terraform_version': plan.get('terraform_version'),
        'workspace': plan.get('workspace'),
        'resources': resources,
        'changed_resources': changed,
    })

    lines = ['Terraform plan', f'Terraform version: {plan.get("terraform_version") or "unknown"}']
    if plan.get('workspace'):
        lines.append(f'Workspace: {plan["workspace"]}')
    lines.append(
        'Changes: {total_changes} (create {create}, update {update}, delete {delete}, replace {replace})'.format(**counts))
    lines.append(f'Destructive production actions: {"yes" if facts["destructive_production"] else "no"}')
    lines.append(f'Network changes: {"yes" if facts["network_change"] else "no"}')
    lines.append(f'Internet gateway removed: {"yes" if facts["internet_gateway_removed"] else "no"}')
    lines.append(
        f'Production database subnet recreated: {"yes" if facts["production_database_recreated"] else "no"}')
    lines.append('')
    lines.append('Changed resources:')
    if not changed:
        lines.append('none')
    for index, resource in enumerate(changed[:MAX_RESOURCES], 1):
        tags = []
        if resource['production']:
            tags.append('production')
        if resource['network']:
            tags.append('network')
        if resource['database']:
            tags.append('database')
        suffix = f' [{" ".join(tags)}]' if tags else ''
        lines.append(f'{index}. {resource["address"]} ({resource["type"]}): {resource["action"]}{suffix}')
        if resource['before']:
            lines.append(f'   before: {_clip(resource["before"])}')
        if resource['after']:
            lines.append(f'   after: {_clip(resource["after"])}')
    if len(changed) > MAX_RESOURCES:
        lines.append(f'... and {len(changed) - MAX_RESOURCES} more changed resources.')
    return '\n'.join(lines), facts


def _evaluate_gate(out, facts, high_severity, destructive_threshold):
    answers = out['answers']
    model_id = out.get('model') or 'the model'
    change_class = answers['change_class']['choice']
    severity = answers['severity']['score']
    destructive = answers['destructive_prod']['noul'] >= destructive_threshold
    uncertain = [qid for qid, answer in answers.items()
                 if (answer.get('uncertainty') or {}).get('needs_review') is True]
    reasons = []
    if facts.get('destructive_production'):
        reasons.append('the plan deletes or replaces a production resource')
    if facts.get('internet_gateway_removed'):
        reasons.append('the plan removes an internet gateway')
    if facts.get('production_database_recreated'):
        reasons.append('the plan recreates a production database subnet')
    if destructive:
        reasons.append(f'{model_id} says the plan contains destructive production actions '
                       f'({answers["destructive_prod"]["noul"]:.0%})')
    if change_class in ('recreate', 'network', 'destroy'):
        reasons.append(f'{model_id} classified the change as {change_class}')
    if severity >= high_severity:
        reasons.append(f'{model_id} severity {severity:.2f}/4 is at or above the review threshold {high_severity:.2f}')
    if uncertain:
        reasons.append(f'{model_id} is uncertain about: ' + ', '.join(uncertain))
    return {
        'approval_required': bool(reasons),
        'action': 'HUMAN_APPROVAL_REQUIRED' if reasons else 'AUTO_APPROVED',
        'reasons': reasons,
        'high_severity_threshold': high_severity,
        'destructive_threshold': destructive_threshold,
    }


def score_plan(plan, model, production=None, production_pattern=PROD_RE,
               high_severity=3.0, destructive_threshold=0.5, allow_truncation=False, source=None):
    """Scan a plan, ask the model, and return a report with the gate decision.

    model is any object with decide(state, questions, allow_truncation=False), such as
    ``s1ced.load("vega-0.8b")``.
    """
    state, facts = build_state(plan, production, production_pattern)
    facts['source'] = source
    out = model.decide(state, QUESTIONS, allow_truncation=allow_truncation)
    check_response(out, {'questions': QUESTIONS})
    answers = out['answers']
    severity = answers['severity']
    probabilities = severity['probabilities']
    top_level = max(probabilities, key=probabilities.get)
    report = {
        'scorer': 'terraform-change-risk',
        'model': out['model'],
        'plan': {
            'kind': facts['kind'],
            'source': facts['source'],
            'terraform_version': facts['terraform_version'],
            'workspace': facts['workspace'],
            'production_mode': facts['production_mode'],
            'counts': facts['counts'],
            'destructive_production': facts['destructive_production'],
            'network_change': facts['network_change'],
            'internet_gateway_removed': facts['internet_gateway_removed'],
            'production_database_recreated': facts['production_database_recreated'],
            'changed_resources': [
                {key: resource[key] for key in ('address', 'type', 'action', 'production', 'network', 'database')}
                for resource in facts['changed_resources']
            ],
        },
        'risk': {
            'change_class': answers['change_class']['choice'],
            'change_class_confidence': answers['change_class']['confidence'],
            'severity': severity['score'],
            'severity_label': severity['legend'][top_level],
            'severity_confidence': severity['confidence'],
            'destructive_production': answers['destructive_prod']['noul'] >= destructive_threshold,
            'destructive_production_probability': answers['destructive_prod']['noul'],
            'needs_review': any((answer.get('uncertainty') or {}).get('needs_review') is True
                                for answer in answers.values()),
        },
        'gate': _evaluate_gate(out, facts, high_severity, destructive_threshold),
        'answers': answers,
        'state': state,
        'usage': out.get('usage'),
        'latency_ms': out.get('latency_ms'),
        'warnings': out.get('warnings') or [],
    }
    return report


def _answer_cell(answer):
    if answer['type'] == 'noul':
        return f'{"yes" if answer["noul"] >= 0.5 else "no"} ({answer["noul"]:.0%})'
    if answer['type'] == 'choice':
        return f'{answer["choice"]} ({answer["confidence"]:.0%})'
    top = max(answer['probabilities'], key=answer['probabilities'].get)
    return f'{answer["score"]:.2f} / {len(answer["legend"]) - 1} - {answer["legend"][top]} ({answer["confidence"]:.0%})'


def format_markdown(report):
    plan, gate = report['plan'], report['gate']
    lines = ['# Terraform change risk', '']
    lines.append(f'**Gate: {gate["action"].replace("_", " ")}**')
    lines.append('')
    lines.append(f'Model: `{report["model"]}`  ')
    if plan.get('source'):
        lines.append(f'Plan: `{plan["source"]}`  ')
    lines.append(f'Changes: {plan["counts"]["total_changes"]} '
                 f'(create {plan["counts"]["create"]}, update {plan["counts"]["update"]}, '
                 f'delete {plan["counts"]["delete"]}, replace {plan["counts"]["replace"]})')
    lines.append('')
    lines.append('| Question | Answer |')
    lines.append('|---|---|')
    labels = {'change_class': 'Change class', 'severity': 'Severity', 'destructive_prod': 'Destructive production'}
    for qid in ('change_class', 'severity', 'destructive_prod'):
        lines.append(f'| {labels[qid]} | {_answer_cell(report["answers"][qid])} |')
    lines.append('')
    lines.append('## Plan scan')
    lines.append('')
    for label, key in (('Destructive production', 'destructive_production'),
                       ('Network change', 'network_change'),
                       ('Internet gateway removed', 'internet_gateway_removed'),
                       ('Production database recreated', 'production_database_recreated')):
        lines.append(f'- {label}: {"yes" if plan[key] else "no"}')
    if plan['changed_resources']:
        lines.append('')
        lines.append('Changed resources:')
        for resource in plan['changed_resources']:
            tags = [tag for tag in ('production', 'network', 'database') if resource[tag]]
            suffix = f' [{" ".join(tags)}]' if tags else ''
            lines.append(f'- `{resource["address"]}` ({resource["type"]}): {resource["action"]}{suffix}')
    if gate['reasons']:
        lines.append('')
        lines.append('## Why a human must approve')
        lines.append('')
        for reason in gate['reasons']:
            lines.append(f'- {reason}')
    if report.get('warnings'):
        lines.append('')
        lines.append('## Warnings')
        lines.append('')
        for warning in report['warnings']:
            lines.append(f'- {warning}')
    return '\n'.join(lines)


def format_json(report, include_state=False):
    data = report if include_state else {key: value for key, value in report.items() if key != 'state'}
    return json.dumps(data, indent=2, ensure_ascii=False)


def _load_model(name, device=None):
    from .models import load
    if name.startswith('jev-'):
        return load(name)
    return load(name, device=device) if device else load(name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='terraform-risk', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plan', help='JSON from "terraform show -json plan.out", plain plan text, or - for stdin')
    parser.add_argument('--model', default='vega-0.8b',
                        help='vega-0.8b (default), vega-4b, s1ced, or a jev- model')
    parser.add_argument('--device', help='Vega or S1-CED device: cuda, mps or cpu')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--production', dest='production', action='store_true', default=None,
                       help='treat every resource as production')
    group.add_argument('--no-production', dest='production', action='store_false',
                       help='do not infer production from names, tags or plan variables')
    parser.add_argument('--production-pattern', default=PROD_RE.pattern,
                        help=f'regex for production markers (default: {PROD_RE.pattern})')
    parser.add_argument('--high-severity', type=float, default=3.0,
                        help='severity score at which a human must approve (default: 3.0 of 4)')
    parser.add_argument('--destructive-threshold', type=float, default=0.5,
                        help='probability above which the model counts as a yes (default: 0.5)')
    parser.add_argument('--format', choices=('markdown', 'json'), default='markdown')
    parser.add_argument('--out', type=Path, help='write the report to this file instead of stdout')
    parser.add_argument('--include-state', action='store_true', help='include the text sent to the model in JSON')
    parser.add_argument('--allow-truncation', action='store_true',
                        help='let the model shorten input it cannot read in full, with a warning')
    args = parser.parse_args(argv)

    try:
        raw = sys.stdin.read() if args.plan == '-' else Path(args.plan).read_text(encoding='utf-8-sig')
    except OSError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    try:
        plan = json.loads(raw)
    except ValueError:
        plan = raw
    try:
        pattern = re.compile(args.production_pattern, re.I)
    except re.error as error:
        print(f'error: --production-pattern: {error}', file=sys.stderr)
        return 1
    try:
        model = _load_model(args.model, args.device)
        report = score_plan(
            plan, model,
            production=args.production,
            production_pattern=pattern,
            high_severity=args.high_severity,
            destructive_threshold=args.destructive_threshold,
            allow_truncation=args.allow_truncation,
            source=None if args.plan == '-' else args.plan,
        )
    except DecisionError as error:
        print(json.dumps({'status': error.status, **error.body()}, indent=2), file=sys.stderr)
        return 1
    except Exception as error:
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1

    text = (format_json(report, args.include_state) if args.format == 'json' else format_markdown(report))
    if args.out:
        try:
            args.out.write_text(text + '\n', encoding='utf-8')
        except OSError as error:
            print(f'error: {error}', file=sys.stderr)
            return 1
    else:
        print(text)
    return 0 if not report['gate']['approval_required'] else 2


if __name__ == '__main__':
    sys.exit(main())
