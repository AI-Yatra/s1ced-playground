"""The decision contract every model in this package follows.

It is TypeSafe's System One API (https://docs.typesafe.ai/api), so a request written for one model works unchanged
on the others, including TypeSafe's own SDKs pointed at the local server. See docs/CONTRACT.md for the rules in prose.

Request:   {"model": "...", "state": str | object | array,
            "questions": {"<id>": {"type": "noul" | "choice" | "score", "instructions": ..., "criteria": ...}}}
Response:  {"model": "<versioned id>", "answers": {"<id>": <answer>}, "usage": {"input_tokens": int, "output_tokens": int},
            "latency_ms": float, "warnings": [str]}
Answers:   noul   {"type", "noul"}
           choice {"type", "choice", "probabilities", "confidence"}
           score  {"type", "score", "legend", "probabilities", "confidence"}
           and, from every model, "uncertainty": {"prediction_set", "abstain", "out_of_scope", "needs_review"}, with
           null for any signal that model does not produce.
Errors:    HTTP status plus {"detail": {"error_type": "...", "message": "...", "loc": [...]}}.
"""
import json, math

QUESTION_TYPES = ('noul', 'choice', 'score')
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
MAX_DEPTH = 64
REQUEST_KEYS = {'model', 'state', 'questions'}
QUESTION_KEYS = {'type', 'instructions', 'criteria'}
UNCERTAINTY_KEYS = ('prediction_set', 'abstain', 'out_of_scope', 'needs_review')
PROB_SUM_TOLERANCE = 0.02


class DecisionError(Exception):
    """A contract error. status follows the TypeSafe API: 400 limits, 401 key, 404 model, 422 invalid request,
    429 rate limit, 502 upstream or invalid model output, 503 model still loading."""

    def __init__(self, status, error_type, message, loc=None, retry_after=None):
        super().__init__(message)
        self.status, self.error_type, self.message, self.loc = status, error_type, message, loc
        self.retry_after = retry_after   # seconds, sent as a Retry-After header so SDKs wait and retry

    def body(self):
        d = {'error_type': self.error_type, 'message': self.message}
        if self.loc: d['loc'] = list(self.loc)
        return {'detail': d}


def _invalid(message, *loc):
    return DecisionError(422, 'invalid_request', message, loc)


def loads(raw):
    """Parse a request body: UTF-8 JSON without NaN or Infinity, at most MAX_DEPTH levels deep."""
    def bad_constant(name): raise ValueError(f'{name} is not a finite number')
    try:
        if isinstance(raw, (bytes, bytearray)): raw = raw.decode('utf-8')
        return json.loads(raw, parse_constant=bad_constant)
    except RecursionError:
        raise DecisionError(400, 'invalid_json', f'JSON is nested more than {MAX_DEPTH} levels deep.')
    except (UnicodeDecodeError, ValueError) as e:
        raise DecisionError(400, 'invalid_json', f'Request body is not valid UTF-8 JSON: {e}')


def _check_value(x, loc, depth=0):
    """state, instructions and criteria hold strings, objects, arrays and finite numbers, at most MAX_DEPTH deep."""
    if depth > MAX_DEPTH: raise DecisionError(400, 'too_deep', f'More than {MAX_DEPTH} nested levels.', loc)
    if isinstance(x, dict):
        for k, v in x.items(): _check_value(v, loc + (k,), depth + 1)
    elif isinstance(x, list):
        for i, v in enumerate(x): _check_value(v, loc + (i,), depth + 1)
    elif isinstance(x, float) and not math.isfinite(x):
        raise DecisionError(400, 'invalid_number', 'Numbers must be finite.', loc)
    elif not (x is None or isinstance(x, (str, int, float, bool))):
        raise _invalid(f'Unsupported value of type {type(x).__name__}.', *loc)


def _text_like(x): return isinstance(x, (str, dict, list))


def text_of(x):
    """A state, instruction or criterion as text, for models that only read text."""
    if x is None: return ''
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def option_labels(q):
    """The labels an answer's probabilities and prediction set use: ['no', 'yes'], the choice names, or '0'..'n-1'."""
    if q['type'] == 'noul': return ['no', 'yes']
    if q['type'] == 'choice': return list(q['criteria'])
    return [str(i) for i in range(len(q['criteria']))]


def check_request(req, model_required=True):
    """Validate a request against the contract and return it. Raises DecisionError (422, or 400 for limits)."""
    if not isinstance(req, dict): raise _invalid('The request must be a JSON object.')
    extra = set(req) - REQUEST_KEYS
    if extra: raise _invalid(f'Unknown field(s): {", ".join(sorted(extra))}. A request has model, state and questions.', sorted(extra)[0])
    if model_required and not (isinstance(req.get('model'), str) and req['model'].strip()):
        raise _invalid('model is required, for example "s1ced", "vega-0.8b" or "jev-latest".', 'model')
    if 'model' in req and not isinstance(req['model'], str): raise _invalid('model must be a string.', 'model')
    if 'state' not in req: raise _invalid('state is required: a string, object or array.', 'state')
    if not _text_like(req['state']): raise _invalid('state must be a string, object or array.', 'state')
    _check_value(req['state'], ('state',))
    qs = req.get('questions')
    if not isinstance(qs, dict) or not qs: raise _invalid('questions must be a non-empty object keyed by question id.', 'questions')
    for qid, q in qs.items():
        loc = ('questions', qid)
        if not isinstance(qid, str) or not qid.strip(): raise _invalid('Question ids must be non-empty strings.', *loc)
        if not isinstance(q, dict): raise _invalid('Each question must be an object.', *loc)
        extra = set(q) - QUESTION_KEYS
        if extra: raise _invalid(f'Unknown question field(s): {", ".join(sorted(extra))}.', *loc, sorted(extra)[0])
        t = q.get('type')
        if t not in QUESTION_TYPES: raise _invalid(f'type must be one of {", ".join(QUESTION_TYPES)}.', *loc, 'type')
        ins = q.get('instructions')
        if not _text_like(ins) or (isinstance(ins, str) and not ins.strip()):
            raise _invalid('instructions are required: a non-empty string, object or array.', *loc, 'instructions')
        _check_value(ins, loc + ('instructions',))
        crit = q.get('criteria')
        if crit is not None: _check_value(crit, loc + ('criteria',))
        if t == 'noul':
            if crit is not None and (not isinstance(crit, dict) or set(crit) - {'true', 'false'}):
                raise _invalid('noul criteria must be an object with only "true" and "false".', *loc, 'criteria')
        elif t == 'choice':
            if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE_OPTIONS:
                raise _invalid(f'choice criteria must map 2 to {MAX_CHOICE_OPTIONS} option names to descriptions or null.', *loc, 'criteria')
            if any(not k.strip() for k in crit): raise _invalid('Option names must be non-empty.', *loc, 'criteria')
            if any(v is not None and not _text_like(v) for v in crit.values()):
                raise _invalid('Option descriptions must be a string, object, array or null.', *loc, 'criteria')
        else:
            if not isinstance(crit, list) or not MIN_SCORE_LEVELS <= len(crit) <= MAX_SCORE_LEVELS:
                raise _invalid(f'score criteria must be a list of {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, lowest first.', *loc, 'criteria')
            if any(not _text_like(c) for c in crit): raise _invalid('Score levels must be strings, objects or arrays.', *loc, 'criteria')
            if len({text_of(c) for c in crit}) != len(crit): raise _invalid('Score levels must be distinct.', *loc, 'criteria')
    return req


def uncertainty(prediction_set=None, abstain=None, out_of_scope=None):
    """The one uncertainty block every model returns. needs_review follows a single rule for all models: true when
    the model abstains, says the question is out of scope, or cannot narrow its prediction set to one answer;
    null when the model produces none of these signals."""
    signals = [s for s in (abstain, out_of_scope) if s is not None]
    if prediction_set is not None: signals.append(len(prediction_set) != 1)
    return {'prediction_set': prediction_set, 'abstain': abstain, 'out_of_scope': out_of_scope,
            'needs_review': any(signals) if signals else None}


def _prob(x, where):
    if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
        raise _bad_output(f'{where} must be a number from 0 to 1.')
    return float(x)


def _bad_output(message): return DecisionError(502, 'invalid_model_response', message)


def check_response(resp, req):
    """Hold a model's response to the contract: same ids, same types, probabilities that sum to 1 over exactly the
    requested options. Raises DecisionError(502) so a broken adapter can never pass as an answer."""
    if not isinstance(resp, dict) or not isinstance(resp.get('model'), str): raise _bad_output('Response needs a model string.')
    answers, qs = resp.get('answers'), req['questions']
    if not isinstance(answers, dict) or set(answers) != set(qs):
        raise _bad_output('Response must answer exactly the requested question ids.')
    for qid, q in qs.items():
        a, where = answers[qid], f'answers.{qid}'
        if not isinstance(a, dict) or a.get('type') != q['type']: raise _bad_output(f'{where} must be a {q["type"]} answer.')
        labels = option_labels(q)
        if q['type'] == 'noul':
            a['noul'] = _prob(a.get('noul'), f'{where}.noul')
        else:
            probs = a.get('probabilities')
            if not isinstance(probs, dict) or set(probs) != set(labels):
                raise _bad_output(f'{where}.probabilities must cover exactly the requested options.')
            a['probabilities'] = {k: _prob(probs[k], f'{where}.probabilities.{k}') for k in labels}
            if abs(sum(a['probabilities'].values()) - 1) > PROB_SUM_TOLERANCE: raise _bad_output(f'{where}.probabilities must sum to 1.')
            a['confidence'] = _prob(a.get('confidence'), f'{where}.confidence')
            if q['type'] == 'choice':
                if a.get('choice') not in labels: raise _bad_output(f'{where}.choice must be one of the requested options.')
            else:
                s = a.get('score')
                if type(s) not in (int, float) or not 0 <= s <= len(labels) - 1: raise _bad_output(f'{where}.score is out of range.')
                a['score'] = float(s)
                if not isinstance(a.get('legend'), dict) or set(a['legend']) != set(labels): raise _bad_output(f'{where}.legend must cover every level.')
        u = a.get('uncertainty')
        if not isinstance(u, dict) or set(u) != set(UNCERTAINTY_KEYS): raise _bad_output(f'{where}.uncertainty must have {", ".join(UNCERTAINTY_KEYS)}.')
        if u['prediction_set'] is not None and (not isinstance(u['prediction_set'], list) or set(u['prediction_set']) - set(labels)):
            raise _bad_output(f'{where}.uncertainty.prediction_set must list requested options.')
        if any(u[k] is not None and not isinstance(u[k], bool) for k in ('abstain', 'out_of_scope', 'needs_review')):
            raise _bad_output(f'{where}.uncertainty flags must be true, false or null.')
    usage = resp.get('usage')
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')):
        raise _bad_output('usage must have whole-number input_tokens and output_tokens.')
    if not isinstance(resp.get('warnings', []), list): raise _bad_output('warnings must be a list.')
    return resp
