# The decision contract

S1-CED, Vega and Jev follow the same rules in this package. A request written for one model works unchanged on the others, and every answer comes back in the same shape. The contract is TypeSafe's [System One API](https://docs.typesafe.ai/api), so code written against the local server also works against TypeSafe, and TypeSafe's own SDKs work against the local server.

The rules are enforced in code, not only described here: [`contract.py`](../src/s1ced/contract.py) checks every request before any model sees it and every response before it leaves. A model that broke a rule would fail with error 502 rather than return a differently shaped answer.

## Three ways in, one contract

| From | Call |
|---|---|
| Python | `s1ced.load("vega-0.8b").decide(state, questions)` or `s1ced.decide({"model": ..., "state": ..., "questions": ...})` |
| HTTP, any language | `POST http://127.0.0.1:8770/v1/systemone` while `s1ced-playground` or `s1ced --serve` runs |
| TypeSafe SDKs | `TypeSafeClient(base_url="http://127.0.0.1:8770", api_key="local", model="s1ced")` |
| Command line | `s1ced --input request.json` |

## Models

| `model` | Runs | Notes |
|---|---|---|
| `s1ced` | your computer, CPU | 38M parameters. Reads up to 510 state tokens and 8 options per question. |
| `vega-0.8b`, `vega-4b` | your computer, GPU recommended | Loads on first use (the 0.8B downloads 1.8 GB). `vega` means `vega-0.8b`. |
| `jev-latest`, `jev-preview`, `jev-1.14.0`, ... | TypeSafe | Needs a TypeSafe key. Billed per input token. |

`GET /v1/models` lists them in TypeSafe's format.

## The request

```json
{"model": "s1ced",
 "state": "text, or any JSON object or array",
 "questions": {
   "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?",
              "criteria": {"true": "blocking or time-sensitive", "false": "can wait"}},
   "team":   {"type": "choice", "instructions": "Which team owns this?",
              "criteria": {"technical": "outages and access", "billing": null}},
   "mood":   {"type": "score", "instructions": "How frustrated is the customer?",
              "criteria": ["Calm", "Annoyed", "Angry"]}}}
```

Rules, the same for every model:

1. A request has exactly `model`, `state` and `questions`. Unknown fields are rejected, so a typo cannot be silently ignored.
2. `state` is required: a string, object or array.
3. Each question has `type`, `instructions` and, depending on the type, `criteria`. There is no default type.
   - `noul`: a yes/no question. `criteria` is optional, with only `true` and `false`.
   - `choice`: `criteria` maps 2 to 255 option names to a description or `null`.
   - `score`: `criteria` is a list of 2 to 10 distinct levels, lowest first.
4. Every model reads the option descriptions. They are part of the question.
5. Numbers must be finite, and values may nest at most 64 levels.
6. Question ids are for your code. No model ever sees them.
7. No model shortens your input without telling you. Input longer than a model can read is refused with `400 max_tokens_exceeded`. In Python and on the command line you can pass `allow_truncation`; the model then shortens it and says so in `warnings`.

A model may support less than the contract allows. It then answers with a clear `400` error, never a quietly different result:

| model | state | instructions | options |
|---|---|---|---|
| `s1ced` | 510 tokens | the question with its longest option at most 91 tokens | at most 8 |
| `vega-0.8b`, `vega-4b` | about 73,000 tokens | 256 tokens | each option with its description at most 32 tokens |
| `jev-*` | 64k tokens per request | within the context | at most 255 |

Vega's own library shortens over-long input without saying so. The adapter measures each request with Vega's own prompt builder first, so Vega follows the same rule as the others.

## The response

```json
{"model": "vega-0.8b",
 "answers": {
   "urgent": {"type": "noul", "noul": 0.94, "uncertainty": {...}},
   "team":   {"type": "choice", "choice": "technical", "probabilities": {"technical": 0.97, "billing": 0.03},
              "confidence": 0.97, "uncertainty": {...}},
   "mood":   {"type": "score", "score": 1.7, "legend": {"0": "Calm", "1": "Annoyed", "2": "Angry"},
              "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}, "confidence": 0.75, "uncertainty": {...}}},
 "usage": {"input_tokens": 246, "output_tokens": 0},
 "latency_ms": 212.4,
 "warnings": []}
```

1. `model` is the exact version that answered, such as `s1ced-v3-deep` or `jev-1.13.0`.
2. Every question id gets exactly one answer of the same type.
3. Probabilities cover exactly the options you asked about and sum to 1. Score levels are keyed `"0"` to `"n-1"`. `score` is the probability-weighted level, so it can fall between levels.
4. `usage.input_tokens` is what the model read. Local models have no output tokens. Jev answers also carry `cost_usd`.
5. `latency_ms` and `warnings` are always present.

### One uncertainty block for every model

Every answer has the same `uncertainty` object. A signal a model does not produce is `null`, never left out:

| field | meaning | S1-CED | Vega | Jev |
|---|---|---|---|---|
| `prediction_set` | answers the model cannot rule out, in the same labels as `probabilities` (`"yes"`/`"no"` for noul) | yes | yes | null |
| `abstain` | the model would rather not stand behind this answer | null | yes | null |
| `out_of_scope` | the question looks unlike anything the model was trained on | null | yes | null |
| `needs_review` | one rule for all: true if it abstains, is out of scope, or keeps more than one answer in its prediction set | yes | yes | null |

`needs_review` is the model's own signal. Your thresholds (the playground's gates) are your policy, and you apply them in your code the same way for every model.

## Errors

Every error, from every model, has an HTTP status and the same body:

```json
{"detail": {"error_type": "invalid_request", "message": "score criteria must be a list of 2 to 10 levels, lowest first.",
            "loc": ["questions", "mood", "criteria"]}}
```

| status | when |
|---|---|
| 400 | a limit: `max_tokens_exceeded`, `too_many_options`, invalid JSON or numbers, nesting over 64 |
| 401 | Jev without a valid TypeSafe key |
| 403 | a request from outside this computer to the local server |
| 404 | an unknown `model` |
| 413 | a body over 2 MB on the local server |
| 415 | a `Content-Encoding` other than gzip, zstd or none |
| 422 | a request that breaks a rule above |
| 429, 5xx from Jev | passed through from TypeSafe |
| 502 | a model returned something that breaks the response rules, or TypeSafe could not be reached |
| 503 | Vega is not installed, failed to load, or is still loading |

In Python the same errors are raised as `s1ced.DecisionError` with `.status`, `.error_type`, `.message` and `.loc`. TypeSafe's SDKs raise their usual typed errors, such as `TypeSafeUnprocessableEntityError` for 422.

## Keys for Jev

The local server uses, in order, `TYPESAFE_API_KEY`, the key saved in the playground, and otherwise the caller's `Authorization: Bearer` token. Placeholder tokens such as `local` are never forwarded. Local models ignore the key.
