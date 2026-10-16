# Use S1-CED or Vega in your own code, the TypeSafe way

No Jev credits? You can still build with TypeSafe's SDKs, docs and cookbooks. This package runs a local copy of TypeSafe's API on your computer, answered by S1-CED or Vega instead of Jev. Your code does not change: the same SDK calls, the same questions, the same answers. When you get credits, you point the same code back at TypeSafe.

The rules both sides follow are in [CONTRACT.md](CONTRACT.md).

## 1. Install

You need Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
```

**Without an NVIDIA GPU** (S1-CED is fast; Vega works but takes 5 to 12 seconds per request):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "s1ced @ git+https://github.com/AI-Yatra/s1ced-playground"
```

**With an NVIDIA GPU** (Vega answers in about 0.2 seconds), install the CUDA build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "s1ced @ git+https://github.com/AI-Yatra/s1ced-playground"
```

Then install TypeSafe's SDK for the language you build in:

```bash
pip install typesafe-sdk          # Python
npm install @typesafe-ai/sdk      # JavaScript or TypeScript
```

## 2. Start the local server

```bash
s1ced --serve --vega-preload
```

Leave it running. It prints its address, normally `http://127.0.0.1:8770`. If that port is busy it uses the next free one; use the printed address in step 3.

- `--vega-preload` starts loading Vega right away. The first time, it downloads about 1.8 GB. Leave it out if you only use S1-CED.
- To keep TypeSafe cookbooks and examples that name `jev-latest` working without credits, add `--jev-fallback vega-0.8b` (see step 5).

## 3. Point the SDK at it

Set three environment variables in the terminal where your code runs. TypeSafe's Python and JavaScript SDKs both read them.

macOS or Linux:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8770
export TYPESAFE_API_KEY=local
export TYPESAFE_DEFAULT_MODEL=vega-0.8b
```

Windows PowerShell:

```powershell
$env:TYPESAFE_BASE_URL = "http://127.0.0.1:8770"
$env:TYPESAFE_API_KEY = "local"
$env:TYPESAFE_DEFAULT_MODEL = "vega-0.8b"
```

`TYPESAFE_API_KEY` must be set because the SDK refuses to start without one. The local models ignore it, so any placeholder such as `local` works.

## 4. Write code exactly as in TypeSafe's docs

Python:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ticket = "Our API integration started returning 500 errors on every request about 20 minutes ago."

with TypeSafeClient() as client:   # reads the three variables from step 3
    response = client.system_one(
        state=ticket,
        questions={
            "department": Choice(instructions="Which team should handle this",
                                 criteria={"billing": "Payment or subscription issues",
                                           "technical": "Bugs or integration problems",
                                           "sales": "Pricing or account questions"}),
            "frustration": Score(instructions="How frustrated the customer appears",
                                 criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]),
            "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
        },
    )

print(response.model)                              # which model answered, e.g. vega-0.8b
print(response.choices["department"].choice)
print(response.scores["frustration"].score)
print(response.nouls["is_urgent"].noul)
```

JavaScript:

```javascript
import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient();   // reads the three variables from step 3
const response = await client.systemOne({
  state: "Our API integration started returning 500 errors on every request about 20 minutes ago.",
  questions: {
    department: choice("Which team should handle this", {
      billing: "Payment or subscription issues",
      technical: "Bugs or integration problems",
      sales: "Pricing or account questions",
    }),
    is_urgent: noul("The message conveys urgency or time-sensitivity"),
  },
});
console.log(response.model, response.answers.department.choice, response.answers.is_urgent.noul);
```

To pick a model in code instead of through the environment, pass it: `TypeSafeClient(model="s1ced")` in Python, `new TypeSafeClient({ defaultModel: "s1ced" })` in JavaScript.

Without the SDK, any HTTP client works. Send the same JSON that TypeSafe's [API reference](https://docs.typesafe.ai/api) describes to `POST http://127.0.0.1:8770/v1/systemone`.

## 5. Use TypeSafe's cookbooks and examples

TypeSafe's examples usually write the model into the code, for example `TypeSafeClient(model="jev-latest")`. Two ways to run them without credits:

- Change `"jev-latest"` to `"vega-0.8b"` or `"s1ced"` in the code.
- Or start the server with `--jev-fallback vega-0.8b` and run the code unchanged. With no TypeSafe key, every `jev-` request is answered by Vega. The answer is honest about it: `response.model` is `vega-0.8b`, and `warnings` says the fallback was used.

## Vega in your harness, just like Jev

Anything that calls Jev can call Vega the same way: an agent loop, an eval harness, a guardrail in front of an LLM, a batch script. With the three variables from step 3 set, `TypeSafeClient()` talks to Vega. The request, the answer objects (`response.choices`, `response.nouls`, `response.scores`), the errors and the retries behave as they do with Jev. Four practical differences:

1. **Load it before your harness starts.** Start the server with `--vega-preload` and wait for `vega-0.8b ready` in its terminal. A request that arrives while Vega is still loading gets `503` with a `Retry-After` header, as a busy hosted API would. The SDK retries twice and then raises an error that says Vega is loading.
2. **Long text on a CPU needs a longer timeout.** The SDKs give up after 10 seconds. Short requests take under a second even on a CPU, but a long document with several questions can take 5 to 12 seconds without a GPU. Raise the timeout in that case: `TypeSafeClient(timeout=60)` in Python, `new TypeSafeClient({ timeout: 60000 })` in JavaScript (milliseconds).
3. **One request at a time.** The server answers Vega requests in order. Parallel requests from your harness work, but they wait their turn, so on a CPU a batch takes the sum of its requests.
4. **Different limits and accuracy.** Vega keeps each option with its description to 32 tokens and instructions to 256, and is less accurate than Jev (see the table below). It also returns an `uncertainty` block with `abstain` and `out_of_scope`, which Jev does not. TypeSafe's SDKs ignore it; read it from the raw response, or from `s1ced.load("vega-0.8b")` in Python, if your harness can use it.

Without the server, Python code can also call Vega directly. The answers are plain dictionaries in the same shape as the HTTP API:

```python
import s1ced

vega = s1ced.load("vega-0.8b")   # starts loading in the background; the first call waits for it
out = vega.decide(state, {"urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}})
print(out["answers"]["urgent"]["noul"], out["answers"]["urgent"]["uncertainty"])
```

## 6. Switch to Jev when you have credits

Remove `TYPESAFE_BASE_URL`, set `TYPESAFE_API_KEY` to your real key, and set the model to `jev-latest`. Nothing else changes.

You can also keep the local server and give it your key (`TYPESAFE_API_KEY` in the server's terminal, or **Add key** in the playground). It then forwards `jev-` requests to TypeSafe, and you can compare all three models from the same code by changing only `model`.

## Which model to use

| | `s1ced` | `vega-0.8b` | `jev-latest` |
|---|---|---|---|
| Runs | your computer | your computer | TypeSafe |
| Needs | nothing extra | 1.8 GB download; a GPU for speed | a TypeSafe key and credits |
| Speed per request | about 0.1 s on CPU | about 0.2 s on GPU, 5 to 12 s on CPU | about 0.4 s |
| On this repo's 38 example answers | 16 right | 20 to 21 right | 35 right |
| Longest state | 510 tokens | 73,728 tokens | 64k tokens |
| Options per choice | 8 | 255; each option with its description at most 32 tokens | 255 |
| Instructions | 91 tokens together with the longest option | 256 tokens | within the context |
| Uncertainty signals | prediction set | prediction set, abstain, out of scope | confidence only |

S1-CED is fast but often wrong on text unlike its training data, so check its answers against your own examples before relying on it. Vega is the stronger local model. Jev is the most accurate of the three.

## What is different from TypeSafe's service

- Local answers carry the shared `uncertainty` block (see [CONTRACT.md](CONTRACT.md)). TypeSafe's SDKs ignore fields they do not know, so this never breaks them; read it from the raw response if you want it.
- Not available locally: `box` questions and images (`assets`). A request that uses them is rejected with a clear error, not ignored. The `TypeSafe-Danger-Zone-Long-Context` header has no effect locally.
- Input longer than a model can read is refused with `400 max_tokens_exceeded` and a message naming the part that is too long, as on TypeSafe. Each model has its own limits (table above). In Python, `decide(..., allow_truncation=True)` lets the model shorten the input instead, and `warnings` says what was cut.
- The local server only accepts requests from your own computer.

## When something goes wrong

| You see | Fix |
|---|---|
| `No API key was provided` | Set `TYPESAFE_API_KEY=local` (step 3). |
| Connection refused | Start the server (step 2) and check the address it printed. |
| `404 not found` | Something else, such as an older copy of the playground, is using that address. Use the address `s1ced --serve` printed, or stop the other program. |
| `401 No TypeSafe key set` | Your code asks for `jev-latest`. Set `TYPESAFE_DEFAULT_MODEL`, change the model in code, or start the server with `--jev-fallback vega-0.8b`. |
| `503` from Vega | It is still loading, or failed to load. The server terminal says which. |
| `400 max_tokens_exceeded` from S1-CED | The state is over 510 tokens, or a question with its longest option is over 91 tokens. Use Vega, or shorten it. |
| `400 max_tokens_exceeded` from Vega | Usually an option description over 32 tokens. Shorten the descriptions. |
| `400 too_many_options` from S1-CED | S1-CED takes up to 8 options. Use Vega, or split the question. |
| `404 model_not_found` | Use `s1ced`, `vega-0.8b`, `vega-4b` or a `jev-` model. `GET /v1/models` lists them. |
