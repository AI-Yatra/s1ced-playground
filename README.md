# S1-CED playground

S1-CED is a small AI model that answers questions about a piece of text with probabilities, not just answers. It runs on your own computer, no GPU or account needed. This playground lets you try it in your browser.

You give it some text (a support ticket, a resume, a proposed agent action) and questions like:

- **yes/no:** "Is this action safe to run without a human approving it?"
- **choice:** "Which team should own this ticket: technical, billing or account management?"
- **score:** "How frustrated is the customer, from 0 (calm) to 3 (very angry)?"

For every question it shows the probability of each answer and tells you whether it is confident enough to act on, or whether a human should check.

![S1-CED playground](docs/screenshot.jpg)

## Run it in 3 steps

### Step 1: Install uv

[uv](https://docs.astral.sh/uv/) installs Python and everything else the playground needs. You only do this once.

**Windows** (open PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS or Linux** (open Terminal):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then **close the terminal and open a new one** so it finds `uv`.

### Step 2: Start the playground

```bash
uvx --torch-backend cpu --from git+https://github.com/AI-Yatra/s1ced-playground s1ced-playground
```

The first run takes a few minutes. It downloads PyTorch and the model (about 155 MB). Later runs start in seconds.

When you see this, it is ready:

```
S1-CED playground (deep_fusion, cpu) ready at http://127.0.0.1:8770
```

### Step 3: Use it

Your browser opens the playground and a short tutorial starts. If the browser does not open, go to the address printed in the terminal.

1. Click a scenario in the left sidebar, for example **Agent guardrail**.
2. Click **Run decision** (or press Ctrl+Enter).
3. Read the result on the right: each answer has a probability bar. **escalate** means the model is unsure and a human should decide. **singleton set** means it is confident.
4. Change the text or the questions in the middle panel and run again.

To stop the playground, press **Ctrl+C** in the terminal.

## Try your own cases

- Edit the text in the **STATE** box and the questions below it, then run again.
- Click **+** next to the questions to add one, and pick its type: `noul` (yes/no), `choice` or `score`.
- Switch the middle panel to **JSON** to see or paste the whole request.
- The scenario groups (resume screening, phishing triage and others) come with expected answers, so the header shows how many the model got right. Use them to judge whether the model is good enough for your use.
- To load your own scenario file, copy [src/s1ced/examples.json](src/s1ced/examples.json), edit it, and start with:

  ```bash
  uvx --torch-backend cpu --from git+https://github.com/AI-Yatra/s1ced-playground s1ced-playground --examples my_cases.json
  ```

## Know its limits

On cases like its training data the model picks the right answer 83% of the time, and when it says it is confident it is right 96% of the time. On text that looks different from its training data it is much weaker, sometimes close to guessing. On the realistic examples in this playground it gets 15 of 38 answers right.

So treat it as a fast first filter with a human fallback, and test it on your own cases before relying on it. The [model card](https://huggingface.co/syedazeez/s1ced-v3-deep) has the full evaluation.

## Optional: compare with Microsoft-Decision-1

The playground can send the same question to Microsoft's `microsoft-decision-1` model on [OpenRouter](https://openrouter.ai) and show both answers side by side. This needs an OpenRouter API key, and OpenRouter charges per request. Everything else is local and free.

1. Click **Add key** in the bottom left and paste your OpenRouter key. Tick **Remember on this computer** to keep it for next time. It is stored encrypted on Windows, or in a file only your user can read on macOS and Linux. It is never sent to the browser.
2. Click **Compare both** at the top of the left sidebar and run a decision.

You can also set the `OPENROUTER_API_KEY` environment variable before starting.

## Other ways to install

**With pip** (Python 3.10 or newer):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu "s1ced @ git+https://github.com/AI-Yatra/s1ced-playground"
s1ced-playground
```

The extra index installs the CPU version of PyTorch, which is much smaller. Leave it out if you want to use an NVIDIA GPU.

**From a clone:**

```bash
git clone https://github.com/AI-Yatra/s1ced-playground
cd s1ced-playground
uv run s1ced-playground
```

## Use the model in your own code

After installing with pip:

```python
from s1ced import S1CED

model = S1CED.from_pretrained()

results = model.decide(
    "Task: clean up inactive accounts. Proposed tool call: delete_rows(table='customers'). No backup was taken today.",
    [
        {"question": "Is this action safe to run without a human approving it first?", "type": "noul"},
        {"question": "What should happen next?", "options": ["run it", "take a backup first", "ask the owner"]},
    ],
)
for r in results:
    print(r["selected"], r["probabilities"], "escalate" if r["should_escalate"] else "confident")
```

Each result contains:

| field | meaning |
|---|---|
| `selected` | the most likely answer |
| `probabilities` | probability of every option |
| `confidence` | probability of the selected answer |
| `prediction_set` | every option the model cannot rule out |
| `should_escalate` | `true` when more than one option is in the set, so a human should check |

There is also a command-line tool:

```bash
s1ced --input request.json      # answer one request from a file
s1ced --serve --port 8000       # HTTP API: POST /decide
s1ced --bench                   # measure speed on your machine
```

A request file looks like this:

```json
{"state": "Policy: refunds within 30 days. Case: bought 12 days ago, unopened.",
 "questions": [{"question": "Is the refund allowed?", "type": "noul"}]}
```

## Troubleshooting

| problem | fix |
|---|---|
| `uvx: command not found` | Close the terminal and open a new one after installing uv. |
| Port 8770 is busy | Nothing to do. The playground picks the next free port and prints it. To choose one: `--port 9000`. |
| The browser did not open | Open the address printed in the terminal. Use `--no-browser` to stop it from opening. |
| The first start is slow | It is downloading PyTorch and the model. This only happens once. |
| You want the tutorial again | Click **Tutorial** at the top right. |
| No internet after the first run | Set `HF_HUB_OFFLINE=1` to skip the update check and use the downloaded model. |

To free disk space later, run `uv cache clean` and delete the `models--syedazeez--s1ced-v3-deep` folder from your Hugging Face cache (`~/.cache/huggingface/hub` on macOS and Linux, `%USERPROFILE%\.cache\huggingface\hub` on Windows).

## Speed

On an ordinary laptop CPU (Intel Core i5-13450HX) one question takes about 50 to 120 ms, and eight questions about the same text take 140 to 310 ms. Run `s1ced --bench` to measure yours.

## For developers

```bash
git clone https://github.com/AI-Yatra/s1ced-playground
cd s1ced-playground
uv sync
uv run python -m unittest discover -s tests       # Python tests, no model download
node --test tests/playground_ui.test.cjs          # playground UI tests (Node 18+)
```

All options for both commands: `s1ced-playground --help` and `s1ced --help`. `--model` (or the `S1CED_MODEL` environment variable) takes a Hugging Face repo id or a local folder with the model files.

## License

Code and model weights: [Apache License 2.0](LICENSE).

The model was pretrained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By 1.0) and fine-tuned only on the CC BY 4.0 rows of [VerQen Decisions](https://huggingface.co/datasets/zorqelis-ai/verqen-decisions) by ZorQelis AI.
