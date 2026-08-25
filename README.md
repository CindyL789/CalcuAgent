# px-calculator

A very small agent that does arithmetic. An LLM does the *orchestration*; a
Python function does the *computing*. No agent framework — just a loop.

The only tool the model gets performs **one binary operation** (`a op b`), so a
composite expression forces it to plan:

```
9 / 3 - 4 * 5
  → calculate(9, "/", 3)   → 3
  → calculate(4, "*", 5)   → 20
  → calculate(3, "-", 20)  → -17
```

## Layout

| File | Role |
|---|---|
| `src/px_calculator/utils/calc_tool.py` | `calculate(a, op, b)` + the provider-neutral tool schema |
| `src/px_calculator/models/llm.py` | Provider backends (OpenAI, Ollama, Anthropic, any OpenAI-compatible endpoint) |
| `src/px_calculator/models/agent.py` | `CalculatorAgent` — the tool-calling loop |
| `src/px_calculator/cli.py` | Command line front end |
| `mains/main.py` | Entry point |
| `tests/test.py` | Offline tests (no API key, no network) |

## Install

```bash
pip install -e ".[all]"      # or ".[openai]" / ".[anthropic]"
```

`ollama` and `openai-compatible` reuse the `openai` SDK, so `.[openai]` covers them.

## Configure

Providers read their credentials from the environment:

| Provider | API key | Base URL | Default model |
|---|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (optional) | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` (optional) | `claude-sonnet-4-5` |
| `ollama` | — | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `llama3.1` |
| `openai-compatible` | `LLM_API_KEY` | `LLM_BASE_URL` | — (set `LLM_MODEL`) |

`LLM_PROVIDER` and `LLM_MODEL` set the defaults for every run.
`LLM_TIMEOUT` (default 30s) and `LLM_MAX_RETRIES` (default 2) bound each request —
without a timeout a dropped connection hangs the agent indefinitely.

```bash
export OPENAI_API_KEY="sk-..."
```

## Use

```bash
python mains/main.py "9/3 - 4*5"            # one shot; tool calls shown as they happen
python mains/main.py -q "2*8 + 1"           # final answer only
python mains/main.py --trace "9/3 + 6 - 20" # + each API round-trip and elapsed time
python mains/main.py -p ollama -m llama3.1  # interactive, local model
python mains/main.py -p anthropic "4/10"
python mains/main.py --timeout 60 "1/3"     # longer wait (0 = wait forever)
python -m px_calculator                     # same CLI
```

Progress is printed as it happens (flushed), so a slow run is visibly working
rather than apparently hung:

```
$ python mains/main.py --trace "9/3 + 6 - 20"
  ... turn 1: asking gpt-4o-mini
  [1] 9 / 3 = 3
  ... turn 2: asking gpt-4o-mini
  [2] 3 + 6 = 9
  ... turn 3: asking gpt-4o-mini
  [3] 9 - 20 = -11
  ... turn 4: asking gpt-4o-mini
  (3 tool call(s), 4 turn(s), 4.2s)
9/3 + 6 - 20 = -11
```

From Python:

```python
from px_calculator.models.agent import CalculatorAgent

agent = CalculatorAgent(
    provider="openai",
    verbose=1,                       # 0 silent, 1 tool calls, 2 + round-trips
    timeout=30,                      # seconds per request; None = wait forever
    on_step=lambda i, step: None,    # optional live callback
)
result = agent.run("9/3 - 4*5")

print(result.answer)                    # "9 / 3 - 4 * 5 = -17"
print([str(s) for s in result.steps])   # ['9 / 3 = 3', '4 * 5 = 20', '3 - 20 = -17']
print(result.turns, result.elapsed)     # round-trips and wall-clock seconds
```

## Adding a provider

```python
from px_calculator.models.llm import LLMBackend, register_provider

class GeminiBackend(LLMBackend):
    name = "gemini"
    def complete(self, system, history, tools): ...

register_provider("gemini", GeminiBackend)
```

Ollama-style endpoints need no new class — point `openai-compatible` at them.

## Test

```bash
PYTHONPATH=src python3 tests/test.py
```

31 tests: the tool's arithmetic and error handling, the agent loop (driven by a
scripted fake LLM, including multi-step chaining, parallel tool calls, error
recovery and the turn cap), the OpenAI/Anthropic message adapters, and the
timeout / verbosity / CLI wiring.
