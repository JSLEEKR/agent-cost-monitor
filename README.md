<div align="center">

# agent-cost-monitor

**Track and analyze AI agent API costs across providers in real-time.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

[Features](#features) · [Quick Start](#quick-start) · [SDK Integration](#sdk-integration) · [API Reference](#api-reference)

</div>

---

## Why?

AI agents chain multiple LLM calls. Costs add up fast. **agent-cost-monitor** gives you:

- **Real-time cost tracking** across Claude, GPT, and Gemini models
- **Budget alerts** before you blow through your API credits
- **Zero-config SDK wrappers** -- plug in and forget
- **Export anywhere** -- JSON, CSV, or formatted terminal reports

---

## Features

| Feature | Description |
|---|---|
| Multi-provider pricing | Built-in rates for Claude, GPT-4o, GPT-4.1, Gemini 2.5, and more |
| Budget enforcement | Callback alerts, exception-based hard stops, or both |
| SDK wrappers | `wrap_anthropic()` and `wrap_openai()` auto-track every API call |
| Decorator | `@track_usage` wraps any function returning an SDK-style response |
| Export | `to_json()`, `to_csv()`, and `report()` for formatted tables |
| History cap | Bounded memory via configurable `max_history` (default 10,000) |
| CLI demo | `python -m agent_cost_monitor demo` for a quick look |

---

## Quick Start

### Install

```bash
pip install -e .
```

### Basic Usage

```python
from agent_cost_monitor import CostTracker

tracker = CostTracker(budget=1.00)

# Record API calls as they happen
tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=800)
tracker.record("gpt-4o", input_tokens=1000, output_tokens=400)

# Check your spend
print(f"Total cost: ${tracker.total_cost:.4f}")
print(f"Over budget: {tracker.is_over_budget}")

# Get a full breakdown
print(tracker.report())
```

---

## SDK Integration

### Anthropic

```python
import anthropic
from agent_cost_monitor import CostTracker, wrap_anthropic

client = anthropic.Anthropic()
tracker = CostTracker(budget=5.00)
wrap_anthropic(client, tracker)

# Every call is now automatically tracked -- no other changes needed
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)

print(f"Running total: ${tracker.total_cost:.6f}")
```

### OpenAI

```python
from openai import OpenAI
from agent_cost_monitor import CostTracker, wrap_openai

client = OpenAI()
tracker = CostTracker(budget=5.00)
wrap_openai(client, tracker)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(f"Running total: ${tracker.total_cost:.6f}")
```

### Decorator

```python
from agent_cost_monitor import CostTracker, track_usage

tracker = CostTracker()

@track_usage(tracker, model="claude-sonnet-4-6")
def call_llm(prompt):
    # Your API call here -- must return an object with .usage and .model
    return api_client.create(prompt=prompt)

response = call_llm("Summarize this document")
print(tracker.summary())
```

---

## Budget Alerts

### Callback

Get notified when spending crosses the threshold:

```python
def alert(usage, tracker):
    print(f"WARNING: Budget exceeded! Current spend: ${tracker.total_cost:.4f}")

tracker = CostTracker(budget=0.50, on_budget_exceeded=alert)
```

### Exception

Hard-stop with an exception to prevent runaway costs:

```python
from agent_cost_monitor import CostTracker, BudgetExceededError

tracker = CostTracker(budget=0.50, raise_on_budget=True)

try:
    tracker.record("claude-opus-4-6", input_tokens=100_000, output_tokens=50_000)
except BudgetExceededError as e:
    print(f"Stopped: {e}")
```

### Both

Use a callback for logging and an exception for enforcement:

```python
tracker = CostTracker(
    budget=1.00,
    on_budget_exceeded=lambda u, t: log.warning(f"Over budget: ${t.total_cost:.4f}"),
    raise_on_budget=True,
)
```

---

## Export & Reporting

### Formatted Report

```python
print(tracker.report())
```

```
+======================================+
|     Agent Cost Monitor Report        |
+======================================+
| Total Cost:        $0.031950         |
| Total Requests:    5                 |
| Budget:            $1.00 (3.2% used) |
+--------------------------------------+
| Cost by Model:                       |
|   claude-sonnet-4-6   $0.021000      |
|   gpt-4o-mini         $0.001950      |
|   gemini-2.5-flash    $0.001170      |
|   gpt-4o              $0.006500      |
+======================================+
```

### JSON

```python
with open("costs.json", "w") as f:
    f.write(tracker.to_json())
```

### CSV

```python
with open("costs.csv", "w") as f:
    f.write(tracker.to_csv())
```

---

## CLI Demo

```bash
python -m agent_cost_monitor demo
```

Runs a quick simulation with multiple providers and prints the report, JSON preview, and CSV export.

---

## Supported Models

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---:|---:|
| Anthropic | `claude-opus-4-6` | $15.00 | $75.00 |
| Anthropic | `claude-sonnet-4-6` | $3.00 | $15.00 |
| Anthropic | `claude-haiku-4-5` | $0.80 | $4.00 |
| OpenAI | `gpt-4o` | $2.50 | $10.00 |
| OpenAI | `gpt-4o-mini` | $0.15 | $0.60 |
| OpenAI | `gpt-4.1` | $2.00 | $8.00 |
| OpenAI | `gpt-4.1-mini` | $0.40 | $1.60 |
| Google | `gemini-2.5-pro` | $1.25 | $10.00 |
| Google | `gemini-2.5-flash` | $0.15 | $0.60 |

Unknown models fall back to default pricing ($3.00 / $15.00 per 1M tokens).

---

## API Reference

### `CostTracker(budget=None, max_history=10_000, on_budget_exceeded=None, raise_on_budget=False)`

Main tracking class.

| Method / Property | Description |
|---|---|
| `record(model, input_tokens, output_tokens)` | Record a single API call. Returns a `Usage` object. |
| `total_cost` | Running total cost in USD. |
| `total_input_tokens` | Total input tokens across all recorded calls. |
| `total_output_tokens` | Total output tokens across all recorded calls. |
| `is_over_budget` | `True` if total cost exceeds budget. |
| `summary()` | Returns a dict with cost, tokens, request count, and budget status. |
| `cost_by_model()` | Returns a dict mapping model names to their total cost. |
| `report()` | Returns a formatted string table. |
| `to_json()` | Returns usage history as a JSON string. |
| `to_csv()` | Returns usage history as a CSV string. |
| `reset()` | Clears all recorded usage data. |

### `track_usage(tracker, model=None)`

Decorator for functions that return SDK-style responses (with `.usage` and `.model` attributes).

### `wrap_anthropic(client, tracker)`

Monkey-patches an Anthropic client so `client.messages.create()` auto-records usage.

### `wrap_openai(client, tracker)`

Monkey-patches an OpenAI client so `client.chat.completions.create()` auto-records usage.

### `BudgetExceededError`

Exception raised when `raise_on_budget=True` and total cost exceeds the budget.

---

## License

[MIT](LICENSE)
