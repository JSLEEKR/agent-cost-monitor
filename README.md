<div align="center">

# agent-cost-monitor

**Stop guessing what your AI agents cost. Start knowing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e.svg?style=flat)]()
[![Tests: 84 passing](https://img.shields.io/badge/tests-84%20passing-22c55e.svg?style=flat)]()
[![pip installable](https://img.shields.io/badge/pip-installable-3776AB.svg?style=flat&logo=pypi&logoColor=white)]()

---

[Why?](#why) · [Features](#features) · [Quick Start](#quick-start) · [SDK Integration](#sdk-integration) · [Sessions](#sessions) · [Budget Alerts](#budget-alerts) · [Anomaly Detection](#anomaly-detection) · [Persistence](#persistence) · [Export & Reporting](#export--reporting) · [Rate Monitoring](#rate-monitoring) · [CLI](#cli) · [Supported Models](#supported-models) · [API Reference](#api-reference)

</div>

---

## Why?

AI agents don't make one API call -- they make dozens. Across different models, different providers, different tasks. Costs spiral silently until the invoice arrives.

- **No visibility** -- you have no idea which agent task burned through your budget until the bill comes
- **No guardrails** -- a single runaway loop can drain your API credits in minutes
- **No attribution** -- when costs spike, you can't pinpoint which model, session, or task is responsible

**agent-cost-monitor** solves all three. Drop it into any Python agent and get real-time cost tracking, budget enforcement, anomaly detection, and per-task attribution -- across Claude, GPT, and Gemini.

---

## Features

| | Feature | What it does |
|---|---|---|
| :bar_chart: | **Multi-provider pricing** | Built-in rates for 12 models across Anthropic, OpenAI, and Google |
| :shield: | **Budget enforcement** | Callback alerts, hard-stop exceptions, or both |
| :electric_plug: | **SDK wrappers** | `wrap_anthropic()` / `wrap_openai()` auto-track every call (sync + async) |
| :label: | **Decorator pattern** | `@track_usage` / `@async_track_usage` for custom functions |
| :file_folder: | **Session tracking** | Per-task cost attribution with named sessions and context managers |
| :floppy_disk: | **Persistence** | `save()` / `load()` / `auto_save` for durable state across restarts |
| :page_facing_up: | **Export** | `to_json()`, `to_csv()`, and `report()` formatted tables |
| :rotating_light: | **Anomaly detection** | Automatic 3x cost-spike alerts with callback hooks |
| :stopwatch: | **Rate tracking** | `cost_per_minute()` and `requests_per_minute()` in real time |
| :computer: | **CLI demo** | `python -m agent_cost_monitor demo` for instant visualization |
| :gear: | **History cap** | Bounded memory via configurable `max_history` (default 10,000) |

---

## Quick Start

### Install

```bash
pip install -e .
```

### 5-Line Usage

```python
from agent_cost_monitor import CostTracker

tracker = CostTracker(budget=1.00)
tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=800)
tracker.record("gpt-4o", input_tokens=1000, output_tokens=400)
print(f"Total: ${tracker.total_cost:.4f} | Over budget: {tracker.is_over_budget}")
```

---

## SDK Integration

### Anthropic (sync)

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

### Anthropic (async)

```python
import anthropic
from agent_cost_monitor import CostTracker, wrap_anthropic_async

client = anthropic.AsyncAnthropic()
tracker = CostTracker(budget=5.00)
wrap_anthropic_async(client, tracker)

response = await client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### OpenAI (sync)

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

### OpenAI (async)

```python
from openai import AsyncOpenAI
from agent_cost_monitor import CostTracker, wrap_openai_async

client = AsyncOpenAI()
tracker = CostTracker(budget=5.00)
wrap_openai_async(client, tracker)

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Decorator Pattern

```python
from agent_cost_monitor import CostTracker, track_usage, async_track_usage

tracker = CostTracker()

@track_usage(tracker, model="claude-sonnet-4-6")
def call_claude(prompt):
    return anthropic_client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

# Async version
@async_track_usage(tracker, model="gpt-4o")
async def call_gpt(prompt):
    return await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

response = call_claude("Summarize this document")
print(tracker.summary())
```

> **Note:** The `model` parameter is optional. If omitted, the decorator reads `response.model` automatically.

---

## Sessions

Track costs per-task with named sessions. Sessions act as scoped windows into the same tracker -- every session record also rolls up into the global total.

```python
from agent_cost_monitor import CostTracker

tracker = CostTracker(budget=10.00)

# Use as a context manager
with tracker.session("research") as s:
    s.record("claude-sonnet-4-6", input_tokens=5000, output_tokens=2000)
    s.record("gemini-2.5-pro", input_tokens=3000, output_tokens=1500)
    print(f"Research cost: ${s.total_cost:.4f}")

with tracker.session("writing") as s:
    s.record("gpt-4o", input_tokens=2000, output_tokens=4000)
    print(f"Writing cost: ${s.total_cost:.4f}")

# See cost breakdown by session
print(tracker.cost_by_session())
# {'research': 0.0405, 'writing': 0.045}

# Global total includes everything
print(f"Total across all sessions: ${tracker.total_cost:.4f}")
```

Sessions expose `total_cost`, `total_input_tokens`, `total_output_tokens`, and `summary()`.

---

## Budget Alerts

### Callback

Get notified when spending crosses the threshold:

```python
def alert(usage, tracker):
    print(f"WARNING: Budget exceeded! Spend: ${tracker.total_cost:.4f}")

tracker = CostTracker(budget=0.50, on_budget_exceeded=alert)
```

### Exception

Hard-stop to prevent runaway costs:

```python
from agent_cost_monitor import CostTracker, BudgetExceededError

tracker = CostTracker(budget=0.50, raise_on_budget=True)

try:
    tracker.record("claude-opus-4-6", input_tokens=100_000, output_tokens=50_000)
except BudgetExceededError as e:
    print(f"Stopped: {e}")
    # Stopped: Budget of 0.5 exceeded: total cost is 5.250000
```

### Both

Use a callback for logging and an exception for enforcement:

```python
import logging
log = logging.getLogger(__name__)

tracker = CostTracker(
    budget=1.00,
    on_budget_exceeded=lambda u, t: log.warning(f"Over budget: ${t.total_cost:.4f}"),
    raise_on_budget=True,
)
```

---

## Anomaly Detection

Automatically detect cost spikes. When any single request costs more than **3x the running average** (after at least 5 prior records), the `on_anomaly` callback fires.

```python
def spike_alert(anomaly, usage, tracker):
    print(f"ANOMALY: {anomaly['type']} detected!")
    print(f"  Cost: ${anomaly['cost']:.4f} (avg: ${anomaly['avg_cost']:.4f})")
    print(f"  Ratio: {anomaly['ratio']:.1f}x the average")

tracker = CostTracker(on_anomaly=spike_alert)

# Build up a baseline of cheap calls
for _ in range(6):
    tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)

# This expensive call triggers the anomaly alert
tracker.record("claude-opus-4-6", input_tokens=50_000, output_tokens=20_000)
# ANOMALY: spike detected!
#   Cost: $2.2500 (avg: $0.0001)
#   Ratio: 30186.2x the average
```

---

## Persistence

### Save and Load

```python
tracker = CostTracker(budget=5.00)
tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)

# Save state to disk
tracker.save("costs.json")

# Load it back later -- budget and history are restored
restored = CostTracker.load("costs.json")
print(f"Restored cost: ${restored.total_cost:.6f}")
```

### Auto-save

Automatically persist after every `record()` call:

```python
tracker = CostTracker(budget=5.00, auto_save="costs.json")

# Every record() call now writes state to disk automatically
tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
# costs.json is updated immediately
```

> **Note:** `load()` returns a fresh empty tracker if the file is missing or corrupted -- no exceptions to handle.

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

### JSON Export

```python
json_str = tracker.to_json()
with open("costs.json", "w") as f:
    f.write(json_str)
```

```json
[
  {
    "timestamp": "2026-03-25T12:00:00+00:00",
    "model": "claude-sonnet-4-6",
    "input_tokens": 2000,
    "output_tokens": 800,
    "cost": 0.018
  }
]
```

### CSV Export

```python
csv_str = tracker.to_csv()
with open("costs.csv", "w") as f:
    f.write(csv_str)
```

```
timestamp,model,input_tokens,output_tokens,cost
2026-03-25T12:00:00+00:00,claude-sonnet-4-6,2000,800,0.018
2026-03-25T12:00:00+00:00,gpt-4o,1000,400,0.0065
```

---

## Rate Monitoring

Track how fast you're spending:

```python
tracker = CostTracker()

# ... after some API calls ...

print(f"Burn rate: ${tracker.cost_per_minute():.4f}/min")
print(f"Request rate: {tracker.requests_per_minute():.1f} req/min")
```

Both methods compute averages from the timestamps of the first and last recorded usage. Returns `0.0` if fewer than 2 records exist.

---

## CLI

Run the built-in demo to see the tracker in action:

```bash
python -m agent_cost_monitor demo
```

**Sample output:**

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

--- JSON export (first 3 lines) ---
[
  {
    "timestamp": "2026-03-25T...",
...

--- CSV export ---
timestamp,model,input_tokens,output_tokens,cost
...
```

---

## Supported Models

All pricing is built-in. No configuration required.

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|:---|:---|---:|---:|
| **Anthropic** | `claude-opus-4-6` | $15.00 | $75.00 |
| **Anthropic** | `claude-sonnet-4-6` | $3.00 | $15.00 |
| **Anthropic** | `claude-haiku-4-5` | $0.80 | $4.00 |
| **OpenAI** | `gpt-4o` | $2.50 | $10.00 |
| **OpenAI** | `gpt-4o-mini` | $0.15 | $0.60 |
| **OpenAI** | `gpt-4.1` | $2.00 | $8.00 |
| **OpenAI** | `gpt-4.1-mini` | $0.40 | $1.60 |
| **Google** | `gemini-2.5-pro` | $1.25 | $10.00 |
| **Google** | `gemini-2.5-flash` | $0.15 | $0.60 |

> **Unknown models** automatically fall back to default pricing ($3.00 / $15.00 per 1M tokens). You never need to configure pricing manually.

---

## API Reference

### `CostTracker`

```python
CostTracker(
    budget=None,              # Optional spending limit in USD
    max_history=10_000,       # Max records kept in memory (oldest evicted)
    on_budget_exceeded=None,  # Callback: fn(usage, tracker)
    raise_on_budget=False,    # Raise BudgetExceededError when over budget
    auto_save=None,           # File path for auto-saving after every record()
    on_anomaly=None,          # Callback: fn(anomaly_dict, usage, tracker)
)
```

#### Methods

| Method | Returns | Description |
|:---|:---|:---|
| `record(model, input_tokens, output_tokens)` | `Usage` | Record a single API call |
| `summary()` | `dict` | Cost, tokens, request count, and budget status |
| `cost_by_model()` | `dict` | Map of model name to total cost |
| `session(name)` | `Session` | Create or retrieve a named session |
| `cost_by_session()` | `dict` | Map of session name to total cost |
| `check_anomaly(usage)` | `dict \| None` | Check if a usage record is anomalous |
| `cost_per_minute()` | `float` | Average cost per minute |
| `requests_per_minute()` | `float` | Average requests per minute |
| `report()` | `str` | Formatted ASCII table report |
| `to_json()` | `str` | Usage history as JSON string |
| `to_csv()` | `str` | Usage history as CSV string |
| `save(path)` | `None` | Save full state to a JSON file |
| `reset()` | `None` | Clear all recorded usage data |

#### Class Methods

| Method | Returns | Description |
|:---|:---|:---|
| `CostTracker.load(path)` | `CostTracker` | Load state from file (returns empty tracker if file missing/corrupt) |

#### Properties

| Property | Type | Description |
|:---|:---|:---|
| `total_cost` | `float` | Running total cost in USD |
| `total_input_tokens` | `int` | Total input tokens across all calls |
| `total_output_tokens` | `int` | Total output tokens across all calls |
| `is_over_budget` | `bool` | `True` if total cost exceeds budget |

---

### `Session`

Returned by `tracker.session(name)`. Supports use as a context manager.

| Member | Type | Description |
|:---|:---|:---|
| `name` | `str` | Session name |
| `record(model, input_tokens, output_tokens)` | `Usage` | Record usage (also recorded on parent tracker) |
| `total_cost` | `float` | Session cost in USD |
| `total_input_tokens` | `int` | Session input tokens |
| `total_output_tokens` | `int` | Session output tokens |
| `summary()` | `dict` | Session name, cost, tokens, and request count |

---

### `Usage`

Dataclass returned by `record()`.

| Field | Type | Description |
|:---|:---|:---|
| `model` | `str` | Model name |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `cost` | `float` | Computed cost in USD (property) |

---

### `BudgetExceededError`

Exception raised when `raise_on_budget=True` and total cost exceeds the budget. Inherits from `Exception`.

---

### Functions

| Function | Description |
|:---|:---|
| `wrap_anthropic(client, tracker)` | Auto-track `client.messages.create()` calls |
| `wrap_openai(client, tracker)` | Auto-track `client.chat.completions.create()` calls |
| `wrap_anthropic_async(client, tracker)` | Auto-track async Anthropic calls |
| `wrap_openai_async(client, tracker)` | Auto-track async OpenAI calls |
| `track_usage(tracker, model=None)` | Sync decorator for functions returning SDK-style responses |
| `async_track_usage(tracker, model=None)` | Async decorator for functions returning SDK-style responses |

---

## License

[MIT](LICENSE)
