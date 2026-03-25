import csv
import io
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

DEFAULT_MAX_HISTORY = 10_000


class BudgetExceededError(Exception):
    pass


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def cost(self):
        rates = PRICING.get(self.model, PRICING["default"])
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
        )


PRICING = {
    "claude-sonnet-4-6": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
    "claude-haiku-4-5": {
        "input": 0.80 / 1_000_000,
        "output": 4.0 / 1_000_000,
    },
    "claude-opus-4-6": {
        "input": 15.0 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.0 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4.1": {
        "input": 2.0 / 1_000_000,
        "output": 8.0 / 1_000_000,
    },
    "gpt-4.1-mini": {
        "input": 0.40 / 1_000_000,
        "output": 1.60 / 1_000_000,
    },
    "gemini-2.5-pro": {
        "input": 1.25 / 1_000_000,
        "output": 10.0 / 1_000_000,
    },
    "gemini-2.5-flash": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "default": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
}


class Session:
    """A named cost tracking scope within a CostTracker."""

    def __init__(self, name: str, tracker: "CostTracker"):
        self.name = name
        self._tracker = tracker
        self._usages: list[Usage] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    def record(self, model, input_tokens, output_tokens):
        """Record usage on both this session and the parent tracker."""
        usage = self._tracker.record(model, input_tokens, output_tokens)
        self._usages.append(usage)
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cost += usage.cost
        return usage

    @property
    def total_cost(self):
        return self._total_cost

    @property
    def total_input_tokens(self):
        return self._total_input_tokens

    @property
    def total_output_tokens(self):
        return self._total_output_tokens

    def summary(self):
        return {
            "session_name": self.name,
            "total_cost": round(self._total_cost, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "num_requests": len(self._usages),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class CostTracker:
    def __init__(self, budget=None, max_history=DEFAULT_MAX_HISTORY,
                 on_budget_exceeded=None, raise_on_budget=False,
                 auto_save=None, on_anomaly=None):
        self.budget = budget
        self.max_history = max_history
        self.on_budget_exceeded = on_budget_exceeded
        self.raise_on_budget = raise_on_budget
        self.auto_save = auto_save
        self.on_anomaly = on_anomaly
        self._usages = deque(maxlen=max_history)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._sessions: dict[str, Session] = {}

    def record(self, model, input_tokens, output_tokens):
        if not isinstance(input_tokens, int) or input_tokens < 0:
            raise ValueError(
                "input_tokens must be a non-negative integer,"
                f" got {input_tokens!r}"
            )
        if not isinstance(output_tokens, int) or output_tokens < 0:
            raise ValueError(
                "output_tokens must be a non-negative integer,"
                f" got {output_tokens!r}"
            )
        usage = Usage(model=model, input_tokens=input_tokens,
                      output_tokens=output_tokens)
        if len(self._usages) == self.max_history:
            evicted = self._usages[0]
            self._total_input_tokens -= evicted.input_tokens
            self._total_output_tokens -= evicted.output_tokens
            self._total_cost -= evicted.cost
        self._usages.append(usage)
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cost += usage.cost
        if self.is_over_budget:
            if self.on_budget_exceeded is not None:
                self.on_budget_exceeded(usage, self)
            if self.raise_on_budget:
                raise BudgetExceededError(
                    f"Budget of {self.budget} exceeded: "
                    f"total cost is {self.total_cost:.6f}"
                )
        anomaly = self.check_anomaly(usage)
        if anomaly is not None and self.on_anomaly is not None:
            self.on_anomaly(anomaly, usage, self)
        if self.auto_save is not None:
            self.save(self.auto_save)
        return usage

    @property
    def total_cost(self):
        return self._total_cost

    @property
    def total_input_tokens(self):
        return self._total_input_tokens

    @property
    def total_output_tokens(self):
        return self._total_output_tokens

    @property
    def is_over_budget(self):
        if self.budget is None:
            return False
        return self.total_cost > self.budget

    def summary(self):
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_requests": len(self._usages),
            "budget": self.budget,
            "over_budget": self.is_over_budget,
        }

    def cost_by_model(self):
        models = {}
        for u in self._usages:
            if u.model not in models:
                models[u.model] = 0.0
            models[u.model] += u.cost
        return {k: round(v, 6) for k, v in models.items()}

    def session(self, name: str) -> Session:
        """Create or retrieve a named session for scoped cost tracking."""
        if name not in self._sessions:
            self._sessions[name] = Session(name, self)
        return self._sessions[name]

    def cost_by_session(self) -> dict:
        """Return cost breakdown by session name."""
        return {
            name: round(s.total_cost, 6)
            for name, s in self._sessions.items()
        }

    def check_anomaly(self, usage) -> dict | None:
        """Check if the given usage is anomalous (cost >3x the running average).

        Returns None if normal or if fewer than 5 previous records exist.
        Returns a dict with type, cost, avg_cost, and ratio if anomalous.
        """
        # Need at least 5 *previous* records (excluding the one just added)
        # The usage passed in is already in self._usages, so we need len >= 6
        if len(self._usages) < 6:
            return None
        # Calculate average cost of all records except the last one
        costs = [u.cost for u in list(self._usages)[:-1]]
        avg_cost = sum(costs) / len(costs)
        if avg_cost == 0:
            return None
        ratio = usage.cost / avg_cost
        if ratio > 3.0:
            return {
                "type": "spike",
                "cost": usage.cost,
                "avg_cost": avg_cost,
                "ratio": ratio,
            }
        return None

    def cost_per_minute(self) -> float:
        """Average cost per minute based on timestamps of recorded usages."""
        if len(self._usages) < 2:
            return 0.0
        usages = list(self._usages)
        first_ts = datetime.fromisoformat(usages[0].timestamp)
        last_ts = datetime.fromisoformat(usages[-1].timestamp)
        elapsed_minutes = (last_ts - first_ts).total_seconds() / 60.0
        if elapsed_minutes <= 0:
            return 0.0
        return self._total_cost / elapsed_minutes

    def requests_per_minute(self) -> float:
        """Average requests per minute based on timestamps of recorded usages."""
        if len(self._usages) < 2:
            return 0.0
        usages = list(self._usages)
        first_ts = datetime.fromisoformat(usages[0].timestamp)
        last_ts = datetime.fromisoformat(usages[-1].timestamp)
        elapsed_minutes = (last_ts - first_ts).total_seconds() / 60.0
        if elapsed_minutes <= 0:
            return 0.0
        return len(self._usages) / elapsed_minutes

    def reset(self):
        self._usages.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

    def to_json(self) -> str:
        records = [
            {
                "timestamp": u.timestamp,
                "model": u.model,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cost": round(u.cost, 6),
            }
            for u in self._usages
        ]
        return json.dumps(records, indent=2)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "model", "input_tokens",
                         "output_tokens", "cost"])
        for u in self._usages:
            writer.writerow([u.timestamp, u.model, u.input_tokens,
                             u.output_tokens, round(u.cost, 6)])
        return buf.getvalue()

    def save(self, path: str) -> None:
        """Save full tracker state to a JSON file."""
        state = {
            "budget": self.budget,
            "max_history": self.max_history,
            "usages": [
                {
                    "model": u.model,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "timestamp": u.timestamp,
                }
                for u in self._usages
            ],
            "sessions": {
                name: [
                    {
                        "model": u.model,
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                        "timestamp": u.timestamp,
                    }
                    for u in s._usages
                ]
                for name, s in self._sessions.items()
            },
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CostTracker":
        """Load tracker state from a JSON file.

        Returns a fresh CostTracker if the file is missing or corrupted.
        """
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls()

        if not isinstance(state, dict):
            return cls()

        budget = state.get("budget")
        max_history = state.get("max_history", DEFAULT_MAX_HISTORY)
        tracker = cls(budget=budget, max_history=max_history)

        for u_data in state.get("usages", []):
            usage = Usage(
                model=u_data["model"],
                input_tokens=u_data["input_tokens"],
                output_tokens=u_data["output_tokens"],
                timestamp=u_data.get("timestamp",
                                     datetime.now(timezone.utc).isoformat()),
            )
            if len(tracker._usages) == tracker.max_history:
                evicted = tracker._usages[0]
                tracker._total_input_tokens -= evicted.input_tokens
                tracker._total_output_tokens -= evicted.output_tokens
                tracker._total_cost -= evicted.cost
            tracker._usages.append(usage)
            tracker._total_input_tokens += usage.input_tokens
            tracker._total_output_tokens += usage.output_tokens
            tracker._total_cost += usage.cost

        for session_name, session_usages in state.get("sessions", {}).items():
            session = tracker.session(session_name)
            for u_data in session_usages:
                usage = Usage(
                    model=u_data["model"],
                    input_tokens=u_data["input_tokens"],
                    output_tokens=u_data["output_tokens"],
                    timestamp=u_data.get(
                        "timestamp",
                        datetime.now(timezone.utc).isoformat()),
                )
                session._usages.append(usage)
                session._total_input_tokens += usage.input_tokens
                session._total_output_tokens += usage.output_tokens
                session._total_cost += usage.cost

        return tracker

    def report(self) -> str:
        w = 40
        total_cost = self.total_cost
        num_requests = len(self._usages)
        by_model = self.cost_by_model()

        lines = []
        lines.append("+" + "=" * (w - 2) + "+")
        lines.append("|" + "Agent Cost Monitor Report".center(w - 2) + "|")
        lines.append("+" + "=" * (w - 2) + "+")
        lines.append(f"| {'Total Cost:':<18} ${total_cost:<17.6f} |")
        lines.append(f"| {'Total Requests:':<18} {num_requests:<18} |")
        if self.budget is not None:
            pct = (total_cost / self.budget * 100) if self.budget > 0 else 0.0
            budget_str = f"${self.budget:.2f} ({pct:.1f}% used)"
            lines.append(f"| {'Budget:':<18} {budget_str:<18} |")
        lines.append("+" + "-" * (w - 2) + "+")
        lines.append(f"| {'Cost by Model:':<{w - 4}} |")
        for model, cost in by_model.items():
            entry = f"  {model:<20} ${cost:.6f}"
            lines.append(f"| {entry:<{w - 4}} |")
        lines.append("+" + "=" * (w - 2) + "+")
        return "\n".join(lines)
