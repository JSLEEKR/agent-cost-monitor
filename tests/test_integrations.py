"""Tests for SDK integration wrappers using mock objects."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_cost_monitor import CostTracker, wrap_anthropic, wrap_openai


# ---------------------------------------------------------------------------
# Helpers to build mock SDK clients
# ---------------------------------------------------------------------------

def _make_anthropic_client(model="claude-sonnet-4-6", input_tokens=100,
                           output_tokens=50, usage=True):
    """Return a mock Anthropic client with a messages.create method."""
    response = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens,
                              output_tokens=output_tokens) if usage else None,
        content=[SimpleNamespace(text="Hello")],
    )
    client = MagicMock()
    client.messages.create = MagicMock(return_value=response)
    return client, response


def _make_openai_client(model="gpt-4o", prompt_tokens=200,
                        completion_tokens=80, usage=True):
    """Return a mock OpenAI client with a chat.completions.create method."""
    response = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens) if usage else None,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hi"),
        )],
    )
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=response)
    return client, response


# ---------------------------------------------------------------------------
# wrap_anthropic tests
# ---------------------------------------------------------------------------

class TestWrapAnthropic:
    def test_records_usage(self):
        client, _ = _make_anthropic_client(input_tokens=100, output_tokens=50)
        tracker = CostTracker()
        wrap_anthropic(client, tracker)

        client.messages.create(model="claude-sonnet-4-6",
                               messages=[{"role": "user", "content": "Hi"}])

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.summary()["num_requests"] == 1

    def test_returns_original_response(self):
        client, expected_response = _make_anthropic_client()
        tracker = CostTracker()
        wrap_anthropic(client, tracker)

        result = client.messages.create(model="claude-sonnet-4-6",
                                        messages=[{"role": "user",
                                                   "content": "Hi"}])
        assert result is expected_response

    def test_uses_response_model(self):
        client, _ = _make_anthropic_client(model="claude-haiku-4-5",
                                           input_tokens=500,
                                           output_tokens=200)
        tracker = CostTracker()
        wrap_anthropic(client, tracker)

        client.messages.create(model="claude-haiku-4-5",
                               messages=[{"role": "user", "content": "Hi"}])

        by_model = tracker.cost_by_model()
        assert "claude-haiku-4-5" in by_model

    def test_no_usage_graceful(self):
        client, _ = _make_anthropic_client(usage=False)
        tracker = CostTracker()
        wrap_anthropic(client, tracker)

        result = client.messages.create(model="claude-sonnet-4-6",
                                        messages=[{"role": "user",
                                                   "content": "Hi"}])

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.summary()["num_requests"] == 0
        assert result is not None

    def test_multiple_calls_accumulate(self):
        client, _ = _make_anthropic_client(input_tokens=100, output_tokens=50)
        tracker = CostTracker()
        wrap_anthropic(client, tracker)

        for _ in range(3):
            client.messages.create(model="claude-sonnet-4-6",
                                   messages=[{"role": "user",
                                              "content": "Hi"}])

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert tracker.summary()["num_requests"] == 3

    def test_budget_alert_fires(self):
        client, _ = _make_anthropic_client(input_tokens=100000,
                                           output_tokens=50000)
        alerts = []
        tracker = CostTracker(budget=0.001,
                              on_budget_exceeded=lambda u, t: alerts.append(True))
        wrap_anthropic(client, tracker)

        client.messages.create(model="claude-sonnet-4-6",
                               messages=[{"role": "user", "content": "Hi"}])

        assert tracker.is_over_budget
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# wrap_openai tests
# ---------------------------------------------------------------------------

class TestWrapOpenAI:
    def test_records_usage(self):
        client, _ = _make_openai_client(prompt_tokens=200,
                                        completion_tokens=80)
        tracker = CostTracker()
        wrap_openai(client, tracker)

        client.chat.completions.create(model="gpt-4o",
                                       messages=[{"role": "user",
                                                  "content": "Hi"}])

        assert tracker.total_input_tokens == 200
        assert tracker.total_output_tokens == 80
        assert tracker.summary()["num_requests"] == 1

    def test_returns_original_response(self):
        client, expected_response = _make_openai_client()
        tracker = CostTracker()
        wrap_openai(client, tracker)

        result = client.chat.completions.create(model="gpt-4o",
                                                messages=[{"role": "user",
                                                           "content": "Hi"}])
        assert result is expected_response

    def test_uses_response_model(self):
        client, _ = _make_openai_client(model="gpt-4o-mini",
                                        prompt_tokens=300,
                                        completion_tokens=100)
        tracker = CostTracker()
        wrap_openai(client, tracker)

        client.chat.completions.create(model="gpt-4o-mini",
                                       messages=[{"role": "user",
                                                  "content": "Hi"}])

        by_model = tracker.cost_by_model()
        assert "gpt-4o-mini" in by_model

    def test_no_usage_graceful(self):
        client, _ = _make_openai_client(usage=False)
        tracker = CostTracker()
        wrap_openai(client, tracker)

        result = client.chat.completions.create(model="gpt-4o",
                                                messages=[{"role": "user",
                                                           "content": "Hi"}])

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.summary()["num_requests"] == 0
        assert result is not None

    def test_multiple_calls_accumulate(self):
        client, _ = _make_openai_client(prompt_tokens=200,
                                        completion_tokens=80)
        tracker = CostTracker()
        wrap_openai(client, tracker)

        for _ in range(3):
            client.chat.completions.create(model="gpt-4o",
                                           messages=[{"role": "user",
                                                      "content": "Hi"}])

        assert tracker.total_input_tokens == 600
        assert tracker.total_output_tokens == 240
        assert tracker.summary()["num_requests"] == 3

    def test_budget_alert_fires(self):
        client, _ = _make_openai_client(prompt_tokens=100000,
                                        completion_tokens=50000)
        alerts = []
        tracker = CostTracker(budget=0.001,
                              on_budget_exceeded=lambda u, t: alerts.append(True))
        wrap_openai(client, tracker)

        client.chat.completions.create(model="gpt-4o",
                                       messages=[{"role": "user",
                                                  "content": "Hi"}])

        assert tracker.is_over_budget
        assert len(alerts) == 1
