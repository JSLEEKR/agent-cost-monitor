"""Tests for async SDK integration wrappers and async decorator."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest
from agent_cost_monitor import (
    CostTracker,
    async_track_usage,
    wrap_anthropic_async,
    wrap_openai_async,
)


# ---------------------------------------------------------------------------
# Helpers to build mock async SDK clients
# ---------------------------------------------------------------------------

def _make_async_anthropic_client(model="claude-sonnet-4-6", input_tokens=100,
                                  output_tokens=50, usage=True):
    """Return a mock async Anthropic client with an async messages.create."""
    response = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens,
                              output_tokens=output_tokens) if usage else None,
        content=[SimpleNamespace(text="Hello")],
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client, response


def _make_async_openai_client(model="gpt-4o", prompt_tokens=200,
                               completion_tokens=80, usage=True):
    """Return a mock async OpenAI client with an async chat.completions.create."""
    response = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens) if usage else None,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hi"),
        )],
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client, response


# ---------------------------------------------------------------------------
# wrap_anthropic_async tests
# ---------------------------------------------------------------------------

class TestWrapAnthropicAsync:
    @pytest.mark.asyncio
    async def test_records_usage(self):
        client, _ = _make_async_anthropic_client(input_tokens=100,
                                                  output_tokens=50)
        tracker = CostTracker()
        wrap_anthropic_async(client, tracker)

        await client.messages.create(model="claude-sonnet-4-6",
                                     messages=[{"role": "user", "content": "Hi"}])

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.summary()["num_requests"] == 1

    @pytest.mark.asyncio
    async def test_returns_original_response(self):
        client, expected_response = _make_async_anthropic_client()
        tracker = CostTracker()
        wrap_anthropic_async(client, tracker)

        result = await client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result is expected_response

    @pytest.mark.asyncio
    async def test_uses_response_model(self):
        client, _ = _make_async_anthropic_client(model="claude-haiku-4-5",
                                                  input_tokens=500,
                                                  output_tokens=200)
        tracker = CostTracker()
        wrap_anthropic_async(client, tracker)

        await client.messages.create(model="claude-haiku-4-5",
                                     messages=[{"role": "user", "content": "Hi"}])

        by_model = tracker.cost_by_model()
        assert "claude-haiku-4-5" in by_model

    @pytest.mark.asyncio
    async def test_no_usage_graceful(self):
        client, _ = _make_async_anthropic_client(usage=False)
        tracker = CostTracker()
        wrap_anthropic_async(client, tracker)

        result = await client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.summary()["num_requests"] == 0
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        client, _ = _make_async_anthropic_client(input_tokens=100,
                                                  output_tokens=50)
        tracker = CostTracker()
        wrap_anthropic_async(client, tracker)

        for _ in range(3):
            await client.messages.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert tracker.summary()["num_requests"] == 3

    @pytest.mark.asyncio
    async def test_budget_alert_fires(self):
        client, _ = _make_async_anthropic_client(input_tokens=100000,
                                                  output_tokens=50000)
        alerts = []
        tracker = CostTracker(
            budget=0.001,
            on_budget_exceeded=lambda u, t: alerts.append(True),
        )
        wrap_anthropic_async(client, tracker)

        await client.messages.create(model="claude-sonnet-4-6",
                                     messages=[{"role": "user", "content": "Hi"}])

        assert tracker.is_over_budget
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# wrap_openai_async tests
# ---------------------------------------------------------------------------

class TestWrapOpenAIAsync:
    @pytest.mark.asyncio
    async def test_records_usage(self):
        client, _ = _make_async_openai_client(prompt_tokens=200,
                                               completion_tokens=80)
        tracker = CostTracker()
        wrap_openai_async(client, tracker)

        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert tracker.total_input_tokens == 200
        assert tracker.total_output_tokens == 80
        assert tracker.summary()["num_requests"] == 1

    @pytest.mark.asyncio
    async def test_returns_original_response(self):
        client, expected_response = _make_async_openai_client()
        tracker = CostTracker()
        wrap_openai_async(client, tracker)

        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result is expected_response

    @pytest.mark.asyncio
    async def test_uses_response_model(self):
        client, _ = _make_async_openai_client(model="gpt-4o-mini",
                                               prompt_tokens=300,
                                               completion_tokens=100)
        tracker = CostTracker()
        wrap_openai_async(client, tracker)

        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
        )

        by_model = tracker.cost_by_model()
        assert "gpt-4o-mini" in by_model

    @pytest.mark.asyncio
    async def test_no_usage_graceful(self):
        client, _ = _make_async_openai_client(usage=False)
        tracker = CostTracker()
        wrap_openai_async(client, tracker)

        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.summary()["num_requests"] == 0
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        client, _ = _make_async_openai_client(prompt_tokens=200,
                                               completion_tokens=80)
        tracker = CostTracker()
        wrap_openai_async(client, tracker)

        for _ in range(3):
            await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert tracker.total_input_tokens == 600
        assert tracker.total_output_tokens == 240
        assert tracker.summary()["num_requests"] == 3

    @pytest.mark.asyncio
    async def test_budget_alert_fires(self):
        client, _ = _make_async_openai_client(prompt_tokens=100000,
                                               completion_tokens=50000)
        alerts = []
        tracker = CostTracker(
            budget=0.001,
            on_budget_exceeded=lambda u, t: alerts.append(True),
        )
        wrap_openai_async(client, tracker)

        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert tracker.is_over_budget
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# async_track_usage decorator tests
# ---------------------------------------------------------------------------

class TestAsyncTrackUsage:
    @pytest.mark.asyncio
    async def test_records_usage(self):
        tracker = CostTracker()
        response = SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=150, output_tokens=75),
        )

        @async_track_usage(tracker)
        async def call_api():
            return response

        result = await call_api()
        assert result is response
        assert tracker.total_input_tokens == 150
        assert tracker.total_output_tokens == 75

    @pytest.mark.asyncio
    async def test_model_override(self):
        tracker = CostTracker()
        response = SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )

        @async_track_usage(tracker, model="custom-model")
        async def call_api():
            return response

        await call_api()
        by_model = tracker.cost_by_model()
        assert "custom-model" in by_model
        assert "claude-sonnet-4-6" not in by_model

    @pytest.mark.asyncio
    async def test_multiple_calls(self):
        tracker = CostTracker()
        response = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )

        @async_track_usage(tracker)
        async def call_api():
            return response

        for _ in range(3):
            await call_api()

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert tracker.summary()["num_requests"] == 3

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        tracker = CostTracker()

        @async_track_usage(tracker)
        async def my_special_function():
            return SimpleNamespace(
                model="gpt-4o",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

        assert my_special_function.__name__ == "my_special_function"
