"""Async SDK integration helpers for auto-tracking costs.

These wrappers monkey-patch async SDK clients so that every API call
automatically records token usage on a CostTracker instance.
Neither the anthropic nor openai package is required at import time.
"""

import functools


def wrap_anthropic_async(client, tracker):
    """Wrap an async Anthropic client's messages.create to auto-track costs.

    Usage:
        import anthropic
        client = anthropic.AsyncAnthropic()
        tracker = CostTracker(budget=1.0)
        wrap_anthropic_async(client, tracker)
        # Now every await client.messages.create() call is auto-tracked
    """
    original_create = client.messages.create

    @functools.wraps(original_create)
    async def wrapped_create(*args, **kwargs):
        response = await original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            model = getattr(response, "model", None) or kwargs.get("model", "unknown")
            tracker.record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return response

    client.messages.create = wrapped_create


def wrap_openai_async(client, tracker):
    """Wrap an async OpenAI client's chat.completions.create to auto-track costs.

    Usage:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        tracker = CostTracker(budget=1.0)
        wrap_openai_async(client, tracker)
        # Now every await client.chat.completions.create() call is auto-tracked
    """
    original_create = client.chat.completions.create

    @functools.wraps(original_create)
    async def wrapped_create(*args, **kwargs):
        response = await original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            model = getattr(response, "model", None) or kwargs.get("model", "unknown")
            tracker.record(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )
        return response

    client.chat.completions.create = wrapped_create
