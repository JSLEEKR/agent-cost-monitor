import functools


def track_usage(tracker, model=None):
    """Decorator that extracts usage from API response and records it.

    Works with responses that have a .usage attribute (Anthropic/OpenAI SDK style)
    with input_tokens and output_tokens fields.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            usage = response.usage
            resolved_model = model or response.model
            tracker.record(
                model=resolved_model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            return response
        return wrapper
    return decorator
