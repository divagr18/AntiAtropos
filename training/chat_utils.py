"""Chat rendering helpers for text-only Qwen/Qwen-VL control prompts."""

from __future__ import annotations

from typing import Any, Dict, List


def render_no_think_chat(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Render a chat prompt with Qwen thinking disabled when supported.

    Qwen3-family templates expose ``enable_thinking`` as a Jinja variable.
    Older templates ignore that keyword, so we fall back cleanly rather than
    failing training for non-Qwen or older tokenizer builds.
    """
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_text_only(tokenizer: Any, input_text: str, device: Any):
    """Tokenize a rendered text prompt without invoking VL image loading.

    Some Qwen-VL processors route the first positional argument to ``images``.
    Passing the transcript through the explicit ``text=`` keyword keeps the
    prompt on the text path and avoids PIL trying to parse chat text as an image.
    """
    try:
        inputs = tokenizer(text=input_text, return_tensors="pt")
    except ValueError as exc:
        if "Incorrect image source" not in str(exc):
            raise
        inputs = tokenizer(text=[input_text], images=None, return_tensors="pt")
    except TypeError:
        inputs = tokenizer(input_text, return_tensors="pt")

    return inputs.to(device)
