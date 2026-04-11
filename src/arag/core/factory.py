"""Factory helpers that bridge ``Config`` and other runtime objects.

This module is the single place that knows about both ``Config`` and
``LLMClient``, keeping those two modules decoupled from each other.
"""

from __future__ import annotations

import os
from typing import Any

from arag.core.config import Config


def resolve_llm_profile(config: Config, role: str = "rag") -> dict[str, Any]:
    """Resolve LLM kwargs for a role (``"rag"`` / ``"eval"`` / ...).

    Reads ``{ROLE}_MODEL`` from the environment to pick a profile under
    ``[llm.<profile>]``, then swaps ``api_key_env`` for the real key pulled
    from that env var. Returns a dict ready to splat into ``LLMClient(...)``.
    """
    env_var = f"{role.upper()}_MODEL"
    profile_name = os.environ.get(env_var)
    assert profile_name, f"{env_var} is not set (expected an [llm.*] profile name)"

    profile = config["llm"][profile_name]  # KeyError gives a clear message
    api_key = os.environ.get(profile["api_key_env"])
    assert api_key, (
        f"{profile['api_key_env']} is not set (required by LLM profile {profile_name!r})"
    )

    return {
        "model": profile["model"],
        "api_key": api_key,
        "base_url": profile.get("base_url", "https://api.openai.com/v1"),
        "temperature": profile.get("temperature", 0.0),
        "max_tokens": profile.get("max_tokens", 16384),
        "reasoning_effort": profile.get("reasoning_effort"),
    }
