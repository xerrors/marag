"""Configuration management for ARAG.

``Config`` is a thin, strict wrapper around a nested ``dict``:

- Load with ``Config.from_file(path)`` (auto-detects ``.toml``/``.yaml``/
  ``.yml``/``.json``) or one of the explicit ``from_*`` constructors.
- Access keys like a dict: ``config["llm"]["gpt-5-mini"]["model"]``. When a
  value is itself a mapping it is wrapped in another ``Config`` so chained
  bracket access keeps working.
- Missing keys raise ``KeyError`` immediately. Use ``config.get(key, default)``
  only when a default is explicitly desired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Config:
    """Strict, nested-dict-like configuration container."""

    def __init__(self, config_dict: dict[str, Any] | None = None):
        self._config: dict[str, Any] = dict(config_dict) if config_dict else {}

    # ------------------------------------------------------------------ loaders

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load a config file, dispatching on its extension."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".toml":
            return cls.from_toml(path)
        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if ext == ".json":
            return cls.from_json(path)
        raise ValueError(
            f"Unsupported config format {ext!r} (expected .toml, .yaml, .yml, or .json)"
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        import tomllib  # Python 3.11+

        with open(path, "rb") as f:
            return cls(tomllib.load(f))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------ dict-like API

    def __getitem__(self, key: str) -> Any:
        value = self._config[key]  # raises KeyError if missing
        if isinstance(value, dict):
            return Config(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __iter__(self):
        return iter(self._config)

    def __len__(self) -> int:
        return len(self._config)

    def __repr__(self) -> str:
        return f"Config({self._config!r})"

    def keys(self):
        return self._config.keys()

    def values(self):
        return self._config.values()

    def items(self):
        return self._config.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the raw value for ``key``, or ``default`` if missing."""
        return self._config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._config)
