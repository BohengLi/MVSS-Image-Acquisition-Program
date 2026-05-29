from __future__ import annotations

import logging
import math
from typing import Any


LOGGER = logging.getLogger("mvss_capture")
TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled"}


def config_bool(config: dict[str, Any] | object, key: str, default: bool) -> bool:
    if not isinstance(config, dict):
        return default
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return default
    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return default


def config_float(config: dict[str, Any] | object, key: str, default: float) -> float:
    if not isinstance(config, dict):
        return default
    value = config.get(key, default)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        LOGGER.warning("Invalid numeric config value for %s=%r; using default %r.", key, value, default)
        return default
    if not math.isfinite(parsed):
        LOGGER.warning("Invalid non-finite config value for %s=%r; using default %r.", key, value, default)
        return default
    return parsed


def config_int(config: dict[str, Any] | object, key: str, default: int) -> int:
    if not isinstance(config, dict):
        return default
    value = config.get(key, default)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        LOGGER.warning("Invalid integer config value for %s=%r; using default %r.", key, value, default)
        return default
