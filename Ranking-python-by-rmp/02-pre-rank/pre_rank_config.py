"""Read the ``config_overrides`` workbook and expose the values the
pre-rank stage needs (number of furnaces and number of passes)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Column names inside the config_overrides sheet.
_VAR_COL = "variable_name"
_VAL_COL = "value"

# Keys we care about (kept as constants so callers don't hard-code strings).
NUM_FURNACES_KEY = "NUM_FURNACES"
NUM_PASSES_KEY = "NUM_PASSES"


@dataclass(frozen=True)
class ConfigValues:
    """Typed view of the config values used by the pre-rank stage."""

    num_furnaces: int
    num_passes: int
    raw: Dict[str, object]  # full variable_name -> value map, for convenience


def _load_map(path: str, sheet_name: str | int | None) -> Dict[str, object]:
    """Return the ``variable_name -> value`` mapping from the config sheet."""
    df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    if _VAR_COL not in df.columns or _VAL_COL not in df.columns:
        raise ValueError(
            f"config workbook must contain columns '{_VAR_COL}' and '{_VAL_COL}'; "
            f"found {list(df.columns)}"
        )
    # Strip whitespace from keys; keep values as-is.
    return {str(k).strip(): v for k, v in zip(df[_VAR_COL], df[_VAL_COL])}


def read_config(path: str, sheet_name: str | int | None = None) -> ConfigValues:
    """Read ``config_overrides`` and return :class:`ConfigValues`.

    Raises ``KeyError`` if NUM_FURNACES / NUM_PASSES are absent, and
    ``ValueError`` if they are not positive integers.
    """
    raw = _load_map(path, sheet_name)

    for key in (NUM_FURNACES_KEY, NUM_PASSES_KEY):
        if key not in raw:
            raise KeyError(f"'{key}' not found in config overrides ({path})")

    def _as_positive_int(key: str) -> int:
        val = raw[key]
        try:
            ival = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{key}' must be an integer, got {val!r}") from exc
        if ival <= 0:
            raise ValueError(f"'{key}' must be positive, got {ival}")
        return ival

    num_furnaces = _as_positive_int(NUM_FURNACES_KEY)
    num_passes = _as_positive_int(NUM_PASSES_KEY)

    logger.info("Config: NUM_FURNACES=%d, NUM_PASSES=%d", num_furnaces, num_passes)
    return ConfigValues(num_furnaces=num_furnaces, num_passes=num_passes, raw=raw)
