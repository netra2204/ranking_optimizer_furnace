"""
module_10_output_format_check.py
=================================
Replicates the "Output_Format_Check" subprocess (RMP lines 9377–9432).

This is a defensive schema-enforcement block, NOT a transformation. The
input from module 09 is already in the correct 4-column long format:
    Timestamp | sub_model_id | tag | value
This module's only jobs are:
  1. Verify those 4 columns exist with compatible types.
  2. Append a single sentinel row of MISSING values (so the schema is always
     visible downstream even if the payload is empty).
  3. If validation fails, raise an exception with the exact RMP message
     "Output is not in correct format".

Operator flow (mirrors RMP exactly):

    Output_Format_Check
    └─ Handle Exception "Output_format_check (2)"
       │
       ├─ TRY arm:
       │   ├─ Create ExampleSet (13)  build 1-row missing-value frame:
       │   │     Timestamp     → DATE,    MISSING (NaT)
       │   │     sub_model_id  → NUMERIC, MISSING (NaN)
       │   │     tag           → NOMINAL, MISSING (None)
       │   │     value         → NUMERIC, MISSING (NaN)
       │   └─ Append (20) merge_type="all"
       │       Vertical concat of input + sentinel row.
       │
       └─ CATCH arm:
             Throw Exception "Output is not in correct format"

The Handle Exception in RapidMiner triggers the CATCH arm when:
  - The Append fails due to schema/type mismatch
  - The input is not a valid example set
We replicate that by validating the schema BEFORE the append, and raising
the same exception string on any mismatch.

Public surface:
    run(df) -> pd.DataFrame
        Returns the input with one MISSING-valued sentinel row appended.
        Raises ValueError("Output is not in correct format") on schema error.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from config import STORE

logger = logging.getLogger(__name__)

# The exact column ordering and roles expected from module 09's output
REQUIRED_COLUMNS = ["Timestamp", "sub_model_id", "tag", "value"]
EXCEPTION_MESSAGE = "Output is not in correct format"


# ===========================================================================
# Helpers
# ===========================================================================
def _remember(name: str, df: pd.DataFrame) -> None:
    STORE[name] = df.copy() if isinstance(df, pd.DataFrame) else df


def _is_datetime_like(series: pd.Series) -> bool:
    """True if the column already is datetime, or every non-null value parses
    cleanly as a datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return True   # vacuously ok
    try:
        parsed = pd.to_datetime(non_null, errors="coerce")
        return parsed.notna().all()
    except Exception:
        return False


def _is_numeric_like(series: pd.Series) -> bool:
    """True if the column is numeric, or every non-null value parses to numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return True
    parsed = pd.to_numeric(non_null, errors="coerce")
    return parsed.notna().all()


def _is_nominal_like(series: pd.Series) -> bool:
    """True for string/object/category — matches RapidMiner's 'nominal'."""
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


# ===========================================================================
# Schema validation — fires the CATCH arm via exception on failure
# ===========================================================================
def _validate_schema(df: pd.DataFrame) -> None:
    """
    Verify the 4 required columns exist with compatible types.
    Raises ValueError(EXCEPTION_MESSAGE) on any mismatch.

    Type compatibility (mirrors RapidMiner's column roles):
        Timestamp     → date / datetime-parseable
        sub_model_id  → numeric / numeric-parseable
        tag           → nominal (string/object)
        value         → numeric / numeric-parseable
    """
    # ── Missing columns ──────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Output_Format_Check FAILED: missing columns %s", missing)
        raise ValueError(EXCEPTION_MESSAGE)

    # ── Type compatibility ──────────────────────────────────────────────
    if not _is_datetime_like(df["Timestamp"]):
        logger.error("Output_Format_Check FAILED: 'Timestamp' is not date-like.")
        raise ValueError(EXCEPTION_MESSAGE)

    if not _is_numeric_like(df["sub_model_id"]):
        logger.error("Output_Format_Check FAILED: 'sub_model_id' is not numeric.")
        raise ValueError(EXCEPTION_MESSAGE)

    if not _is_nominal_like(df["tag"]):
        logger.error("Output_Format_Check FAILED: 'tag' is not nominal/string.")
        raise ValueError(EXCEPTION_MESSAGE)

    if not _is_numeric_like(df["value"]):
        logger.error("Output_Format_Check FAILED: 'value' is not numeric.")
        raise ValueError(EXCEPTION_MESSAGE)


# ===========================================================================
# Create ExampleSet (13) — build the 1-row sentinel of MISSING values
# ===========================================================================
def _build_missing_row() -> pd.DataFrame:
    """
    RapidMiner's Create ExampleSet (13) builds a 1-example dataframe whose
    cells are all MISSING but whose attribute *types* are pinned to:
        Timestamp    : DATE
        sub_model_id : NUMERIC
        tag          : NOMINAL
        value        : NUMERIC
    """
    return pd.DataFrame([{
        "Timestamp":    pd.NaT,
        "sub_model_id": np.nan,
        "tag":          None,
        "value":        np.nan,
    }])


# ===========================================================================
# Append (20) — vertical concat, merge_type="all"
# ===========================================================================
def _append_missing_row(df: pd.DataFrame, sentinel: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate sentinel on top of df (RMP wires Append's `example set 1` to
    the input and `example set 2` to Create ExampleSet, so input rows come
    first and the sentinel goes last). merge_type="all" preserves all columns
    from both sides.
    """
    # Align to REQUIRED_COLUMNS order — guards against accidental column shuffles
    df_in       = df[REQUIRED_COLUMNS].copy()
    sentinel_in = sentinel[REQUIRED_COLUMNS].copy()
    appended    = pd.concat([df_in, sentinel_in], axis=0, ignore_index=True)
    return appended


# ===========================================================================
# Public entry point
# ===========================================================================
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Output_Format_Check.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format output from module 09 with columns:
        Timestamp | sub_model_id | tag | value

    Returns
    -------
    pd.DataFrame
        The input with one extra row of MISSING values appended.

    Raises
    ------
    ValueError("Output is not in correct format")
        If the schema does not match the required 4-column long format.
    """
    logger.info("=== MODULE 10 – OUTPUT FORMAT CHECK ===")

    if df is None:
        logger.error("Output_Format_Check FAILED: input is None.")
        raise ValueError(EXCEPTION_MESSAGE)

    # ── Handle Exception TRY arm ─────────────────────────────────────────
    try:
        _validate_schema(df)
        sentinel = _build_missing_row()
        appended = _append_missing_row(df, sentinel)

    except ValueError:
        # CATCH arm: re-raise the RMP exception message verbatim
        raise

    except Exception as exc:
        # Any other failure (e.g. concat blowing up on a weird dtype) is
        # treated as a schema failure — mirrors RapidMiner's Handle Exception.
        logger.error("Output_Format_Check FAILED: unexpected error: %s", exc)
        raise ValueError(EXCEPTION_MESSAGE) from exc

    _remember("df_output_long", appended)
    logger.info(
        "OUTPUT FORMAT CHECK PASSED – %d input rows + 1 sentinel = %d total rows",
        len(df), len(appended),
    )
    return appended
