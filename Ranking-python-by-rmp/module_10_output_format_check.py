"""
module_10_output_format_check.py
=================================
Replicates the "Output_Format_Check" subprocess.

The RapidMiner block uses a Handle Exception pattern:
  Try:
    Validate that the output DataFrame has exactly the 4 required columns
    in the correct types:
      Timestamp   – date/datetime (or parseable string)
      sub_model_id – numeric
      tag          – nominal (string)
      value        – numeric
    If valid, append a single dummy row of MISSING values to ensure the schema
    is always passed through cleanly.
  Except:
    Throw exception "Output is not in correct format"

In Python we:
  1. Validate required columns exist and have compatible types.
  2. If valid, return df_output as-is (with a schema check pass flag).
  3. If invalid, raise a ValueError with a descriptive message.

The output of this module is the *long* (melted) format used for DB writing:
  Timestamp | sub_model_id | tag | value

Inputs  (STORE)
------
    "df_final_output"

Outputs
-------
    df_long : pd.DataFrame   – long-format output ready for DB insert
    STORE["df_output_long"]
"""

import pandas as pd
import numpy as np
import logging

from config import MACROS, STORE, DB_CONFIG

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Timestamp", "sub_model_id", "tag", "value"]
MODEL_ID = DB_CONFIG.get("model_id", 520)


def _recall(name):
    return STORE.get(name, pd.DataFrame())


def _remember(name, df):
    STORE[name] = df.copy()


# ---------------------------------------------------------------------------
# Convert wide df_final_output → long format (Timestamp | sub_model_id | tag | value)
# ---------------------------------------------------------------------------
def pivot_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide final-output DataFrame into a long format suitable for the
    Furnace_Output DB table.

    Each furnace × parameter combination becomes one row:
      Timestamp    – from df
      sub_model_id – constant MODEL_ID
      tag          – column name (prefixed with entity_name if needed)
      value        – numeric cell value
    """
    if df.empty:
        logger.warning("pivot_to_long: input DataFrame is empty.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    id_vars = ["Timestamp", "entity_name"] if "entity_name" in df.columns else ["Timestamp"]

    # Exclude non-numeric / structural columns from the melt
    exclude = set(id_vars + ["_Timestamp_dt"])
    value_vars = [c for c in df.columns if c not in exclude]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="tag",
        value_name="value"
    )

    # Prefix tag with entity_name to make it unique (mirrors the tag naming in RapidMiner)
    if "entity_name" in df_long.columns:
        df_long["tag"] = df_long["entity_name"].astype(str) + "." + df_long["tag"]
        df_long.drop(columns=["entity_name"], inplace=True)

    df_long["sub_model_id"] = MODEL_ID

    # Coerce value to numeric; non-parseable → NaN
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Reorder columns
    df_long = df_long[REQUIRED_COLUMNS]
    logger.info("Long format: %d rows", len(df_long))
    return df_long


# ---------------------------------------------------------------------------
# Validation
# Mirrors: Handle Exception → Throw Exception "Output is not in correct format"
# ---------------------------------------------------------------------------
def validate_output(df_long: pd.DataFrame) -> bool:
    """
    Check that all required columns exist and have roughly the expected types.
    Raises ValueError if invalid.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df_long.columns]
    if missing:
        raise ValueError(
            f"Output is not in correct format – missing columns: {missing}"
        )

    # Timestamp must be parseable
    try:
        pd.to_datetime(df_long["Timestamp"].dropna().iloc[:1])
    except Exception:
        raise ValueError("Output is not in correct format – 'Timestamp' not parseable.")

    # sub_model_id and value must be numeric
    for col in ("sub_model_id", "value"):
        if not pd.api.types.is_numeric_dtype(df_long[col]):
            raise ValueError(
                f"Output is not in correct format – column '{col}' is not numeric."
            )

    # tag must be string/object
    if not pd.api.types.is_string_dtype(df_long["tag"]) and \
       not pd.api.types.is_object_dtype(df_long["tag"]):
        raise ValueError("Output is not in correct format – 'tag' is not nominal/string.")

    logger.info("Output format validation PASSED.")
    return True


# ---------------------------------------------------------------------------
# Append dummy MISSING row (mirrors the Append (20) in the Handle Exception try-branch)
# ---------------------------------------------------------------------------
def append_schema_row(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Append a single all-NaN sentinel row so the schema is always visible
    downstream even if the result set is empty.
    """
    dummy = pd.DataFrame([{
        "Timestamp":    pd.NaT,
        "sub_model_id": np.nan,
        "tag":          None,
        "value":        np.nan,
    }])
    return pd.concat([df_long, dummy], ignore_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_final: pd.DataFrame = None) -> pd.DataFrame:
    """
    Execute the Output_Format_Check subprocess.

    Parameters
    ----------
    df_final : pd.DataFrame, optional
        Wide final-output DataFrame.  If None, recalled from STORE.

    Returns
    -------
    df_long : pd.DataFrame
        Validated long-format output.

    Raises
    ------
    ValueError
        If the output does not conform to the required schema.
    """
    logger.info("=== MODULE 10 – OUTPUT FORMAT CHECK ===")

    if df_final is None or (isinstance(df_final, pd.DataFrame) and df_final.empty):
        df_final = _recall("df_final_output")

    try:
        # Convert to long format
        df_long = pivot_to_long(df_final)

        # Validate
        validate_output(df_long)

        # Append dummy schema row
        df_long = append_schema_row(df_long)

    except ValueError as exc:
        logger.error("Output format check FAILED: %s", exc)
        raise

    _remember("df_output_long", df_long)
    logger.info("OUTPUT FORMAT CHECK complete – %d rows (incl. schema sentinel).",
                len(df_long))
    return df_long
