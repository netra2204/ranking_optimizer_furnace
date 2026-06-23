"""
module_03_parameterization.py
=============================
Replicates the "Parameterization" subprocess inside "United OLF".

Responsibilities
----------------
1.  Recall tag_parameter_mapping from STORE.
2.  Parse nominal columns to numeric (Parse Numbers + Numerical to Real).
3.  De-pivot on tag_name to long format (all columns except Timestamp).
4.  Inner-join to tag_parameter_mapping on tag_name ↔ short_name.
5.  Remove duplicates on (tag_name, Timestamp).
6.  Pivot back to wide format: rows = (Timestamp × entity_name),
    columns = parameter_name, aggregation = first.
7.  Rename aggregated columns (strip 'first(value)_' prefix).
8.  Return the wide DataFrame for pre-processing.

Inputs  (from STORE / args)
------
    "tag_parameter_mapping"  – from initialization (or STORE)
    df_main                  – main furnace dataset

Outputs  (return value)
-------
    df_param : pd.DataFrame
        Wide table with columns:
          Timestamp, entity_name, <parameter_name_1>, <parameter_name_2>, …
"""

import pandas as pd
import numpy as np
import logging
import re

from config import MACROS, STORE

logger = logging.getLogger(__name__)


def _recall(name: str) -> pd.DataFrame:
    df = STORE.get(name, pd.DataFrame())
    if df.empty:
        logger.warning("recall('%s') returned empty DataFrame.", name)
    return df


# ---------------------------------------------------------------------------
# Step 1 – Parse nominal → numeric (all columns)
# Mirrors: Parse Numbers (25) + Numerical to Real (11)
# ---------------------------------------------------------------------------
def parse_and_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try converting every column (except Timestamp) to float.
    Columns that cannot be parsed are left as object.
    Mirrors 'skip attribute' unparsable behaviour.
    """
    df = df.copy()
    for col in df.columns:
        if col in ("Timestamp", "_Timestamp_dt"):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------------
# Step 2 – De-Pivot to long format
# Mirrors: De-Pivot (10)
#   index_attribute = tag_name
#   keep_missings   = True
#   attribute_name regex: ^(?!Timestamp$).+   (all columns except Timestamp)
# ---------------------------------------------------------------------------
def de_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt all non-Timestamp columns into a long table:
      Timestamp | tag_name | value
    The 'tag_name' column holds the original column name.
    """
    id_vars = [c for c in ["Timestamp", "_Timestamp_dt"] if c in df.columns]
    value_vars = [c for c in df.columns if c not in id_vars]

    df_long = df.melt(id_vars=id_vars,
                      value_vars=value_vars,
                      var_name="tag_name",
                      value_name="value")
    logger.info("After de-pivot: %d rows", len(df_long))
    return df_long


# ---------------------------------------------------------------------------
# Step 3 – Join to tag_parameter_mapping on tag_name ↔ short_name
# Mirrors: Join (74)
# ---------------------------------------------------------------------------
def join_parameter_mapping(df_long: pd.DataFrame, df_tpm: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join df_long (tag_name) to df_tpm (short_name).
    Brings in entity_name and parameter_name.
    """
    if df_tpm.empty:
        logger.warning("tag_parameter_mapping empty – adding dummy entity_name.")
        df_long["entity_name"]  = "FS"
        df_long["parameter_name"] = df_long["tag_name"]
        return df_long

    # Rename short_name → tag_name for the merge key
    df_tpm_merge = df_tpm.rename(columns={"short_name": "tag_name"})

    merged = pd.merge(df_long, df_tpm_merge, on="tag_name", how="inner",
                      suffixes=("", "_map"))
    dup_cols = [c for c in merged.columns if c.endswith("_map")]
    merged.drop(columns=dup_cols, inplace=True)
    logger.info("After join to parameter_mapping: %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Step 4 – Remove duplicates on (tag_name, Timestamp)
# Mirrors: Remove Duplicates (17)  subset = tag_name|Timestamp
# ---------------------------------------------------------------------------
def dedup_tag_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    subset = [c for c in ["tag_name", "Timestamp"] if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first")
    logger.info("After dedup (tag_name, Timestamp): %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 5 – Pivot to wide format
# Mirrors: Pivot (10)
#   group_by = Timestamp | entity_name
#   column_grouping = parameter_name
#   aggregation = first(value)
# ---------------------------------------------------------------------------
def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot df so each unique parameter_name becomes a column.
    """
    index_cols = [c for c in ["Timestamp", "entity_name"] if c in df.columns]
    if "parameter_name" not in df.columns or "value" not in df.columns:
        logger.warning("Missing parameter_name / value columns – returning as-is.")
        return df

    df_pivot = df.pivot_table(
        index=index_cols,
        columns="parameter_name",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Flatten MultiIndex columns if present
    df_pivot.columns.name = None
    logger.info("After pivot: %d rows × %d cols", len(df_pivot), len(df_pivot.columns))
    return df_pivot


# ---------------------------------------------------------------------------
# Step 6 – Rename: strip 'first(value)_' prefix (left by RapidMiner pivot)
# Mirrors: Rename by Replacing (35)  replace_what = first\(value\)_
# ---------------------------------------------------------------------------
def rename_pivot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"^first\(value\)_", "", str(c)) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Parameterization subprocess.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main dataset from module_01_inputs (post initialization).

    Returns
    -------
    df_param : pd.DataFrame
        Wide per-(Timestamp × entity_name) table with one column per parameter.
    """
    logger.info("=== MODULE 03 – PARAMETERIZATION ===")

    df_tpm = _recall("tag_parameter_mapping")

    # Step 1: cast numerics
    df = parse_and_cast_numeric(df_main)

    # Step 2: de-pivot to long
    df_long = de_pivot(df)

    # Step 3: join parameter mapping
    df_long = join_parameter_mapping(df_long, df_tpm)

    # Step 4: dedup on (tag_name, Timestamp)
    df_long = dedup_tag_timestamp(df_long)

    # Step 5: pivot to wide
    df_param = pivot_wide(df_long)

    # Step 6: rename columns
    df_param = rename_pivot_columns(df_param)

    STORE["df_parameterized"] = df_param.copy()
    logger.info("PARAMETERIZATION complete – %d rows × %d cols",
                len(df_param), len(df_param.columns))
    return df_param
