"""
module_02_initialization.py
===========================
Replicates the "Initialization (3)" subprocess inside "United OLF".

Responsibilities
----------------
1.  Join the tag master table and tag-details to the main dataset on the
    'name' / 'short_name' key.
2.  Remove duplicate tag_id rows (keep first occurrence).
3.  Filter the ROPT_extract_macro_value rows (pipeline_location contains
    "ROPT_extract_macro_value_furnace" → entity_name = "F1", else "FS").
4.  Filter the deviation-check rows
    (pipeline_location contains "deviation_check_furnace_system").
5.  Join tag_parameter_mapping on 'short_name'; deduplicate on parameter_id.
6.  Select entity_name + parameter_name columns → store as
    ROPT_extract_macro_value.
7.  Split inferred-tag rows (0_1, 0_2, … prefix) into four separate stores:
    inferred_tags_1 … inferred_tags_4.

Inputs  (from STORE)
------
    "tag"               – tag master dataframe
    "tag_details"       – tag details / coilsim formula dataframe
    "tag_parameter_mapping" – parameter-to-tag mapping dataframe
    "df_main_inputs"    – main furnace dataset from module 01

Outputs  (to STORE)
-------
    "ROPT_extract_macro_value"      – entity / parameter mapping
    "deviation_check_furnace_system"– sorted deviation rows
    "inferred_tags_1" … "_4"        – split inferred-tag formula tables
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


def _remember(name: str, df: pd.DataFrame):
    STORE[name] = df.copy()
    logger.debug("Stored '%s'  (%d × %d)", name, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Step 1 – Join tag + tag_details on 'name' (inner join)
# Mirrors: Join (24)
# ---------------------------------------------------------------------------
def join_tag_details(df_tag: pd.DataFrame, df_tag_details: pd.DataFrame) -> pd.DataFrame:
    """Inner join on 'name' column; remove duplicate attributes."""
    if df_tag.empty or df_tag_details.empty:
        logger.info("Tag / tag_details empty – skipping join, returning tag as-is.")
        return df_tag

    # Suffix suffixes for overlapping columns (RapidMiner keeps the left side)
    merged = pd.merge(df_tag, df_tag_details, on="name", how="inner",
                      suffixes=("", "_detail"))
    # Drop duplicate columns (those ending in _detail override nothing)
    dup_cols = [c for c in merged.columns if c.endswith("_detail")]
    merged.drop(columns=dup_cols, inplace=True)
    logger.info("After join tag+tag_details: %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Step 2 – Remove duplicates on tag_id (keep first)
# Mirrors: Remove Duplicates (11)
# ---------------------------------------------------------------------------
def deduplicate_tag_id(df: pd.DataFrame) -> pd.DataFrame:
    if "tag_id" not in df.columns:
        return df
    df = df.drop_duplicates(subset=["tag_id"], keep="first")
    logger.info("After dedup on tag_id: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3 – Filter ROPT_extract_macro_value rows; set entity_name
# Mirrors: Filter Examples (83) + Generate Attributes (90)
# ---------------------------------------------------------------------------
def filter_ropt_extract_macro_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows where pipeline_location contains 'ROPT_extract_macro_value'.
    Then assign entity_name:
      'F1'  if pipeline_location contains 'ROPT_extract_macro_value_furnace'
      'FS'  otherwise
    """
    if "pipeline_location" not in df.columns:
        logger.warning("'pipeline_location' column missing – returning as-is.")
        return df

    mask = df["pipeline_location"].str.contains("ROPT_extract_macro_value", na=False)
    df_ropt = df[mask].copy()

    df_ropt["entity_name"] = df_ropt["pipeline_location"].apply(
        lambda loc: "F1"
        if "ROPT_extract_macro_value_furnace" in str(loc)
        else "FS"
    )
    logger.info("ROPT_extract_macro_value rows: %d", len(df_ropt))
    return df_ropt


# ---------------------------------------------------------------------------
# Step 4 – Filter deviation_check_furnace_system rows (sorted desc by name)
# Mirrors: Filter Examples (85) + Sort (3) + Remember (3)
# ---------------------------------------------------------------------------
def filter_deviation_check(df: pd.DataFrame) -> pd.DataFrame:
    if "pipeline_location" not in df.columns:
        return pd.DataFrame()

    mask = df["pipeline_location"].str.contains("deviation_check_furnace_system", na=False)
    df_dev = df[mask][["name"]].copy()
    df_dev = df_dev.sort_values("name", ascending=False)
    logger.info("deviation_check_furnace_system rows: %d", len(df_dev))
    return df_dev


# ---------------------------------------------------------------------------
# Step 5 – Join tag_parameter_mapping on short_name; dedup on parameter_id
# Mirrors: Join (41) + Remove Duplicates (30) + Select Attributes (103)
# ---------------------------------------------------------------------------
def join_parameter_mapping(df_ropt: pd.DataFrame,
                            df_tpm: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join df_ropt to the tag_parameter_mapping table on 'short_name'.
    Keep only entity_name and parameter_name columns.
    Deduplicate on parameter_id before final selection.
    """
    if df_tpm.empty:
        logger.warning("tag_parameter_mapping empty – returning ropt as-is.")
        return df_ropt[["entity_name"]].copy() if "entity_name" in df_ropt.columns else pd.DataFrame()

    merged = pd.merge(df_ropt, df_tpm, on="short_name", how="inner",
                      suffixes=("", "_map"))
    # Drop _map suffixed duplicates
    dup_cols = [c for c in merged.columns if c.endswith("_map")]
    merged.drop(columns=dup_cols, inplace=True)

    # Dedup on parameter_id
    if "parameter_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["parameter_id"], keep="first")

    # Select only entity_name + parameter_name
    keep_cols = [c for c in ["entity_name", "parameter_name"] if c in merged.columns]
    result = merged[keep_cols].copy()
    logger.info("After parameter_mapping join: %d rows, cols=%s", len(result), list(result.columns))
    return result


# ---------------------------------------------------------------------------
# Step 6 – Split inferred-tag rows into groups 1..4
# Mirrors: Subprocess (10): Split on '0_' → filter name_1 contains '1'|'2'|..
# ---------------------------------------------------------------------------
def split_inferred_tags(df: pd.DataFrame) -> dict:
    """
    Select rows with name + formula_expression columns.
    Split the 'name' column on the '0_' delimiter into name_1 and name_2.
    Then partition into four groups based on name_1 containing '1','2','3','4'.

    Returns a dict: {1: df_inferred_1, 2: df_inferred_2, ...}
    """
    needed = {"name", "formula_expression"}
    if not needed.issubset(set(df.columns)):
        logger.warning("Columns for inferred-tag split not found; skipping.")
        return {}

    df_sel = df[["name", "formula_expression"]].copy()

    # Split 'name' on '0_' (ordered split – first occurrence)
    split_parts = df_sel["name"].str.split("0_", n=1, expand=True)
    df_sel["name_1"] = split_parts[0] if 0 in split_parts else ""
    df_sel["name_2"] = split_parts[1] if 1 in split_parts else df_sel["name"]

    # Rename name_2 → Inferred_tag, formula_expression → Inferred_tag_formula
    df_sel = df_sel.rename(columns={"name_2": "Inferred_tag",
                                     "formula_expression": "Inferred_tag_formula"})

    groups = {}
    for i in range(1, 5):
        mask = df_sel["name_1"].str.contains(str(i), na=False)
        grp = df_sel[mask][["Inferred_tag", "Inferred_tag_formula"]].copy()
        grp = grp.reset_index(drop=True)
        groups[i] = grp
        logger.info("inferred_tags_%d: %d rows", i, len(grp))

    return groups


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Initialization (3) subprocess.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main dataset from module_01_inputs.

    Returns
    -------
    df_main : pd.DataFrame
        Unchanged pass-through; initialization primarily populates STORE.
    """
    logger.info("=== MODULE 02 – INITIALIZATION ===")

    # Recall stored data-frames
    df_tag         = _recall("tag")
    df_tag_details = _recall("tag_details")
    df_tpm         = _recall("tag_parameter_mapping")  # may be empty initially

    # If tag data was not separately loaded, treat df_main as the unified source
    if df_tag.empty:
        logger.info("No separate tag store found – using df_main as tag source.")
        df_tag = df_main.copy()

    # --- Steps 1 & 2: join + dedup ---
    df_joined = join_tag_details(df_tag, df_tag_details)
    df_joined = deduplicate_tag_id(df_joined)

    # --- Step 3: ROPT extract-macro-value filter ---
    df_ropt = filter_ropt_extract_macro_value(df_joined)

    # --- Step 4: deviation-check filter ---
    df_dev = filter_deviation_check(df_joined)
    _remember("deviation_check_furnace_system", df_dev)

    # --- Step 5: join parameter mapping ---
    df_ropt_mapped = join_parameter_mapping(df_ropt, df_tpm)
    _remember("ROPT_extract_macro_value", df_ropt_mapped)

    # Update macro with count of ROPT extract macro rows
    MACROS["extract_value_count"] = len(df_ropt_mapped)
    logger.info("extract_value_count = %d", MACROS["extract_value_count"])

    # --- Step 6: split inferred tags ---
    inferred_groups = split_inferred_tags(df_joined)
    for idx, df_inf in inferred_groups.items():
        _remember(f"inferred_tags_{idx}", df_inf)

    # If no inferred-tag data found, store empty fallbacks
    for i in range(1, 5):
        if f"inferred_tags_{i}" not in STORE:
            _remember(f"inferred_tags_{i}", pd.DataFrame(columns=["Inferred_tag", "Inferred_tag_formula"]))

    logger.info("INITIALIZATION complete.")
    return df_main
