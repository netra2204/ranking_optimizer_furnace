"""
module_09_generate_output.py
============================
Replicates the "Generate Output" logic that follows Post Grid.

In the RapidMiner process, after Post Grid the result is either:
  (a) The fresh optimised output (deviation_exists == 1 path), or
  (b) The previous hour's output joined with current state columns
      (deviation_exists == 0 path – reuse prev output).

Both paths then produce the final wide output DataFrame which is written
to the database / returned as the pipeline result.

Responsibilities
----------------
1.  Branch on deviation_exists:
      == 1  → use df_post_grid as the output base
      == 0  → recall prev_timestamp_ranking_output, join current columns,
               set ranking_cause_indicator = 99
2.  Build the final output columns required downstream:
      entity_name, Timestamp, overall_ranking, Feed_flow, New_Feed_flow,
      Overall_conversion, New_Overall_conversion, feed_bias, conversion_bias,
      ethylene_production, del_ethylene, ranking_opportunity,
      Furnace_condition, ranking_cause_indicator, sum_del_ethylene_final
3.  Add a 'change_in_furnace' column (0/1 – any bias applied).
4.  Return the final DataFrame.

Inputs  (STORE + MACROS)
------
    "df_post_grid"
    "prev_timestamp_ranking_output"
    MACROS["deviation_exists"]
    MACROS["ranking_cause_indicator"]
    MACROS["sum_del_ethylene_final"]

Outputs
-------
    df_output : pd.DataFrame – final ranked output
    STORE["df_final_output"]
"""

import pandas as pd
import numpy as np
import logging

from config import MACROS, STORE

logger = logging.getLogger(__name__)


def _m(key, default=0):
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _recall(name):
    return STORE.get(name, pd.DataFrame())


def _remember(name, df):
    STORE[name] = df.copy()


# ---------------------------------------------------------------------------
# Helper: columns present in final output
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "entity_name",
    "Timestamp",
    "overall_ranking",
    "Feed_flow",
    "New_Feed_flow",
    "Overall_conversion",
    "New_Overall_conversion",
    "feed_bias",
    "conversion_bias",
    "ethylene_production",
    "del_ethylene",
    "ranking_opportunity",
    "Furnace_condition",
    "ranking_cause_indicator",
    "sum_del_ethylene_final",
    "change_in_furnace",
    "days_remaining",
    "max_coke_thickness",
    "specific_energy_consumption",
    "shc_ratio",
]


# ---------------------------------------------------------------------------
# Path A – deviation_exists == 1  (fresh optimiser output)
# ---------------------------------------------------------------------------
def build_fresh_output(df_post_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Use the post-grid result directly. Add constant columns from MACROS.
    Mirrors the main output assembly after Post Grid.
    """
    df = df_post_grid.copy()

    # Stamp global macros onto every row
    df["ranking_cause_indicator"] = int(_m("ranking_cause_indicator", 1))
    df["sum_del_ethylene_final"]   = float(_m("sum_del_ethylene_final", 0))

    # change_in_furnace: 1 if any bias was applied, else 0
    fb = df["feed_bias"].abs()       if "feed_bias"       in df.columns else pd.Series(0, index=df.index)
    cb = df["conversion_bias"].abs() if "conversion_bias" in df.columns else pd.Series(0, index=df.index)
    df["change_in_furnace"] = np.where((fb > 0) | (cb > 0), 1, 0)

    logger.info("Fresh output built: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Path B – deviation_exists == 0  (reuse previous hour output)
# Mirrors: Branch (114) + Recall (43) + join + Set Macros (15) + Generate Attributes (130)
# ---------------------------------------------------------------------------
def build_reuse_output(df_current: pd.DataFrame) -> pd.DataFrame:
    """
    Recall the previous hour's ranking output and join with current-state
    columns (entity_name key).  Set ranking_cause_indicator = 99.
    """
    df_prev = _recall("prev_timestamp_ranking_output")

    if df_prev.empty:
        logger.warning("prev_timestamp_ranking_output is empty – falling back to current data.")
        return build_fresh_output(df_current)

    # Exclude bias / optimizer columns from prev output that should be reset
    exclude_cols = [
        "change_in_furnace", "cit_bias", "conversion_bias", "cot_bias",
        "feed_bias", "heat_bias", "shc_bias", "total_optimizer_run_check"
    ]
    df_prev_clean = df_prev.drop(
        columns=[c for c in exclude_cols if c in df_prev.columns],
        errors="ignore"
    )

    # Inner join on entity_name
    if "entity_name" not in df_current.columns or "entity_name" not in df_prev_clean.columns:
        return build_fresh_output(df_current)

    # Keep current-state columns from df_current (suffixed _curr to avoid collision)
    curr_keep = [c for c in df_current.columns if c not in df_prev_clean.columns or c == "entity_name"]
    df_merged = pd.merge(
        df_prev_clean,
        df_current[curr_keep],
        on="entity_name",
        how="inner",
        suffixes=("", "_curr")
    )

    # Set ranking_cause_indicator = 99 (no optimizer run)
    df_merged["ranking_cause_indicator"] = 99
    MACROS["ranking_cause_indicator"]    = 99

    # Zero out bias columns
    for col in ["feed_bias", "conversion_bias"]:
        df_merged[col] = 0.0

    # New values = current values (no change)
    if "Feed_flow" in df_merged.columns:
        df_merged["New_Feed_flow"] = df_merged["Feed_flow"]
    if "Overall_conversion" in df_merged.columns:
        df_merged["New_Overall_conversion"] = df_merged["Overall_conversion"]

    df_merged["del_ethylene"]         = 0.0
    df_merged["ranking_opportunity"]  = 0.0
    df_merged["sum_del_ethylene_final"] = 0.0
    df_merged["change_in_furnace"]    = 0

    logger.info("Reuse-previous-output: %d rows", len(df_merged))
    return df_merged


# ---------------------------------------------------------------------------
# Ensure required columns exist (fill with NaN / 0 where missing)
# ---------------------------------------------------------------------------
def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_pre_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the final output DataFrame.

    Parameters
    ----------
    df_pre_grid : pd.DataFrame
        The pre-grid / post-grid furnace data (used as current state for
        the reuse path).

    Returns
    -------
    df_output : pd.DataFrame
        Final ranked output ready for Output_Format_Check.
    """
    logger.info("=== MODULE 09 – GENERATE OUTPUT ===")

    deviation_exists = int(_m("deviation_exists", 1))
    df_post_grid     = _recall("df_post_grid")

    if deviation_exists == 1:
        # Fresh optimiser output
        if df_post_grid.empty:
            logger.warning("df_post_grid empty – using pre_grid data as fallback.")
            df_post_grid = df_pre_grid.copy()
        df_output = build_fresh_output(df_post_grid)
    else:
        # Reuse previous output
        df_output = build_reuse_output(
            df_post_grid if not df_post_grid.empty else df_pre_grid
        )

    # Ensure all required columns are present
    df_output = ensure_output_columns(df_output)

    # Sort by overall_ranking
    if "overall_ranking" in df_output.columns:
        df_output = df_output.sort_values("overall_ranking").reset_index(drop=True)

    _remember("df_final_output", df_output)
    logger.info("GENERATE OUTPUT complete – %d rows × %d cols",
                len(df_output), len(df_output.columns))
    return df_output
