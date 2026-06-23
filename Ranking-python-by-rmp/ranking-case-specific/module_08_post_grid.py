"""
module_08_post_grid.py
======================
Replicates the "Post Grid" subprocess (RMP lines 8391–8732).

Flow (mirrors the RMP operator chain exactly):

    Post Grid
    │
    ├─ Subprocess (64) "get the best bias"
    │   │
    │   ├─ Upper path (the input dataframe):
    │   │   ├─ Select Attributes (127)    exclude conversion_bias, feed_bias
    │   │   └─ Sort (31)                  by overall_ranking ascending
    │   │
    │   ├─ Lower path (the merged log):
    │   │   ├─ recall LOG if created  (Handle Exception)
    │   │   │     try:  recall Feed_Conversion_merged_log
    │   │   │     catch: Create ExampleSet (4) — zero-valued dummy log
    │   │   ├─ Parse Numbers (148)        coerce all cols to numeric
    │   │   └─ Subprocess (68) "best bias extractor"
    │   │       ├─ Sort (118)             sum_del_ethylene descending
    │   │       ├─ Filter Example Range (11)   keep row 1 (the winner)
    │   │       ├─ Transpose (106)        columns → rows  (id, att_1)
    │   │       ├─ Filter Examples (901)  id.contains "_feed_delta"
    │   │       ├─ Filter Examples (903)  id.contains "_conversion_delta"
    │   │       │     (operates on the "original" output of 901)
    │   │       ├─ Select Attributes (449) keep only att_1
    │   │       ├─ Rename (195)           att_1 → conversion_bias
    │   │       ├─ Select Attributes (436) keep only att_1
    │   │       ├─ Rename "feed bias"     att_1 → feed_bias
    │   │       └─ Merge Attributes (6)   positional column-bind →
    │   │                                 9 rows × (feed_bias, conversion_bias)
    │   │
    │   └─ Merge Attributes (55)          positional column-bind of upper-path
    │                                     sorted-input with lower-path biases
    │
    ├─ Filter Examples (332)              overall_ranking is_not_missing
    ├─ Generate Attributes (880)          New_Overall_conversion, New_Feed_flow
    │
    ├─ inf tag final (subprocess)
    │   ├─ recall inferred_tags_1
    │   ├─ Extract Macro (430)            inferred_tags_egs = row count
    │   └─ Loop (50) ×inferred_tags_egs
    │       ├─ Extract Macro (431)        Inferred_tag + Inferred_tag_formula
    │       └─ Generate Attributes (683)  <tag> = eval(<formula>)
    │
    ├─ Aggregate (80)                     sum(del_ethylene)
    ├─ Extract Macro                      sum_del_ethylene_final
    └─ ranking_opportunity                Generate Attributes [del_ethylene]*24

Key contract changes vs. the prior implementation
-------------------------------------------------
1. The merged log is now READ from STORE["Feed_Conversion_merged_log"]
   (populated by module_07a). It is NEVER synthesised from macros.
2. The winning bias values come from the *log row* (post sort+filter+transpose),
   not directly from Row_N_feed_delta macros.
3. `del_ethylene` is computed by inferred_tags_1 formulas, not by a hard-coded
   linear approximation. The aggregator just sums whatever the tags produced.
4. No more `good_tubes_calculated = 250.0` hack. If a downstream formula needs
   it, it must come from upstream modules.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from config import MACROS, STORE, NUM_FURNACES

logger = logging.getLogger(__name__)

# Columns produced by Create ExampleSet (4) in the RMP — the zero-valued
# fallback log when no real merged log exists.
_LOG_COLS = (
    [f"Row_{i}_feed_delta"             for i in range(1, NUM_FURNACES + 1)]
    + [f"Grid_Row_{i}_conversion_delta" for i in range(1, NUM_FURNACES + 1)]
    + ["sum_del_ethylene", "sum_Change_in_Recycle_Ethane_Feed"]
)


# ---------------------------------------------------------------------------
# Macro / store helpers
# ---------------------------------------------------------------------------
def _m(key: str, default: float = 0.0) -> float:
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _recall(name: str) -> pd.DataFrame:
    return STORE.get(name, pd.DataFrame())


def _remember(name: str, df: pd.DataFrame) -> None:
    STORE[name] = df.copy() if isinstance(df, pd.DataFrame) else df


# ===========================================================================
# Select Attributes (127) — exclude bias columns from the input
# ===========================================================================
def _exclude_input_bias_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: if df was reused across runs and already carries conversion_bias
    or feed_bias, drop them so the upcoming merge writes fresh values.
    """
    return df.drop(columns=["conversion_bias", "feed_bias"], errors="ignore")


# ===========================================================================
# recall LOG if created  (Handle Exception)
# ===========================================================================
def _recall_or_build_zero_log() -> pd.DataFrame:
    """
    Try-arm: recall the real Feed_Conversion_merged_log written by module_07a
             during the FEED grid enumeration.
    Catch-arm: Create ExampleSet (4) — a single-row, all-zeros dummy with the
               same 20-column schema.
    """
    log = _recall("Feed_Conversion_merged_log")
    if log is not None and not log.empty:
        logger.info("Feed_Conversion_merged_log recalled: %d rows × %d cols",
                    len(log), len(log.columns))
        return log.copy()

    logger.warning("Feed_Conversion_merged_log empty/missing – using zero-valued dummy log.")
    return pd.DataFrame([{c: 0.0 for c in _LOG_COLS}])


# ===========================================================================
# Parse Numbers (148) — coerce every column to numeric
# ===========================================================================
def _parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric coercion across every column. Non-parseable values become NaN."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ===========================================================================
# Subprocess (68) — extract per-furnace feed_bias / conversion_bias
# ===========================================================================
def _extract_best_biases(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    The RMP performs:
        Sort (118)        – sum_del_ethylene descending
        Filter Range (11) – keep row 1 (winner)
        Transpose (106)   – columns become rows; new id-col + att_1
        Filter Examples (901)  id.contains "_feed_delta"        → 9 rows
        Filter Examples (903)  id.contains "_conversion_delta"  → 9 rows
                                (from the "original" port of 901,
                                 i.e. the un-filtered transpose)
        Select + Rename: feed_bias  /  conversion_bias
        Merge Attributes (6) – positional column-bind → 9 rows × 2 cols

    We achieve the same result directly: pick the winning row from the log,
    then read its `Row_i_feed_delta` and `Grid_Row_i_conversion_delta`
    values in furnace-index order.
    """
    if df_log.empty:
        return pd.DataFrame({"feed_bias":       [0.0] * NUM_FURNACES,
                             "conversion_bias": [0.0] * NUM_FURNACES})

    # Sort (118) + Filter Example Range (11)
    if "sum_del_ethylene" not in df_log.columns:
        logger.warning("merged log missing 'sum_del_ethylene' – taking first row as winner.")
        winner = df_log.iloc[0]
    else:
        winner = df_log.sort_values("sum_del_ethylene", ascending=False).iloc[0]

    feed_biases = []
    conv_biases = []
    for i in range(1, NUM_FURNACES + 1):
        fb_col = f"Row_{i}_feed_delta"
        cb_col = f"Grid_Row_{i}_conversion_delta"
        feed_biases.append(float(winner[fb_col]) if fb_col in winner.index else 0.0)
        conv_biases.append(float(winner[cb_col]) if cb_col in winner.index else 0.0)

    bias_frame = pd.DataFrame({"feed_bias":       feed_biases,
                               "conversion_bias": conv_biases})
    logger.info("Best biases extracted: feed_sum=%+.2f  conv_sum=%+.2f",
                bias_frame["feed_bias"].sum(),
                bias_frame["conversion_bias"].sum())
    return bias_frame


# ===========================================================================
# Merge Attributes (55) — positional column-bind of data + biases
# ===========================================================================
def _merge_biases_into_df(df_sorted: pd.DataFrame,
                          bias_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Positional merge. RMP's operator_toolbox:merge keeps the longer side
    and pads the shorter side with NaN.

    df_sorted is the input dataframe sorted by overall_ranking ascending.
    bias_frame is the 9-row bias table from Subprocess (68). When df_sorted
    has fewer than 9 rows (Number_of_rows < 9), we trim the bias_frame to
    match the dataframe length so we don't introduce phantom rows.
    """
    df = df_sorted.copy().reset_index(drop=True)
    n  = len(df)

    # Trim or pad the bias frame to len(df). The merged log always has
    # MAX_FURNACES bias columns; furnaces beyond Number_of_rows simply carry 0.
    bf = bias_frame.iloc[:n].copy().reset_index(drop=True)
    while len(bf) < n:
        bf = pd.concat([bf, pd.DataFrame({"feed_bias": [0.0],
                                          "conversion_bias": [0.0]})],
                       ignore_index=True)

    df["feed_bias"]       = bf["feed_bias"].values
    df["conversion_bias"] = bf["conversion_bias"].values
    return df


# ===========================================================================
# Filter Examples (332) — overall_ranking is_not_missing
# ===========================================================================
def _filter_ranked(df: pd.DataFrame) -> pd.DataFrame:
    if "overall_ranking" not in df.columns:
        return df
    return df[df["overall_ranking"].notna()].copy()


# ===========================================================================
# Generate Attributes (880) — New_Overall_conversion, New_Feed_flow
# ===========================================================================
def _compute_new_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Overall_conversion" in df.columns:
        df["New_Overall_conversion"] = (
            df["Overall_conversion"].astype(float) + df["conversion_bias"].astype(float)
        )
    if "Feed_flow" in df.columns:
        df["New_Feed_flow"] = (
            df["Feed_flow"].astype(float) + df["feed_bias"].astype(float)
        )
    return df


# ===========================================================================
# inf tag final  → Loop (50) over inferred_tags_1
# ===========================================================================
def _evaluate_inferred_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every (Inferred_tag, Inferred_tag_formula) row from STORE
    `inferred_tags_1` to df via df.eval().

    `del_ethylene` is itself one of these inferred tags. We DO NOT
    pre-compute or overwrite it — the formula owns that column.
    """
    df_tags = _recall("inferred_tags_1")
    if df_tags is None or df_tags.empty:
        logger.info("inferred_tags_1 empty – skipping per-row inferred-tag eval.")
        return df
    if {"Inferred_tag", "Inferred_tag_formula"}.issubset(df_tags.columns) is False:
        logger.warning("inferred_tags_1 missing required columns – skipping.")
        return df

    # Extract Macro (430) — inferred_tags_egs (row count)
    n_tags = len(df_tags)
    MACROS["inferred_tags_egs"] = n_tags

    out = df.copy()
    for k, tag_row in enumerate(df_tags.itertuples(index=False), start=1):
        MACROS["iteration_inf_tags"] = k

        tag      = str(getattr(tag_row, "Inferred_tag", "")).strip()
        formula  = str(getattr(tag_row, "Inferred_tag_formula", "")).strip()
        if not tag or not formula or tag.lower() == "nan":
            continue
        try:
            out[tag] = out.eval(formula)
        except Exception as exc:
            logger.warning("Post-grid inferred tag '%s' eval failed: %s", tag, exc)

    return out


# ===========================================================================
# Aggregate (80) + Extract Macro sum_del_ethylene_final
# ===========================================================================
def _aggregate_and_extract_final(df: pd.DataFrame) -> None:
    """
    Pure sum over del_ethylene. RMP wires the *original* port of Aggregate (80)
    onwards (the un-aggregated df), so we don't need to materialise the
    one-row aggregated table — we just compute the scalar and stamp the macro.
    """
    if "del_ethylene" not in df.columns:
        logger.warning("'del_ethylene' missing after inferred-tag eval – "
                       "sum_del_ethylene_final set to 0.")
        MACROS["sum_del_ethylene_final"] = 0.0
        return
    total = float(pd.to_numeric(df["del_ethylene"], errors="coerce").fillna(0.0).sum())
    MACROS["sum_del_ethylene_final"] = round(total, 4)
    logger.info("sum_del_ethylene_final = %.4f t/h", total)


# ===========================================================================
# ranking_opportunity Generate Attributes — [del_ethylene] * 24
# ===========================================================================
def _add_ranking_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "del_ethylene" in df.columns:
        df["ranking_opportunity"] = pd.to_numeric(df["del_ethylene"], errors="coerce") * 24.0
    else:
        df["ranking_opportunity"] = 0.0
    return df


# ===========================================================================
# Public entry point
# ===========================================================================
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Post-Grid subprocess.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-grid furnace dataframe; one row per active furnace.

    Returns
    -------
    df_result : pd.DataFrame
        Same df with `feed_bias`, `conversion_bias`, `New_Overall_conversion`,
        `New_Feed_flow`, all inferred_tags_1 columns including `del_ethylene`,
        and `ranking_opportunity`. Filtered to rows where overall_ranking is
        not missing.

    Side effects
    ------------
    MACROS["sum_del_ethylene_final"]
    MACROS["inferred_tags_egs"]
    MACROS["iteration_inf_tags"]
    STORE["df_post_grid"]
    """
    logger.info("=== MODULE 08 – POST GRID ===")

    # ── Subprocess (64) – upper path ─────────────────────────────────────
    df_upper = _exclude_input_bias_cols(df)
    if "overall_ranking" in df_upper.columns:
        df_upper = df_upper.sort_values("overall_ranking", ascending=True).reset_index(drop=True)

    # ── Subprocess (64) – lower path ─────────────────────────────────────
    df_log     = _recall_or_build_zero_log()
    df_log     = _parse_numbers(df_log)
    bias_frame = _extract_best_biases(df_log)

    # ── Merge Attributes (55) — positional column-bind ────────────────────
    df_merged = _merge_biases_into_df(df_upper, bias_frame)

    # ── Filter Examples (332) — overall_ranking not missing ──────────────
    df_filtered = _filter_ranked(df_merged)

    # ── Generate Attributes (880) — New_Overall_conversion, New_Feed_flow ─
    df_new = _compute_new_values(df_filtered)

    # ── inf tag final — Loop (50) over inferred_tags_1 ────────────────────
    df_with_tags = _evaluate_inferred_tags(df_new)

    # ── Aggregate (80) + Extract Macro sum_del_ethylene_final ─────────────
    _aggregate_and_extract_final(df_with_tags)

    # ── ranking_opportunity Generate Attributes ───────────────────────────
    df_result = _add_ranking_opportunity(df_with_tags)

    _remember("df_post_grid", df_result)
    logger.info("POST GRID complete – %d rows × %d cols",
                len(df_result), len(df_result.columns))
    return df_result
