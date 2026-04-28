"""
module_08_post_grid.py
======================
Replicates the "Post Grid" subprocess.

Responsibilities
----------------
1.  Recall the grid log (Feed_Conversion_merged_log) produced during the grid
    search, or create a default zero-value log if none exists.
2.  Parse numeric columns in the log.
3.  Find the best row (max sum_del_ethylene) from the log.
4.  Transpose and split: extract conversion_bias and feed_bias columns.
5.  Merge feed_bias and conversion_bias back onto the main df.
6.  Filter to rows with overall_ranking not missing.
7.  Compute New_Overall_conversion and New_Feed_flow.
8.  Evaluate inferred tags (inferred_tags_1 store).
9.  Aggregate sum(del_ethylene).
10. Compute ranking_opportunity = del_ethylene * 24.

Inputs  (STORE + MACROS)
------
    df_pre_grid   – main furnace DataFrame
    feed_combo    – dict {row_num: feed_delta}
    conv_combo    – dict {row_num: conv_delta}

Outputs  (return + MACROS)
-------
    df_result    – final per-furnace DataFrame with new columns
    MACROS["sum_del_ethylene_final"]
"""

import pandas as pd
import numpy as np
import logging

from config import MACROS, STORE

logger = logging.getLogger(__name__)

MAX_FURNACES = 9


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
# Step 1 – Build or recall the grid log
# Mirrors: 'recall LOG if created' Handle Exception + Feed grid log / Create ES
# ---------------------------------------------------------------------------
def build_grid_log() -> pd.DataFrame:
    """
    Construct the feed/conversion log table from the best grid result
    stored in MACROS (Row_N_feed_delta, Grid_Row_N_conversion_delta).
    This replicates what the RapidMiner Log operator records during the grid.
    """
    feed_combo = STORE.get("feed_combo", {})
    conv_combo = STORE.get("conv_combo", {})

    n = int(_m("Number_of_rows", 0))

    row = {}
    total_eth  = 0.0
    total_recycle = 0.0

    for i in range(1, MAX_FURNACES + 1):
        fd = feed_combo.get(i, 0.0)
        cd = conv_combo.get(i, 0.0)
        row[f"Row_{i}_feed_delta"]          = float(fd)
        row[f"Grid_Row_{i}_conversion_delta"] = float(cd)

        # Estimate delta ethylene
        feed    = _m(f"Row_{i}_Feed_flow", 0)
        eth     = _m(f"Row_{i}_Ethylene_Production", 0)
        shc     = _m("shc_ratio", 0)

        if feed > 0:
            del_eth_feed = eth * (fd / feed)
        else:
            del_eth_feed = 0.0
        del_eth_conv = eth * 0.8 * (cd / 100.0) if eth > 0 else 0.0

        total_eth += del_eth_feed + del_eth_conv

        # Recycle ethane change
        ethane_feed = feed / (1.0 + shc) if (1.0 + shc) > 0 else 0.0
        conv        = _m(f"Row_{i}_Conversion", 0)
        new_conv    = conv + cd
        recycle_change = ethane_feed * (100 - new_conv) / 100 - ethane_feed * (100 - conv) / 100
        total_recycle += recycle_change

    row["sum_del_ethylene"]               = round(total_eth, 4)
    row["sum_Change_in_Recycle_Ethane_Feed"] = round(total_recycle, 4)

    df_log = pd.DataFrame([row])
    logger.info("Grid log built: sum_del_ethylene=%.4f", total_eth)
    return df_log


# ---------------------------------------------------------------------------
# Step 2 – Select best row; extract conversion_bias + feed_bias
# Mirrors: Sort (118) + Filter Example Range (11) + Transpose + Filter
# ---------------------------------------------------------------------------
def extract_best_biases(df_log: pd.DataFrame) -> tuple[float, float]:
    """
    Sort by sum_del_ethylene descending; take the first row.
    Extract scalar conversion_bias (sum of Grid_Row_*_conversion_delta)
    and feed_bias (sum of Row_*_feed_delta).

    Returns (conversion_bias, feed_bias) as scalars for a per-furnace merge.
    """
    if df_log.empty:
        return 0.0, 0.0

    df_log = df_log.sort_values("sum_del_ethylene", ascending=False)
    best   = df_log.iloc[0]

    feed_cols = [c for c in df_log.columns if "_feed_delta" in c]
    conv_cols = [c for c in df_log.columns if "_conversion_delta" in c]

    # Return per-furnace dicts rather than scalars
    feed_bias_dict = {c: best[c] for c in feed_cols}
    conv_bias_dict = {c: best[c] for c in conv_cols}

    return conv_bias_dict, feed_bias_dict


# ---------------------------------------------------------------------------
# Step 3 – Merge biases back onto df; compute New_ columns
# Mirrors: Merge Attributes (55) + Generate Attributes (880)
# ---------------------------------------------------------------------------
def merge_biases(df: pd.DataFrame,
                 conv_bias_dict: dict,
                 feed_bias_dict: dict) -> pd.DataFrame:
    """
    Assign conversion_bias and feed_bias per furnace row (by sorted rank order).
    Then compute:
      New_Overall_conversion = Overall_conversion + conversion_bias
      New_Feed_flow          = Feed_flow + feed_bias
    """
    df = df.copy()

    if "overall_ranking" in df.columns:
        df = df.sort_values("overall_ranking").reset_index(drop=True)

    feed_combo = STORE.get("feed_combo", {})
    conv_combo = STORE.get("conv_combo", {})

    # Assign per-row biases
    def _get_feed_bias(i):
        row_num = i + 1
        # Try Row_N_feed_delta macro first, then feed_combo
        val = MACROS.get(f"Row_{row_num}_feed_delta", feed_combo.get(row_num, 0.0))
        try:
            return float(val)
        except Exception:
            return 0.0

    def _get_conv_bias(i):
        row_num = i + 1
        val = MACROS.get(f"Grid_Row_{row_num}_conversion_delta", conv_combo.get(row_num, 0.0))
        try:
            return float(val)
        except Exception:
            return 0.0

    n = len(df)
    df["feed_bias"]       = [_get_feed_bias(i)       for i in range(n)]
    df["conversion_bias"] = [_get_conv_bias(i)       for i in range(n)]

    # Compute new values
    if "Overall_conversion" in df.columns:
        df["New_Overall_conversion"] = df["Overall_conversion"].astype(float) + df["conversion_bias"]
    if "Feed_flow" in df.columns:
        df["New_Feed_flow"] = df["Feed_flow"].astype(float) + df["feed_bias"]

    return df


# ---------------------------------------------------------------------------
# Step 4 – Filter to ranked rows only
# Mirrors: Filter Examples (332) overall_ranking.is_not_missing.
# ---------------------------------------------------------------------------
def filter_ranked(df: pd.DataFrame) -> pd.DataFrame:
    if "overall_ranking" not in df.columns:
        return df
    return df[df["overall_ranking"].notna()].copy()


# ---------------------------------------------------------------------------
# Step 5 – Evaluate inferred tags (inferred_tags_1)
# Mirrors: 'inf tag final' subprocess (Loop over inferred_tags_1)
# ---------------------------------------------------------------------------
def evaluate_inferred_tags(df: pd.DataFrame) -> pd.DataFrame:
    df_tags = _recall("inferred_tags_1")
    if df_tags.empty:
        return df

    df = df.copy()
    for _, tag_row in df_tags.iterrows():
        tag_name    = tag_row.get("Inferred_tag", "")
        formula_str = tag_row.get("Inferred_tag_formula", "")
        if not tag_name or not formula_str:
            continue
        try:
            df[tag_name] = df.eval(formula_str)
        except Exception as e:
            logger.warning("Post-grid inferred tag '%s' eval failed: %s", tag_name, e)

    return df


# ---------------------------------------------------------------------------
# Step 6 – Compute del_ethylene per furnace; aggregate; ranking_opportunity
# Mirrors: Aggregate (80) + sum_del_ethylene_final + ranking_opportunity
# ---------------------------------------------------------------------------
def compute_ethylene_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    del_ethylene per furnace ≈ Ethylene_Production * feed_delta / Feed_flow
                               + Ethylene_Production * 0.8 * conv_delta / 100
    ranking_opportunity = del_ethylene * 24   (hourly → daily)
    """
    df = df.copy()

    eth  = df["ethylene_production"].astype(float) if "ethylene_production" in df.columns else pd.Series(0.0, index=df.index)
    feed = df["Feed_flow"].astype(float) if "Feed_flow" in df.columns else pd.Series(1.0, index=df.index)
    fd   = df["feed_bias"].astype(float) if "feed_bias" in df.columns else pd.Series(0.0, index=df.index)
    cd   = df["conversion_bias"].astype(float) if "conversion_bias" in df.columns else pd.Series(0.0, index=df.index)

    del_eth_feed = np.where(feed > 0, eth * (fd / feed), 0.0)
    del_eth_conv = eth * 0.8 * (cd / 100.0)

    df["del_ethylene"]        = del_eth_feed + del_eth_conv
    df["ranking_opportunity"] = df["del_ethylene"] * 24.0

    # Update global macro
    total = float(df["del_ethylene"].sum())
    MACROS["sum_del_ethylene_final"] = round(total, 4)
    logger.info("sum_del_ethylene_final=%.4f t/h", total)

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Post-Grid subprocess.

    Parameters
    ----------
    df : pd.DataFrame  – pre-grid furnace data

    Returns
    -------
    df_result : pd.DataFrame  – final per-furnace result with bias columns
    """
    logger.info("=== MODULE 08 – POST GRID ===")

    # Step 1: Build/recall grid log
    df_log = build_grid_log()
    _remember("Feed_Conversion_merged_log", df_log)

    # Step 2: Extract best biases
    conv_bias_dict, feed_bias_dict = extract_best_biases(df_log)

    # Step 3: Merge biases; compute New_ columns
    df_result = merge_biases(df, conv_bias_dict, feed_bias_dict)

    # Step 4: Filter to ranked rows
    df_result = filter_ranked(df_result)

    # Step 5: Evaluate inferred tags
    df_result = evaluate_inferred_tags(df_result)

    # Step 6: Compute del_ethylene + ranking_opportunity
    df_result = compute_ethylene_delta(df_result)

    _remember("df_post_grid", df_result)
    logger.info("POST GRID complete – %d rows × %d cols",
                len(df_result), len(df_result.columns))
    return df_result
