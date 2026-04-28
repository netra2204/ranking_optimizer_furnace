"""
module_06_pre_grid.py
=====================
Replicates the "Pre Grid" subprocess (child of "main" → "Ranking optimizaion main").

Responsibilities
----------------
1.  Initialise per-furnace computed columns:
      Ethane_Feed, Current_Recycle_Ethane_Feed, factor, feed_reduction_potential.
2.  Extract Number_of_rows and shc_ratio macros.
3.  Filter to Good-condition furnaces; compute upper_limit_feed per furnace
    based on Margin_condition_type and saturator_margin.
4.  Aggregate sum_upper_limit_feed and sum_feed_reduction_potential.
5.  Balance-feed loop (While loops in RapidMiner) that distributes available
    feed margin across furnaces (biasing_condition != 2 path).
6.  Compute mixed_feed_margin, Extra_Recycle_Ethane, and recycle-ethane limits.
7.  Apply fresh_feed_change == -1 branch (reduction-only limits).
8.  Apply biasing_condition == 1 feed balancing.
9.  Apply biasing_condition == 3 conversion limit logic.
10. Extract per-row macros (Row_N_*) for the grid optimizer.

Inputs  (STORE + MACROS)
------
    df_preprocessed, good_fur_data, no_good_fur_data

Outputs  (MACROS)
-------
    Row_1_* … Row_9_*  feed/conversion limits and furnace metadata
    sum_upper_limit_feed, mixed_feed_margin, Extra_Recycle_Ethane, etc.
"""

import pandas as pd
import numpy as np
import logging
import math

from config import MACROS, STORE

logger = logging.getLogger(__name__)

MAX_FURNACES = 9


def _m(key, default=0):
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _ms(key, default=""):
    return str(MACROS.get(key, default))


def _remember(name, df):
    STORE[name] = df.copy()


def _recall(name):
    return STORE.get(name, pd.DataFrame())


# ---------------------------------------------------------------------------
# Step 1 – Compute initial derived columns
# Mirrors: Generate Attributes (906)
# ---------------------------------------------------------------------------
def compute_initial_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    shc = df["shc_ratio"].fillna(0).astype(float) if "shc_ratio" in df.columns else 0.0

    # Ethane_Feed = Feed_flow / (1 + shc_ratio)
    if "Feed_flow" in df.columns:
        df["Ethane_Feed"] = df["Feed_flow"].astype(float) / (1.0 + shc)
    else:
        df["Ethane_Feed"] = 0.0

    # Current_Recycle_Ethane_Feed
    if "Overall_conversion" in df.columns:
        conv = df["Overall_conversion"].astype(float)
        df["Current_Recycle_Ethane_Feed"] = df["Ethane_Feed"] * (100.0 - conv) / 100.0
    else:
        df["Current_Recycle_Ethane_Feed"] = 0.0

    # factor
    if "percent_above_threshold" in df.columns and "Overall_conversion" in df.columns:
        pat = df["percent_above_threshold"].astype(float)
        conv = df["Overall_conversion"].astype(float)
        df["factor"] = np.where(
            pat > 0,
            1.0 / conv.replace(0, np.nan).fillna(1),
            -1.0 * conv
        )
    else:
        df["factor"] = 0.0

    # Initialise grid limit columns to 0
    df["lower_limit_feed"]             = 0.0
    df["upper_limit_feed"]             = 0.0
    df["flag_conversion_part_override"] = 0
    df["step_size_conversion"]          = 0.0

    # feed_reduction_potential: cap value 3 → 2
    if "feed_reduction_potential" in df.columns:
        df["feed_reduction_potential"] = df["feed_reduction_potential"].apply(
            lambda v: 2 if v == 3 else v
        )
    else:
        df["feed_reduction_potential"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Step 2 – Extract Number_of_rows and shc_ratio macros
# Mirrors: Number_of_rows (11) + Extract Macro (434)
# ---------------------------------------------------------------------------
def extract_row_macros(df: pd.DataFrame):
    MACROS["Number_of_rows"] = len(df)
    if "shc_ratio" in df.columns and len(df) > 0:
        MACROS["shc_ratio"] = float(df.iloc[0]["shc_ratio"])
    logger.info("Number_of_rows=%d  shc_ratio=%.4f", MACROS["Number_of_rows"], _m("shc_ratio"))


# ---------------------------------------------------------------------------
# Step 3 – Filter Good furnaces; compute upper_limit_feed
# Mirrors: Filter Examples (346) + Generate Attributes (833)
# ---------------------------------------------------------------------------
def compute_upper_limit_feed(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Good-condition furnaces, compute Margin_condition_type and
    upper_limit_feed / step_size_feed.
    """
    if "Furnace_condition" not in df.columns:
        return df

    df_good = df[df["Furnace_condition"] == "Good"].copy()

    if df_good.empty:
        return df

    margin_in_feed       = df_good["Margin_in_Feed"].astype(float) if "Margin_in_Feed" in df_good.columns else 0
    margin_lower_check   = df_good["Margin_in_Feed_lower_check"].astype(float) if "Margin_in_Feed_lower_check" in df_good.columns else 0
    saturator_margin     = df_good["saturator_margin"].astype(float) if "saturator_margin" in df_good.columns else 0
    max_potential        = df_good["max_potential_total_feed"].astype(float).apply(lambda v: 2 if v == 3 else v) if "max_potential_total_feed" in df_good.columns else 0

    # Margin_condition_type
    df_good["Margin_condition_type"] = np.where(
        (margin_in_feed == 1) & (margin_lower_check == 0), 0,
        np.where(
            (margin_in_feed == 1) & (margin_lower_check == 1), 1,
            1000
        )
    )

    # upper_limit_feed
    mct = df_good["Margin_condition_type"]
    df_good["upper_limit_feed"] = np.where(
        saturator_margin == 1,
        np.where(mct == 1, 4, np.where(mct == 0, 2, 0)),
        np.where(mct == 1, 2, np.where(mct == 0, 1, 0))
    )
    # Cap by max_potential_total_feed
    df_good["upper_limit_feed"] = np.minimum(df_good["upper_limit_feed"], max_potential)

    # step_size_feed
    df_good["step_size_feed"] = np.where(df_good["upper_limit_feed"] > 1, 2, df_good["upper_limit_feed"])

    # Merge back into full df
    update_cols = ["Margin_condition_type", "upper_limit_feed", "step_size_feed"]
    for col in update_cols:
        if col in df_good.columns:
            df.loc[df_good.index, col] = df_good[col]

    return df


# ---------------------------------------------------------------------------
# Step 4 – Aggregate sums
# Mirrors: Good (19) aggregate sum(upper_limit_feed)
#          No good (7) aggregate sum(feed_reduction_potential)
# ---------------------------------------------------------------------------
def aggregate_sums(df: pd.DataFrame):
    df_good    = df[df.get("Furnace_condition", pd.Series([])) == "Good"] if "Furnace_condition" in df.columns else pd.DataFrame()
    df_no_good = df[df.get("Furnace_condition", pd.Series([])) != "Good"] if "Furnace_condition" in df.columns else pd.DataFrame()

    sum_upper = float(df_good["upper_limit_feed"].sum()) if "upper_limit_feed" in df_good.columns else 0.0
    sum_fp    = float(df_no_good["feed_reduction_potential"].sum()) if "feed_reduction_potential" in df_no_good.columns else 0.0

    MACROS["sum_upper_limit_feed"]        = sum_upper
    MACROS["sum_feed_reduction_potential"] = sum_fp
    logger.info("sum_upper_limit_feed=%.2f  sum_feed_reduction_potential=%.2f", sum_upper, sum_fp)


# ---------------------------------------------------------------------------
# Step 5 – Balance-feed While loop  (biasing_condition != 2 path)
# Mirrors: Subprocess (94) + multiple Loop (While) operators
# ---------------------------------------------------------------------------
def balance_feed_loop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute available feed-reduction potential to No-Good furnaces
    and feed-increase margin to Good furnaces in a balanced way.

    The logic replicates the two While loops in the RMP that iterate until
    all furnaces have been allocated or the budget is consumed.
    """
    biasing_cond = int(_m("biasing_condition", 0))
    if biasing_cond == 2:
        return df   # bypass

    fresh_feed_change = int(_m("fresh_feed_change", 0))

    if "Furnace_condition" not in df.columns:
        return df

    df = df.copy()

    # --- Sub-path: fresh_feed_change == -1  (force reduction)
    if fresh_feed_change == -1:
        # upper_limit_feed = 0, lower_limit_feed = -feed_reduction_potential
        if "feed_reduction_potential" in df.columns:
            df["upper_limit_feed"] = 0.0
            df["lower_limit_feed"] = -df["feed_reduction_potential"].astype(float)
            df["step_size_feed"]   = 1.0
        return df

    # --- Normal path: sort by factor ascending (most needy first)
    if "factor" not in df.columns:
        df["factor"] = 0.0

    df_good    = df[df["Furnace_condition"] == "Good"].copy().sort_values("factor")
    df_no_good = df[df["Furnace_condition"] != "Good"].copy().sort_values("factor")

    # Allocate reduction to no-good furnaces (zeroing their feed_reduction_potential)
    # then adjust upper limits
    balance   = float(_m("sum_feed_reduction_potential", 0))
    iteration = 0
    total_fur = int(_m("total_fur_available_for_bias", len(df)))

    # While loop 1: allocate reduction
    for idx in df_no_good.index:
        if iteration >= total_fur:
            break
        fp = float(df_no_good.loc[idx, "feed_reduction_potential"]) if "feed_reduction_potential" in df_no_good.columns else 0
        taken = min(fp, balance)
        df.loc[idx, "lower_limit_feed"] = -taken
        balance -= taken
        iteration += 1

    # While loop 2: reallocate remaining balance to good furnaces
    balance2   = float(_m("sum_upper_limit_feed", 0))
    iteration2 = 0
    for idx in df_good.index:
        if iteration2 >= total_fur:
            break
        ulf = float(df_good.loc[idx, "upper_limit_feed"]) if "upper_limit_feed" in df_good.columns else 0
        taken2 = min(ulf, balance2)
        df.loc[idx, "upper_limit_feed"] = taken2
        balance2 -= taken2
        iteration2 += 1

    return df


# ---------------------------------------------------------------------------
# Step 6 – Compute mixed_feed_margin and Extra_Recycle_Ethane
# Mirrors: Generate Macro (149)
# ---------------------------------------------------------------------------
def compute_recycle_ethane_macros(df: pd.DataFrame):
    fresh_feed_change = int(_m("fresh_feed_change", 0))
    sum_ulf           = _m("sum_upper_limit_feed", 0)
    good_fur_count    = _m("count_of_good_fur", 0)
    shc               = _m("shc_ratio", 0)
    ff_qty            = _m("fresh_feed_quantity", 0)

    if fresh_feed_change != 0:
        mixed_feed_margin = round(sum_ulf)
    else:
        mixed_feed_margin = round(min(sum_ulf, good_fur_count * 2))

    MACROS["mixed_feed_margin"] = mixed_feed_margin

    extra_re = mixed_feed_margin / (1.0 + shc) - ff_qty if (1.0 + shc) != 0 else 0.0
    MACROS["Extra_Recycle_Ethane"] = extra_re

    re_upper = extra_re + _m("Fur_change_recycle_ethane_upper_limit", 0.3)
    re_lower = extra_re + _m("Fur_change_recycle_ethane_lower_limit", -0.3)
    MACROS["upper_limit_change_in_recycle_ethane"] = re_upper
    MACROS["lower_limit_change_in_recycle_ethane"] = re_lower

    logger.info("mixed_feed_margin=%.2f  Extra_Recycle_Ethane=%.4f", mixed_feed_margin, extra_re)


# ---------------------------------------------------------------------------
# Step 7 – Compute conversion limits per furnace
# Mirrors: Branch (66) → Generate Attributes (1039 / 43)
# ---------------------------------------------------------------------------
def compute_conversion_limits(df: pd.DataFrame) -> pd.DataFrame:
    biasing_cond = int(_m("biasing_condition", 0))
    df = df.copy()

    cv_upper_thresh = _m("conversion_bias_threshold_upper_limit", 1)
    cv_lower_thresh = _m("conversion_bias_threshold_lower_limit", -1)
    cv_upper_expand = _m("conversion_upper_limit_expansion_max_limit", 0)
    cv_lower_expand = _m("conversion_lower_limit_expansion_max_limit", 0)

    if "Overall_conversion" not in df.columns:
        df["conversion_lower_limit_in_grid"] = 0.0
        df["conversion_upper_limit_in_grid"]  = 0.0
        df["step_size_conversion"]            = 0.0
        return df

    conv = df["Overall_conversion"].astype(float)

    # Lower limit
    df["conversion_lower_limit_in_grid"] = df.apply(
        lambda r: min(0, max(cv_lower_thresh,
                             -math.floor((float(r["Overall_conversion"]) - cv_lower_expand) / 0.5) * 0.5)),
        axis=1
    )

    if biasing_cond == 3:
        # Conversion biasing active
        pat = df["percent_above_threshold"].astype(float) if "percent_above_threshold" in df.columns else 0
        fur_cond = df["Furnace_condition"] if "Furnace_condition" in df.columns else ""

        df["conversion_upper_limit_in_grid"] = df.apply(
            lambda r: max(0, min(cv_upper_thresh,
                                 math.floor((cv_upper_expand - float(r["Overall_conversion"])) / 0.5) * 0.5))
            if (r.get("Furnace_condition") == "Semi Good" and float(r.get("percent_above_threshold", 0)) > 0)
            else 0,
            axis=1
        )

        no_good_count = int(_m("count_of_no_good_fur", 0))
        if   no_good_count < 6:  step = 5
        elif no_good_count == 6: step = 3
        elif no_good_count == 9: step = 1
        else:                    step = 2
        df["step_size_conversion"] = step
    else:
        fresh_fc = int(_m("fresh_feed_change", 0))
        frl_rank = df["Forecasted_runlength_rank"].astype(float) if "Forecasted_runlength_rank" in df.columns else pd.Series(100, index=df.index)

        df["conversion_upper_limit_in_grid"] = df.apply(
            lambda r: max(0, min(cv_upper_thresh,
                                 math.floor((cv_upper_expand - float(r["Overall_conversion"])) / 0.5) * 0.5))
            if ((r.get("Furnace_condition") in {"Good", "Semi Good"}) and
                (fresh_fc != 0 or float(r.get("Forecasted_runlength_rank", 100)) != 100))
            else 0,
            axis=1
        )

    return df


# ---------------------------------------------------------------------------
# Step 8 – Finalise limits; handle lower/upper symmetry
# Mirrors: Generate Attributes (328)
# ---------------------------------------------------------------------------
def finalise_limits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lower_limit_feed" not in df.columns:
        df["lower_limit_feed"] = 0.0
    if "upper_limit_feed" not in df.columns:
        df["upper_limit_feed"] = 0.0

    ulf = df["upper_limit_feed"].astype(float)
    llf = df["lower_limit_feed"].astype(float)

    lower_limit_feed_org = llf.copy()

    # If ulf == llf_org → both 0
    both_zero = ulf == lower_limit_feed_org
    df["lower_limit_feed"] = np.where(both_zero, 0.0,
        np.where(ulf != 0, ulf, llf))
    df["upper_limit_feed"] = np.where(both_zero, 0.0,
        np.where(llf < 0, llf, ulf))

    df["step_size_feed"] = np.where(
        (df["lower_limit_feed"] + df["upper_limit_feed"]) == 0, 0.0, 1.0
    )
    return df


# ---------------------------------------------------------------------------
# Step 9 – Initialise Row macros to 0 (loop 1-9)
# Mirrors: Loop (48) + Set Macros (28)
# ---------------------------------------------------------------------------
def reset_row_macros():
    for i in range(1, MAX_FURNACES + 1):
        MACROS[f"Row_{i}_upper_limit_feed"]      = 0
        MACROS[f"Row_{i}_lower_limit_feed"]      = 0
        MACROS[f"Row_{i}_step_size_feed"]        = 0
        MACROS[f"Grid_Row_{i}_conversion_delta"] = 0
        MACROS[f"Row_{i}_step_size_conversion"]  = 0
        MACROS[f"Row_{i}_upper_limit_conversion"] = 0
        MACROS[f"Row_{i}_lower_limit_conversion"] = 0
        MACROS[f"Row_{i}_Furnace"]               = 0
        MACROS[f"Row_{i}_part_override"]         = 0


# ---------------------------------------------------------------------------
# Step 10 – Extract per-row macros from df
# Mirrors: Sort (106) + extract row loop (11) + Extract Macro (435)
# ---------------------------------------------------------------------------
def extract_row_macros_from_df(df: pd.DataFrame):
    """
    Sort by overall_ranking ascending; then for each row extract key values
    into Row_N_* macros.
    """
    if "overall_ranking" in df.columns:
        df = df.sort_values("overall_ranking").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = min(len(df), MAX_FURNACES)
    MACROS["Number_of_rows"] = n

    col_map = {
        "Feed_flow":                "Feed_flow",
        "entity_name":              "Furnace",
        "specific_energy_consumption": "Specific_Energy_consumption",
        "Overall_conversion":       "Conversion",
        "Furnace_condition":        "Furnace_condition",
        "ethylene_production":      "Ethylene_Production",
        "lower_limit_feed":         "lower_limit_feed",
        "upper_limit_feed":         "upper_limit_feed",
        "step_size_feed":           "step_size_feed",
        "Current_Recycle_Ethane_Feed": "Current_Recycle_Ethane_Feed",
        "conversion_lower_limit_in_grid": "lower_limit_conversion",
        "conversion_upper_limit_in_grid": "upper_limit_conversion",
        "step_size_conversion":     "step_size_conversion",
        "flag_conversion_part_override": "part_override",
    }

    for i in range(n):
        row = df.iloc[i]
        row_num = i + 1
        for df_col, macro_suffix in col_map.items():
            val = row.get(df_col, 0)
            MACROS[f"Row_{row_num}_{macro_suffix}"] = val

    logger.info("Row macros extracted for %d furnaces.", n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Pre-Grid subprocess.

    Parameters
    ----------
    df : pd.DataFrame  – pre-processed furnace data

    Returns
    -------
    df : pd.DataFrame  – with additional pre-grid columns
    """
    logger.info("=== MODULE 06 – PRE-GRID ===")

    # Step 1
    df = compute_initial_columns(df)

    # Step 2
    extract_row_macros(df)

    # Step 3
    df = compute_upper_limit_feed(df)

    # Step 4
    aggregate_sums(df)

    # Step 5 – balance feed loop
    df = balance_feed_loop(df)

    # Step 6 – recycle ethane macros
    compute_recycle_ethane_macros(df)

    # Step 7 – conversion limits
    df = compute_conversion_limits(df)

    # Step 8 – finalise limits
    df = finalise_limits(df)

    # Step 9 – reset row macros
    reset_row_macros()

    # Step 10 – extract per-row macros
    extract_row_macros_from_df(df)

    STORE["df_pre_grid"] = df.copy()
    logger.info("PRE-GRID complete – %d rows, Row macros set for %d furnaces.",
                len(df), int(_m("Number_of_rows")))
    return df
