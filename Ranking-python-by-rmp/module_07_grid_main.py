"""
module_07_grid_main.py
======================
Replicates the "Grid-Main" subprocess.

This is the core combinatorial optimiser.  In RapidMiner it uses
"Optimize Parameters (Grid)" (FEED GRID) which enumerates all combinations of
feed-delta values for each furnace (Row_1 … Row_9) and picks the combination
that maximises ethylene benefit.

Python implementation strategy
-------------------------------
Instead of a full Cartesian-product grid search (which can be millions of
combinations), we implement the same logic as:
  1. FEED GRID: enumerate discrete feed-delta values per furnace constrained
     by [lower_limit_feed … upper_limit_feed] with step_size_feed.
  2. For each combination, evaluate the benefit (delta ethylene) using the
     linear model embedded in the inner subprocess.
  3. Pick the best combination → store in Row_N_feed_delta macros.
  4. MAIN CONVERSION GRID: for the best feed combination, enumerate
     conversion-delta values per furnace and pick the best.
  5. Store final results in Row_N_feed_delta and Grid_Row_N_conversion_delta.

The inner performance model is:
    del_ethylene ≈ sum_i [ Row_i_Ethylene_Production *
                           (New_Feed / Current_Feed - 1) ]
This mirrors the RapidMiner 'Performance' operator with the sum_del_ethylene
metric driving the grid search.

Inputs  (MACROS)
------
    Row_N_lower/upper/step_size_feed      – from Pre-Grid
    Row_N_lower/upper/step_size_conversion
    Row_N_Feed_flow, Row_N_Ethylene_Production, Row_N_Conversion
    Min_target_sum_feed_bias
    fresh_feed_change, biasing_condition

Outputs  (MACROS)
-------
    Row_N_feed_delta              – best feed delta for each furnace
    Grid_Row_N_conversion_delta   – best conversion delta for each furnace
    sum_del_Feed_flow
    sum_del_ethylene_final
    Feed_Grid_Character
    Conversion_Grid_Success
    ranking_cause_indicator
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


def _set(key, val):
    MACROS[key] = val


# ---------------------------------------------------------------------------
# Helper: build discrete range for a furnace's grid parameter
# ---------------------------------------------------------------------------
def _build_range(lower: float, upper: float, step: float) -> list:
    """
    Build a list of evenly spaced values [lower, lower+step, …, upper].
    Returns [0] if step == 0 (no movement).
    Mirrors the [lower;upper;step;linear] syntax in RapidMiner's grid.
    """
    if step <= 0 or lower == upper:
        return [lower if lower == upper else 0.0]

    values = []
    v = lower
    while v <= upper + 1e-9:
        values.append(round(v, 6))
        v += step
    if not values:
        values = [lower]
    return values


# ---------------------------------------------------------------------------
# Objective function: compute total delta ethylene for a feed combo
# Mirrors the inner subprocess of FEED GRID + Performance operator
# ---------------------------------------------------------------------------
def _evaluate_feed_combo(feed_deltas: list, n_rows: int) -> tuple[float, float]:
    """
    Given a list of feed_delta values (one per furnace row, length n_rows),
    compute:
      sum_del_ethylene   – total change in ethylene production (t/h)
      sum_del_Feed_flow  – total change in feed (t/h)

    Model: ΔEthylene_i ≈ Ethylene_i * (ΔFeed_i / Feed_i)
    (linear approximation; coilsim equation replaced by proportional scaling)
    """
    total_ethylene = 0.0
    total_feed     = 0.0

    for i, delta in enumerate(feed_deltas):
        row_num = i + 1
        feed    = _m(f"Row_{row_num}_Feed_flow", 0)
        eth     = _m(f"Row_{row_num}_Ethylene_Production", 0)

        if feed > 0:
            total_ethylene += eth * (delta / feed)
        total_feed += delta

    return total_ethylene, total_feed


# ---------------------------------------------------------------------------
# FEED GRID – exhaustive (within budget)
# ---------------------------------------------------------------------------
def run_feed_grid(df: pd.DataFrame) -> dict:
    """
    Enumerate feed-delta combinations for active furnaces (up to MAX_FURNACES).
    Returns best_combo: {row_num: best_feed_delta}.

    Mirrors: FEED GRID (2) (concurrency:optimize_parameters_grid)
    """
    n = int(_m("Number_of_rows", 0))
    if n == 0:
        logger.warning("Number_of_rows=0 – no furnaces to optimise.")
        return {}

    fresh_feed_change      = int(_m("fresh_feed_change", 0))
    min_target_sum         = _m("Min_target_sum_feed_bias", 0)
    compare_log            = int(_m("compare_log_curr_feed_delta", 0))

    # Build per-furnace range lists
    ranges = []
    for i in range(1, n + 1):
        lo   = _m(f"Row_{i}_lower_limit_feed", 0)
        hi   = _m(f"Row_{i}_upper_limit_feed", 0)
        step = _m(f"Row_{i}_step_size_feed", 0)
        ranges.append(_build_range(lo, hi, step))

    logger.info("Feed grid: %d furnaces, range sizes: %s",
                n, [len(r) for r in ranges])

    # Guard against combinatorial explosion: cap to 10k evaluations
    total_combos = math.prod(len(r) for r in ranges)
    MAX_COMBOS   = 50_000
    best_eth     = -np.inf
    best_combo   = [0.0] * n
    best_sum     = 0.0

    if total_combos <= MAX_COMBOS:
        current_combo = [0.0] * n

        def _nested_search(depth):
            nonlocal best_eth, best_combo, best_sum
            if depth == n:
                eth, total_feed = _evaluate_feed_combo(current_combo, n)
                if fresh_feed_change == -1 and compare_log != 0:
                    if total_feed < min_target_sum:
                        return
                if eth > best_eth:
                    best_eth   = eth
                    best_combo = current_combo[:]
                    best_sum   = total_feed
                return
            for val in ranges[depth]:
                current_combo[depth] = val
                _nested_search(depth + 1)

        _nested_search(0)
    else:
        # Greedy per-furnace optimisation when full grid is too large
        logger.warning("Total combos=%d > %d; using greedy per-furnace optimisation.",
                       total_combos, MAX_COMBOS)
        best_combo = []
        for i, rng in enumerate(ranges):
            best_local = max(rng)   # take max positive (or least negative for reduction)
            best_combo.append(best_local)
        _, best_sum = _evaluate_feed_combo(best_combo, n)
        best_eth, _ = _evaluate_feed_combo(best_combo, n)

    # Store results in MACROS
    for i, delta in enumerate(best_combo):
        row_num = i + 1
        MACROS[f"Row_{row_num}_feed_delta"] = round(delta, 4)

    # Compute and store sum_del_Feed_flow
    sum_feed = sum(best_combo)
    MACROS["sum_del_Feed_flow"] = round(sum_feed, 4)
    logger.info("Feed grid best: sum_del_Feed=%+.2f  est_del_eth=%.4f t/h",
                sum_feed, best_eth)

    # Feed_Grid_Character flag
    if best_eth > 0:
        MACROS["Feed_Grid_Character"] = "positive"
    elif best_eth < 0:
        MACROS["Feed_Grid_Character"] = "negative"
    else:
        MACROS["Feed_Grid_Character"] = "neutral"

    # compare_log update (mirrors Generate Macro (20))
    if fresh_feed_change == -1:
        if compare_log != 0:
            MACROS["compare_log_curr_feed_delta"] = 1
        else:
            MACROS["compare_log_curr_feed_delta"] = 1 if sum_feed >= min_target_sum else 0

    return {i + 1: v for i, v in enumerate(best_combo)}


# ---------------------------------------------------------------------------
# CONVERSION GRID – per-furnace after best feed combo is locked
# Mirrors: MAIN CONVERSION GRID → GRID -Conversion → GRID (2)
# ---------------------------------------------------------------------------
def run_conversion_grid(df: pd.DataFrame, feed_combo: dict) -> dict:
    """
    For the chosen feed combination, optimise conversion deltas.
    Returns best_conversion_combo: {row_num: best_conversion_delta}.
    """
    n = int(_m("Number_of_rows", 0))
    if n == 0:
        return {}

    Conversion_Grid_Success = 0

    ranges = []
    for i in range(1, n + 1):
        lo   = _m(f"Row_{i}_lower_limit_conversion", 0)
        hi   = _m(f"Row_{i}_upper_limit_conversion", 0)
        step = _m(f"Row_{i}_step_size_conversion", 0)
        ranges.append(_build_range(lo, hi, step))

    total_combos = math.prod(len(r) for r in ranges)
    logger.info("Conversion grid: %d furnaces, total combos=%d", n, total_combos)

    best_eth_conv = -np.inf
    best_combo    = [0.0] * n

    MAX_COMBOS = 50_000
    if total_combos <= MAX_COMBOS:
        current_combo = [0.0] * n

        def _nested_search_conv(depth):
            nonlocal best_eth_conv, best_combo
            if depth == n:
                eth_inc = _evaluate_conversion_combo(current_combo, feed_combo, n)
                if eth_inc > best_eth_conv:
                    best_eth_conv = eth_inc
                    best_combo    = current_combo[:]
                return
            for val in ranges[depth]:
                current_combo[depth] = val
                _nested_search_conv(depth + 1)

        _nested_search_conv(0)
    else:
        best_combo = [r[-1] if r else 0.0 for r in ranges]
        best_eth_conv, _ = _evaluate_feed_combo(best_combo, n)

    if best_eth_conv > 0:
        Conversion_Grid_Success = 1

    MACROS["Conversion_Grid_Success"] = Conversion_Grid_Success

    result = {}
    for i, delta in enumerate(best_combo):
        row_num = i + 1
        MACROS[f"Grid_Row_{row_num}_conversion_delta"] = round(delta, 4)
        result[row_num] = round(delta, 4)

    logger.info("Conversion grid best est_del_eth=%.4f, success=%d",
                best_eth_conv, Conversion_Grid_Success)
    return result


def _evaluate_conversion_combo(conv_deltas: list, feed_combo: dict, n_rows: int) -> float:
    """
    Estimate ethylene benefit from conversion changes.
    ΔEthylene_i ≈ Feed_i * Δconversion_i / 100 * specific_yield
    (simplified; RapidMiner uses the coilsim model here)
    """
    total = 0.0
    YIELD_FACTOR = 0.8   # approximate ethylene yield per unit conversion change

    for i, conv_delta in enumerate(conv_deltas):
        row_num = i + 1
        feed    = _m(f"Row_{row_num}_Feed_flow", 0) + feed_combo.get(row_num, 0)
        eth     = _m(f"Row_{row_num}_Ethylene_Production", 0)

        if feed > 0 and eth > 0:
            # Proportional ethylene gain from conversion improvement
            total += eth * YIELD_FACTOR * (conv_delta / 100.0)

    return total


# ---------------------------------------------------------------------------
# Post-grid: compute sum_del_ethylene_final and update ranking_cause_indicator
# Mirrors: Generate Macro (45) fallback in Handle Exception (12)
# ---------------------------------------------------------------------------
def finalise_grid_results(df: pd.DataFrame, feed_combo: dict, conv_combo: dict):
    n = int(_m("Number_of_rows", 0))

    total_eth_feed = 0.0
    total_eth_conv = 0.0
    for i in range(1, n + 1):
        feed    = _m(f"Row_{i}_Feed_flow", 0)
        eth_prod = _m(f"Row_{i}_Ethylene_Production", 0)
        fd       = feed_combo.get(i, 0)
        cd       = conv_combo.get(i, 0)

        if feed > 0:
            total_eth_feed += eth_prod * (fd / feed)
        if eth_prod > 0:
            total_eth_conv += eth_prod * 0.8 * (cd / 100.0)

    MACROS["sum_del_ethylene_final"] = round(total_eth_feed + total_eth_conv, 4)

    # ranking_cause_indicator update (failure fallback)
    ric = int(_m("ranking_cause_indicator", 1))
    if ric == 5:
        MACROS["ranking_cause_indicator"] = -5
    elif ric == 6:
        MACROS["ranking_cause_indicator"] = -6
    else:
        pass   # stays positive if grid ran successfully

    logger.info("sum_del_ethylene_final=%.4f t/h", _m("sum_del_ethylene_final"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Grid-Main subprocess.

    Parameters
    ----------
    df : pd.DataFrame  – pre-grid furnace data

    Returns
    -------
    df : pd.DataFrame  – unchanged (outputs are MACROS)
    """
    logger.info("=== MODULE 07 – GRID MAIN ===")

    # Initialise grid macros
    _set("Min_target_sum_feed_bias",
         _m("fresh_feed_quantity", 0) * (1.0 + _m("shc_ratio", 0)))
    _set("Max_Benefit",     -1000 if _m("fresh_feed_change") == -1 else 0)
    _set("Max_Benefit_SPC", 1000)
    _set("ranking_cause_indicator", 1)

    try:
        # Run FEED GRID
        feed_combo = run_feed_grid(df)

        # Run CONVERSION GRID using best feed combo
        conv_combo = run_conversion_grid(df, feed_combo)

        # Finalise
        finalise_grid_results(df, feed_combo, conv_combo)

    except Exception as e:
        logger.error("Grid optimisation failed: %s", e)
        # Failure path: set negative indicator
        ric = int(_m("ranking_cause_indicator", 1))
        _set("ranking_cause_indicator", -abs(ric) if ric > 0 else ric)
        feed_combo = {i: 0.0 for i in range(1, MAX_FURNACES + 1)}
        conv_combo = {i: 0.0 for i in range(1, MAX_FURNACES + 1)}

    STORE["feed_combo"] = feed_combo
    STORE["conv_combo"] = conv_combo

    logger.info("GRID MAIN complete. ranking_cause_indicator=%s",
                MACROS.get("ranking_cause_indicator"))
    return df
