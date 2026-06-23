"""
module_07a_grid_feed.py
=======================
Chunk A of the Grid-Main subprocess: the **FEED-GRID layer**.

This file replicates everything in the RapidMiner `Grid-Main` subprocess
*except* the inner `MAIN CONVERSION GRID` and `GRID -Conversion` blocks.
Those live in `module_07b_grid_conversion.py`.

RapidMiner blocks covered (.rmp lines 5492 – 8390):
---------------------------------------------------
    • Generate ID (25)              – assign a 1..N id per furnace row
    • Set Role (21)                 – id = regular
    • Generate Macro (17)           – init Min_target_sum_feed_bias,
                                      Max_Benefit, Max_Benefit_SPC,
                                      ranking_cause_indicator = 1
    • Handle Exception (12)         – wraps the entire grid; fallback sets
                                      ranking_cause_indicator = -5/-6/-1
    • FEED GRID (2)                 – grid over Row_1..9_feed_delta with
                                      `Loop (49)` step-size override
    • sum_del_Feed_flow / Feed_Grid_Character / handle dupe (Branch 81)
                                    – signature log to skip duplicate combos
    • Generate Macro (20)           – fresh_feed_change == -1 rule
    • Branch (90)                   – IF compare_log_curr_feed_delta != 0
                                      THEN skip, ELSE: build per-row table
                                      and call the MAIN CONVERSION GRID
    • Loop (144) + Generate Attr 334 – per-row del_Feed_flow / New_Feed_flow
    • Append (27) + exclude (2)     – rebuild ExampleSet with new feeds
    • Branch (113)                  – if sum_del_ethylene == Max_Benefit
                                      → commit feed_delta_1..9 and
                                        conversion_delta_best_1..9 as
                                        new global best
    • Generate Attributes (914)     – sum(del_Feed_flow), mixed_feed_margin
                                      (informational; carried on the df)
    • Set Macros (3)                – update Min_target_sum_feed_bias and
                                      Max_Benefit_SPC for next iteration
    • Log "Feed_Conversion_merged_log" – running log of every (feed,
                                          conversion, sum_del_ethylene) row
    • Free Memory (41)

Macros consumed
---------------
    Row_N_lower_limit_feed, Row_N_upper_limit_feed, Row_N_step_size_feed
    Number_of_rows, fresh_feed_change, fresh_feed_quantity, shc_ratio,
    mixed_feed_margin, Min_target_sum_feed_bias

    Plus everything Chunk B needs (passed through MACROS).

Macros produced (after grid completion)
---------------------------------------
    Row_N_feed_delta              – best feed delta for furnace N
    Grid_Row_N_conversion_delta   – best conversion delta for furnace N
    sum_del_Feed_flow             – Σ Row_N_feed_delta of winning combo
    sum_del_ethylene              – sum delta ethylene of winning combo
    Max_Benefit                   – best benefit seen across all combos
    Max_Benefit_SPC               – SPC for winning combo
    Min_target_sum_feed_bias      – updated for next iteration
    Conversion_Grid_Success       – 0/1
    Feed_Grid_Character           – signature string of last combo
    ranking_cause_indicator       – 1 on success, -5/-6/-1 on failure

STORE entries
-------------
    "feed_grid_character_log"     – pd.DataFrame of signatures seen
    "Feed_Conversion_merged_log"  – pd.DataFrame; one row per combo evaluated
                                    columns: Row_1..9_feed_delta,
                                             Grid_Row_1..9_conversion_delta,
                                             sum_del_ethylene,
                                             sum_Change_in_Recycle_Ethane_Feed,
                                             … (other diagnostic cols)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MACROS, STORE, NUM_FURNACES, NUM_PASSES

logger = logging.getLogger(__name__)

# Optional import — Chunk B may not exist yet on first generation.
try:
    import module_07b_grid_conversion as gridB
    _CONV_AVAILABLE = True
except ImportError:
    gridB = None
    _CONV_AVAILABLE = False
    logger.warning("module_07b_grid_conversion not found – conversion grid "
                   "will be stubbed (sum_del_ethylene=0, Conversion_Grid_Success=0).")
logger.info("conversion availability: %s", _CONV_AVAILABLE)
print(f"[module_07a import] _CONV_AVAILABLE = {_CONV_AVAILABLE}")

# Hard cap on combinatorial enumeration. 1M is plenty given that in practice
# most furnaces have step_size_feed == 0 and contribute exactly one value.
MAX_FEED_COMBOS = 1_000_000


# ---------------------------------------------------------------------------
# Macro helpers (small wrappers to keep the rest of the file terse)
# ---------------------------------------------------------------------------
def _m(key: str, default: float = 0.0) -> float:
    """Resolve a macro to float (RapidMiner's parse(%{key}))."""
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _mi(key: str, default: int = 0) -> int:
    """Resolve a macro to int."""
    try:
        return int(round(_m(key, default)))
    except Exception:
        return int(default)


def _ms(key: str, default: str = "") -> str:
    return str(MACROS.get(key, default))


def _set(key: str, val) -> None:
    MACROS[key] = val


def _recall(name: str) -> pd.DataFrame:
    return STORE.get(name, pd.DataFrame())


def _remember(name: str, df: pd.DataFrame) -> None:
    STORE[name] = df.copy() if isinstance(df, pd.DataFrame) else df


# ===========================================================================
# Generate ID (25) + Set Role (21)
# ===========================================================================
def _add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 1..N id column. RapidMiner's `Generate ID (25)` then `Set Role (21)`
    flips id to a regular attribute so downstream filters can match on it.
    """
    df = df.copy()
    df["id"] = np.arange(1, len(df) + 1, dtype=int)
    return df


# ===========================================================================
# Generate Macro (17) – grid initialisation
# ===========================================================================
def _init_grid_macros() -> None:
    """
    Min_target_sum_feed_bias = fresh_feed_quantity * (1 + shc_ratio)
    Max_Benefit              = -1000 if fresh_feed_change == -1 else 0
    Max_Benefit_SPC          = 1000
    ranking_cause_indicator  = 1
    """
    fresh_feed_quantity = _m("fresh_feed_quantity", 0.0)
    shc_ratio           = _m("shc_ratio", 0.0)
    fresh_feed_change   = _mi("fresh_feed_change", 0)

    _set("Min_target_sum_feed_bias", fresh_feed_quantity * (1.0 + shc_ratio))
    _set("Max_Benefit",     -1000.0 if fresh_feed_change == -1 else 0.0)
    _set("Max_Benefit_SPC", 1000.0)
    _set("ranking_cause_indicator", 1)

    # Initialise best-so-far trackers (mirror feed_delta_* / conversion_delta_best_*).
    # These are the macros Subprocess (132) writes; we initialise to 0 so that
    # if no combo ever wins, post-grid sees zeros.
    for i in range(1, NUM_FURNACES + 1):
        _set(f"feed_delta_{i}", 0.0)
        _set(f"conversion_delta_best_{i}", 0.0)
        _set(f"Row_{i}_feed_delta", 0.0)
        _set(f"Grid_Row_{i}_conversion_delta", 0.0)

    # sum_del_ethylene starts at the "negative infinity" sentinel (Set Macro 43
    # uses -10000 inside the catch path; we just track a running max benefit).
    _set("sum_del_ethylene", 0.0)
    _set("sum_del_Feed_flow", 0.0)
    _set("Conversion_Grid_Success", 0)
    _set("curr_sum_SPC_Furnace", 1000.0)


# ===========================================================================
# FEED GRID (2) — build the per-furnace value list
# ===========================================================================
def _linear_range(lower: float, upper: float, step: float) -> List[float]:
    """
    Replicates RapidMiner's `[lower;upper;step;linear]` grid syntax.
    Returns an evenly-spaced list of floats. A step of 0 collapses the
    "range" to the lower endpoint only (single value).
    """
    if step <= 0 or upper < lower:
        return [round(float(lower), 6)]

    n_steps = int(np.floor((upper - lower) / step + 1e-9)) + 1
    vals = [round(lower + k * step, 6) for k in range(n_steps)]
    # Ensure upper is included even with float drift
    if vals[-1] < upper - 1e-9:
        vals.append(round(upper, 6))
    return vals


def _build_feed_ranges(n: int) -> List[List[float]]:
    """
    Build the per-furnace feed-delta value list, one list per Row_1..n.

    Row_N_lower_limit_feed / upper / step come from the pre-grid module.
    """
    ranges = []
    for i in range(1, n + 1):
        lo   = _m(f"Row_{i}_lower_limit_feed", 0.0)
        hi   = _m(f"Row_{i}_upper_limit_feed", 0.0)
        step = _m(f"Row_{i}_step_size_feed", 0.0)
        ranges.append(_linear_range(lo, hi, step))
    return ranges


# ---------------------------------------------------------------------------
# Loop (49)  — Generate Macro (158) + (159) per-iteration override
# ---------------------------------------------------------------------------
def _apply_loop49_override(feed_deltas: List[float]) -> List[float]:
    """
    For each furnace i:
        Generate Macro (158):
            Row_i_feed_delta = if(Row_i_step_size_feed == 0, 0, Row_i_feed_delta)
        Generate Macro (159):
            Row_i_feed_delta = if(Row_i_step_size_feed != 0, Row_i_feed_delta,
                                  Row_i_lower_limit_feed)

    Net effect: when step_size == 0, the value is forced to lower_limit_feed.
    When step_size != 0, the grid value is kept as-is.

    (The two Generate Macros look contradictory in isolation but compose:
     158 zeroes it, 159 then replaces 0 with lower_limit when step==0.)
    """
    result = []
    for i, v in enumerate(feed_deltas, start=1):
        step = _m(f"Row_{i}_step_size_feed", 0.0)
        lo   = _m(f"Row_{i}_lower_limit_feed", 0.0)
        if step == 0:
            result.append(lo)
        else:
            result.append(float(v))
    return result


# ---------------------------------------------------------------------------
# Feed_Grid_Character signature & deduplication log
# ---------------------------------------------------------------------------
def _signature(feed_deltas: List[float]) -> str:
    """
    Mirror of Generate Macro (28):
        concat(str(d1),"#",str(d2),"#",…,str(d9),"#")

    Pads with 0.0 when fewer than 9 furnaces.
    """
    full = list(feed_deltas) + [0.0] * (NUM_FURNACES - len(feed_deltas))
    return "#".join(str(round(v, 6)) for v in full) + "#"


def _is_duplicate_signature(sig: str) -> bool:
    """
    Mirror of `handle dupe` Branch (81): look up sig in the running log;
    if already present, this combo is a duplicate and should be skipped.
    """
    log = STORE.get("feed_grid_character_log")
    if log is None or len(log) == 0:
        return False
    return bool((log["Feed_Grid_Character"] == sig).any())


def _record_signature(sig: str) -> None:
    log = STORE.get("feed_grid_character_log")
    new_row = pd.DataFrame([{"Feed_Grid_Character": sig}])
    STORE["feed_grid_character_log"] = (
        new_row if log is None or len(log) == 0
        else pd.concat([log, new_row], ignore_index=True)
    )


def _compute_compare_log(sum_del_feed: float, is_duplicate: bool) -> int:
    """
    Combined logic of:
        - Generate Macro (29)  (the dupe-check expression):
              if(fresh_feed_change == -1, 0,
                 if(mixed_feed_margin == sum_del_Feed_flow, 0, 1))
        - Branch (81) inner Filter+Extract that sets it to # of duplicates
        - Generate Macro (20) (the post-log expression):
              if(fresh_feed_change == -1,
                 if(compare != 0, 1,
                    if(sum_del_Feed_flow >= Min_target_sum_feed_bias, 0, 1)),
                 compare)

    Returns the final compare_log_curr_feed_delta value (0 = run this combo,
    non-zero = skip).
    """
    fresh_feed_change = _mi("fresh_feed_change", 0)
    mixed_feed_margin = _m("mixed_feed_margin", 0.0)
    min_target        = _m("Min_target_sum_feed_bias", 0.0)

    # ── Generate Macro (29) ───────────────────────────────────────────────
    if fresh_feed_change == -1:
        compare = 0
    else:
        # Use small tolerance for float equality
        compare = 0 if abs(mixed_feed_margin - sum_del_feed) < 1e-6 else 1

    # ── Branch (81) duplicate count override ───────────────────────────────
    # If this signature was already seen, Filter Examples will yield rows
    # (count >= 1), so compare gets set to that count (non-zero → skip).
    if is_duplicate:
        compare = 1  # any non-zero means "skip"

    # ── Generate Macro (20) – fresh_feed_change == -1 special rule ────────
    if fresh_feed_change == -1:
        if compare != 0:
            return 1
        return 0 if sum_del_feed >= min_target else 1

    return compare


# ---------------------------------------------------------------------------
# Loop (144) + Append (27) + exclude(id) — build the per-row table that
# feeds into MAIN CONVERSION GRID.
# ---------------------------------------------------------------------------
def _build_per_row_table(df: pd.DataFrame, feed_deltas: List[float]) -> pd.DataFrame:
    """
    For each row i (id=i):
      del_Feed_flow  = Row_i_feed_delta
      New_Feed_flow  = Feed_flow + del_Feed_flow

    Then drop the id column (exclude (2)).
    Note: only the first `len(feed_deltas)` rows are touched; if df has
    extra rows they are kept as-is with del_Feed_flow = 0.
    """
    df = df.copy()
    df["del_Feed_flow"] = 0.0
    n = min(len(df), len(feed_deltas))
    for k in range(n):
        df.iat[k, df.columns.get_loc("del_Feed_flow")] = float(feed_deltas[k])

    if "Feed_flow" in df.columns:
        df["New_Feed_flow"] = df["Feed_flow"].astype(float) + df["del_Feed_flow"]
    else:
        # Fall back – very rare; pre-grid should always provide Feed_flow
        df["New_Feed_flow"] = df["del_Feed_flow"]
        logger.warning("Feed_flow column missing in df; New_Feed_flow = del_Feed_flow only.")

    df.drop(columns=["id"], errors="ignore", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Update Row_N_feed_delta macros (so inner subprocesses see the current combo)
# ---------------------------------------------------------------------------
def _set_row_feed_delta_macros(feed_deltas: List[float]) -> None:
    for i, v in enumerate(feed_deltas, start=1):
        _set(f"Row_{i}_feed_delta", float(v))
    # Pad if fewer than NUM_FURNACES
    for i in range(len(feed_deltas) + 1, NUM_FURNACES + 1):
        _set(f"Row_{i}_feed_delta", 0.0)
    _set("sum_del_Feed_flow", float(sum(feed_deltas)))


# ---------------------------------------------------------------------------
# Branch (113) — commit best-so-far trackers when this combo wins
# ---------------------------------------------------------------------------
def _commit_best_if_winner() -> None:
    """
    Mirror of Branch (113): expression  sum_del_ethylene == Max_Benefit
    When true, Subprocess (132) writes feed_delta_1..9 and
    conversion_delta_best_1..9 with the *current* Row_N_feed_delta and
    Grid_Row_N_conversion_delta values, and Set Macros (3) updates
    Min_target_sum_feed_bias and Max_Benefit_SPC.
    """
    sum_eth     = _m("sum_del_ethylene", -1e9)
    max_benefit = _m("Max_Benefit", -1e9)

    # Strict equality is correct here — Generate Macro (44) just *set*
    # Max_Benefit = sum_del_ethylene if all flags passed. If that fired
    # this iteration, the two will be equal exactly.
    if abs(sum_eth - max_benefit) > 1e-9:
        return

    # ── Subprocess (132) — commit current combo as new best ───────────────
    for i in range(1, NUM_FURNACES + 1):
        _set(f"feed_delta_{i}",            _m(f"Row_{i}_feed_delta", 0.0))
        _set(f"conversion_delta_best_{i}", _m(f"Grid_Row_{i}_conversion_delta", 0.0))

    # ── Set Macros (3) ────────────────────────────────────────────────────
    _set("Min_target_sum_feed_bias", _m("sum_del_Feed_flow", 0.0))
    _set("Max_Benefit_SPC",          _m("curr_sum_SPC_Furnace", 1000.0))

    logger.debug(
        "  ▶ NEW BEST: sum_del_eth=%.4f  sum_del_feed=%.2f  "
        "feed_deltas=%s",
        sum_eth, _m("sum_del_Feed_flow"),
        [round(_m(f"feed_delta_{i}"), 2) for i in range(1, NUM_FURNACES + 1)],
    )


# ---------------------------------------------------------------------------
# Log "Feed_Conversion_merged_log" — append one row per evaluated combo
# ---------------------------------------------------------------------------
_MERGED_LOG_COLS = (
    [f"Row_{i}_feed_delta"            for i in range(1, NUM_FURNACES + 1)]
    + ["sum_del_ethylene", "sum_Change_in_Recycle_Ethane_Feed"]
    + [f"Grid_Row_{i}_conversion_delta" for i in range(1, NUM_FURNACES + 1)]
)


def _append_merged_log() -> None:
    """
    Append a row to STORE["Feed_Conversion_merged_log"] capturing the
    current state of all per-row deltas, sum_del_ethylene and
    sum_Change_in_Recycle_Ethane_Feed. Post-grid reads this log.
    """
    row = {col: float(MACROS.get(col, 0.0)) for col in _MERGED_LOG_COLS}
    log = STORE.get("Feed_Conversion_merged_log")
    new_row = pd.DataFrame([row])
    STORE["Feed_Conversion_merged_log"] = (
        new_row if log is None or len(log) == 0
        else pd.concat([log, new_row], ignore_index=True)
    )


# ===========================================================================
# Conversion-grid hook (calls Chunk B; placeholder if not loaded yet)
# ===========================================================================
def _run_conversion_grid(df_per_row: pd.DataFrame) -> pd.DataFrame:
    """
    Hand the per-row dataframe to MAIN CONVERSION GRID (Chunk B). That
    subprocess is responsible for setting:
      sum_del_ethylene, sum_Change_in_Recycle_Ethane_Feed,
      Grid_Row_N_conversion_delta, Conversion_Grid_Success,
      Max_Benefit (via Generate Macro 44),
      flag_benefit, flag_energy_consumption, flag_specific_energy_consumption,
      curr_sum_SPC_Furnace, sum_SPC_Furnace, sum_flag_nox.

    On failure or absence of Chunk B, we set safe defaults that effectively
    skip the combo (sum_del_ethylene stays at its sentinel).
    """
    if _CONV_AVAILABLE and gridB is not None:
        try:
            return gridB.run(df_per_row)
        except Exception as exc:
            logger.error("Conversion grid raised %s – treating combo as failed.",
                         exc, exc_info=False)

    # Stub / failure path
    _set("sum_del_ethylene", -1e4)
    _set("sum_Change_in_Recycle_Ethane_Feed", 0.0)
    _set("Conversion_Grid_Success", 0)
    for i in range(1, NUM_FURNACES + 1):
        _set(f"Grid_Row_{i}_conversion_delta", 0.0)
    return df_per_row


# ===========================================================================
# The main grid search
# ===========================================================================
def _enumerate_feed_grid(df: pd.DataFrame, ranges: List[List[float]]) -> Dict:
    """
    Walk every combination of feed deltas, applying the full RapidMiner
    pipeline (override → signature → dedupe → conversion grid → log →
    best-tracker update).

    Returns a summary dict:
        {"combos_total", "combos_evaluated", "combos_skipped_dupe",
         "combos_skipped_target", "winners"}
    """
    n = len(ranges)
    total_combos = 1
    for r in ranges:
        total_combos *= max(1, len(r))

    logger.info(
        "FEED GRID: %d furnaces, %d total combos "
        "(per-furnace value counts: %s)",
        n, total_combos, [len(r) for r in ranges],
    )

    if total_combos > MAX_FEED_COMBOS:
        logger.warning(
            "FEED GRID: %d combos exceeds cap %d – will iterate but warn.",
            total_combos, MAX_FEED_COMBOS,
        )

    stats = dict(
        combos_total=total_combos,
        combos_evaluated=0,
        combos_skipped_dupe=0,
        combos_skipped_target=0,
        winners=0,
    )

    # Reset per-iteration logs
    STORE["feed_grid_character_log"]   = pd.DataFrame(columns=["Feed_Grid_Character"])
    STORE["Feed_Conversion_merged_log"] = pd.DataFrame(columns=_MERGED_LOG_COLS)

    # Iterate Cartesian product
    indices = [0] * n
    sizes   = [len(r) for r in ranges]
    if any(s == 0 for s in sizes):
        logger.warning("FEED GRID: at least one range is empty – nothing to do.")
        return stats

    while True:
        # Build current combo
        combo = [ranges[i][indices[i]] for i in range(n)]

        # Loop (49) override (force step==0 values to lower_limit)
        combo = _apply_loop49_override(combo)

        # Update macros (other subprocesses read Row_N_feed_delta)
        _set_row_feed_delta_macros(combo)

        sum_del_feed = float(sum(combo))
        sig          = _signature(combo)

        # handle dupe / Branch (81) / Generate Macro (20)
        is_dupe = _is_duplicate_signature(sig)
        compare = _compute_compare_log(sum_del_feed, is_dupe)

        # Log the signature (Log Feed_Grid_Character is unconditional)
        _record_signature(sig)
        _set("Feed_Grid_Character", sig)

        if compare != 0:
            if is_dupe:
                stats["combos_skipped_dupe"] += 1
            else:
                stats["combos_skipped_target"] += 1
        else:
            # Build per-row table and run the conversion grid
            df_per_row = _build_per_row_table(df, combo)

            # Snapshot Max_Benefit before the conversion grid mutates it
            max_before = _m("Max_Benefit", -1e9)

            _run_conversion_grid(df_per_row)

            max_after = _m("Max_Benefit", -1e9)
            stats["combos_evaluated"] += 1

            # Append to merged log regardless (post-grid will pick the winner)
            _append_merged_log()

            # Branch (113): if conversion grid bumped Max_Benefit, this combo
            # is the new best — commit feed_delta_* / conversion_delta_best_*
            if max_after > max_before + 1e-9:
                _commit_best_if_winner()
                stats["winners"] += 1

        # Advance the odometer
        k = n - 1
        while k >= 0:
            indices[k] += 1
            if indices[k] < sizes[k]:
                break
            indices[k] = 0
            k -= 1
        if k < 0:
            break  # exhausted all combos

    logger.info(
        "FEED GRID done: evaluated=%d, skipped_dupe=%d, skipped_target=%d, winners=%d",
        stats["combos_evaluated"],
        stats["combos_skipped_dupe"],
        stats["combos_skipped_target"],
        stats["winners"],
    )
    return stats


# ===========================================================================
# Handle Exception (12) — fallback if anything inside blows up
# ===========================================================================
def _on_grid_failure(exc: Exception) -> None:
    """
    Mirror of Generate Macro (45) in the catch arm:
        ranking_cause_indicator = if(==5, -5, if(==6, -6, -1))
    """
    ric = _mi("ranking_cause_indicator", 1)
    if ric == 5:
        _set("ranking_cause_indicator", -5)
    elif ric == 6:
        _set("ranking_cause_indicator", -6)
    else:
        _set("ranking_cause_indicator", -1)
    logger.error("Grid-Main FAILED (%s) – ranking_cause_indicator = %s",
                 exc, _m("ranking_cause_indicator"))


# ===========================================================================
# Public entry point
# ===========================================================================
def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Grid-Main subprocess (Chunk A drives, Chunk B is called for
    each surviving feed combo).

    Parameters
    ----------
    df : pd.DataFrame
        Pre-grid furnace data (one row per furnace, Number_of_rows valid).

    Returns
    -------
    df : pd.DataFrame
        Same df, unchanged. Outputs go to MACROS and STORE.
    """
    logger.info("=== MODULE 07 – GRID MAIN (Chunk A: Feed Grid) ===")

    # ── Generate ID (25) + Set Role (21) ─────────────────────────────────
    df_with_id = _add_row_id(df)

    # ── Generate Macro (17) ──────────────────────────────────────────────
    _init_grid_macros()

    # ── Handle Exception (12) wraps everything ───────────────────────────
    try:
        n = _mi("Number_of_rows", 0)
        if n <= 0:
            logger.warning("Number_of_rows=%d → nothing to optimise.", n)
            return df

        # FEED GRID (2) parameter ranges
        ranges = _build_feed_ranges(n)

        # Enumerate the grid (calls conversion grid for each non-skipped combo)
        _enumerate_feed_grid(df_with_id, ranges)

        # ── After the loop: Row_N_feed_delta currently hold the *last*
        # combo, not the winning one. Restore the best.
        for i in range(1, NUM_FURNACES + 1):
            _set(f"Row_{i}_feed_delta", _m(f"feed_delta_{i}", 0.0))
            _set(f"Grid_Row_{i}_conversion_delta",
                 _m(f"conversion_delta_best_{i}", 0.0))

        _set("sum_del_Feed_flow",
             float(sum(_m(f"feed_delta_{i}", 0.0) for i in range(1, NUM_FURNACES + 1))))

        # Feed_Grid_Character of the WINNING combo (informational)
        winning = [_m(f"feed_delta_{i}", 0.0) for i in range(1, NUM_FURNACES + 1)]
        _set("Feed_Grid_Character", _signature(winning))

    except Exception as exc:
        _on_grid_failure(exc)

    # ── Final log line for operator visibility ───────────────────────────
    logger.info(
        "GRID MAIN summary: Max_Benefit=%.4f  sum_del_Feed_flow=%+.2f  "
        "ranking_cause_indicator=%s",
        _m("Max_Benefit"),
        _m("sum_del_Feed_flow"),
        MACROS.get("ranking_cause_indicator"),
    )
    return df
