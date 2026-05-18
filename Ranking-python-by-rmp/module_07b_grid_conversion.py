"""
module_07b_grid_conversion.py
=============================
Chunk B of the Grid-Main subprocess: the **CONVERSION-GRID layer**.

Called once per surviving feed combo by `module_07a_grid_feed.py`. The input
DataFrame already carries:
    Feed_flow, New_Feed_flow, del_Feed_flow, Current_Recycle_Ethane_Feed,
    Overall_conversion, shc_ratio, percent_above_threshold, Furnace_condition,
    conversion_lower_limit_in_grid, conversion_upper_limit_in_grid,
    step_size_conversion (all from pre-grid + Loop (144) in Chunk A),
    Heat (current furnace heat – used for energy-consumption flags)

RapidMiner blocks covered (.rmp lines 5798 – 8390):
---------------------------------------------------
    MAIN CONVERSION GRID subprocess
      └─ Sort (25) by overall_ranking ascending
      └─ Subprocess (16)
          ├─ min fur check (2)              [>= 2 examples gate]
          ├─ Branch (20) fresh_feed_change != 0 path:
          │    ├─ Generate Macro (163): Extra_Recycle_Ethane,
          │    │                        upper/lower_limit_change_in_recycle_ethane,
          │    │                        Conversion_Limits_given = 0
          │    ├─ Subprocess (102) "only upper extra recycle check"
          │    │   ├─ Generate "3": New_Overall_conversion, New_Recycle_Ethane_Feed,
          │    │   │   New_Extra_Recycle_Ethane (max-band probe)
          │    │   ├─ Filter conversion_upper_limit_in_grid > 0 → Aggregate sum
          │    │   ├─ conversion_max_check_enough_for_upper_lower + _enough
          │    │   ├─ Branch (228) IF conversion_max_check_enough==1:
          │    │   │   THEN: Conversion_Limits_given=1; per-row
          │    │   │     New_Overall_conversion_limit; Branch (22) channel swap;
          │    │   │     Subprocess (17) Loop-While dummy-min/max convergence
          │    │   │     of New_Overall_conversion towards limit
          │    │   │   ELSE: leave bounds alone
          │    │   ├─ Filter Considered_for_conversion_expansion==0 →
          │    │   │   Generate Attributes (342) update conversion_*_in_grid
          │    │   ├─ Extract count_no_fixed_grid_fur
          │    │   └─ Branch (25): step_size_conversion assignment
          │    │       (Generate Attributes 343/339)
          │    ├─ Select Attributes (110), Join (31)
          │    └─ — fall through to inferred_tags_3 loop —
          │
          ├─ Branch "conversion according to bias (2)": biasing_condition == 3
          │    └─ count_upper_limit_available_fur ; Conversion_Limits_given
          │
          ├─ Subprocess (20)/(50): filter+sort+append by
          │    percent_above_threshold/Furnace_condition
          │    + Loop-While topN bound assignment
          │    + Branch (27) ROPT_all_furnace_for_conversion_biasing
          │    + Generate Attributes (132)/(48)/(49)/(134)
          │
          ├─ recall inferred_tags_3 → eval per-row
          ├─ Rename (2) New_Overall_conversion → New_Overall_Conversion_for_Single_Furnace
          ├─ Branch (29): grid_condition
          ├─ Extract Macro (11), Generate Macro (21) → Conversion_Grid_Success,
          │    Conversion_Limits_given
          ├─ exclude (3), Select Attributes (45), Join (46), Sort (28)
          │
          └─ Branch "GRID" Conversion_Limits_given == 1:
              └─ Handle Exception (13)
                  ├─ Loop (97) x9: init Grid_Row_N_*_conversion_limit / step / part_override
                  ├─ GRID -Conversion subprocess
                  │    ├─ factor (2): size_part, flag_conversion_part_override
                  │    ├─ exclude (6) drop New_Overall_conversion
                  │    ├─ Extract No_of_rows
                  │    ├─ Loop (95) over rows: Extract Macro (437) extracts
                  │    │    Grid_Row_i_* (Feed_flow, Furnace, SPC, Conversion,
                  │    │    Furnace_condition, Ethylene_Production,
                  │    │    lower/upper_conversion_limit, step_size_conversion,
                  │    │    part_override)
                  │    └─ GRID (2) subprocess
                  │        ├─ recall inferred_tags_2
                  │        └─ Conversion inside Feed grid (2)
                  │            ←── inner exhaustive grid over per-furnace
                  │                conversion deltas
                  │            Each combo:
                  │              - Loop (96): rounding & part_override via floor()
                  │              - Loop (145): build per-row table with
                  │                  New_Overall_conversion, etc.
                  │              - Inf tags calculations (2): Loop (81)
                  │                  evaluate inferred_tags_2 formulas
                  │              - Aggregate (84): sums of del_ethylene,
                  │                  Change_in_Recycle_Ethane_Feed, Heat,
                  │                  Heat_new, Temp_Ethylene_Production,
                  │                  New_Ethylene_Production
                  │              - flag_energy_consumption, flag_specific…
                  │              - Generate Macro (46): compare_log_curr_conver_delta
                  │                gate (recycle bounds + Max_Benefit +
                  │                energy flags)
                  │              - Branch (96): if not skipped → set 1st..9th_conver,
                  │                Log Conv_GRID_LOG(feed)-main, Performance (3) RMSE
                  │
                  ├─ Subprocess (38): pull conversion log,
                  │    sort, transpose, rename → conversion_bias per row,
                  │    Merge Attributes (104) feed_delta_best columns
                  ├─ Generate Attributes (888): final columns
                  ├─ Set Role (230), Set Macro (42) sum_del_ethylene
                  └─ Branch (98): IF Conversion_Grid_Success == 1
                       └─ Subprocess (44)/(131):
                            - Recall (41) inferred_tags_2 → Loop (66) re-eval
                            - Generate Attributes (52): SPC_*, flag_nox
                            - Aggregate (85): final sums
                            - Extract sum_*_final
                            - Generate Macro (23): flag_SPC, benefit_percent_*
                            - Generate Attributes (53): flag_energy_consumption
                              (with sum_flag_nox rule), flag_specific…, flag_benefit
                            - Generate Macro (44) → Max_Benefit update

The single most important rule (Generate Macro 44):

    Max_Benefit ←
        sum_del_ethylene
        IF  lower_limit_change_in_recycle_ethane ≤ sum_Change_in_Recycle_Ethane_Feed
                                                  ≤ upper_limit_change_in_recycle_ethane
        AND flag_benefit == 1
        AND flag_energy_consumption == 1
        AND flag_specific_energy_consumption == 1
        ELSE Max_Benefit (unchanged)

This is what tells Chunk A "this combo is the new best" so it can commit
the corresponding feed_delta_* / conversion_delta_best_* macros.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MACROS, STORE

logger = logging.getLogger(__name__)

MAX_FURNACES = 9

# Cap on per-feed-combo conversion-grid combinations. The inner grid runs
# once per surviving feed combo so it needs to stay fast.
MAX_CONV_COMBOS = 1_000_000


# ===========================================================================
# Macro / store helpers (same shape as Chunk A — kept module-local for clarity)
# ===========================================================================
def _m(key: str, default: float = 0.0) -> float:
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _mi(key: str, default: int = 0) -> int:
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


# ===========================================================================
# Inferred-tag evaluation
# Mirrors Loop (3) / Loop (81) / Loop (66) — all do the same thing:
#   for each (tag_name, formula) in inferred_tags_N:
#       df[tag_name] = df.eval(formula)
# ===========================================================================
def _evaluate_inferred_tags(df: pd.DataFrame, tags_store_name: str) -> pd.DataFrame:
    """Apply formulas from STORE[tags_store_name] to df in place (returns df)."""
    df_tags = _recall(tags_store_name)
    if df_tags is None or df_tags.empty:
        logger.info("inferred tag data not found")
        return df

    if "Inferred_tag" not in df_tags.columns or "Inferred_tag_formula" not in df_tags.columns:
        logger.debug("inferred-tag store '%s' missing required columns – skipping eval.",
                     tags_store_name)
        return df

    df = df.copy()
    for _, row in df_tags.iterrows():
        tag = str(row["Inferred_tag"])
        formula = str(row["Inferred_tag_formula"])
        if not tag or not formula or tag.lower() == "nan":
            continue
        try:
            df[tag] = df.eval(formula)
        except Exception as exc:
            logger.warning("inferred-tag '%s' eval failed (%s) – column left unchanged.",
                         tag, exc)
    return df


# ===========================================================================
# Subprocess (102) — recycle-ethane probe & bound expansion
# (mirrors Generate "3" + Filter + Aggregate + Generate Macro (150))
# ===========================================================================
def _probe_recycle_bounds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Probe whether moving every furnace to its max-allowed conversion is
    enough to reach Extra_Recycle_Ethane. Returns the three macro values
    set by Generate Macro (150):
        sum_New_Extra_Recycle_Ethane_max_check_upper
        sum_New_Extra_Recycle_Ethane_max_check_lower
        sum_New_Extra_Recycle_Ethane_max_check
    plus the two `enough` flags.
    """
    fresh_feed_change = _mi("fresh_feed_change", 0)
    conv_upper_max    = _m("conversion_upper_limit_expansion_max_limit", 0.0)
    conv_lower_max    = _m("conversion_lower_limit_expansion_max_limit", 0.0)
    max_single_fur    = _m("max_conversion_single_furnace_limit", 0.0)
    extra_recycle     = _m("Extra_Recycle_Ethane", 0.0)

    if df.empty:
        return dict(sum_New_Extra_Recycle_Ethane_max_check_upper=0.0,
                    sum_New_Extra_Recycle_Ethane_max_check_lower=0.0,
                    sum_New_Extra_Recycle_Ethane_max_check=0.0,
                    conversion_max_check_enough_for_upper_lower=0,
                    conversion_max_check_enough=0)

    work = df.copy()
    OC   = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else pd.Series(0.0, index=work.index)
    NFF  = work["New_Feed_flow"].astype(float)      if "New_Feed_flow"      in work.columns else pd.Series(0.0, index=work.index)
    SHC  = work["shc_ratio"].astype(float)          if "shc_ratio"          in work.columns else pd.Series(0.0, index=work.index)
    CRE  = work["Current_Recycle_Ethane_Feed"].astype(float) if "Current_Recycle_Ethane_Feed" in work.columns else pd.Series(0.0, index=work.index)
    CULG = work["conversion_upper_limit_in_grid"].astype(float) if "conversion_upper_limit_in_grid" in work.columns else pd.Series(0.0, index=work.index)

    frac = OC - np.floor(OC)
    if fresh_feed_change > 0:
        # Push conversion up
        noc = np.minimum(conv_upper_max + frac - 1.0, OC + max_single_fur)
    else:
        # Push conversion down
        noc = np.maximum(conv_lower_max + frac, OC - max_single_fur)

    new_recycle = (NFF / (1.0 + SHC)) * (100.0 - noc) / 100.0
    new_extra   = new_recycle - CRE

    # max_check uses *all* rows; max_check_upper/_lower filter on
    # conversion_upper_limit_in_grid > 0 (the rows with conversion headroom).
    mask_pos = CULG > 0

    sum_max_check       = float(new_extra.sum())
    sum_max_check_upper = float(new_extra[mask_pos].sum())
    sum_max_check_lower = float(new_extra[mask_pos].sum())  # same filter; RM duplicates it

    if fresh_feed_change > 0:
        # Need new_extra <= Extra_Recycle_Ethane to reach the target
        enough_ul = 1 if sum_max_check_upper <= extra_recycle else 0
        enough    = 1 if sum_max_check       <= extra_recycle else 0
    else:
        enough_ul = 1 if sum_max_check_lower >= extra_recycle else 0
        enough    = 1 if sum_max_check       >= extra_recycle else 0

    return dict(
        sum_New_Extra_Recycle_Ethane_max_check_upper=sum_max_check_upper,
        sum_New_Extra_Recycle_Ethane_max_check_lower=sum_max_check_lower,
        sum_New_Extra_Recycle_Ethane_max_check=sum_max_check,
        conversion_max_check_enough_for_upper_lower=enough_ul,
        conversion_max_check_enough=enough,
    )


# ---------------------------------------------------------------------------
# Subprocess (17) — Loop (While) dummy-min/max convergence
# ---------------------------------------------------------------------------
def _converge_conversion_band(df: pd.DataFrame,
                              considered_flag: int) -> pd.DataFrame:
    """
    Iteratively nudge `New_Overall_conversion` towards `New_Overall_conversion_limit`
    by ±1 (for the row(s) at the current min/max) until the cumulative
    `New_Extra_Recycle_Ethane` reaches `Extra_Recycle_Ethane`.

    Termination: condition_satisfy == 1, or no row left with
    delta_conversion_limit != 0.

    `considered_flag` carries Generate Attributes (68)/(69):
        IF row was "considered for conversion expansion" (==1):
            New_Overall_conversion stays where probe placed it (= limit-based)
        ELSE (==0):
            New_Overall_conversion becomes New_Overall_conversion_limit
            (i.e. force to the limit so it can drift back)
    """
    fresh_feed_change = _mi("fresh_feed_change", 0)
    extra_recycle     = _m("Extra_Recycle_Ethane", 0.0)

    if df.empty or "New_Overall_conversion_limit" not in df.columns:
        return df

    work = df.copy()
    if "Considered_for_conversion_expansion" not in work.columns:
        work["Considered_for_conversion_expansion"] = considered_flag

    # Seed New_Overall_conversion from the recycle probe (Subprocess 102 →
    # Generate "3" computed it from min/max formulas). If the caller hasn't
    # populated it, start from Overall_conversion.
    if "New_Overall_conversion" not in work.columns:
        if "Overall_conversion" in work.columns:
            work["New_Overall_conversion"] = work["Overall_conversion"].astype(float)
        else:
            work["New_Overall_conversion"] = 0.0

    # Generate Attributes (68): when considered==0 → New_Overall_conversion := limit
    mask = work["Considered_for_conversion_expansion"] == 0
    if mask.any():
        work.loc[mask, "New_Overall_conversion"] = work.loc[mask, "New_Overall_conversion_limit"]

    # Loop-While body
    MAX_ITERS = 50
    for _ in range(MAX_ITERS):
        # delta_conversion_limit per row
        delta = (work["New_Overall_conversion_limit"] - work["New_Overall_conversion"]).abs()
        # Filter: rows where delta != 0  OR  Considered_for_conversion_expansion == 0
        active_mask = (delta != 0) | (work["Considered_for_conversion_expansion"] == 0)
        active = work[active_mask]
        if active.empty:
            break

        # Pick min or max based on direction
        if fresh_feed_change == 1:
            target = float(active["New_Overall_conversion"].min())
            mask_t = active["New_Overall_conversion"] == target
            # Generate Attributes (71) for fresh_feed_change == 1: bump min row(s) up by 1
            new_val = np.minimum(
                active.loc[mask_t, "New_Overall_conversion_limit"],
                active.loc[mask_t, "New_Overall_conversion"] + 1.0,
            )
        else:
            target = float(active["New_Overall_conversion"].max())
            mask_t = active["New_Overall_conversion"] == target
            new_val = np.maximum(
                active.loc[mask_t, "New_Overall_conversion_limit"],
                active.loc[mask_t, "New_Overall_conversion"] - 1.0,
            )

        # Apply update back to `work`
        idx_to_update = active.loc[mask_t].index
        work.loc[idx_to_update, "New_Overall_conversion"] = new_val.values

        # Recompute Recycle quantities
        NFF = work["New_Feed_flow"].astype(float)
        SHC = work["shc_ratio"].astype(float) if "shc_ratio" in work.columns else 0.0
        work["New_Recycle_Ethane_Feed"] = (NFF / (1.0 + SHC)) * (100.0 - work["New_Overall_conversion"]) / 100.0
        work["New_Extra_Recycle_Ethane"] = work["New_Recycle_Ethane_Feed"] - work["Current_Recycle_Ethane_Feed"].astype(float)

        sum_curr = float(work["New_Extra_Recycle_Ethane"].sum())

        # condition_satisfy
        if fresh_feed_change == 1:
            if sum_curr <= extra_recycle:
                break
        else:
            if sum_curr >= extra_recycle:
                break

    # Generate Attributes (72) at the tail
    OC = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else 0.0
    work["Considered_for_conversion_expansion"] = ((work["New_Overall_conversion"] - OC).abs() > 0).astype(int)

    return work


# ---------------------------------------------------------------------------
# Generate Attributes (342) + count_no_fixed_grid_fur + Branch (25)
# (final conversion_lower/upper_limit_in_grid + step_size_conversion)
# ---------------------------------------------------------------------------
def _finalise_conversion_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows with Considered_for_conversion_expansion==1, the lower-limit
    becomes fresh_feed_change * floor(|ΔConv| / 0.5) * 0.5, and the upper
    becomes equal to the lower (zero-width slot — already chosen).

    Then count_no_fixed_grid_fur counts the rows that *weren't* expanded
    (==0) and Branch (25) assigns step_size_conversion accordingly.
    """
    fresh_feed_change = _mi("fresh_feed_change", 0)

    if df.empty:
        return df

    work = df.copy()

    needed = {"Considered_for_conversion_expansion", "New_Overall_conversion",
              "Overall_conversion", "conversion_lower_limit_in_grid",
              "conversion_upper_limit_in_grid"}
    if not needed.issubset(work.columns):
        # If pre-grid already provided these and Subprocess(17) wasn't called,
        # just return as-is.
        return work

    delta_conv = (work["New_Overall_conversion"] - work["Overall_conversion"]).abs()
    new_lower  = np.where(
        work["Considered_for_conversion_expansion"] == 0,
        work["conversion_lower_limit_in_grid"].astype(float),
        fresh_feed_change * np.floor(delta_conv / 0.5) * 0.5,
    )
    new_upper  = np.where(
        work["Considered_for_conversion_expansion"] == 0,
        work["conversion_upper_limit_in_grid"].astype(float),
        new_lower,
    )
    work["conversion_lower_limit_in_grid"] = new_lower
    work["conversion_upper_limit_in_grid"] = new_upper

    count_no_fixed = int((work["Considered_for_conversion_expansion"] == 0).sum())
    _set("count_no_fixed_grid_fur", count_no_fixed)

    # Branch (25)
    if count_no_fixed == 0:
        # THEN: Sort by percent_above_threshold desc + ID + step_size mapping
        if "percent_above_threshold" in work.columns:
            work = work.sort_values("percent_above_threshold", ascending=False).reset_index(drop=True)
        work["id"] = np.arange(1, len(work) + 1)
        id_max = int(work["id"].max())
        # Top-(id_max-2) rows get step=0, the last 2 get upper*2
        work["step_size_conversion"] = np.where(
            work["id"] <= id_max - 2,
            0.0,
            work["conversion_upper_limit_in_grid"].astype(float) * 2.0,
        )
    else:
        # ELSE: step_size_conversion table
        def _step_for_count(c: int, considered: int) -> float:
            if considered == 1:
                return 0.0
            if c < 6:
                return 5.0
            if c == 6:
                return 3.0
            if c == 9:
                return 1.0
            return 2.0

        work["step_size_conversion"] = [
            _step_for_count(count_no_fixed, int(c))
            for c in work["Considered_for_conversion_expansion"]
        ]

    return work


# ---------------------------------------------------------------------------
# Subprocess (20)/(50) tail — biasing_condition != 3 path (typical)
# Loop-While topN bound assignment + Branch (27) ROPT_all_furnace…
# ---------------------------------------------------------------------------
def _topN_bound_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Increase the count_of_top_rows until cumulative New_Extra_Recycle_Ethane
    exceeds Extra_Recycle_Ethane, then assign per-row conversion bands based
    on whether each row was in the "top N" (gets the lower-limit pushed) or
    not.

    Termination conditions are Conversion_Limits_given == 1 OR
    iter > Number_of_rows.
    """
    if df.empty:
        _set("Conversion_Limits_given", 0)
        return df

    work = df.copy()

    # Subprocess (50): filter rows with percent_above_threshold < 0 AND Furnace_condition == "Good"
    # then append "rest" sorted by overall_conversion_rank descending.
    if "percent_above_threshold" in work.columns and "Furnace_condition" in work.columns:
        good_mask = (work["percent_above_threshold"] < 0) & (work["Furnace_condition"].astype(str) == "Good")
        good = work[good_mask]
        rest = work[~good_mask]
        if "overall_conversion_rank" in rest.columns:
            rest = rest.sort_values("overall_conversion_rank", ascending=False)
        work = pd.concat([good, rest], ignore_index=True)

    work["id"] = np.arange(1, len(work) + 1)

    _set("Conversion_Limits_given", 0)
    extra_recycle = _m("Extra_Recycle_Ethane", 0.0)
    number_of_rows = _mi("Number_of_rows", 0)

    count_of_top_rows = 2
    iter_loop = 0
    MAX_ITER = max(number_of_rows + 5, 15)

    while True:
        iter_loop += 1
        if iter_loop > MAX_ITER:
            break

        # Generate Attributes (21)
        OC  = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else 0.0
        CLL = work["conversion_lower_limit_in_grid"].astype(float)
        NFF = work["New_Feed_flow"].astype(float) if "New_Feed_flow" in work.columns else 0.0
        SHC = work["shc_ratio"].astype(float) if "shc_ratio" in work.columns else 0.0
        CRE = work["Current_Recycle_Ethane_Feed"].astype(float) if "Current_Recycle_Ethane_Feed" in work.columns else 0.0

        new_conv = np.where(work["id"] <= count_of_top_rows, OC + CLL, OC)
        new_recycle = (NFF / (1.0 + SHC)) * (100.0 - new_conv) / 100.0
        new_extra   = new_recycle - CRE
        sum_new_extra = float(np.sum(new_extra))

        work["New_Overall_conversion"]       = new_conv
        work["New_Recycle_Ethane_Feed"]      = new_recycle
        work["New_Extra_Recycle_Ethane"]     = new_extra

        # checks
        count_of_top_rows += 1
        if sum_new_extra > extra_recycle:
            _set("Conversion_Limits_given", 1)
            break
        if iter_loop > number_of_rows:
            break

    # Branch (27) — ROPT_all_furnace_for_conversion_biasing
    biasing_active = _ms("ROPT_all_furnace_for_conversion_biasing", "active") != "inactive"

    if not biasing_active:
        # Generate Macro (19): count_of_top_rows -= 1
        count_of_top_rows -= 1

        # Generate Attributes (132): update for non-top rows
        conv_given = _mi("Conversion_Limits_given", 0)
        NFF = work["New_Feed_flow"].astype(float) if "New_Feed_flow" in work.columns else 0.0
        SHC = work["shc_ratio"].astype(float) if "shc_ratio" in work.columns else 0.0
        CRE = work["Current_Recycle_Ethane_Feed"].astype(float) if "Current_Recycle_Ethane_Feed" in work.columns else 0.0
        OC  = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else 0.0

        if conv_given == 0:
            work["New_Overall_conversion"] = OC

        work["New_Recycle_Ethane_Feed"]  = (NFF / (1.0 + SHC)) * (100.0 - work["New_Overall_conversion"]) / 100.0
        work["New_Extra_Recycle_Ethane"] = work["New_Recycle_Ethane_Feed"] - CRE
        work["conversion_lower_limit_in_grid"] = work["New_Overall_conversion"] - OC
        work["conversion_upper_limit_in_grid"] = 0.0

        # Generate Attributes (48): upper := lower for id < count_of_top_rows - 1
        work["conversion_upper_limit_in_grid"] = np.where(
            work["id"] < count_of_top_rows - 1,
            work["conversion_lower_limit_in_grid"],
            work["conversion_upper_limit_in_grid"],
        )
        # Reset Generate Macro (24): count_of_top_rows = 2 (not used downstream here)
    else:
        # Generate Macro (25): count_of_top_rows = min(count-1, Number_of_rows-2)
        count_of_top_rows = min(count_of_top_rows - 1, number_of_rows - 2)
        # Generate Attributes (49): for top-N rows: upper := lower
        work["conversion_upper_limit_in_grid"] = np.where(
            work["id"] <= count_of_top_rows,
            work["conversion_lower_limit_in_grid"],
            work["conversion_upper_limit_in_grid"],
        )

    # Generate Attributes (134): step_size_conversion based on count_of_top_rows
    def _step_for_topN(c: int, upper_eq_lower: bool) -> float:
        if upper_eq_lower:
            return 0.0
        if c < 5:
            return 5.0
        if c == 5:
            return 3.0
        if c == 6:
            return 2.0
        return 1.0

    upper_eq_lower = (work["conversion_upper_limit_in_grid"] == work["conversion_lower_limit_in_grid"]).values
    work["step_size_conversion"] = [
        _step_for_topN(count_of_top_rows, bool(u)) for u in upper_eq_lower
    ]

    return work


# ===========================================================================
# Subprocess (16) — orchestrates the bound-update logic
# ===========================================================================
def _update_conversion_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full mirror of Subprocess (16):
      • If fresh_feed_change != 0 → recycle-bound expansion path
        (Subprocess 102 → optional Subprocess 17 → Generate Attributes 342 →
         Branch 25 step assignment)
      • Else → biasing_condition path (Subprocess 20/50 topN loop)

    After both paths: evaluate inferred_tags_3, check grid_condition,
    set Conversion_Grid_Success and Conversion_Limits_given.
    """
    fresh_feed_change = _mi("fresh_feed_change", 0)

    # min fur check (2): >= 2 examples gate
    if len(df) < 2:
        _set("Conversion_Limits_given", 0)
        _set("Conversion_Grid_Success", 0)
        return df

    work = df.copy().sort_values("overall_ranking" if "overall_ranking" in df.columns else df.columns[0]).reset_index(drop=True)

    if fresh_feed_change != 0:
        # ── Generate Macro (163) ──────────────────────────────────────────
        sum_del_feed       = _m("sum_del_Feed_flow", 0.0)
        shc_ratio          = _m("shc_ratio", 0.0)
        fresh_feed_quantity= _m("fresh_feed_quantity", 0.0)
        re_ul_lim          = _m("change_recycle_ethane_upper_limit", 0.0)
        re_ll_lim          = _m("change_recycle_ethane_lower_limit", 0.0)

        if fresh_feed_change == -1:
            extra_recycle = (sum_del_feed / (1.0 + shc_ratio)) - fresh_feed_quantity
            _set("Extra_Recycle_Ethane", extra_recycle)
        else:
            extra_recycle = _m("Extra_Recycle_Ethane", 0.0)

        _set("upper_limit_change_in_recycle_ethane", extra_recycle + re_ul_lim)
        _set("lower_limit_change_in_recycle_ethane", extra_recycle + re_ll_lim)
        _set("Conversion_Limits_given", 0)

        # ── Subprocess (102): probe ──────────────────────────────────────
        probe = _probe_recycle_bounds(work)
        for k, v in probe.items():
            _set(k, v)

        # ── Branch (228) ─────────────────────────────────────────────────
        if probe["conversion_max_check_enough"] == 1:
            _set("Conversion_Limits_given", 1)
            _set("condition_satisfy", 0)

            # Generate Attributes (46): per-row New_Overall_conversion_limit
            OC   = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else 0.0
            frac = OC - np.floor(OC)
            ulim = _m("conversion_upper_limit_expansion_max_limit", 0.0)
            llim = _m("conversion_lower_limit_expansion_max_limit", 0.0)
            mfsl = _m("max_conversion_single_furnace_limit", 0.0)

            if fresh_feed_change == 1:
                work["New_Overall_conversion_limit"] = np.minimum(
                    ulim + frac - 1.0,
                    OC + mfsl,
                )
            else:
                work["New_Overall_conversion_limit"] = np.maximum(
                    llim + frac,
                    OC - mfsl,
                )

            # Subprocess (17): converge bounds
            work = _converge_conversion_band(
                work, considered_flag=probe["conversion_max_check_enough_for_upper_lower"]
            )
            # Generate Attributes (342) + Branch (25): final bands + step
            work = _finalise_conversion_bands(work)
        else:
            # leave bands alone but make sure step_size_conversion exists
            if "step_size_conversion" not in work.columns:
                work["step_size_conversion"] = 0.0
            # Seed New_Overall_conversion so downstream checks don't crash
            if "New_Overall_conversion" not in work.columns and "Overall_conversion" in work.columns:
                work["New_Overall_conversion"] = work["Overall_conversion"].astype(float)

    else:
        # biasing_condition path
        biasing_cond = _mi("biasing_condition", 0)
        if biasing_cond == 3:
            # "conversion according to bias (2)" — count_upper_limit_available_fur
            if "conversion_upper_limit_in_grid" in work.columns:
                avail = int((work["conversion_upper_limit_in_grid"].astype(float) > 0).sum())
            else:
                avail = 0
            if avail > 0:
                _set("Conversion_Limits_given", 1)
            else:
                _set("Conversion_Limits_given", 0)
                _set("ranking_cause_indicator", 6)
        else:
            work = _topN_bound_assignment(work)

    # ── Evaluate inferred_tags_3 (per-row formulas) ───────────────────────
    work = _evaluate_inferred_tags(work, "inferred_tags_3")

    # ── Rename New_Overall_conversion → New_Overall_Conversion_for_Single_Furnace (Rename 2) ──
    if "New_Overall_Conversion_for_Single_Furnace" in work.columns and "New_Overall_conversion" not in work.columns:
        work = work.rename(columns={"New_Overall_Conversion_for_Single_Furnace": "New_Overall_conversion"})

    # ── Branch (29) grid_condition ─────────────────────────────────────────
    grid_condition = _check_grid_condition(work)
    _set("grid_condition", grid_condition)

    # ── Generate Macro (21) ───────────────────────────────────────────────
    if grid_condition == 1:
        _set("Conversion_Grid_Success", 1)
        _set("Conversion_Limits_given", 2)
    else:
        _set("Conversion_Grid_Success", 0)
        _set("Conversion_Limits_given", 0)

    return work


def _check_grid_condition(df: pd.DataFrame) -> int:
    """
    Mirror of Branch (29):
      IF fresh_feed_change != 0:
          THEN compute New_Overall_conversion_limit & grid_condition (Gen Attr 50)
      ELSE compute delta_conversion & grid_condition (Gen Attr 1043)
    Then Extract Macro (11) takes example_index=1 grid_condition.
    """
    if df.empty:
        return 0

    fresh_feed_change = _mi("fresh_feed_change", 0)

    if fresh_feed_change != 0:
        OC   = df["Overall_conversion"].astype(float) if "Overall_conversion" in df.columns else pd.Series(0.0)
        frac = OC - np.floor(OC)
        ulim = _m("conversion_upper_limit_expansion_max_limit", 0.0)
        llim = _m("conversion_lower_limit_expansion_max_limit", 0.0)
        mfsl = _m("max_conversion_single_furnace_limit", 0.0)

        if fresh_feed_change == 1:
            limit = np.minimum(ulim + frac - 1.0, OC + mfsl)
            if "New_Overall_conversion" in df.columns:
                cond = (limit >= df["New_Overall_conversion"].astype(float)).astype(int)
            else:
                cond = pd.Series([1] * len(df))
        else:
            limit = np.maximum(llim + frac, OC - mfsl)
            if "New_Overall_conversion" in df.columns:
                cond = (limit <= df["New_Overall_conversion"].astype(float)).astype(int)
            else:
                cond = pd.Series([1] * len(df))
    else:
        if "New_Overall_conversion" in df.columns and "Overall_conversion" in df.columns:
            delta = df["New_Overall_conversion"].astype(float) - df["Overall_conversion"].astype(float)
        else:
            delta = pd.Series([0.0] * len(df))
        CLL = df["conversion_lower_limit_in_grid"].astype(float) if "conversion_lower_limit_in_grid" in df.columns else pd.Series([0.0] * len(df))
        CUL = df["conversion_upper_limit_in_grid"].astype(float) if "conversion_upper_limit_in_grid" in df.columns else pd.Series([0.0] * len(df))
        cond = ((delta >= CLL) & (delta <= CUL)).astype(int)

    return int(cond.iloc[0]) if len(cond) > 0 else 0


# ===========================================================================
# GRID -Conversion  → Extract per-row Grid_Row_i_* macros
# ===========================================================================
def _extract_grid_row_macros(df: pd.DataFrame) -> int:
    """
    Mirror of Loop (95): iterate over each row and populate the macros
    Grid_Row_i_Feed_flow, _Furnace, _Specific_Energy_consumption,
    _Conversion, _Furnace_condition, _Ethylene_Production,
    _lower_conversion_limit, _upper_conversion_limit, _step_size_conversion,
    _part_override.

    Returns the number of rows extracted (= No_of_rows).
    """
    # factor (2): size_part, flag_conversion_part_override
    if not df.empty and "conversion_lower_limit_in_grid" in df.columns:
        CLL  = df["conversion_lower_limit_in_grid"].astype(float)
        CUL  = df["conversion_upper_limit_in_grid"].astype(float)
        SSC  = df["step_size_conversion"].astype(float) if "step_size_conversion" in df.columns else 0.0
        size_part = (CLL - CUL) / SSC.replace(0, np.nan)
        size_part = size_part.fillna(0)

        flag_override = np.where(
            SSC == 3,
            CLL * (CLL - size_part) * (CLL - 2 * size_part) * CUL,
            np.where(SSC == 2, CLL * (CLL - size_part) * CUL, 0.0)
        )
        df = df.copy()
        df["size_part"] = size_part
        df["flag_conversion_part_override"] = flag_override

    n = len(df)
    _set("No_of_rows", n)
    if n == 0:
        return 0

    # Reset all Grid_Row_i_* macros first
    for i in range(1, MAX_FURNACES + 1):
        _set(f"Grid_Row_{i}_Feed_flow", 0.0)
        _set(f"Grid_Row_{i}_Furnace", "")
        _set(f"Grid_Row_{i}_Specific_Energy_consumption", 0.0)
        _set(f"Grid_Row_{i}_Conversion", 0.0)
        _set(f"Grid_Row_{i}_Furnace_condition", "")
        _set(f"Grid_Row_{i}_Ethylene_Production", 0.0)
        _set(f"Grid_Row_{i}_lower_conversion_limit", 0.0)
        _set(f"Grid_Row_{i}_upper_conversion_limit", 0.0)
        _set(f"Grid_Row_{i}_step_size_conversion", 0.0)
        _set(f"Grid_Row_{i}_part_override", 0.0)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if idx > MAX_FURNACES:
            break
        _set(f"Grid_Row_{idx}_Feed_flow",
             float(row.get("Feed_flow", 0.0)))
        _set(f"Grid_Row_{idx}_Furnace",
             str(row.get("entity_name", "")))
        _set(f"Grid_Row_{idx}_Specific_Energy_consumption",
             float(row.get("specific_energy_consumption", 0.0)))
        _set(f"Grid_Row_{idx}_Conversion",
             float(row.get("Overall_conversion", 0.0)))
        _set(f"Grid_Row_{idx}_Furnace_condition",
             str(row.get("Furnace_condition", "")))
        _set(f"Grid_Row_{idx}_Ethylene_Production",
             float(row.get("ethylene_production", 0.0)))
        _set(f"Grid_Row_{idx}_lower_conversion_limit",
             float(row.get("conversion_lower_limit_in_grid", 0.0)))
        _set(f"Grid_Row_{idx}_upper_conversion_limit",
             float(row.get("conversion_upper_limit_in_grid", 0.0)))
        _set(f"Grid_Row_{idx}_step_size_conversion",
             float(row.get("step_size_conversion", 0.0)))
        _set(f"Grid_Row_{idx}_part_override",
             float(row.get("flag_conversion_part_override", 0.0)))

    return n


# ===========================================================================
# Loop (96) — rounding & part_override application
# ===========================================================================
def _round_and_override_conversion_deltas(combo: List[float]) -> List[float]:
    """
    For each furnace i, Generate Macro (185) + (201):
        if step_size_conversion_i == 0:
            delta_i = 0
        elif part_override_i != 0:
            floor_num = floor(delta_i / 0.25) * 0.25
            delta_i = 0 if |floor_num| <= 0.5 else floor_num
        else:
            floor_num = floor(delta_i / 0.25) * 0.25
            delta_i = 0 if |floor_num| < 0.5 else floor_num
    """
    result = []
    for i, v in enumerate(combo, start=1):
        step = _m(f"Grid_Row_{i}_step_size_conversion", 0.0)
        override = _m(f"Grid_Row_{i}_part_override", 0.0)
        lo   = _m(f"Grid_Row_{i}_lower_conversion_limit", 0.0)
        if step == 0:
            result.append(0.0)
            continue
        floor_num = math.floor(float(v) / 0.25) * 0.25
        if override != 0:
            result.append(0.0 if abs(floor_num) <= 0.5 else floor_num)
        else:
            result.append(0.0 if abs(floor_num) <  0.5 else floor_num)
    # If `combo` shorter than MAX, pad zeros
    while len(result) < MAX_FURNACES:
        result.append(0.0)
    return result


# ===========================================================================
# Conversion-grid inner: enumerate and evaluate combos
# ===========================================================================
def _build_conv_ranges(n: int) -> List[List[float]]:
    """
    Build per-furnace conversion-delta value lists from
    Grid_Row_i_lower/upper_conversion_limit, Grid_Row_i_step_size_conversion.
    """
    ranges = []
    for i in range(1, n + 1):
        lo   = _m(f"Grid_Row_{i}_lower_conversion_limit", 0.0)
        hi   = _m(f"Grid_Row_{i}_upper_conversion_limit", 0.0)
        step = _m(f"Grid_Row_{i}_step_size_conversion", 0.0)
        # Reuse Chunk-A-style linear range
        if step <= 0 or hi < lo:
            ranges.append([round(lo, 6)])
            continue
        n_steps = int(np.floor((hi - lo) / step + 1e-9)) + 1
        vals = [round(lo + k * step, 6) for k in range(n_steps)]
        if vals[-1] < hi - 1e-9:
            vals.append(round(hi, 6))
        ranges.append(vals)
    # Pad to MAX_FURNACES with [0.0]
    while len(ranges) < MAX_FURNACES:
        ranges.append([0.0])
    return ranges


def _build_per_row_conv_table(df: pd.DataFrame, conv_deltas: List[float]) -> pd.DataFrame:
    """
    Loop (145) + Generate Attributes (335): for each row i (1-indexed),
    set conversion_delta = Grid_Row_i_conversion_delta and
    New_Overall_conversion = Overall_conversion + conversion_delta.
    """
    work = df.copy()
    if "id" not in work.columns:
        work["id"] = np.arange(1, len(work) + 1)

    work["conversion_bias"]      = 0.0
    work["New_Overall_conversion"] = work["Overall_conversion"].astype(float) if "Overall_conversion" in work.columns else 0.0

    n = min(len(work), len(conv_deltas))
    for k in range(n):
        delta_k = float(conv_deltas[k])
        work.iat[k, work.columns.get_loc("conversion_bias")]      = delta_k
        work.iat[k, work.columns.get_loc("New_Overall_conversion")] = (
            float(work.iat[k, work.columns.get_loc("Overall_conversion")]) + delta_k
            if "Overall_conversion" in work.columns else delta_k
        )

    # Recycle / change in recycle  — these are part of the standard derived columns
    if {"New_Feed_flow", "shc_ratio", "Current_Recycle_Ethane_Feed"}.issubset(work.columns):
        work["New_Recycle_Ethane_Feed"]  = (
            work["New_Feed_flow"].astype(float) / (1.0 + work["shc_ratio"].astype(float))
        ) * (100.0 - work["New_Overall_conversion"]) / 100.0
        work["Change_in_Recycle_Ethane_Feed"] = (
            work["New_Recycle_Ethane_Feed"] - work["Current_Recycle_Ethane_Feed"].astype(float)
        )

    return work


def _aggregate_combo(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate (84): sum of del_ethylene, Change_in_Recycle_Ethane_Feed,
    Heat, Heat_new, Temp_Ethylene_Production, New_Ethylene_Production.
    Then Generate Attributes (51): flag_energy_consumption /
    flag_specific_energy_consumption.

    Returns a dict of scalar macros that callers extract.
    """
    def _sum(col):
        return float(df[col].astype(float).sum()) if col in df.columns else 0.0

    sum_del_eth    = _sum("del_ethylene")
    sum_chg_rec    = _sum("Change_in_Recycle_Ethane_Feed")
    sum_heat       = _sum("Heat")
    sum_heat_new   = _sum("Heat_new")
    sum_temp_eth   = _sum("Temp_Ethylene_Production")
    sum_new_eth    = _sum("New_Ethylene_Production")

    sum_curr_spc   = sum_heat     / sum_temp_eth if sum_temp_eth != 0 else 0.0
    sum_new_spc    = sum_heat_new / sum_new_eth  if sum_new_eth  != 0 else 0.0

    ric_improve_ec   = _mi("ranking_improve_energy_consumption", 0)
    ric_improve_spec = _mi("ranking_improve_specific_energy_consumption", 0)

    flag_ec      = 1 if (ric_improve_ec == 0 or sum_heat_new <= sum_heat) else 0
    flag_spec_ec = 1 if (ric_improve_spec == 0 or sum_new_spc <= sum_curr_spc) else 0

    return dict(
        sum_del_ethylene=sum_del_eth,
        sum_Change_in_Recycle_Ethane_Feed=sum_chg_rec,
        sum_Heat=sum_heat, sum_Heat_new=sum_heat_new,
        sum_Temp_Ethylene_Production=sum_temp_eth,
        sum_New_Ethylene_Production=sum_new_eth,
        sum_Current_specific_energy_consumption=sum_curr_spc,
        sum_New_specific_energy_consumption=sum_new_spc,
        flag_energy_consumption=flag_ec,
        flag_specific_energy_consumption=flag_spec_ec,
    )


def _enumerate_conversion_grid(df_per_row: pd.DataFrame) -> Dict[str, float]:
    """
    Mirror of `Conversion inside Feed grid (2)`:
      • Build per-furnace conversion-delta ranges from Grid_Row_i_* macros.
      • Walk the Cartesian product.
      • For each combo:
          - apply Loop (96) rounding/override
          - skip if same as a previous combo (compare_log_curr_conver_delta dedupe)
          - build per-row table, evaluate inferred_tags_2, aggregate,
            compute compare_log_curr_conver_delta (Generate Macro 46)
          - if passes the gate, accept this conversion combo as new best
            (record Grid_Row_i_conversion_delta macros + log entry)

    The gate is:
        recycle bounds OK    AND
        sum_del_ethylene > Max_Benefit   AND
        flag_energy_consumption == 1    AND
        flag_specific_energy_consumption == 1

    Returns a summary dict of the winning combo's aggregated values.
    """
    n = int(min(_mi("No_of_rows", 0), MAX_FURNACES))

    if n == 0 or df_per_row.empty:
        return dict(sum_del_ethylene=-1e4, sum_Change_in_Recycle_Ethane_Feed=0.0,
                    Conversion_Grid_Success=0)

    ranges = _build_conv_ranges(n)
    total_combos = 1
    for r in ranges[:n]:
        total_combos *= max(1, len(r))

    if total_combos > MAX_CONV_COMBOS:
        logger.warning("CONV GRID: %d combos exceeds cap %d – capping enumeration.",
                       total_combos, MAX_CONV_COMBOS)

    logger.debug("CONV GRID: %d furnaces, %d total combos", n, total_combos)

    # Bounds for the gate
    re_ul = _m("upper_limit_change_in_recycle_ethane", 1e9)
    re_ll = _m("lower_limit_change_in_recycle_ethane", -1e9)
    max_benefit_in = _m("Max_Benefit", -1e9)

    # Local "Conv_GRID_LOG(feed)-main" — RM persists this per inner conversion-grid run
    log_rows = []

    # Best combo for this feed iteration
    best = dict(
        sum_del_ethylene=-1e4,
        sum_Change_in_Recycle_Ethane_Feed=0.0,
        combo=[0.0] * n,
        accepted=False,
    )

    # Pre-extract per-row derived columns once (we'll only mutate
    # conversion_delta / New_Overall_conversion per combo).
    df_base = df_per_row.copy()
    if "id" not in df_base.columns:
        df_base["id"] = np.arange(1, len(df_base) + 1)

    indices = [0] * n
    sizes   = [max(1, len(ranges[i])) for i in range(n)]
    combos_done = 0

    while True:
        combo = [ranges[i][indices[i]] for i in range(n)]

        # Loop (96) override/round
        combo_full = _round_and_override_conversion_deltas(combo)
        combo_n    = combo_full[:n]

        # Push Grid_Row_i_conversion_delta macros for inferred-tag eval
        for i, v in enumerate(combo_full, start=1):
            _set(f"Grid_Row_{i}_conversion_delta", v)

        # Build per-row table for this combo
        df_combo = _build_per_row_conv_table(df_base, combo_n)

        # Evaluate inferred_tags_2 (the formulas for del_ethylene, Heat,
        # Temp_Ethylene_Production, etc.)
        df_combo = _evaluate_inferred_tags(df_combo, "inferred_tags_2")

        # Aggregate (84)
        agg = _aggregate_combo(df_combo)

        # Generate Macro (46): gate
        passes = (
            re_ll <= agg["sum_Change_in_Recycle_Ethane_Feed"] <= re_ul
            and agg["sum_del_ethylene"] > max_benefit_in
            and agg["flag_energy_consumption"] == 1
            and agg["flag_specific_energy_consumption"] == 1
        )

        if passes:
            # New best for *this* feed combo
            if agg["sum_del_ethylene"] > best["sum_del_ethylene"]:
                best.update(agg)
                best["combo"] = combo_full[:]
                best["accepted"] = True

            # Conv_GRID_LOG(feed)-main row
            log_rows.append({
                **{f"Grid_Row_{i}_conversion_delta": combo_full[i - 1]
                   for i in range(1, MAX_FURNACES + 1)},
                "sum_del_ethylene": agg["sum_del_ethylene"],
                "sum_Change_in_Recycle_Ethane_Feed":
                    agg["sum_Change_in_Recycle_Ethane_Feed"],
            })

        combos_done += 1
        if combos_done >= MAX_CONV_COMBOS:
            break

        # Advance odometer
        k = n - 1
        while k >= 0:
            indices[k] += 1
            if indices[k] < sizes[k]:
                break
            indices[k] = 0
            k -= 1
        if k < 0:
            break

    # Persist log
    if log_rows:
        prev = STORE.get("Conv_GRID_LOG_feed_main")
        new_df = pd.DataFrame(log_rows)
        STORE["Conv_GRID_LOG_feed_main"] = (
            new_df if prev is None or len(prev) == 0
            else pd.concat([prev, new_df], ignore_index=True)
        )

    # Push winning Grid_Row_i_conversion_delta macros (or zeros if nothing accepted)
    for i, v in enumerate(best["combo"], start=1):
        _set(f"Grid_Row_{i}_conversion_delta", float(v))
        _set(f"Grid_Row_{i}_conversion_delta_best", float(v))

    if not best["accepted"]:
        # Set Macro (43) sentinel
        best["sum_del_ethylene"] = -1e4

    best["Conversion_Grid_Success"] = 1 if best["accepted"] else 0
    return best


# ===========================================================================
# Subprocess (131)/(44) — post-grid final aggregation + Generate Macro (44)
# Updates Max_Benefit per the canonical rule.
# ===========================================================================
def _finalise_combo_and_update_max_benefit(df_per_row: pd.DataFrame,
                                          best: Dict) -> None:
    """
    Apply the winning conversion deltas, re-evaluate inferred_tags_2 (RM
    Recall (41) inside Subprocess (131) – same store as the inner grid),
    compute Aggregate (85) sums + SPC + nox + flag_benefit, then run
    Generate Macro (44) to potentially update Max_Benefit.
    """
    if best.get("Conversion_Grid_Success", 0) != 1:
        # No conversion grid hit — leave Max_Benefit untouched.
        _set("sum_del_ethylene", -1e4)
        _set("sum_Change_in_Recycle_Ethane_Feed", 0.0)
        _set("curr_sum_SPC_Furnace", _m("Max_Benefit_SPC", 1000.0))
        return

    # Build per-row with the winning conversion combo
    combo = best.get("combo", [0.0] * MAX_FURNACES)
    n     = int(min(_mi("No_of_rows", 0), len(combo), len(df_per_row)))
    df_win = _build_per_row_conv_table(df_per_row, combo[:n])

    # Recall (41) inferred_tags_2 + re-eval (RM uses tags_2 here too)
    df_win = _evaluate_inferred_tags(df_win, "inferred_tags_2")

    # Generate Attributes (52): SPC_*, flag_nox
    if "conversion_bias" not in df_win.columns:
        df_win["conversion_bias"] = 0.0
    if "del_Feed_flow" not in df_win.columns:
        df_win["del_Feed_flow"] = 0.0

    df_win["SPC_conversion_bias"] = (df_win["conversion_bias"].astype(float) != 0).astype(int)
    df_win["SPC_feed_bias"]       = (df_win["del_Feed_flow"].astype(float)   != 0).astype(int)
    df_win["SPC_Furnace"]         = df_win["SPC_feed_bias"] + df_win["SPC_conversion_bias"]

    if "nox_margin" in df_win.columns and "Heat" in df_win.columns and "Heat_new" in df_win.columns:
        df_win["flag_nox"] = np.where(
            df_win["nox_margin"].astype(float) == 0, 0,
            np.where(df_win["Heat_new"].astype(float) > df_win["Heat"].astype(float), 1, 0)
        )
    else:
        df_win["flag_nox"] = 0

    # Aggregate (85): sum of everything
    sums = _aggregate_combo(df_win)
    sum_spc = float(df_win["SPC_Furnace"].sum()) if "SPC_Furnace" in df_win.columns else 0.0
    sum_nox = float(df_win["flag_nox"].sum())    if "flag_nox"    in df_win.columns else 0.0

    _set("sum_del_ethylene",                  sums["sum_del_ethylene"])
    _set("sum_Change_in_Recycle_Ethane_Feed", sums["sum_Change_in_Recycle_Ethane_Feed"])
    _set("curr_sum_SPC_Furnace",              sum_spc)
    _set("sum_flag_nox",                      sum_nox)

    # Generate Macro (23): flag_SPC, benefit_percent_upper/lower
    max_benefit_spc = _m("Max_Benefit_SPC", 1000.0)
    bpt             = _m("benefit_percent_threshold", 0.0)
    flag_spc = 1 if sum_spc < max_benefit_spc else 0
    benefit_pct_upper = 1.0 + bpt
    benefit_pct_lower = 1.0 - bpt
    _set("flag_SPC", flag_spc)
    _set("benefit_percent_upper", benefit_pct_upper)
    _set("benefit_percent_lower", benefit_pct_lower)

    # Generate Attributes (53): flag_energy_consumption (with sum_flag_nox rule),
    # flag_specific_energy_consumption, flag_benefit
    ric_improve_ec   = _mi("ranking_improve_energy_consumption", 0)
    ric_improve_spec = _mi("ranking_improve_specific_energy_consumption", 0)

    if ric_improve_ec == 0:
        flag_ec = 1 if sum_nox == 0 else 0
    else:
        flag_ec = 1 if sums["sum_Heat_new"] <= sums["sum_Heat"] else 0
    _set("flag_energy_consumption", flag_ec)

    if ric_improve_spec == 0:
        flag_spec_ec = 1
    else:
        flag_spec_ec = 1 if sums["sum_New_specific_energy_consumption"] <= sums["sum_Current_specific_energy_consumption"] else 0
    _set("flag_specific_energy_consumption", flag_spec_ec)

    max_benefit = _m("Max_Benefit", -1e9)
    sum_del_eth = sums["sum_del_ethylene"]
    # flag_benefit:
    #   if sum_del_eth > Max_Benefit * benefit_pct_upper: 1
    #   elif Max_Benefit * benefit_pct_lower < sum_del_eth < Max_Benefit * benefit_pct_upper
    #        AND flag_SPC == 1: 1
    #   else 0
    bench_upper = max_benefit * benefit_pct_upper
    bench_lower = max_benefit * benefit_pct_lower
    if sum_del_eth > bench_upper:
        flag_benefit = 1
    elif (bench_lower < sum_del_eth < bench_upper) and flag_spc == 1:
        flag_benefit = 1
    else:
        flag_benefit = 0
    _set("flag_benefit", flag_benefit)

    # ── Generate Macro (44) — THE acceptance rule ─────────────────────────
    re_ul = _m("upper_limit_change_in_recycle_ethane", 1e9)
    re_ll = _m("lower_limit_change_in_recycle_ethane", -1e9)
    sum_chg_rec = sums["sum_Change_in_Recycle_Ethane_Feed"]

    if (re_ll <= sum_chg_rec <= re_ul
        and flag_benefit == 1
        and flag_ec == 1
        and flag_spec_ec == 1):
        _set("Max_Benefit", sum_del_eth)
        logger.debug("  ← Max_Benefit updated to %.4f (combo accepted)", sum_del_eth)


# ===========================================================================
# Public entry point — Chunk A calls this once per surviving feed combo
# ===========================================================================
def run(df_per_row: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full MAIN CONVERSION GRID + GRID -Conversion for ONE feed combo.

    Inputs (in df_per_row):
        Feed_flow, New_Feed_flow, del_Feed_flow, shc_ratio,
        Overall_conversion, Current_Recycle_Ethane_Feed,
        conversion_lower_limit_in_grid, conversion_upper_limit_in_grid,
        step_size_conversion, Furnace_condition, ethylene_production,
        percent_above_threshold, overall_ranking, entity_name,
        specific_energy_consumption, Heat (current heat), …

    Side effects (MACROS):
        sum_del_ethylene, sum_Change_in_Recycle_Ethane_Feed,
        Grid_Row_1..9_conversion_delta, Conversion_Grid_Success,
        Max_Benefit (possibly updated), flag_benefit, flag_energy_consumption,
        flag_specific_energy_consumption, curr_sum_SPC_Furnace, sum_flag_nox,
        Conversion_Limits_given, condition_satisfy, count_no_fixed_grid_fur

    Returns df_per_row (unchanged shape; intermediate columns dropped).
    """
    logger.info("07b-conversion grid submodule started")
    # ── Subprocess (16): bound updates ───────────────────────────────────
    df_bands = _update_conversion_bands(df_per_row)

    # ── Branch GRID: only run inner grid if Conversion_Limits_given == 1 ──
    conv_limits_given = _mi("Conversion_Limits_given", 0)
    if conv_limits_given != 1 and conv_limits_given != 2:
        # No inner grid; sentinel result.
        _set("sum_del_ethylene", -1e4)
        _set("sum_Change_in_Recycle_Ethane_Feed", 0.0)
        _set("Conversion_Grid_Success", 0)
        for i in range(1, MAX_FURNACES + 1):
            _set(f"Grid_Row_{i}_conversion_delta", 0.0)
            _set(f"Grid_Row_{i}_conversion_delta_best", 0.0)
        return df_per_row

    # ── Handle Exception (13) wrapper around the inner grid ──────────────
    try:
        # Loop (97): init dummy macros (done implicitly by _extract_grid_row_macros)
        _extract_grid_row_macros(df_bands)

        # Conversion inside Feed grid (2)
        best = _enumerate_conversion_grid(df_bands)

        # Subprocess (44)/(131): finalisation + Max_Benefit update
        _finalise_combo_and_update_max_benefit(df_bands, best)

    except Exception as exc:
        logger.error("Conversion grid raised %s – treating combo as failed.",
                     exc, exc_info=False)
        _set("sum_del_ethylene", -1e4)
        _set("sum_Change_in_Recycle_Ethane_Feed", 0.0)
        _set("Conversion_Grid_Success", 0)
        for i in range(1, MAX_FURNACES + 1):
            _set(f"Grid_Row_{i}_conversion_delta", 0.0)
            _set(f"Grid_Row_{i}_conversion_delta_best", 0.0)

    return df_per_row
