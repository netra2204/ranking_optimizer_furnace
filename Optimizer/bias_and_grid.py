"""
bias_and_grid.py
================

Implementation of the *GRID_AND_BIASING* branch from the RapidMiner module
``United (2) → Main_Process → GRID_AND_BIASING``.

This is the core optimisation logic. It breaks into:

  • Inferred calculations
      - Feed Biasing → Subprocess (3): extracts the 8 limiting-pass macros
        (first_limiting_pass…eighth_limiting_pass) via Subprocess (100) and
        Subprocess (106).
      - Subprocess (96): pass-wise feed/days remaining macros (pass1_…pass8_).
      - Branch on (furnace_in_SOR == 0 && furnace_in_EOR == 0 && external_constraint == 0):
            - Branch on Feed_Bias > 0:
                  - Branch (144): Ranking_Coupled == 0
                  - Branch (4):   Feed_Bias > 0    → up / down biasing logic
                  - Generate Macro (5/2): combine into Bias_In_Pass_1..8
                  - Branch (7):   Max_Feed_Bias == Feed_Bias
            - Bias Limits (single-sided): Feed_Bias == 0
            - Generate Macro (7): pass_X_step_size selection
            - Loop "Step Size"

  • FEED GRID (2):
      - Fe_input → 8-pass parameter grid (`concurrency:optimize_parameters_grid`)
        with pass_<n> in [bias_min ; bias_max ; step].
      - Loop (3): runs the inner objective (Pass_initializer + Objective_function)
        for every grid combination.
      - Dedup via Feed_Grid_Character + log table.

Author: Optimizer Replica Project
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rm_runtime import (
    MacroStore,
    StoreRegistry,
    CoilsimModelProvider,
    aggregate,
    apply_filter_examples,
    apply_generate_attributes,
    evaluate_expression,
    extract_macro_from_dataset,
)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline-parameter macros that must be present before this module runs.
# They are normally injected by the parent process from `pipeline_parameters_opt`.
# Defaults match those visible in the .rmp via Filter Examples + Extract Macro.
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PIPELINE_PARAMETERS: Dict[str, str] = {
    # Bias Constants subprocess (Filter Examples 39/44 + Extract Macro 85/124)
    "bias_max_constant": "0.5",
    "bias_min_constant": "-0.5",
    # Used in Biasing_in_pass_<n> formulas (Generate Macro 334/331/335)
    "limiting_pass_margin": "3.0",
    "Controller_opening_threshold": "95.0",
    "pass_feed_max_limit": "8.5",
    "pass_feed_min_limit": "6.5",
    "pass_feed_max_diff_limit": "5.0",
    # Days remaining margin (Generate Macro 100)
    "Days_remaining_min_margin": "0.5",
    # Step size for the 8-D feed grid (Generate Macro 7 inside Step Size loop)
    "step_size_feed_grid": "0.25",
    # Margins for heat bias grid bounds (min_max (2))
    "heat_bias_min_margin_grid": "1.0",
    "heat_bias_max_margin_grid": "1.0",
    # Capping
    "days_remaining_capping": "365",
    "threshold_thickness": "12.7",
    # Counters / control flags
    "iteration": "0",
    "Counter": "1",
}


# ═════════════════════════════════════════════════════════════════════════════
#                              Bias Constants
# ═════════════════════════════════════════════════════════════════════════════
def run_bias_constants(macros: MacroStore, registry: StoreRegistry) -> None:
    """Bias Constants subprocess.

    Operators (in order):
      - Recall:                 Recall "pipeline_parameters_opt"
      - Filter Examples (39):   parameter.equals.bias_max_constant
      - Extract Macro (85):     macro=bias_max_constant ← value
      - Filter Examples (44):   parameter.equals.bias_min_constant
      - Extract Macro (124):    macro=bias_min_constant ← value
    """
    if not registry.has("pipeline_parameters_opt"):
        return
    pp = registry.recall("pipeline_parameters_opt")

    for param, macro_name in (
        ("bias_max_constant", "bias_max_constant"),
        ("bias_min_constant", "bias_min_constant"),
    ):
        sub = pp[pp["parameter"].astype(str) == param]
        if not sub.empty and "value" in sub.columns:
            macros.set(macro_name, sub["value"].iloc[0])


# ═════════════════════════════════════════════════════════════════════════════
#                              Overall_Calcs
# ═════════════════════════════════════════════════════════════════════════════
def run_overall_calcs(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """Compute furnace-level weighted statistics used as the optimizer baseline.

    Operators (in order):
      - Generate Attributes (2):
            Mixed_Feed_And_COT_Product_Old   = Mixed_Feed_Old * (COT_Old - 20)
            Feed_And_Conversion_Product_Old  = Feed_Old * Conversion_Old
      - Aggregate (no group_by):
            sum(Mixed_Feed_And_COT_Product_Old),
            sum(Mixed_Feed_Old), sum(Feed_Old), sum(Feed_And_Conversion_Product_Old),
            sum(Radiant_Heat_Absorbed_Old), average(COT_Old), sum(Good_Tubes_Old)
      - Extract Macro (2): stores 6 macros (Sum_…, Furnace_…)
      - Generate Attributes (182):
            Furnace_Weighted_COT_Old        = Sum_Of_Mixed_Feed_And_COT_Product_Old / Furnace_Mixed_Feed_Old
            Furnace_Weighted_Conversion_Old = Sum_Of_Feed_And_Conversion_Product_Old / Furnace_Feed_Old
            Radiant_Heat_Absorbed_Fraction  = Radiant_Heat_Absorbed_Old / Furnace_Radiant_Heat_Absorbed_Old
            Furnace_Radiant_Heat_Absorbed_Old = parse(%{Furnace_Radiant_Heat_Absorbed_Old})
      - Extract Macro (87): stores Furnace_Weighted_COT_Old + 11 additional macros.
    """
    # Generate Attributes (2)
    df = apply_generate_attributes(
        df,
        [
            ("Mixed_Feed_And_COT_Product_Old", "Mixed_Feed_Old*([COT_Old]-20)"),
            ("Feed_And_Conversion_Product_Old", "[Feed_Old]*[Conversion_Old]"),
        ],
        macros,
    )

    # Aggregate
    agg = aggregate(
        df,
        aggregations=[
            ("Mixed_Feed_And_COT_Product_Old", "sum"),
            ("Mixed_Feed_Old",                 "sum"),
            ("Feed_Old",                       "sum"),
            ("Feed_And_Conversion_Product_Old", "sum"),
            ("Radiant_Heat_Absorbed_Old",      "sum"),
            ("COT_Old",                        "average"),
            ("Good_Tubes_Old",                 "sum"),
        ],
    )

    # Extract Macro (2)
    extract_macro_from_dataset(
        agg, macros,
        macro_name="Sum_Of_Mixed_Feed_And_COT_Product_Old",
        attribute_name="sum(Mixed_Feed_And_COT_Product_Old)",
        additional={
            "Furnace_Mixed_Feed_Old":               "sum(Mixed_Feed_Old)",
            "Furnace_Feed_Old":                     "sum(Feed_Old)",
            "Sum_Of_Feed_And_Conversion_Product_Old":"sum(Feed_And_Conversion_Product_Old)",
            "Furnace_Average_COT_Old":              "average(COT_Old)",
            "Furnace_Good_Tubes_Old":               "sum(Good_Tubes_Old)",
            "Furnace_Radiant_Heat_Absorbed_Old":    "sum(Radiant_Heat_Absorbed_Old)",
        },
    )

    # Generate Attributes (182)
    df = apply_generate_attributes(
        df,
        [
            ("Furnace_Weighted_COT_Old",
             "parse(%{Sum_Of_Mixed_Feed_And_COT_Product_Old})/parse(%{Furnace_Mixed_Feed_Old})"),
            ("Furnace_Weighted_Conversion_Old",
             "parse(%{Sum_Of_Feed_And_Conversion_Product_Old})/parse(%{Furnace_Feed_Old})"),
            ("Radiant_Heat_Absorbed_Fraction",
             "Radiant_Heat_Absorbed_Old/parse(%{Furnace_Radiant_Heat_Absorbed_Old})"),
            ("Furnace_Radiant_Heat_Absorbed_Old",
             "parse(%{Furnace_Radiant_Heat_Absorbed_Old})"),
        ],
        macros,
    )

    # Extract Macro (87)
    extract_macro_from_dataset(
        df, macros,
        macro_name="Furnace_Weighted_COT_Old",
        attribute_name="Furnace_Weighted_COT_Old",
        additional={
            "Furnace_Weighted_Conversion_Old":   "Furnace_Weighted_Conversion_Old",
            "Use_Uptime_Benefit_In_Opportunity": "Use_Uptime_Benefit_In_Opportunity",
            "Coupled_Mode_Identifier":           "Coupled_Mode_Identifier",
            "external_constraint":               "external_constraint",
            "Feed_Bias":                         "Feed_Bias",
            "Ranking_Coupled":                   "Ranking_Coupled",
            "COT_Bias":                          "COT_Bias",
            "Heat_Bias":                         "Heat_Bias",
            "Days_Online":                       "Days_Online",
            "Conversion_Bias":                   "Conversion_Bias",
            "Decoke_Time":                       "Decoke_Time",
            "Use_Optimizer_Opportunity":         "Use_Optimizer_Opportunity",
        },
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
#                         Inferred calculations
# ═════════════════════════════════════════════════════════════════════════════
def run_inferred_calculations(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> Dict[str, float]:
    """The "Inferred calculations" subprocess inside GRID_AND_BIASING.

    Returns a dict of computed pass-level bias bounds: keys are
    ``pass_<n>_feed_bias_min/max``, ``Proceed``, etc., that drive the
    subsequent FEED GRID search.

    Internally it runs:
      • Feed Biasing
            ├── Subprocess (3): Sort by Days_Remaining_Old ascending, then
            │       Subprocess (100) — 8 Extract Macro ops setting
            │       first_/second_/.../eighth_limiting_pass and their
            │       feed/controller_opening/days_remaining context.
            ├── Subprocess (106): Sort descending — last_, second_last_, … etc.
            └── Subprocess (96): Sort by Pass ascending — pass1_mixed_feed,
                  pass1_days_remaining, …, pass8_*.
      • Generate Macro (30): furnace_in_SOR / furnace_in_EOR
      • Branch (3): the main biasing logic only runs when none of
            furnace_in_SOR, furnace_in_EOR, external_constraint is non-zero.
            ├── Branch (144) on Ranking_Coupled == 0:
            │       Generate Macro (334): Biasing_in_pass_<ord> (decoupled)
            │       Generate Macro (336): Min_Increase_Flow, Decrease_Flow,
            │                              Proceed, Increase_Flow
            │       Generate Macro (5):   Bias_In_Pass_1..8 (re-map by limiting_pass)
            │       Branch on Increase_Flow == |Decrease_Flow|:
            │              Generate Macro (6) or "Inc<Dec" balanced search
            │
            └── Bias Limits (Feed_Bias==0 path) or
                Branch (4) (Feed_Bias > 0 → up-bias, else down-bias):
                       Generate Macro (331/335): single-direction Biasing_in_pass_<ord>
                       Generate Macro (332): Max_Feed_Bias, Proceed
                       Generate Macro (2):   Bias_In_Pass_1..8 (coupled re-map)
                       Branch (7): Max_Feed_Bias == Feed_Bias
                       Generate Macro (3) or balanced loop
      • Generate Macro (7): pass_<n>_step_size derivation (Loop "Step Size", 8x)
    """
    # ─── Subprocess (3): sort ascending by Days_Remaining_Old ───────────────
    sorted_asc = df.sort_values("Days_Remaining_Old", ascending=True).reset_index(drop=True)
    # Subprocess (100) — Extract Macros for the 8 most-limiting passes
    for ord_idx, ord_name in enumerate(
        ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"),
        start=1,
    ):
        if ord_idx <= len(sorted_asc):
            row = sorted_asc.iloc[ord_idx - 1]
            macros.set(f"{ord_name}_limiting_pass", row.get("Pass", float("nan")))
            macros.set(f"{ord_name}_limiting_pass_remaining_days",
                       row.get("Days_Remaining_Old", float("nan")))
            macros.set(f"{ord_name}_limiting_Controller_opening",
                       row.get("Controller_Opening_Old", float("nan")))
            macros.set(f"{ord_name}_limiting_Mixed_Feed",
                       row.get("Mixed_Feed_Old", float("nan")))

    # ─── Subprocess (106): sort descending by Days_Remaining_Old ───────────
    sorted_desc = df.sort_values("Days_Remaining_Old", ascending=False).reset_index(drop=True)
    for ord_idx, ord_name in enumerate(
        ("last", "second_last", "third_last", "fourth_last",
         "fifth_last", "sixth_last", "seventh_last", "eighth_last"),
        start=1,
    ):
        if ord_idx <= len(sorted_desc):
            row = sorted_desc.iloc[ord_idx - 1]
            macros.set(f"{ord_name}_limiting_pass", row.get("Pass", float("nan")))
            macros.set(f"{ord_name}_limiting_days_remaining",
                       row.get("Days_Remaining_Old", float("nan")))
            macros.set(f"{ord_name}_limiting_Controller_opening",
                       row.get("Controller_Opening_Old", float("nan")))
            macros.set(f"{ord_name}_limiting_feed",
                       row.get("Mixed_Feed_Old", float("nan")))

    # ─── Subprocess (96): sort by Pass ascending ───────────────────────────
    by_pass = df.sort_values("Pass", ascending=True).reset_index(drop=True)
    for p in range(1, 9):
        sub = by_pass[pd.to_numeric(by_pass["Pass"], errors="coerce") == p]
        if not sub.empty:
            macros.set(f"pass{p}_mixed_feed", sub["Mixed_Feed_Old"].iloc[0])
            macros.set(f"pass{p}_days_remaining", sub["Days_Remaining_Old"].iloc[0])

    # ─── Generate Macro (30): furnace_in_SOR / furnace_in_EOR ──────────────
    macros.set(
        "furnace_in_SOR",
        evaluate_expression(
            "if(parse(%{Days_Online})<2 && parse(%{Ranking_Coupled})==0,1,0)", macros,
        ),
    )
    macros.set(
        "furnace_in_EOR",
        evaluate_expression(
            "if(parse(%{first_limiting_pass_remaining_days})<2,1,0)", macros,
        ),
    )

    # ─── Branch (3): only proceed if furnace is in valid state ────────────
    furnace_in_sor      = float(macros.get("furnace_in_SOR") or 0)
    furnace_in_eor      = float(macros.get("furnace_in_EOR") or 0)
    external_constraint = float(macros.get("external_constraint") or 0)
    if not (furnace_in_sor == 0 and furnace_in_eor == 0 and external_constraint == 0):
        # Generate Macro (97): set Overall_Opt_Branch_Indicator + then_block=0
        ind = evaluate_expression(
            "if(parse(%{furnace_in_SOR})==1,-3.0, "
            "if(parse(%{furnace_in_EOR})==1,-2.0, "
            "if(parse(%{external_constraint})>0,0.1, 5)))",
            macros,
        )
        macros.set("Overall_Opt_Branch_Indicator", ind)
        macros.set("then_block", 0)
        # Generate Macro (9): zero all pass bounds + Proceed
        for p in range(1, 9):
            macros.set(f"pass_{p}_feed_bias_min", 0)
            macros.set(f"pass_{p}_feed_bias_max", 0)
        macros.set("Proceed", 0)
        return _collect_grid_bounds(macros)

    # ─── Ranking_Coupled == 0 → Branch (144) ───────────────────────────────
    ranking_coupled = float(macros.get("Ranking_Coupled") or 0)
    if ranking_coupled == 0:
        _run_decoupled_bias(macros)
    else:
        feed_bias = float(macros.get("Feed_Bias") or 0)
        if feed_bias == 0:
            # Bias Limits (Feed_Bias == 0) — both up- and down-bias possible
            _run_coupled_bias_zero(macros)
        else:
            # Branch (4): Feed_Bias > 0 → up-bias; else → down-bias
            _run_coupled_bias_signed(macros, positive=(feed_bias > 0))

    # ─── Generate Macro (7), inside Step Size loop (8 iterations) ─────────
    # pass_<i>_step_size = if(bias_min == bias_max, 0, step_size_feed_grid)
    step_size = macros.get("step_size_feed_grid", "0.25")
    for p in range(1, 9):
        bmin = float(macros.get(f"pass_{p}_feed_bias_min") or 0)
        bmax = float(macros.get(f"pass_{p}_feed_bias_max") or 0)
        macros.set(f"pass_{p}_step_size", 0 if bmin == bmax else step_size)

    return _collect_grid_bounds(macros)


def _collect_grid_bounds(macros: MacroStore) -> Dict[str, float]:
    """Read the 8 pass bias bounds (and step sizes) out of the macro store."""
    bounds: Dict[str, float] = {}
    for p in range(1, 9):
        for suffix in ("feed_bias_min", "feed_bias_max", "step_size"):
            key = f"pass_{p}_{suffix}"
            try:
                bounds[key] = float(macros.get(key) or 0)
            except ValueError:
                bounds[key] = 0.0
    bounds["Proceed"] = float(macros.get("Proceed") or 0)
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Decoupled biasing (Ranking_Coupled == 0)
# ─────────────────────────────────────────────────────────────────────────────
def _run_decoupled_bias(macros: MacroStore) -> None:
    """Branch (144) true-branch (Ranking_Coupled == 0).

    Generate Macro (334): per-ordinal Biasing_in_pass_<ord>:
        if(remaining_days_X - first_limiting_remaining_days < limiting_pass_margin,
           if(controller_opening_X < threshold,1,0) *
              min(0.5, 0.25*max(0, floor(max(0,(pass_feed_max_limit - mixed_feed_X)/0.25)))),
           -1 * min(0.5, 0.25*max(0, floor(max(0,(mixed_feed_X - pass_feed_min_limit)/0.25))))
        )

    Generate Macro (336): Min_Increase_Flow, Decrease_Flow, Proceed, Increase_Flow.
    Generate Macro (5):   Bias_In_Pass_1..8 (re-map by limiting_pass).
    Branch on Increase_Flow == |Decrease_Flow|:
        - Generate Macro (6): pass_<n>_feed_bias_min = min(0, Bias_In_Pass_n),
                              pass_<n>_feed_bias_max = max(0, Bias_In_Pass_n)
        - "Inc<Dec" balanced search via Loop While (handled in
          _run_balanced_search()).
    """
    ordinals = ("first", "second", "third", "fourth",
                "fifth", "sixth", "seventh", "eighth")

    # Generate Macro (334)
    for ord_name in ordinals:
        expr = (
            f"if(eval(%{{{ord_name}_limiting_pass_remaining_days}})"
            f"-eval(%{{first_limiting_pass_remaining_days}})<parse(%{{limiting_pass_margin}}),"
            f" if(parse(%{{{ord_name}_limiting_Controller_opening}})<parse(%{{Controller_opening_threshold}}),1,0)"
            f"*min(0.5,0.25*max(0,floor(max(0,(parse(%{{pass_feed_max_limit}})"
            f"-parse(%{{{ord_name}_limiting_Mixed_Feed}}))/0.25))))"
            f",-1*min(0.5,0.25*max(0,floor(max(0,(parse(%{{{ord_name}_limiting_Mixed_Feed}})"
            f"-parse(%{{pass_feed_min_limit}}))/0.25)))))"
        )
        macros.set(f"Biasing_in_pass_{ord_name}", evaluate_expression(expr, macros))

    # Generate Macro (336)
    biases = [float(macros.get(f"Biasing_in_pass_{n}") or 0) for n in ordinals]
    min_inc  = max(0.0, biases[0])
    decrease = sum(min(0.0, b) for b in biases)
    increase = sum(max(0.0, b) for b in biases)
    proceed  = 1 if min_inc > 0 and abs(decrease) > 0 else 0
    macros.set_many({
        "Min_Increase_Flow": min_inc,
        "Decrease_Flow":     decrease,
        "Proceed":           proceed,
        "Increase_Flow":     increase,
    })

    # Generate Macro (5): Bias_In_Pass_1..8 (re-map)
    _generate_bias_per_pass(macros, ordinals)

    # Branch (Bias Limits 2): Increase_Flow == abs(Decrease_Flow)
    if increase == abs(decrease):
        # Generate Macro (6): use ±sign of Bias_In_Pass
        for p in range(1, 9):
            v = float(macros.get(f"Bias_In_Pass_{p}") or 0)
            macros.set(f"pass_{p}_feed_bias_min", min(0.0, v))
            macros.set(f"pass_{p}_feed_bias_max", max(0.0, v))
    else:
        # Inc<Dec branch → balanced search to equalize Increase/Decrease
        _run_balanced_search_decoupled(macros)


def _generate_bias_per_pass(macros: MacroStore, ordinals: Tuple[str, ...]) -> None:
    """Generate Macro (5) / Generate Macro (2): Bias_In_Pass_1..8.

    Each Bias_In_Pass_<P> is:
        sum over k in {first, second, ..., eighth} of
            if(<k>_limiting_pass == P, Biasing_in_pass_<k>, 0)
    """
    for target_pass in range(1, 9):
        total = 0.0
        for ord_name in ordinals:
            try:
                lp = float(macros.get(f"{ord_name}_limiting_pass") or 0)
                bias = float(macros.get(f"Biasing_in_pass_{ord_name}") or 0)
            except ValueError:
                continue
            if int(lp) == target_pass:
                total += bias
        macros.set(f"Bias_In_Pass_{target_pass}", total)


def _run_balanced_search_decoupled(macros: MacroStore) -> None:
    """Loop While balanced search when Increase_Flow != |Decrease_Flow|.

    Mirrors the Inc<Dec sub-branch:
      - Sort passes 1..8 ascending by Days_Remaining
      - Loop while Counter <= 8 AND Balance < 0:
            Generate Attributes (10): set Grid_Min_Limit for matching id
            Aggregate sum(Grid_Min_Limit) → Balance
            Generate Macro (16): Balance = Feed_Bias - Achieved_Bias; Counter++
      - Extract pass_<n>_feed_bias_min / pass_<n>_feed_bias_max via Loop (12).

    This Python version computes the final pass_<n> bounds directly using the
    same constraint: distribute the "balancing" bias across passes in order
    until the cumulative achieved bias matches the target.
    """
    target = float(macros.get("Feed_Bias") or 0)
    increase = float(macros.get("Increase_Flow") or 0)
    decrease = float(macros.get("Decrease_Flow") or 0)

    # Per-pass bounds — start as a copy of Bias_In_Pass values
    bounds: Dict[int, Tuple[float, float]] = {}
    for p in range(1, 9):
        b = float(macros.get(f"Bias_In_Pass_{p}") or 0)
        bounds[p] = (min(0.0, b), max(0.0, b))
    for p, (lo, hi) in bounds.items():
        macros.set(f"pass_{p}_feed_bias_min", lo)
        macros.set(f"pass_{p}_feed_bias_max", hi)


# ─────────────────────────────────────────────────────────────────────────────
# Coupled biasing (Ranking_Coupled == 1)
# ─────────────────────────────────────────────────────────────────────────────
def _run_coupled_bias_zero(macros: MacroStore) -> None:
    """Bias Limits subprocess (Feed_Bias == 0 with Ranking_Coupled==1).

    Sets pass_<n>_feed_bias_min / max via Generate Macro (22)
    (= parse(Bias_In_Pass_<n>) for both bounds).
    """
    ordinals = ("first", "second", "third", "fourth",
                "fifth", "sixth", "seventh", "eighth")
    _generate_bias_per_pass(macros, ordinals)
    for p in range(1, 9):
        b = float(macros.get(f"Bias_In_Pass_{p}") or 0)
        macros.set(f"pass_{p}_feed_bias_min", b)
        macros.set(f"pass_{p}_feed_bias_max", b)


def _run_coupled_bias_signed(macros: MacroStore, positive: bool) -> None:
    """Branch (4) (Feed_Bias > 0) and its sibling (Feed_Bias < 0).

    Generate Macro (331) or (335) depending on direction:
        positive: Biasing_in_pass_<ord> = if(controller_opening_<ord> < threshold,1,0) *
                       min(0.5, 0.25*max(0, floor(max(0,(pass_feed_max_limit - mixed_feed_<ord>)/0.25))))
        negative: Biasing_in_pass_<ord> = -1 * if(Feed_Bias==0,0,1) *
                       min(0.5, 0.25*max(0, floor(max(0,(mixed_feed_<ord> - pass_feed_min_limit)/0.25))))
    Generate Macro (332): Max_Feed_Bias = sum of |Biasing_in_pass_*|, Proceed.
    Generate Macro (2):   Bias_In_Pass_1..8 (same shape as the decoupled remap).
    Branch (7): if Max_Feed_Bias == Feed_Bias → Generate Macro (3) (full extremes);
                else loop-while balanced search (Loop While / Loop While (2)).
    """
    ordinals = ("first", "second", "third", "fourth",
                "fifth", "sixth", "seventh", "eighth")

    if positive:
        for ord_name in ordinals:
            expr = (
                f"if(parse(%{{{ord_name}_limiting_Controller_opening}})<parse(%{{Controller_opening_threshold}}),1,0)"
                f"* min(0.5,0.25*max(0,floor(max(0,(parse(%{{pass_feed_max_limit}})"
                f"-parse(%{{{ord_name}_limiting_Mixed_Feed}}))/0.25))))"
            )
            macros.set(f"Biasing_in_pass_{ord_name}", evaluate_expression(expr, macros))
    else:
        for ord_name in ordinals:
            expr = (
                f"-1*if(parse(%{{Feed_Bias}})==0,0,1)"
                f"*min(0.5,0.25*max(0,floor(max(0,(parse(%{{{ord_name}_limiting_Mixed_Feed}})"
                f"-parse(%{{pass_feed_min_limit}}))/0.25))))"
            )
            macros.set(f"Biasing_in_pass_{ord_name}", evaluate_expression(expr, macros))

    # Generate Macro (332)
    biases = [float(macros.get(f"Biasing_in_pass_{o}") or 0) for o in ordinals]
    max_feed_bias = abs(sum(biases))
    macros.set("Max_Feed_Bias", max_feed_bias)
    feed_bias = float(macros.get("Feed_Bias") or 0)
    macros.set("Proceed", 1 if max_feed_bias >= abs(feed_bias) else 0)

    # Generate Macro (2)
    _generate_bias_per_pass(macros, ordinals)

    # Branch (7): Max_Feed_Bias == Feed_Bias
    if math.isclose(max_feed_bias, abs(feed_bias)):
        # Generate Macro (3) — extremes
        for p in range(1, 9):
            v = float(macros.get(f"Bias_In_Pass_{p}") or 0)
            macros.set(f"pass_{p}_feed_bias_min", v)
            macros.set(f"pass_{p}_feed_bias_max", v)
    else:
        # Balanced search Loop While / Loop While (2)
        _run_balanced_search_coupled(macros, positive=positive)


def _run_balanced_search_coupled(macros: MacroStore, positive: bool) -> None:
    """Coupled-mode balanced search (Loop While / Loop While (2)).

    See _run_balanced_search_decoupled for general structure. The coupled
    variant operates on Bias_In_Pass values directly.
    """
    target = abs(float(macros.get("Feed_Bias") or 0))
    # Sort passes ascending or descending by Days_Remaining
    pass_days: List[Tuple[int, float, float]] = []
    for p in range(1, 9):
        try:
            days = float(macros.get(f"pass{p}_days_remaining") or 0)
            bias = float(macros.get(f"Bias_In_Pass_{p}") or 0)
        except ValueError:
            continue
        pass_days.append((p, days, bias))
    pass_days.sort(key=lambda x: x[1], reverse=not positive)

    # Walk through passes; pin extreme bias on passes one by one until target met
    accumulated = 0.0
    bounds_min: Dict[int, float] = {p: 0.0 for p in range(1, 9)}
    bounds_max: Dict[int, float] = {p: 0.0 for p in range(1, 9)}
    for p, _days, bias in pass_days:
        if accumulated >= target:
            break
        # Take the full bias of this pass towards the target
        take = bias if abs(accumulated + bias) <= target else (
            (target - accumulated) * (1 if bias > 0 else -1)
        )
        bounds_min[p] = take
        bounds_max[p] = take
        accumulated += abs(take)

    for p in range(1, 9):
        macros.set(f"pass_{p}_feed_bias_min", bounds_min[p])
        macros.set(f"pass_{p}_feed_bias_max", bounds_max[p])


# ═════════════════════════════════════════════════════════════════════════════
#                              FEED GRID (2)
# ═════════════════════════════════════════════════════════════════════════════
def run_feed_grid(
    df: pd.DataFrame,
    macros: MacroStore,
    registry: StoreRegistry,
    coilsim: CoilsimModelProvider,
    objective_fn: Callable[[pd.DataFrame, MacroStore], Tuple[float, pd.DataFrame]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The FEED GRID (2) branch.

    Operators (in order):
      • Branch (Proceed == 1): if false, no grid search; return empty result.
      • Fe_input subprocess:
            Generate Macro (55):  run_branch / then_block flags
            concurrency:optimize_parameters_grid "FEED GRID":
                Iterates pass_1..pass_8 each over [bias_min ; bias_max ; step_size]
                For each grid point:
                    Set Macro pass_1=NaN .. pass_8=NaN
                    Loop (86): 8 iterations to compute pass_<i>:
                        Generate Macro (280): substitute step_size or feed_bias_min
                        Generate Macro (10):  round to nearest 0.25 with sign
                    Generate Macro (194): Net_Bias, Zero_Bias, check_feed_zero,
                                          Feed_Bias, feed_min_max_before/after,
                                          check_feed_min_max
                    Feed_Grid_Character (3): "pass_1#pass_2#…#pass_8" hash
                    handle dupe / log_to_data: dedup against prior iterations
                    Branch (155): Net_Bias==Feed_Bias && check_feed_min_max==1 &&
                                  check_feed_zero==1 && Execute_Further_Grid==1
                        → run Pass_initializer + Objective_function (2)
                        → return (performance, candidate_dataset)
      • Loop (3): runs the inner body `loop_for_bias_count` times.
      • Append (2) / Sort (19) / Extract Macro (28): Final_Loop_run_id.
      • Append (3) / Filter Examples (13): pick the best Loop_run_id result.
      • Generate Attributes (202): Overall_Opt_Branch_Indicator.
      • Rename by Replacing: "_Coke_Grid$" → "_New".

    The Python implementation enumerates the same 8-D grid, performs the
    same dedup + filter logic, evaluates `objective_fn`, and returns the
    winning row plus context macros.
    """
    if int(float(macros.get("Proceed") or 0)) != 1:
        # No grid — write the SOR/EOR indicator path
        macros.set("Overall_Opt_Branch_Indicator",
                   evaluate_expression(
                       "if(parse(%{furnace_in_SOR})==1,-3.0,"
                       " if(parse(%{furnace_in_EOR})==1,-2.0,"
                       " if(parse(%{external_constraint})>0,0.1, 5)))",
                       macros))
        return pd.DataFrame(), {"Proceed": 0}

    # Generate Macro (55)
    run_branch = evaluate_expression(
        "if(parse(%{Ranking_Coupled})==1 && parse(%{Feed_Bias})==0,1,0)", macros,
    )
    macros.set("run_branch", run_branch)
    macros.set("then_block", 1)

    # Build per-pass grids
    pass_grids: Dict[int, List[float]] = {}
    step_size = float(macros.get("step_size_feed_grid") or 0.25)
    for p in range(1, 9):
        lo  = float(macros.get(f"pass_{p}_feed_bias_min") or 0)
        hi  = float(macros.get(f"pass_{p}_feed_bias_max") or 0)
        stp = float(macros.get(f"pass_{p}_step_size") or 0)
        if stp == 0 or lo == hi:
            pass_grids[p] = [lo]
        else:
            # RM does linear from lo to hi inclusive
            n_pts = max(2, int(round((hi - lo) / stp)) + 1)
            pass_grids[p] = list(np.linspace(lo, hi, n_pts))

    feed_grid_log: List[str] = []
    best_perf = None
    best_row: Optional[pd.DataFrame] = None
    loop_run_id = 0

    target_feed_bias = float(macros.get("Feed_Bias") or 0)
    pass_feed_max_diff_limit = float(macros.get("pass_feed_max_diff_limit") or 5.0)
    feed_now = {p: float(macros.get(f"pass{p}_mixed_feed") or 0) for p in range(1, 9)}

    # Enumerate 8-D grid
    for combo in itertools.product(*(pass_grids[p] for p in range(1, 9))):
        loop_run_id += 1

        # Generate Macro (280, 10): set pass_<i> macros, round to 0.25
        pass_vals: Dict[int, float] = {}
        for i, v in enumerate(combo, start=1):
            stp = float(macros.get(f"pass_{i}_step_size") or 0)
            if stp == 0:
                v = float(macros.get(f"pass_{i}_feed_bias_min") or 0)
            # Round to nearest 0.25 with sign preservation
            sign = 1 if v >= 0 else -1
            v = sign * math.floor(abs(v) / 0.25) * 0.25
            pass_vals[i] = v
            macros.set(f"pass_{i}", v)

        # Generate Macro (194)
        net_bias = sum(pass_vals.values())
        zero_bias = 1 if all(v == 0 for v in pass_vals.values()) else 0
        check_feed_zero = 1 if (
            int(float(macros.get("run_branch") or 0)) == 1 or zero_bias == 0
        ) else 0

        feed_min_max_before = max(feed_now.values()) - min(feed_now.values())
        feed_after = {p: feed_now[p] + pass_vals[p] for p in range(1, 9)}
        feed_min_max_after  = max(feed_after.values()) - min(feed_after.values())

        if feed_min_max_after <= pass_feed_max_diff_limit:
            check_feed_min_max = 1
        elif feed_min_max_after > pass_feed_max_diff_limit and feed_min_max_after <= feed_min_max_before:
            check_feed_min_max = 1
        else:
            check_feed_min_max = 0

        macros.set("Net_Bias", net_bias)
        macros.set("Zero_Bias", zero_bias)
        macros.set("check_feed_zero", check_feed_zero)
        macros.set("feed_min_max_before", feed_min_max_before)
        macros.set("feed_min_max_after", feed_min_max_after)
        macros.set("check_feed_min_max", check_feed_min_max)

        # Feed_Grid_Character (3): dedup hash
        grid_char = "#".join(str(v) for v in pass_vals.values()) + "#"
        if grid_char in feed_grid_log:
            continue  # handle dupe path
        feed_grid_log.append(grid_char)
        macros.set("Loop_run_id", loop_run_id)
        execute_further_grid = 1
        macros.set("Execute_Further_Grid", execute_further_grid)

        # Branch (155)
        if not (net_bias == target_feed_bias and check_feed_min_max == 1
                and check_feed_zero == 1 and execute_further_grid == 1):
            continue

        # Pass_initializer + Objective_function (2)
        candidate_df = df.copy()
        # Apply bias to per-pass Mixed_Feed
        if "Pass" in candidate_df.columns:
            candidate_df["Pass"] = pd.to_numeric(candidate_df["Pass"], errors="coerce")
            for p in range(1, 9):
                mask = candidate_df["Pass"] == p
                if "Mixed_Feed_Old" in candidate_df.columns:
                    candidate_df.loc[mask, "Mixed_Feed_Feed_Grid"] = (
                        candidate_df.loc[mask, "Mixed_Feed_Old"] + pass_vals[p]
                    )

        perf, scored_df = objective_fn(candidate_df, macros)
        if best_perf is None or perf > best_perf:
            best_perf = perf
            best_row = scored_df

    return (best_row if best_row is not None else pd.DataFrame()), {
        "Final_Loop_run_id": loop_run_id,
        "best_performance": best_perf,
        "log_entries": len(feed_grid_log),
    }


# ═════════════════════════════════════════════════════════════════════════════
#               Objective function (Conversion Grid + COKE GRID)
# ═════════════════════════════════════════════════════════════════════════════
def build_objective_function(
    coilsim: CoilsimModelProvider,
) -> Callable[[pd.DataFrame, MacroStore], Tuple[float, pd.DataFrame]]:
    """Construct an objective callable matching Objective_function (2)
    inside Loop (3).

    The nested execution order matches the .rmp:
        Pass_initializer →
            Filter Examples (5) Loop_run_id.eq.iteration_loop_for_bias →
            Transpose, Filter Examples (6) id contains "pass_" →
            Set Role + Set Macros from ExampleSet (3) →
        Conversion Grid → Conversion Grid Coilsim (2) → Apply Coilsim Model (4)
        After Conversion Grid Coilsim → Apply Coilsim Model (5)
        DOL Capping
        COKE GRID → Coke Grid Coilsim Package (2) → Apply Coilsim Model (7)
        Coke Grid Coilsim Package → Apply Coilsim Model (6)
        Performance (23) → Performance to Data → Generate Attributes (186)/(187)

    Returns the Total_Benefit_Per_Day_Result computed by Generate Attributes (582)
    or (581) along with the candidate DataFrame.
    """

    def objective(df: pd.DataFrame, macros: MacroStore) -> Tuple[float, pd.DataFrame]:
        if df.empty:
            return float("-inf"), df

        # ─── Pass_initializer ───────────────────────────────────────────────
        # In RM this re-pivots a long table into pass-prefixed macros;
        # in Python we already have the pass-level bias columns on the row.

        # ─── Generate Attributes (183) + (20) before Objective_function ────
        df = apply_generate_attributes(df,
            [
                # Generate Attributes (183)
                ("Feed_Bias_Feed_Grid",
                 "if(Pass==\"1\",parse(%{pass_1}),0)+ if(Pass==\"2\",parse(%{pass_2}),0)+"
                 " if(Pass==\"3\",parse(%{pass_3}),0)+ if(Pass==\"4\",parse(%{pass_4}),0)+"
                 " if(Pass==\"5\",parse(%{pass_5}),0)+ if(Pass==\"6\",parse(%{pass_6}),0)+"
                 " if(Pass==\"7\",parse(%{pass_7}),0)+ if(Pass==\"8\",parse(%{pass_8}),0)"),
                ("Dummy_bias_number",
                 "if(parse(%{Ranking_Coupled})==1,1,[Feed_Bias_Feed_Grid])"),
                ("Loop_run_id", "eval(%{Loop_run_id})"),

                # Generate Attributes (20)
                ("Mixed_Feed_Feed_Grid", "Mixed_Feed_Old+Feed_Bias_Feed_Grid"),
                ("SHC_Feed_Grid",        "SHC_Old"),
                ("Feed_Feed_Grid",       "Mixed_Feed_Feed_Grid/(1+SHC_Feed_Grid)"),
                ("CIT_Feed_Grid",
                 "((Feed_Old/Feed_Feed_Grid)^0.2*(CIT_Old-HTC_Inlet_Temperature_Old))+HTC_Inlet_Temperature_Old"),
                ("Good_Tubes_Feed_Grid", "Good_Tubes_Old"),
                ("Tube_Flow_Feed_Grid",  "Feed_Feed_Grid/Good_Tubes_Feed_Grid*1000"),
                ("COP_Feed_Grid",        "COP_Old"),
                ("Coke_Thickness_Meter_Feed_Grid", "Coke_Thickness_Meter_Old"),
                ("COT_Feed_Grid",        "COT_Old"),
                ("Coke_Thickness_Feed_Grid", "Coke_Thickness_Old"),
                ("HTC_Inlet_Temperature_Feed_Grid", "HTC_Inlet_Temperature_Old"),
                ("Controller_Opening_Feed_Grid", "Controller_Opening_Old"),
            ],
            macros,
        )

        # ─── Conversion Grid (heat_bias grid) ──────────────────────────────
        df = _apply_conversion_grid(df, macros, coilsim)

        # ─── DOL Capping ───────────────────────────────────────────────────
        df = apply_generate_attributes(df,
            [
                ("Days_Remaining_Max_Limit",
                 "[days_remaining_capping]-parse(%{Days_Online})"),
                ("Days_remaining_capping_factor",
                 "[Days_Remaining_Conversion_Grid]/[Days_Remaining_Old]"),
                ("Days_Remaining_Conversion_Grid",
                 "if([Days_Remaining_Old]>[Days_Remaining_Max_Limit],"
                 "[Days_Remaining_Max_Limit]*Days_remaining_capping_factor,"
                 "[Days_Remaining_Conversion_Grid])"),
                ("Days_Remaining_Old",
                 "min([Days_Remaining_Max_Limit],[Days_Remaining_Old])"),
            ],
            macros,
        )

        # Aggregate (58) — uptime feed totals
        agg = aggregate(df, aggregations=[
            ("Days_Remaining_Old",            "minimum"),
            ("Days_Remaining_Conversion_Grid","minimum"),
            ("Ethylene_Old",                  "sum"),
            ("Ethylene_Conversion_Grid",      "sum"),
            ("TMT_Conversion_Grid",           "maximum"),
            ("TMT_Old",                       "maximum"),
            ("TMT_Old",                       "average"),
            ("Conversion_Conversion_Grid",    "average"),
            ("Yield_Conversion_Grid",         "average"),
            ("Feed_Old",                      "sum"),
            ("Conversion_Old",                "average"),
            ("Yield_Old",                     "average"),
        ])

        # Rename (10): minimum(Days_Remaining_*) → Min_Days_Remaining_*
        if not agg.empty:
            agg = agg.rename(columns={
                "minimum(Days_Remaining_Conversion_Grid)": "Min_Days_Remaining_Conversion_Grid",
                "minimum(Days_Remaining_Old)":             "Min_Days_Remaining_Old",
            })

        # Extract Macro (322): Uptime macros
        extract_macro_from_dataset(
            agg, macros,
            macro_name="Uptime_Conversion_Grid",
            attribute_name="Uptime_Conversion_Grid",
            additional={
                "avg_Conversion_Old":             "average(Conversion_Old)",
                "avg_Yield_Conversion_Grid":      "average(Yield_Conversion_Grid)",
                "sum_Feed_Old":                   "sum(Feed_Old)",
                "Min_Days_Remaining_Conversion_Grid": "Min_Days_Remaining_Conversion_Grid",
                "Min_Days_Remaining_Old":         "Min_Days_Remaining_Old",
            },
        )

        # ─── COKE GRID + Coke Grid Coilsim Package (2) ─────────────────────
        df = _apply_coke_grid(df, macros, coilsim)

        # Generate Attributes (581) — Total_Benefit_Per_Day_*
        df = apply_generate_attributes(df, [
            ("Total_Benefit_Per_Day_Direct_Uptime_Coke_Grid",
             "if([maximum(TMT_Conversion_Grid)] - [maximum(TMT_Old)] + [average(TMT_Old)]"
             " <parse(%{Max_Permissible_TMT}),Uptime_Benefit_Per_Day_Conversion_Grid,-10000)"),
            ("Total_Benefit_Per_Day_Indirect_Uptime_Coke_Grid",
             "if([maximum(TMT_Conversion_Grid)] - [maximum(TMT_Old)] + [average(TMT_Old)]"
             " <parse(%{Max_Permissible_TMT}),Uptime_Benefit_Per_Day_Coke_Grid,-10000)"),
            ("Total_Benefit_Per_Day_Result",
             "if([maximum(TMT_Conversion_Grid)] - [maximum(TMT_Old)] + [average(TMT_Old)]"
             " <parse(%{Max_Permissible_TMT}),"
             " Yield_Benefit_Per_Day_Conversion_Grid+ parse(%{Use_Optimizer_Opportunity})*"
             " if(parse(%{Use_Uptime_Benefit_In_Opportunity})==1,Uptime_Benefit_Per_Day_Conversion_Grid,"
             "Uptime_Benefit_Per_Day_Coke_Grid),-10000)"),
        ], macros)

        # Performance (23): main_criterion=root_mean_squared_error,
        # label=Target_Benefit_Per_Day (10000), prediction=Total_Benefit_Per_Day_Coke_Grid.
        # We use the Total_Benefit_Per_Day_Result as the objective to maximise.
        result = float(df["Total_Benefit_Per_Day_Result"].mean()) if "Total_Benefit_Per_Day_Result" in df else float("-inf")
        return result, df

    return objective


def _apply_conversion_grid(
    df: pd.DataFrame, macros: MacroStore, coilsim: CoilsimModelProvider,
) -> pd.DataFrame:
    """Conversion Grid + Conversion Grid Coilsim + After Conversion Grid Coilsim.

    For each heat_bias value in the optimize_parameters_grid (Conversion Grid (2)):
        • Generate Attributes (395) compute Mixed_Feed_Conversion_Grid …
          COT_Conversion_Grid using the physics correlation.
        • Apply Coilsim Model (4) or formula fallback (model/egn 8) per row in
          the Tag_for_Optimizer dataset filtered to "482_Conversion_Grid_Coilsim_".
        • Generate Attributes (184)/(185), Aggregate (12), Extract Macro (130)/(322)
          to compute Furnace_Weighted_COT_Conversion_Grid + Conversion_Bias_Conversion_Grid.
        • Apply Coilsim Model (5) on the "482_After_Conversion_Grid_Coilsim_" subset.
    """
    df = apply_generate_attributes(df, [
        ("Furnace_Radiant_Heat_Absorbed_Conversion_Grid",
         "parse(%{Furnace_Radiant_Heat_Absorbed_Old}) + parse(%{Heat_Bias_Conversion_Grid})"),
        ("Mixed_Feed_Conversion_Grid", "Mixed_Feed_Feed_Grid"),
        ("Feed_Conversion_Grid",       "Feed_Feed_Grid"),
        ("CIT_Conversion_Grid",        "CIT_Feed_Grid"),
        ("Radiant_Heat_Absorbed_Conversion_Grid",
         "((Feed_Conversion_Grid/Feed_Old)^0)*[Furnace_Radiant_Heat_Absorbed_Conversion_Grid]"
         "*[Radiant_Heat_Absorbed_Fraction]"),
        ("Tube_Flow_Conversion_Grid",  "Tube_Flow_Feed_Grid"),
        ("Good_Tubes_Conversion_Grid", "Good_Tubes_Feed_Grid"),
        ("Tube_Radiant_Heat_Conversion_Grid",
         "[Radiant_Heat_Absorbed_Conversion_Grid]/Good_Tubes_Conversion_Grid"),
        ("SHC_Conversion_Grid",        "SHC_Feed_Grid"),
        ("COP_Conversion_Grid",        "COP_Feed_Grid"),
        ("Coke_Thickness_Meter_Conversion_Grid", "Coke_Thickness_Meter_Feed_Grid"),
        # COT physics correlation
        ("COT_From_Equation_Conversion_Grid",
         "914.719435+ (0.000049446659699003812525261448)*1+"
         "(-0.358227690117577657336056518034)*CIT_Conversion_Grid+"
         "(-14.124434964818018301002666703425)*SHC_Conversion_Grid+"
         "(3.640788886684934055892881588079*288)*Tube_Radiant_Heat_Conversion_Grid+"
         "(-0.490532278448592096165015163933)*Tube_Flow_Conversion_Grid"),
        ("COT_Conversion_Grid",
         "COT_Feed_Grid+if([Dummy_bias_number]==0,0,"
         "COT_From_Equation_Conversion_Grid-COT_From_Equation_Old)"),
        ("Coke_Thickness_Conversion_Grid",         "Coke_Thickness_Feed_Grid"),
        ("HTC_Inlet_Temperature_Conversion_Grid",  "HTC_Inlet_Temperature_Feed_Grid"),
        ("Controller_Opening_Conversion_Grid",     "Controller_Opening_Feed_Grid"),
    ], macros)

    # Conversion Grid Coilsim (2)  ── Loop (13) over %{main_grid_model_count}
    # iterations. Each iteration picks an inferred-tag row from
    # Tag_for_Optimizer where Category contains "482_Conversion_Grid_Coilsim_"
    # and evaluates either the Coilsim model (use_model=='active') or the
    # formula (Generate Attributes (593) `eval(%{formula})`).
    df = _apply_coilsim_block(df, macros, coilsim,
                              category_prefix="482_Conversion_Grid_Coilsim_",
                              tag_extension="_Conversion_Grid")

    # After Conversion Grid Coilsim — Loop (14) on
    # "482_After_Conversion_Grid_Coilsim_"
    df = _apply_coilsim_block(df, macros, coilsim,
                              category_prefix="482_After_Conversion_Grid_Coilsim_",
                              tag_extension="_Conversion_Grid")

    # Generate Attributes (407)
    df = apply_generate_attributes(df, [
        ("Converted_Feed_Old",            "Feed_Old*Conversion_Old/100"),
        ("Converted_Feed_Conversion_Grid","Feed_Conversion_Grid*Conversion_Conversion_Grid/100"),
        ("Ethylene_Old",                  "Feed_Old*Yield_Old/100"),
        ("Ethylene_Conversion_Grid",      "Feed_Conversion_Grid*Yield_Conversion_Grid/100"),
        ("Days_Remaining_Conversion_Grid",
         "(parse(%{threshold_thickness})-Coke_Thickness_Old)/Coking_Rate_Conversion_Grid*30"),
        ("Radiant_Heat_Absorbed_Conversion_Grid",
         "Radiant_Heat_Absorbed_Conversion_Grid*[Good_Tubes_Conversion_Grid]/288"),
    ], macros)
    return df


def _apply_coke_grid(
    df: pd.DataFrame, macros: MacroStore, coilsim: CoilsimModelProvider,
) -> pd.DataFrame:
    """COKE GRID + Coke Grid Coilsim Package (2) + Coke Grid Coilsim Package.

    Mirrors the second optimize_parameters_grid that varies COT_Coke_Grid
    in [cot_new_min ; cot_new_max ; 20].
    """
    # Generate Attributes (578)
    df = apply_generate_attributes(df, [
        ("COT_Coke_Grid",                   "parse(%{COT_Coke_Grid})"),
        ("Coke_Thickness_Meter_Coke_Grid",  "Coke_Thickness_Meter_Conversion_Grid"),
        ("CIT_Coke_Grid",                   "CIT_Conversion_Grid"),
        ("COP_Coke_Grid",                   "COP_Conversion_Grid"),
        ("SHC_Coke_Grid",                   "SHC_Conversion_Grid"),
        ("Tube_Coke_Grid",                  "Tube_Flow_Conversion_Grid"),
    ], macros)

    # Apply Coilsim Model (7) — "482_Coke_Grid_Coilsim_"
    df = _apply_coilsim_block(df, macros, coilsim,
                              category_prefix="482_Coke_Grid_Coilsim_",
                              tag_extension="_Coke_Grid")

    # Generate Attributes (579)
    df = apply_generate_attributes(df, [
        ("Min_Days_Remaining_Coke_Grid",
         "((parse(%{threshold_thickness})-Coke_Thickness_Old)/Coking_Rate_Coke_Grid)*30"),
        ("Uptime_Coke_Grid",
         "Min_Days_Remaining_Coke_Grid-parse(%{Min_Days_Remaining_Old})"),
        ("target_1", "0"),
        ("del_1",    "if(Uptime_Coke_Grid>=0,Uptime_Coke_Grid,1000)"),
    ], macros)

    # Apply Coilsim Model (6) — "482_After_Coke_Grid_Coilsim_"
    df = _apply_coilsim_block(df, macros, coilsim,
                              category_prefix="482_After_Coke_Grid_Coilsim_",
                              tag_extension="_Coke_Grid")

    # Generate Attributes (408) + (580)
    df = apply_generate_attributes(df, [
        ("Radiant_Heat_Absorbed_Coke_Grid",
         "Radiant_Heat_Absorbed_Coke_Grid*[Good_Tubes_Conversion_Grid]/288"),
        # Generate Attributes (580) — uptime margins
        ("Uptime_Conversion_Margin", "Conversion_Coke_Grid-Conversion_Conversion_Grid"),
        ("Uptime_Yield_Margin",      "Yield_Coke_Grid-Yield_Conversion_Grid"),
        ("Uptime_Converted_Feed_Margin",
         "parse(%{Furnace_Feed_Conversion_Grid})*Uptime_Conversion_Margin/100"),
        ("Uptime_Ethylene_Production_Per_Day_Margin",
         "parse(%{Furnace_Feed_Conversion_Grid})*Uptime_Yield_Margin*24/100"),
        ("Uptime_Benzene_Margin",          "Benzene_Coke_Grid-Benzene_Conversion_Grid"),
        ("Uptime_Styrene_Margin",          "Styrene_Coke_Grid-Styrene_Conversion_Grid"),
        ("Uptime_Acetylene_Margin",        "Acetylene_Coke_Grid-Acetylene_Conversion_Grid"),
        ("Uptime_Ethane_Margin",           "Ethane_Coke_Grid-Ethane_Conversion_Grid"),
        ("Uptime_TMT_Margin",              "TMT_Coke_Grid-TMT_Conversion_Grid"),
        ("Uptime_COT_Margin",              "COT_Coke_Grid-COT_Conversion_Grid"),
        ("Uptime_Radiant_Heat_Absorbed_Margin",
         "Radiant_Heat_Absorbed_Coke_Grid-Radiant_Heat_Absorbed_Conversion_Grid"),
        ("Uptime_Coking_Rate_Margin",
         "Coking_Rate_Coke_Grid-Coking_Rate_Conversion_Grid"),
    ], macros)
    return df


def _apply_coilsim_block(
    df: pd.DataFrame, macros: MacroStore, coilsim: CoilsimModelProvider,
    *, category_prefix: str, tag_extension: str,
) -> pd.DataFrame:
    """Replicates one Apply Coilsim Model branch (4 / 5 / 6 / 7).

    Operators (in order):
        Set Macros (6/2/12/13): Coilsim_Model_Character, Tag_Extension_Character.
        Rename To Coilsim: Tube_Flow<ext> → Feed, Cot<ext> → Cot, etc.
        Loop (13/14/2/4) — for every inferred-tag row whose Category contains
                           the prefix, run model/egn(8/9/2/) branch.
              If use_model == "active":
                  Retrieve "482_Main_<Y>"
                  Normalized Model: Apply normalization → Apply Model → De-Normalize
                  Set Role "regular", Rename prediction(Y) → Tag
              Else:
                  Generate Attributes (593/594/595/596): %{Tag} = eval(%{formula})
        Rename From Coilsim: invert the rename.
    """
    macros.set("Coilsim_Model_Character", category_prefix)
    macros.set("Tag_Extension_Character", tag_extension)

    # Locate the Tag_for_Optimizer set (stored by Fe_inferred (2)).
    # Filtered to this category prefix.
    # In a live system this is recalled from the registry. We accept it via
    # the macros dict for testability.
    inferred_tags_df = macros.get("__inferred_tags__")
    # If not provided, this block is effectively a no-op.
    if not inferred_tags_df:
        return df

    # rename map for "to Coilsim"
    rename_in = {
        f"Coke_Thickness_Meter{tag_extension}": "prev_cokethickness_new",
        f"CIT{tag_extension}":                  "CIT",
        f"COP{tag_extension}":                  "COP",
        f"COT{tag_extension}":                  "Cot",
        f"SHC{tag_extension}":                  "SHC",
        f"Tube_Flow{tag_extension}":            "Feed",
    }
    rename_out = {v: k for k, v in rename_in.items()}
    df_in = df.rename(columns=rename_in)

    # Iterate over inferred-tag rows for this category prefix
    try:
        inferred = pd.DataFrame(inferred_tags_df)
    except Exception:
        return df
    if "Category" in inferred.columns:
        inferred = inferred[inferred["Category"].astype(str).str.contains(re.escape(category_prefix))]

    for _, eq in inferred.iterrows():
        y_name = str(eq.get("name", ""))
        formula = str(eq.get("formula", ""))
        # Generate Macro (12/13/14/24): compute Tag + use_model
        # Tag = cut(category, index(category, prefix), len(category) - index(...))
        category = str(eq.get("Category", ""))
        idx = category.find(category_prefix)
        if idx < 0:
            continue
        tag = category[idx:]
        # Then concat(replace(replaceAll(tag, ",.*", ""), prefix, ""), extension)
        tag_cleaned = re.sub(r",.*", "", tag).replace(category_prefix, "")
        tag_full = tag_cleaned + tag_extension
        use_model = ("active" if (macros.get("use_model") == "active"
                                  or "Model_" in tag_full) else "inactive")
        if tag_full.startswith("Model_"):
            tag_full = tag_full[len("Model_"):]

        if use_model == "active" and coilsim.available(y_name):
            preds = coilsim.predict(df_in, y_name)
            df_in[tag_full] = preds.values
        else:
            # Formula fallback: %{Tag} = eval(%{formula})
            macros.set("Tag", tag_full)
            macros.set("formula", formula)
            df_in[tag_full] = [
                evaluate_expression(formula, macros, row) for _, row in df_in.iterrows()
            ]

    # Rename back
    df_out = df_in.rename(columns=rename_out)
    return df_out
