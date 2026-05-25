"""
act_opt.py
==========

Python replica of the *ACT=OPT* and *COUPLED_CCP_USED_CHECK* subprocesses
from the United (2) optimizer block.

ACT=OPT is run after GRID_AND_BIASING. Its job is to take whatever the
grid produced, derive a parallel set of "_New" attributes from the
existing "_Old" attributes (when the grid did not run — i.e. then_block == 0),
and then build the final output schema that uses `_actual` / `_furnace_optimum`
suffix instead of `_Old` / `_New`.

Operators in source order:

  Branch (9): condition `parse(%{then_block}) == 1`
      true  → grid ran successfully → pass-through to Subprocess (43).
      false → no grid → build *_New copies from *_Old:
              Recall (125) → Extract Macro (184) Total_Benefit_Per_Day_Coke_Grid
              Recall (12)  → Loop Attributes (7) over `.*_Old`
                  Generate Macro (34): name=<attr>, name_new=replace(name,"_Old","_New")
                  Generate Attributes (23): %{name_new} = #{name}
              Set Macro (13): Total_Benefit_Per_Day_Coke_Grid = -10000

  Subprocess (43):
      Generate Attributes (211): COT_New-=20, COT_Old-=20, days-remaining capping
      Remember (82)
      Aggregate (25) group_by=Furnace: sum(Mixed_Feed_Old), avg(SHC_New), sum(Mixed_Feed_New)
      Sort (24) Days_Remaining_New asc, Sort (42) Days_Remaining_Old asc
      Mass Cot Calc (4) — Generate Attributes (228) + Aggregate (26) + Extract Macro (174)
                         + Generate Attributes (238)
      Extract Macro (186): Furnace_Weighted_COT + 3 add. macros
      Limiting_pass (4) / Limiting_pass_new (4): Extract Macro for Pass
      Branch (10): then_block==1
          Generate Macro (100): min_days_diff_check logic
          Generate Macro (101): min_days_diff_check = 0 (else)
      Rename (11): sum(Mixed_Feed_New) → Furnace_Mixed_Feed_New, etc.
      Generate Attributes (239): Limiting_pass, Overall_Cot, Overall_conversion macros
      Numerical to Real (8), Remember (83)
      Branch (45): %{min_days_diff_check} == 1
          Subprocess (4): the BIG final rename + output formatting subprocess
              Multiply (20): fork stream
              Select Attributes (6) + Rename (917):
                  TMT_New → TMT_furnace_optimum, TMT_Old → TMT_actual, etc.
              Subprocess (149): Loop Values (10) over Pass → rename to Fur#_Pass#_<attr>
              Merge Attributes (6)
              Filter Example Range (7) + Multiply (49) + Select Attributes (247) + (341)
              Join (12) + Remove Duplicates + Parse Numbers + Numerical to Real
              "overall_average tags (4)" recall + Select Attributes (248)
              Generate Attributes (244): Overall_SHC_Ratio_new = SHC_New
              Merge Attributes (211), Numerical to Real (54)
              Rename (102): Furnace_Mixed_Feed_Old → Total_Mixed_Feed_optimizer_actual, …
              Loop Attributes (13): rename remaining attrs to Fur#_<attr>
              Branch (18) on then_block==1:
                  benefit_tags (5): build benefit-side outputs
                  benefit_tags (6): fallback with Create ExampleSet (11) (zeros)
              Merge final outputs
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rm_runtime import (
    MacroStore,
    StoreRegistry,
    aggregate,
    apply_generate_attributes,
    evaluate_expression,
    extract_macro_from_dataset,
)


# ═════════════════════════════════════════════════════════════════════════════
#                                  ACT=OPT
# ═════════════════════════════════════════════════════════════════════════════
def run_act_opt(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> pd.DataFrame:
    """Top-level ACT=OPT subprocess.

    Returns a DataFrame in the *actual* / *furnace_optimum* wide schema.
    """
    # ──────────────────────────────────────────────────────────────────────
    # Branch (9): parse(%{then_block}) == 1
    # ──────────────────────────────────────────────────────────────────────
    then_block = float(macros.get("then_block") or 0)
    if then_block == 1:
        # True branch — grid ran, output already has *_New columns.
        df_with_new = df
    else:
        # False branch — copy *_Old into *_New.
        df_with_new = _copy_old_to_new(df, macros, registry)

    # ──────────────────────────────────────────────────────────────────────
    # Subprocess (43)
    # ──────────────────────────────────────────────────────────────────────
    return _run_subprocess_43(df_with_new, macros, registry)


# ─────────────────────────────────────────────────────────────────────────────
# Branch (9) — false branch: derive *_New columns from *_Old when no grid ran
# ─────────────────────────────────────────────────────────────────────────────
def _copy_old_to_new(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> pd.DataFrame:
    """Replicates Branch (9) false-branch:

      Recall (125):              Recall "Total_Benefit_Per_Day_Coke_Grid"  (best-effort)
      Extract Macro (184):       macro=Total_Benefit_Per_Day_Coke_Grid
      Recall (12):               Recall the data stream
      Loop Attributes (7) ∀ .*_Old:
          Generate Macro (34):   name=loop_attr, name_new=replace(name,"_Old","_New")
          Generate Attributes (23): create %{name_new} = #{name}
      Set Macro (13):            Total_Benefit_Per_Day_Coke_Grid = -10000
    """
    # Recall (125) + Extract Macro (184)
    if registry.has("Total_Benefit_Per_Day_Coke_Grid"):
        tb_df = registry.recall("Total_Benefit_Per_Day_Coke_Grid")
        if isinstance(tb_df, pd.DataFrame) and not tb_df.empty:
            extract_macro_from_dataset(
                tb_df, macros,
                macro_name="Total_Benefit_Per_Day_Coke_Grid",
                attribute_name="Total_Benefit_Per_Day_Coke_Grid",
            )

    # Loop Attributes (7) over `.*_Old`
    df = df.copy()
    for col in [c for c in df.columns if c.endswith("_Old")]:
        new_col = col.replace("_Old", "_New")
        df[new_col] = df[col]

    # Set Macro (13)
    macros.set("Total_Benefit_Per_Day_Coke_Grid", -10000)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess (43) — heart of ACT=OPT
# ─────────────────────────────────────────────────────────────────────────────
def _run_subprocess_43(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> pd.DataFrame:
    """Subprocess (43): aggregate the per-pass *_Old/*_New table into per-furnace
    output rows, compute pass-level limiting metrics, and hand off to
    Subprocess (4) for the final actual/furnace_optimum rename.
    """
    # ──────────────────────────────────────────────────────────────────────
    # Generate Attributes (211): COT subtraction + days-remaining capping
    # ──────────────────────────────────────────────────────────────────────
    df = apply_generate_attributes(df,
        [
            ("COT_New", "[COT_New]-20"),
            ("COT_Old", "[COT_Old]-20"),
            ("Days_Remaining_Max_Limit",
             "[days_remaining_capping]-parse(%{Days_Online})"),
            ("Days_remaining_capping_factor",
             "[Days_Remaining_New]/[Days_Remaining_Old]"),
            ("Days_Remaining_New",
             "if([Days_Remaining_Old]>[Days_Remaining_Max_Limit],"
             "[Days_Remaining_Max_Limit]*Days_remaining_capping_factor,"
             "[Days_Remaining_New])"),
            ("Days_Remaining_Old",
             "min([Days_Remaining_Max_Limit],[Days_Remaining_Old])"),
        ],
        macros,
    )
    # Remember (82): store this intermediate
    registry.remember("act_opt_intermediate", df.copy())

    # ──────────────────────────────────────────────────────────────────────
    # Aggregate (25): group_by=Furnace; sum(Mixed_Feed_Old), avg(SHC_New),
    #                  sum(Mixed_Feed_New)
    # ──────────────────────────────────────────────────────────────────────
    agg25 = aggregate(df,
        aggregations=[
            ("Mixed_Feed_Old", "sum"),
            ("SHC_New",        "average"),
            ("Mixed_Feed_New", "sum"),
        ],
        group_by=["Furnace"] if "Furnace" in df.columns else None,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Sort (24): by Days_Remaining_New ascending
    # ──────────────────────────────────────────────────────────────────────
    if "Days_Remaining_New" in df.columns:
        sorted_new = df.sort_values("Days_Remaining_New", ascending=True).reset_index(drop=True)
    else:
        sorted_new = df.copy()

    # ──────────────────────────────────────────────────────────────────────
    # Sort (42): by Days_Remaining_Old ascending
    # ──────────────────────────────────────────────────────────────────────
    if "Days_Remaining_Old" in df.columns:
        sorted_old = df.sort_values("Days_Remaining_Old", ascending=True).reset_index(drop=True)
    else:
        sorted_old = df.copy()

    # ──────────────────────────────────────────────────────────────────────
    # Mass Cot Calc (4): per-furnace weighted COT/conversion for *_New
    # ──────────────────────────────────────────────────────────────────────
    df = _run_mass_cot_calc(df, macros)

    # ──────────────────────────────────────────────────────────────────────
    # Extract Macro (186): Furnace_Weighted_COT + cot_current_weighted_avg_new
    # ──────────────────────────────────────────────────────────────────────
    if "Furnace_Weighted_COT_Old" in df.columns:
        extract_macro_from_dataset(
            df, macros,
            macro_name="Furnace_Weighted_COT",
            attribute_name="Furnace_Weighted_COT_Old",
            additional={
                "cot_current_weighted_avg_new":      "cot_new_current_new",
                "conversion_current_weighted_avg_new":"conversion_new_current_new",
                "Furnace_Weighted_Conversion_Old":   "Furnace_Weighted_Conversion_Old",
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Limiting_pass (4)/Limiting_pass_new (4): Extract Macro Pass + min days
    # ──────────────────────────────────────────────────────────────────────
    if not sorted_old.empty:
        extract_macro_from_dataset(
            sorted_old, macros,
            macro_name="Limiting_pass",
            attribute_name="Pass",
            additional={"min_Days_remaining_old": "Days_Remaining_Old"},
        )
    if not sorted_new.empty:
        extract_macro_from_dataset(
            sorted_new, macros,
            macro_name="Limiting_pass_new",
            attribute_name="Pass",
            additional={"min_Days_remaining_new": "Days_Remaining_New"},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Branch (10): %{then_block} == 1 → Generate Macro (100); else (101)
    # ──────────────────────────────────────────────────────────────────────
    then_block = float(macros.get("then_block") or 0)
    if then_block == 1:
        # Generate Macro (100): full min_days_diff_check chain
        macros.set(
            "min_days_diff_check",
            evaluate_expression(
                "if(eval(%{min_Days_remaining_new})>=eval(%{min_Days_remaining_old})"
                "+eval(%{Days_remaining_min_margin}),1,0)",
                macros,
            ),
        )
        macros.set(
            "min_days_diff_check",
            evaluate_expression(
                "if(parse(%{Ranking_Coupled})==1,1,parse(%{min_days_diff_check}))",
                macros,
            ),
        )
        macros.set(
            "min_days_diff_check",
            evaluate_expression(
                "if(parse(%{Total_Benefit_Per_Day_Coke_Grid})==-1000 || "
                "parse(%{Total_Benefit_Per_Day_Coke_Grid})==-10000,0,"
                "parse(%{min_days_diff_check}))",
                macros,
            ),
        )
        macros.set(
            "min_days_diff_check",
            evaluate_expression(
                "if(parse(%{then_block})==0,0,%{min_days_diff_check})",
                macros,
            ),
        )
        macros.set("then_block", macros.get("min_days_diff_check"))
    else:
        # Generate Macro (101)
        macros.set("min_days_diff_check", 0)

    # ──────────────────────────────────────────────────────────────────────
    # Rename (11): sum(Mixed_Feed_New) → Furnace_Mixed_Feed_New, etc.
    # ──────────────────────────────────────────────────────────────────────
    if not agg25.empty:
        agg25 = agg25.rename(columns={
            "sum(Mixed_Feed_New)":   "Furnace_Mixed_Feed_New",
            "average(SHC_New)":      "SHC_New",
            "sum(Mixed_Feed_Old)":   "Furnace_Mixed_Feed_Old",
        })

    # ──────────────────────────────────────────────────────────────────────
    # Generate Attributes (239)
    # ──────────────────────────────────────────────────────────────────────
    agg25 = apply_generate_attributes(agg25,
        [
            ("Limiting_pass_new",       "eval(%{Limiting_pass_new})"),
            ("Limiting_pass",           "eval(%{Limiting_pass})"),
            ("Overall_Cot_old",         "eval(%{Furnace_Weighted_COT})"),
            ("Overall_Cot",             "eval(%{cot_current_weighted_avg_new})"),
            ("Overall_conversion_old",  "eval(%{Furnace_Weighted_Conversion_Old})"),
            ("Overall_conversion",      "eval(%{conversion_current_weighted_avg_new})"),
        ],
        macros,
    )

    # Numerical to Real (8) — no-op in pandas (already numeric)
    # Remember (83)
    registry.remember("act_opt_furnace_aggregate", agg25.copy())

    # ──────────────────────────────────────────────────────────────────────
    # Branch (45): %{min_days_diff_check} == 1 → Subprocess (4); else passthrough
    # ──────────────────────────────────────────────────────────────────────
    min_days_check = float(macros.get("min_days_diff_check") or 0)
    if min_days_check != 1:
        # Generate Macro (104) — emit indicator
        macros.set(
            "Overall_Opt_Branch_Indicator",
            evaluate_expression(
                "if(parse(%{Total_Benefit_Per_Day_Coke_Grid})<=-1000,2.1,2.0)",
                macros,
            ),
        )
        # Recall (14) + Loop Attributes (12) over `.*_Old` → produce *_New copies
        df = _loop_attrs_old_to_new(df, macros)
        # Generate Attributes (215): same as Subprocess (43) generate-attributes (211)
        # plus the Overall_Opt_Branch_Indicator setter
        df = apply_generate_attributes(df,
            [
                ("COT_New", "[COT_New]-20"),
                ("COT_Old", "[COT_Old]-20"),
                ("Days_Remaining_Max_Limit",
                 "[days_remaining_capping]-parse(%{Days_Online})"),
                ("Days_remaining_capping_factor",
                 "[Days_Remaining_New]/[Days_Remaining_Old]"),
                ("Days_Remaining_New",
                 "if([Days_Remaining_Old]>[Days_Remaining_Max_Limit],"
                 "[Days_Remaining_Max_Limit]*Days_remaining_capping_factor,"
                 "[Days_Remaining_New])"),
                ("Days_Remaining_Old",
                 "min([Days_Remaining_Max_Limit],[Days_Remaining_Old])"),
                ("Overall_Opt_Branch_Indicator",
                 "parse(%{Overall_Opt_Branch_Indicator})"),
                ("Overall_Opt_Branch_Indicator_new",
                 "[Overall_Opt_Branch_Indicator]"),
            ],
            macros,
        )
        return df

    # Subprocess (4) → final actual/furnace_optimum schema
    return _run_subprocess_4(df, agg25, macros, registry)


def _loop_attrs_old_to_new(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """Loop Attributes (12): generate <attr>_New for every <attr>_Old column."""
    df = df.copy()
    for col in [c for c in df.columns if c.endswith("_Old")]:
        new_col = col.replace("_Old", "_New")
        if new_col not in df.columns:
            df[new_col] = df[col]
    return df


def _run_mass_cot_calc(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """Mass Cot Calc (4) subprocess.

    Generate Attributes (228): sum_feed_and_cot_new, sum_feed_and_conversion_new.
    Aggregate (26):            sum the above plus Mixed_Feed_New, Feed_New.
    Extract Macro (174):       publish sum_feed_and_cot_new + 3 more macros.
    Generate Attributes (238): cot_new_current_new, conversion_new_current_new.
    """
    df = apply_generate_attributes(df,
        [
            ("sum_feed_and_cot_new",        "Mixed_Feed_New*COT_New"),
            ("sum_feed_and_conversion_new", "Feed_New*Conversion_New"),
        ],
        macros,
    )
    agg = aggregate(df,
        aggregations=[
            ("sum_feed_and_cot_new",        "sum"),
            ("Mixed_Feed_New",              "sum"),
            ("Feed_New",                    "sum"),
            ("sum_feed_and_conversion_new", "sum"),
        ],
    )
    extract_macro_from_dataset(
        agg, macros,
        macro_name="sum_feed_and_cot_new",
        attribute_name="sum(sum_feed_and_cot_new)",
        additional={
            "sum_Mixed_Feed_new":           "sum(Mixed_Feed_New)",
            "sum_Feed_new":                 "sum(Feed_New)",
            "sum_feed_and_conversion_new":  "sum(sum_feed_and_conversion_new)",
        },
    )
    df = apply_generate_attributes(df,
        [
            ("cot_new_current_new",
             "parse(%{sum_feed_and_cot_new})/parse(%{sum_Mixed_Feed_new})"),
            ("conversion_new_current_new",
             "parse(%{sum_feed_and_conversion_new})/parse(%{sum_Feed_new})"),
        ],
        macros,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess (4) — final actual/furnace_optimum schema
# ─────────────────────────────────────────────────────────────────────────────
_RENAME_OLD_NEW_TO_ACTUAL_OPTIMUM: Dict[str, str] = {
    "TMT_New": "TMT_furnace_optimum",
    "Yield_New": "Ethylene_Yield_furnace_optimum",
    "Radiant_Heat_Absorbed_New": "Heat_Absorbed_furnace_optimum",
    "Conversion_New": "Conversion_furnace_optimum",
    "Conversion_Old": "Conversion_actual",
    "Coking_Rate_New": "Coking_Rate_furnace_optimum",
    "Coking_Rate_Old": "Coking_Rate_actual",
    "TMT_Old": "TMT_actual",
    "Yield_Old": "Ethylene_Yield_actual",
    "Radiant_Heat_Absorbed_Old": "Heat_Absorbed_actual",
    "Coke_Thickness_New": "Coke_thickness_furnace_optimum",
    "Coke_Thickness_Old": "Coke_thickness_actual",
    "CIT_New": "CIT_furnace_optimum",
    "CIT_Old": "CIT_actual",
    "COT_New": "Cot_optimizer_furnace_optimum",
    "COT_Old": "Cot_optimizer_actual",
    "Days_Remaining_Old": "Days_remaining_actual",
    "Days_Remaining_New": "Days_remaining_furnace_optimum",
    "Styrene_Old": "Styrene_actual",
    "Styrene_New": "Styrene_furnace_optimum",
    "Acetylene_Old": "Acetylene_actual",
    "Acetylene_New": "Acetylene_furnace_optimum",
    "Ethane_Old": "Ethane_actual",
    "Ethane_New": "Ethane_furnace_optimum",
    "Mixed_Feed_Old": "Mixed_Feed_Flow_actual",
    "Mixed_Feed_New": "Mixed_Feed_Flow_furnace_optimum",
    "Benzene_Old": "Benzene_actual",
    "Benzene_New": "Benzene_furnace_optimum",
}

_RENAME_AGGREGATE_TO_ACTUAL_OPTIMUM: Dict[str, str] = {
    "Furnace_Mixed_Feed_Old": "Total_Mixed_Feed_optimizer_actual",
    "Furnace_Mixed_Feed_New": "Total_Mixed_Feed_optimizer_furnace_optimum",
    "Overall_SHC_Ratio_new":  "Overall_SHC_Ratio_optimizer_furnace_optimum",
    "SHC_New":                "Overall_SHC_Ratio_optimizer_actual",
    "Overall_Cot_old":        "Overall_Cot_optimizer_actual",
    "Overall_Cot":            "Overall_Cot_optimizer_furnace_optimum",
    "Overall_conversion_old": "Overall_conversion_optimizer_actual",
    "Overall_conversion":     "Overall_conversion_optimizer_furnace_optimum",
    "Limiting_pass":          "Limiting_pass_optimizer_actual",
    "Limiting_pass_new":      "Limiting_pass_optimizer_furnace_optimum",
    "Overall_Opt_Branch_Indicator":     "Overall_Opt_Branch_Indicator_actual",
    "Overall_Opt_Branch_Indicator_new": "Overall_Opt_Branch_Indicator_furnace_optimum",
}

_RENAME_BENEFIT_TAGS: Dict[str, str] = {
    "total_benefit_per_day_with_direct_uptime":     "Total_Benefit_Per_Day_With_Direct_Uptime_actual",
    "total_benefit_per_day_with_direct_uptime_new": "Total_Benefit_Per_Day_With_Direct_Uptime_furnace_optimum",
    "total_benefit_per_day_with_indirect_uptime":   "Total_Benefit_Per_Day_With_Indirect_Uptime_actual",
    "total_benefit_per_day_with_indirect_uptime_new":"Total_Benefit_Per_Day_With_Indirect_Uptime_furnace_optimum",
    "achieved_heat_bias":                           "achieved_heat_bias_actual",
    "achieved_heat_bias_new":                       "achieved_heat_bias_furnace_optimum",
    "net_uptime":                                   "Net_Uptime_actual",
    "net_uptime_new":                               "Net_uptime_furnace_optimum",
    "Uptime_Benefit_Per_Day_Conversion_Grid_Old":   "Uptime_Benefit_Per_Day_actual",
    "Uptime_Benefit_Per_Day_Conversion_Grid_New":   "Uptime_Benefit_Per_Day_furnace_optimum",
    "yield_benefit_per_day_old":                    "Yield_Benefit_Per_Day_actual",
    "yield_benefit_per_day_new":                    "Yield_Benefit_Per_Day_furnace_optimum",
    "extra_ethylene_produced_per_day_new":          "Extra_Ethylene_Produced_Per_Day_furnace_optimum",
    "extra_ethylene_produced_per_day_old":          "Extra_Ethylene_Produced_Per_Day_actual",
    "Total_Benefit_Per_Day_Result":                 "Total_Benefit_Per_Day_Result_actual",
    "Total_Benefit_Per_Day_Result_new":             "Total_Benefit_Per_Day_Result_furnace_optimum",
}


def _run_subprocess_4(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    macros: MacroStore,
    registry: StoreRegistry,
) -> pd.DataFrame:
    """Subprocess (4) — final actual/furnace_optimum schema construction.

    The .rmp wires this as a long sequence of multiplexed renames + merges.
    We replicate the logical effect: produce one output frame in which every
    operational tag carries `_actual` and `_furnace_optimum` suffixes, plus
    the furnace-level aggregates and the benefit-day metrics.
    """
    # Multiply (20) → two streams: (a) selected-and-renamed, (b) range-filtered.

    # Stream A: Select Attributes (6) keep the *_New/*_Old pairs we care about.
    pair_cols = [
        c for c in df.columns
        if c in _RENAME_OLD_NEW_TO_ACTUAL_OPTIMUM
    ]
    stream_a = df[pair_cols + [c for c in ("Furnace", "Pass", "Timestamp", "sub_model_id") if c in df.columns]].copy()
    # Rename (917)
    stream_a = stream_a.rename(columns=_RENAME_OLD_NEW_TO_ACTUAL_OPTIMUM)

    # Subprocess (149): Loop Values (10) over Pass — rename per-pass attrs to
    # Fur<#>_Pass<#>_<attr>. We do this only for the pass-wise output.
    stream_a = _rename_pass_attributes(stream_a, macros)

    # Stream B: aggregate side → rename to *_optimizer_actual / *_furnace_optimum
    stream_b = agg_df.copy()
    stream_b = stream_b.rename(columns=_RENAME_AGGREGATE_TO_ACTUAL_OPTIMUM)
    # Rename remaining aggregate columns with Fur<#>_<attr> prefix
    stream_b = _rename_furnace_prefix(stream_b, macros)

    # Benefit tags branch (Branch (18))
    then_block = float(macros.get("then_block") or 0)
    if then_block == 1:
        benefit_df = _build_benefit_tags(macros, registry)
    else:
        benefit_df = _build_benefit_tags_zero(macros)

    # Final merge of streams a + b + benefit_df by Timestamp / sub_model_id
    parts: List[pd.DataFrame] = []
    if not stream_a.empty:
        parts.append(stream_a)
    if not stream_b.empty:
        parts.append(stream_b)
    if benefit_df is not None and not benefit_df.empty:
        parts.append(benefit_df)
    if not parts:
        return pd.DataFrame()

    # Outer concat on common index columns
    return pd.concat(parts, axis=1).reset_index(drop=True)


def _rename_pass_attributes(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """Subprocess (149) Loop Values (10) over Pass:

    For each unique Pass value in the row, rename all non-key attributes to
    `Fur<Furnace_number>_Pass<Pass_number>_<attr>` (Rename by Replacing (49)
    with replace_what=(.+) and replace_by=%{prefix}_$1).
    """
    if "Pass" not in df.columns or "Furnace" not in df.columns:
        return df

    furnace_num = macros.get("Furnace_number", "0")
    out_rows = []
    for _, row in df.iterrows():
        pass_no = row["Pass"]
        prefix = f"Fur{furnace_num}_Pass{pass_no}"
        renamed = {}
        for col, val in row.items():
            if col in ("Timestamp", "sub_model_id", "Pass", "Furnace"):
                renamed[col] = val
            else:
                renamed[f"{prefix}_{col}"] = val
        out_rows.append(renamed)
    if not out_rows:
        return df.head(0)
    return pd.DataFrame(out_rows)


def _rename_furnace_prefix(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """Loop Attributes (13)/(14): rename remaining columns to Fur<#>_<col>."""
    furnace_num = macros.get("Furnace_number", "0")
    new_cols = {}
    for c in df.columns:
        if c in ("Timestamp", "sub_model_id", "Furnace"):
            continue
        new_cols[c] = f"Fur{furnace_num}_{c}"
    return df.rename(columns=new_cols)


def _build_benefit_tags(macros: MacroStore, registry: StoreRegistry) -> pd.DataFrame:
    """benefit_tags (5): Recall "overall_average tags (4)" + Set Role +
    Select Attributes (343) + Rename (103/104) + Generate Attributes (245) +
    Loop Attributes (14) → Merge Attributes (213/54)."""
    if not registry.has("overall_average_tags"):
        return pd.DataFrame()

    base = registry.recall("overall_average_tags").copy()
    # Generate Attributes (245) — populate the benefit columns
    base = apply_generate_attributes(base,
        [
            ("total_benefit_per_day_with_indirect_uptime_new",
             "[Total_Benefit_Per_Day_Indirect_Uptime_Coke_Grid]"),
            ("total_benefit_per_day_with_direct_uptime_new",
             "[Total_Benefit_Per_Day_Direct_Uptime_Coke_Grid]"),
            ("uptime_benefit_per_day_new",
             "[Uptime_Benefit_Per_Day_Conversion_Grid]"),
            ("yield_benefit_per_day_new",
             "[Yield_Benefit_Per_Day_Conversion_Grid]"),
            ("extra_ethylene_produced_per_day_new",
             "[Total_Benefit_Per_Day_Coke_Grid]"),
            ("net_uptime",      "Uptime_Conversion_Grid"),
            ("net_uptime_new",  "Uptime_Conversion_Grid"),
            ("Uptime_Benefit_Per_Day_Conversion_Grid_Old", "Uptime_Benefit_Per_Day_Conversion_Grid"),
            ("Uptime_Benefit_Per_Day_Conversion_Grid_New", "Uptime_Benefit_Per_Day_Conversion_Grid"),
            ("yield_benefit_per_day_old",
             "[Yield_Benefit_Per_Day_Conversion_Grid]"),
            ("extra_ethylene_produced_per_day_old",
             "[Total_Benefit_Per_Day_Coke_Grid]"),
            ("total_benefit_per_day_with_indirect_uptime",
             "[Total_Benefit_Per_Day_Indirect_Uptime_Coke_Grid]"),
            ("total_benefit_per_day_with_direct_uptime",
             "[Total_Benefit_Per_Day_Direct_Uptime_Coke_Grid]"),
            ("Total_Benefit_Per_Day_Result_new",
             "[Total_Benefit_Per_Day_Result]"),
        ],
        macros,
    )
    return base.rename(columns=_RENAME_BENEFIT_TAGS)


def _build_benefit_tags_zero(macros: MacroStore) -> pd.DataFrame:
    """benefit_tags (6): Create ExampleSet (11) with zero values.

    CSV:
        Net_Uptime,Uptime_Benefit_Per_Day,Yield_Benefit_Per_Day,
        Extra_Ethylene_Produced_Per_Day,Total_Benefit_Per_Day_With_Direct_Uptime,
        Total_Benefit_Per_Day_With_Indirect_Uptime,achieved_heat_bias,
        Total_Benefit_Per_Day_Result
        0,0,0,0,0,0,100,0

    + Generate Attributes (246) + Rename (21) + Rename by Replacing (3)
    """
    zeros = pd.DataFrame([{
        "Net_Uptime": 0.0,
        "Uptime_Benefit_Per_Day": 0.0,
        "Yield_Benefit_Per_Day": 0.0,
        "Extra_Ethylene_Produced_Per_Day": 0.0,
        "Total_Benefit_Per_Day_With_Direct_Uptime": 0.0,
        "Total_Benefit_Per_Day_With_Indirect_Uptime": 0.0,
        "achieved_heat_bias": 100.0,
        "Total_Benefit_Per_Day_Result": 0.0,
    }])
    zeros = apply_generate_attributes(zeros,
        [
            ("Net_Uptime_new",                       "[Net_Uptime]"),
            ("Uptime_Benefit_Per_Day_new",           "[Uptime_Benefit_Per_Day]"),
            ("Yield_Benefit_Per_Day_new",            "[Yield_Benefit_Per_Day]"),
            ("Extra_Ethylene_Produced_Per_Day_new",  "[Extra_Ethylene_Produced_Per_Day]"),
            ("Total_Benefit_Per_Day_With_Direct_Uptime_new",
             "[Total_Benefit_Per_Day_With_Direct_Uptime]"),
            ("Total_Benefit_Per_Day_With_Indirect_Uptime_new",
             "[Total_Benefit_Per_Day_With_Indirect_Uptime]"),
            ("Overall_Opt_Branch_Indicator",     "parse(%{Overall_Opt_Branch_Indicator})"),
            ("Overall_Opt_Branch_Indicator_new", "[Overall_Opt_Branch_Indicator]"),
            ("achieved_heat_bias_new",           "[achieved_heat_bias]"),
            ("Total_Benefit_Per_Day_Result_new", "[Total_Benefit_Per_Day_Result]"),
        ],
        macros,
    )
    # Rename (21)
    zeros = zeros.rename(columns={
        "Total_Benefit_Per_Day_With_Indirect_Uptime":     "Total_Benefit_Per_Day_With_Indirect_Uptime_actual",
        "Total_Benefit_Per_Day_With_Indirect_Uptime_new": "Total_Benefit_Per_Day_With_Indirect_Uptime_furnace_optimum",
        "Total_Benefit_Per_Day_With_Direct_Uptime":       "Total_Benefit_Per_Day_With_Direct_Uptime_actual",
        "Total_Benefit_Per_Day_With_Direct_Uptime_new":   "Total_Benefit_Per_Day_With_Direct_Uptime_furnace_optimum",
        "Net_Uptime":                       "Net_Uptime_actual",
        "Net_Uptime_new":                   "Net_Uptime_furnace_optimum",
        "Uptime_Benefit_Per_Day":           "Uptime_Benefit_Per_Day_actual",
        "Uptime_Benefit_Per_Day_new":       "Uptime_Benefit_Per_Day_furnace_optimum",
        "Yield_Benefit_Per_Day":            "Yield_Benefit_Per_Day_actual",
        "Yield_Benefit_Per_Day_new":        "Yield_Benefit_Per_Day_furnace_optimum",
        "Extra_Ethylene_Produced_Per_Day":  "Extra_Ethylene_Produced_Per_Day_actual",
        "Extra_Ethylene_Produced_Per_Day_new": "Extra_Ethylene_Produced_Per_Day_furnace_optimum",
        "Overall_Opt_Branch_Indicator":     "Overall_Opt_Branch_Indicator_actual",
        "Overall_Opt_Branch_Indicator_new": "Overall_Opt_Branch_Indicator_furnace_optimum",
        "achieved_heat_bias":               "achieved_heat_bias_actual",
        "achieved_heat_bias_new":           "achieved_heat_bias_furnace_optimum",
        "Total_Benefit_Per_Day_Result":     "Total_Benefit_Per_Day_Result_actual",
        "Total_Benefit_Per_Day_Result_new": "Total_Benefit_Per_Day_Result_furnace_optimum",
    })
    # Rename by Replacing (3): replace_what=(.+), replace_by=Fur<#>_$1
    furnace_num = macros.get("Furnace_number", "0")
    zeros.columns = [f"Fur{furnace_num}_{c}" for c in zeros.columns]
    return zeros


# ═════════════════════════════════════════════════════════════════════════════
#                       COUPLED_CCP_USED_CHECK
# ═════════════════════════════════════════════════════════════════════════════
def run_coupled_ccp_used_check(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> pd.DataFrame:
    """COUPLED_CCP_USED_CHECK subprocess (line 5329 of the .rmp).

    Operators:
      - Extract Macro (189):       macro=sub_model_id, attr=sub_model_id
      - Select Attributes (344):   exclude sub_model_id
      - Set Role (66):             Timestamp → id
      - De-Pivot (6):              long format; regex `^(?!.*\\bTimestamp\\b).*`
                                   → "tag" column
      - Generate Attributes (622): sub_model_id = %{sub_model_id}
      - Branch (23): condition_type=macro_defined,
                     value=post_optimizer_transformation_utd
            true:  Branch (33) expression "%{post_optimizer_transformation_utd}=='active'"
                   then: Generate Attributes (623): coupled_mode = %{Ranking_Coupled}
            false: pass-through
    """
    if df.empty:
        return df

    # Extract Macro (189)
    extract_macro_from_dataset(
        df, macros,
        macro_name="sub_model_id",
        attribute_name="sub_model_id",
    )

    # Select Attributes (344) — exclude sub_model_id
    excl_cols = [c for c in df.columns if c != "sub_model_id"]
    work = df[excl_cols].copy()

    # Set Role (66) Timestamp → id (no pandas semantic change; tracked logically)

    # De-Pivot (6): unpivot all non-Timestamp columns into long format
    id_cols = ["Timestamp"] if "Timestamp" in work.columns else []
    value_cols = [c for c in work.columns if c not in id_cols]
    long = work.melt(id_vars=id_cols, value_vars=value_cols,
                     var_name="tag", value_name="value")

    # Generate Attributes (622): sub_model_id = %{sub_model_id}
    long["sub_model_id"] = macros.get("sub_model_id")

    # Branch (23): macro_defined post_optimizer_transformation_utd
    if not macros.has("post_optimizer_transformation_utd"):
        return long

    # Branch (33): %{post_optimizer_transformation_utd}=="active"
    flag = macros.get("post_optimizer_transformation_utd")
    if flag == "active":
        # Generate Attributes (623): coupled_mode = %{Ranking_Coupled}
        long["coupled_mode"] = macros.get("Ranking_Coupled")
    return long
