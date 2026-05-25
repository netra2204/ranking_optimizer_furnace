"""
united_optimizer.py
===================

Python replica of the RapidMiner sub-process ``United (2)`` exported from
``united-optimizer-main-block.rmp``. The replica follows the same operator
execution order as the .rmp.

The .rmp top-level structure (and therefore this module's outline) is::

    United (2)
    ├── Fe_inferred (2)
    ├── Remove Duplicates (16)
    ├── Handle Timestamp
    ├── Re-formatting
    │   ├── Generate Attributes (25)         (Pass, Furnace derivation)
    │   ├── Filtering_Data
    │   │   ├── Gen_Pass_Values
    │   │   ├── Filter Examples (30)
    │   │   ├── Renaming_Systemwise
    │   │   ├── Renaming_Passwise
    │   │   └── Subprocess (27)
    │   ├── Renaming as per optimizer (2)    (Create ExampleSet → Remember "Rename_data")
    │   ├── Join (74)                        (Left-join "Rename_data")
    │   ├── Generate Attributes (28)         (new_name fallback)
    │   ├── Pivot (10)                       (pivot to wide format)
    │   ├── Rename by Replacing (7)          (strip "average(value)_")
    │   ├── Numerical to Polynominal (6)     (Pass|Furnace → polynominal)
    │   ├── Set Role (10)                    (Timestamp → id)
    │   ├── Generate Attributes              (COP_Old +1, COT_Old +20)
    │   └── ccp_status_reform (4)
    ├── Generate tags (2)                    (derived tags: Tube_Flow, COT_From_Equation…)
    ├── Date to Nominal (7)
    ├── Join (2)                             (with "constraint" recall)
    └── Main_Process
        ├── Extract Macro (88)               (Furnace_Status, ccp_status_curr, total_optimizer_run_check)
        ├── Bias Constants
        ├── Overall_Calcs
        ├── GRID_AND_BIASING                 (the heart of the optimizer)
        │   ├── Inferred calculations
        │   │   ├── Feed Biasing             (8-pass limiting calc + Bias_In_Pass_1..8 + grid bounds)
        │   │   └── …
        │   ├── FEED GRID (2)
        │   │   ├── Fe_input                 (concurrency:optimize_parameters_grid)
        │   │   └── Loop (3)
        │   ├── Generate Attributes (619)
        │   └── Generate Macro (25)
        ├── ACT=OPT
        │   ├── Branch (9)
        │   └── Subprocess (43)
        └── COUPLED_CCP_USED_CHECK
    ├── Remove Duplicates (37)
    ├── Parse Numbers (34)
    └── Numerical to Real (57)

Each top-level subprocess corresponds to one function below, named ``_run_<name>``.
Within each function, RapidMiner operators appear in source order; before the
Python code for each operator there is a comment block carrying the operator
number, the operator name, and its responsibility.

Author: Optimizer Replica Project
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
#                     Per-furnace sub_model_id ↔ furnace_number map
# ═════════════════════════════════════════════════════════════════════════════
# Derived from Generate Attributes (25) inside Re-formatting:
#   sub_model_id==488 → 1, 489 → 2, …, 496 → 9
SUB_MODEL_TO_FURNACE: Dict[int, int] = {
    488: 1, 489: 2, 490: 3, 491: 4, 492: 5,
    493: 6, 494: 7, 495: 8, 496: 9,
}

# Generate Attributes (36) inside ccp_status_reform (4):
#   entity_id==3796 → 488, …, 3804 → 496
ENTITY_TO_SUB_MODEL: Dict[int, int] = {
    3796: 488, 3797: 489, 3798: 490, 3799: 491, 3800: 492,
    3801: 493, 3802: 494, 3803: 495, 3804: 496,
}


# ═════════════════════════════════════════════════════════════════════════════
#                                  Fe_inferred (2)
# ═════════════════════════════════════════════════════════════════════════════
#
# Purpose: builds the "Tag_for_Optimizer" reference table that downstream
# blocks read via Recall. Combines tag, tag_child and tag_details, then
# pulls out two flavours of inferred tag formulas:
#   * those whose `pipeline_location` contains "Optimizer_Input_Name_Macros"
#     (their formula strings become *macros* via Set Macros from ExampleSet (2))
#   * those whose `pipeline_location` contains any of the four Coilsim grid
#     stages (their formulas stay as data rows for later Loop iteration).
#
def _run_fe_inferred(
    tag: pd.DataFrame,
    tag_child: pd.DataFrame,
    tag_details: pd.DataFrame,
    macros: MacroStore,
    registry: StoreRegistry,
) -> pd.DataFrame:
    """Replicates the Fe_inferred (2) subprocess.

    Inputs (recalled in the .rmp):
      - tag          : Recall "tag"
      - tag_child    : Recall "tag_child"
      - tag_details  : Recall "tag_details"

    Outputs:
      - Stores `Tag_for_Optimizer` ExampleSet in `registry`
      - Adds *macros* to `macros` whose names equal the tag names and whose
        values are the inferred formula strings (used later as
        %{TMT_Name}, %{Mixed_Feed_Name}, etc.).
    """
    # ──────────────────────────────────────────────────────────────────────
    # tag (132), tag_child (57), tag_details (110): Recall operators —
    # values come in from the caller, no Python work needed.
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Join (10): INNER join on tag.tag_id == tag_child.parent_tag_id
    # ──────────────────────────────────────────────────────────────────────
    joined = tag.merge(
        tag_child, left_on="tag_id", right_on="parent_tag_id", how="inner"
    )

    # ──────────────────────────────────────────────────────────────────────
    # Join (98): INNER join on `name` with tag_details
    # ──────────────────────────────────────────────────────────────────────
    joined = joined.merge(tag_details, on="name", how="inner")

    # ──────────────────────────────────────────────────────────────────────
    # Filter Examples (105): keep rows where `type` contains "inferred"
    # ──────────────────────────────────────────────────────────────────────
    if "type" in joined.columns:
        joined = joined[joined["type"].astype(str).str.contains("inferred")]

    # Multiply (12) just duplicates output to two consumers; not modelled here.

    # ──────────────────────────────────────────────────────────────────────
    # Path A — Filter Examples (111): pipeline_location contains
    #          "Optimizer_Input_Name_Macros"
    # ──────────────────────────────────────────────────────────────────────
    path_a = joined.copy()
    if "pipeline_location" in path_a.columns:
        path_a = path_a[path_a["pipeline_location"].astype(str)
                          .str.contains("Optimizer_Input_Name_Macros")]

    # Sort (16): tag_order ascending
    if "tag_order" in path_a.columns:
        path_a = path_a.sort_values("tag_order", ascending=True)

    # Select Attributes (239): keep only ['name', 'formula']
    path_a = path_a[["name", "formula"]] if {"name", "formula"} <= set(path_a.columns) else path_a

    # Set Macros from ExampleSet (2): for each row, macro[name] = formula
    for _, row in path_a.iterrows():
        macros.set(str(row["name"]), row.get("formula", ""))

    # ──────────────────────────────────────────────────────────────────────
    # Path B — Filter Examples (112): pipeline_location contains any of
    #          the four Coilsim stage prefixes (OR logic).
    # ──────────────────────────────────────────────────────────────────────
    path_b = joined.copy()
    coilsim_prefixes = [
        "482_After_Coke_Grid_Coilsim_",
        "482_Coke_Grid_Coilsim_",
        "482_Conversion_Grid_Coilsim_",
        "482_After_Conversion_Grid_Coilsim_",
    ]
    if "pipeline_location" in path_b.columns:
        mask = pd.Series(False, index=path_b.index)
        for p in coilsim_prefixes:
            mask = mask | path_b["pipeline_location"].astype(str).str.contains(p)
        path_b = path_b[mask]

    # Sort (17): tag_order ascending
    if "tag_order" in path_b.columns:
        path_b = path_b.sort_values("tag_order", ascending=True)

    # Select Attributes (240): keep ['formula', 'name', 'pipeline_location']
    keep = [c for c in ("formula", "name", "pipeline_location") if c in path_b.columns]
    path_b = path_b[keep]

    # Rename (9): pipeline_location → Category
    path_b = path_b.rename(columns={"pipeline_location": "Category"})

    # Remember (3): store as "Tag_for_Optimizer"
    registry.remember("Tag_for_Optimizer", path_b)
    return path_b


# ═════════════════════════════════════════════════════════════════════════════
#                              Handle Timestamp
# ═════════════════════════════════════════════════════════════════════════════
def _run_handle_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Run the Handle Timestamp subprocess.

    Contains a single operator:

    ─ Filter Examples (19): keep rows where Timestamp is not missing.
    """
    if "Timestamp" not in df.columns:
        return df
    mask = df["Timestamp"].notna() & (df["Timestamp"].astype(str).str.strip() != "")
    return df[mask].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#                              Re-formatting (top)
# ═════════════════════════════════════════════════════════════════════════════
def _run_reformatting(
    df: pd.DataFrame,
    macros: MacroStore,
    registry: StoreRegistry,
) -> pd.DataFrame:
    """The Re-formatting subprocess.

    The work breaks down into Generate Attributes (25), then a nested
    Filtering_Data subprocess (Gen_Pass_Values + Renaming_Systemwise +
    Renaming_Passwise + Subprocess (27)), then Renaming as per optimizer (2),
    Join (74), Generate Attributes (28), Pivot (10), Rename by Replacing (7),
    Numerical to Polynominal (6), Set Role (10), Generate Attributes,
    and ccp_status_reform (4).
    """
    # ──────────────────────────────────────────────────────────────────────
    # Generate Attributes (25): adds `Pass` and `Furnace`
    #
    #   Pass = if(contains(name,"Pass1"),1, … if(contains(name,"Pass8"),8,
    #                                            MISSING_NUMERIC))
    #   Furnace = if(parse(%{sub_model_id})==488,1, … 496→9, MISSING_NUMERIC)
    # ──────────────────────────────────────────────────────────────────────
    def _derive_pass(name: Any) -> float:
        if not isinstance(name, str):
            return float("nan")
        for i in range(1, 9):
            if f"Pass{i}" in name:
                return float(i)
        return float("nan")

    sub_model_id = float(macros.get("sub_model_id") or 0)
    furnace_num  = float(SUB_MODEL_TO_FURNACE.get(int(sub_model_id), 0))
    df = df.copy()
    if "name" in df.columns:
        df["Pass"] = df["name"].apply(_derive_pass)
    df["Furnace"] = furnace_num

    # Multiply (40) simply forks the stream into two paths; both go into
    # Filtering_Data.

    # ──────────────────────────────────────────────────────────────────────
    # Filtering_Data subprocess
    # ──────────────────────────────────────────────────────────────────────
    df_filtered = _run_filtering_data(df, macros, registry)

    # ──────────────────────────────────────────────────────────────────────
    # Renaming as per optimizer (2)
    #
    # Create ExampleSet: builds a (name,new_name) map using the macros set
    # in Fe_inferred (2). After macro substitution, this gives an explicit
    # rename table from "Furnace<n>_<tag>" → canonical "<tag>_Old".
    # ──────────────────────────────────────────────────────────────────────
    rename_pairs: List[Tuple[str, str]] = []
    for macro_name, new_name in _RENAME_TARGETS:
        original = macros.get(macro_name)
        if original:
            rename_pairs.append((original, new_name))
    rename_df = pd.DataFrame(rename_pairs, columns=["name", "new_name"])
    registry.remember("Rename_data", rename_df)

    # ──────────────────────────────────────────────────────────────────────
    # Recall (74) + Join (74): LEFT join on `name`
    # ──────────────────────────────────────────────────────────────────────
    joined = df_filtered.merge(rename_df, on="name", how="left")

    # ──────────────────────────────────────────────────────────────────────
    # Generate Attributes (28): new_name = if(missing(new_name), name, new_name)
    # ──────────────────────────────────────────────────────────────────────
    if "new_name" in joined.columns:
        joined["new_name"] = joined.apply(
            lambda r: r["name"] if pd.isna(r["new_name"]) else r["new_name"], axis=1,
        )
    else:
        joined["new_name"] = joined["name"]

    # ──────────────────────────────────────────────────────────────────────
    # Pivot (10): group_by=Furnace|Pass|Timestamp|sub_model_id,
    #             column_grouping_attribute=new_name, aggregation=average(value)
    # ──────────────────────────────────────────────────────────────────────
    pivot_cols = ["Furnace", "Pass", "Timestamp", "sub_model_id"]
    pivot_cols = [c for c in pivot_cols if c in joined.columns]
    pivoted = (
        joined.pivot_table(
            index=pivot_cols,
            columns="new_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    # ──────────────────────────────────────────────────────────────────────
    # Rename by Replacing (7): strip the "average(value)_" prefix introduced
    # by the Pivot.
    # ──────────────────────────────────────────────────────────────────────
    pivoted.columns = [
        re.sub(r"^average\(value\)_", "", str(c)) for c in pivoted.columns
    ]

    # ──────────────────────────────────────────────────────────────────────
    # Numerical to Polynominal (6): Pass|Furnace from numeric → string
    # ──────────────────────────────────────────────────────────────────────
    for col in ("Pass", "Furnace"):
        if col in pivoted.columns:
            pivoted[col] = pivoted[col].astype(str)

    # ──────────────────────────────────────────────────────────────────────
    # Set Role (10): Timestamp → id
    # (No pandas equivalent; tracked as metadata only.)
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Generate Attributes (unnamed): COP_Old +=1 ; COT_Old += 20
    # ──────────────────────────────────────────────────────────────────────
    for col, delta in (("COP_Old", 1), ("COT_Old", 20)):
        if col in pivoted.columns:
            pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce") + delta

    # ──────────────────────────────────────────────────────────────────────
    # ccp_status_reform (4)
    # ──────────────────────────────────────────────────────────────────────
    _run_ccp_status_reform(macros, registry)
    return pivoted


# Original (name,new_name) mapping defined inside the Create ExampleSet of
# Renaming as per optimizer (2). The left side is the macro-name carrying the
# customer-specific tag identifier, the right side is the canonical column.
_RENAME_TARGETS: List[Tuple[str, str]] = [
    ("TMT_Name", "TMT_Old"),
    ("Mixed_Feed_Name", "Mixed_Feed_Old"),
    ("Feed_Bias_Name", "Feed_Bias"),
    ("COT_Bias_Name", "COT_Bias"),
    ("Coke_Thickness_Name", "Coke_Thickness_Old"),
    ("SHC_Name", "SHC_Old"),
    ("Good_Tubes_Name", "Good_Tubes_Old"),
    ("Coupled_Mode_Identifier_Name", "Coupled_Mode_Identifier"),
    ("Conversion_Name", "Conversion_Old"),
    ("COT_Name", "COT_Old"),
    ("Heat_Bias_Name", "Heat_Bias"),
    ("Conversion_Bias_Name", "Conversion_Bias"),
    ("Days_Online_Name", "Days_Online"),
    ("Furnace_Status_Name", "Furnace_Status"),
    ("Radiant_Heat_Absorbed_Name", "Radiant_Heat_Absorbed_Old"),
    ("Yield_Name", "Yield_Old"),
    ("Days_Remaining_Name", "Days_Remaining_Old"),
    ("HTC_Inlet_Temperature_Name", "HTC_Inlet_Temperature_Old"),
    ("Controller_Opening_Name", "Controller_Opening_Old"),
    ("Decoke_Time_Name", "Decoke_Time"),
    ("COP_Name", "COP_Old"),
    ("CIT_Name", "CIT_Old"),
    ("Tube_Flow_Name", "Tube_Flow_Old"),
    ("Coking_Rate_Name", "Coking_Rate_Old"),
    ("Benzene_Name", "Benzene_Old"),
    ("Ethane_Name", "Ethane_Old"),
    ("Styrene_Name", "Styrene_Old"),
    ("Acetylene_Name", "Acetylene_Old"),
    ("Use_Optimizer_Opportunity_Name", "Use_Optimizer_Opportunity"),
    ("Use_Uptime_Benefit_In_Opportunity_Name", "Use_Uptime_Benefit_In_Opportunity"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Filtering_Data subprocess (nested inside Re-formatting)
# ─────────────────────────────────────────────────────────────────────────────
def _run_filtering_data(
    df: pd.DataFrame, macros: MacroStore, registry: StoreRegistry,
) -> pd.DataFrame:
    """Splits passwise vs systemwide tags, renames each appropriately, and
    re-joins the two streams into a single long-format frame ready for Pivot.
    """
    # ──────────────────────────────────────────────────────────────────────
    # Gen_Pass_Values: fan out pass-numbered rows so every (Timestamp,Pass)
    # combination is represented (Loop Values 6 × inner Loop 9: 1..8).
    # ──────────────────────────────────────────────────────────────────────
    pass_grid = _run_gen_pass_values(df)

    # ──────────────────────────────────────────────────────────────────────
    # Filter Examples (30): keep rows where `Pass` is not missing.
    # ──────────────────────────────────────────────────────────────────────
    if "Pass" in df.columns:
        passwise = df[df["Pass"].notna()].copy()
        systemwide = df[df["Pass"].isna()].copy()
    else:
        passwise = df.iloc[0:0].copy()
        systemwide = df.copy()

    # ──────────────────────────────────────────────────────────────────────
    # Renaming_Systemwise: Replace (3) → strip the "Fur<digit>_" prefix
    # then Select Attributes (69) → drop the Pass column
    # ──────────────────────────────────────────────────────────────────────
    systemwide["name"] = systemwide["name"].astype(str).str.replace(
        r"^Fur\d_(.+)$", r"\1", regex=True
    )
    if "Pass" in systemwide.columns:
        systemwide = systemwide.drop(columns=["Pass"])

    # ──────────────────────────────────────────────────────────────────────
    # Renaming_Passwise: Replace (9) → strip the "Fur<digit>_Pass<digit>_" prefix
    # ──────────────────────────────────────────────────────────────────────
    passwise["name"] = passwise["name"].astype(str).str.replace(
        r"^Fur\d_Pass\d_(.+)$", r"\1", regex=True
    )

    # ──────────────────────────────────────────────────────────────────────
    # Subprocess (27)
    #   Join (25): LEFT join (pass_grid × systemwide) on name
    #   Parse Numbers (5): Pass → numeric
    #   Numerical to Real: Pass → real
    #   Append (48): append the passwise stream
    # ──────────────────────────────────────────────────────────────────────
    # Join systemwide rows with pass_grid so each system tag gets duplicated
    # across all 8 passes (mirrors the RM left-join semantics):
    if not systemwide.empty and not pass_grid.empty:
        sys_expanded = pass_grid.merge(systemwide, on="Timestamp", how="left")
    else:
        sys_expanded = systemwide.copy()
        if "Pass" not in sys_expanded.columns:
            sys_expanded["Pass"] = float("nan")
    if "Pass" in sys_expanded.columns:
        sys_expanded["Pass"] = pd.to_numeric(sys_expanded["Pass"], errors="coerce")

    combined = pd.concat([sys_expanded, passwise], axis=0, ignore_index=True, sort=False)
    return combined


def _run_gen_pass_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gen_Pass_Values subprocess: produce one row per (Timestamp × Pass=1..8).

    Operators:
      - Select Attributes (33): keep only the Timestamp column.
      - Remove Duplicates (18): unique Timestamps.
      - Date to Nominal (3):    string-format Timestamp.
      - Loop Values (6) over Timestamp_nominal:
          - Loop (9) over pass=1..8:
              - Generate Attributes (27): Pass = %{pass}
              - Append (43)
      - Append (46), Select Attributes (68): drop Timestamp_nominal
    """
    if "Timestamp" not in df.columns:
        return pd.DataFrame(columns=["Timestamp", "Pass"])
    timestamps = df[["Timestamp"]].drop_duplicates()
    rows = []
    for ts in timestamps["Timestamp"]:
        for p in range(1, 9):
            rows.append({"Timestamp": ts, "Pass": float(p)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ccp_status_reform (4)
# ─────────────────────────────────────────────────────────────────────────────
def _run_ccp_status_reform(
    macros: MacroStore, registry: StoreRegistry,
) -> None:
    """The ccp_status_reform (4) subprocess (called from inside Re-formatting).

    Operators (in order):
      - Recall (86):                  Recall "ccp_status"
      - Generate Attributes (36):     derive sub_model_id from entity_id
      - Multiply (16):                fork stream
      - Filter Examples (45):         keep sub_model_id == %{sub_model_id}
      - Remember (43):                store as "ccp_status_utd_sub_model"
      - Select Attributes (16):       keep sub_model_id, Timestamp, ccp_status
      - Remember (65):                store as "ccp_status"
    """
    if not registry.has("ccp_status"):
        return
    ccp_status = registry.recall("ccp_status").copy()

    # Generate Attributes (36): sub_model_id derivation
    if "entity_id" in ccp_status.columns:
        ccp_status["sub_model_id"] = (
            ccp_status["entity_id"]
            .map(ENTITY_TO_SUB_MODEL)
            .fillna(float("nan"))
        )

    sub_model_id = float(macros.get("sub_model_id") or 0)
    filtered = ccp_status[
        pd.to_numeric(ccp_status["sub_model_id"], errors="coerce") == sub_model_id
    ].copy()
    registry.remember("ccp_status_utd_sub_model", filtered)

    keep = [c for c in ("sub_model_id", "Timestamp", "ccp_status")
            if c in filtered.columns]
    registry.remember("ccp_status", filtered[keep])


# ═════════════════════════════════════════════════════════════════════════════
#                       Generate tags (2)
# ═════════════════════════════════════════════════════════════════════════════
def _run_generate_tags_2(df: pd.DataFrame, macros: MacroStore) -> pd.DataFrame:
    """The Generate tags (2) operator at the top of Main_Process.

    A long list of derived attributes that turn raw tags into the optimizer's
    canonical *_Old set.
    """
    formulas: List[Tuple[str, str]] = [
        # 1. Coke thickness in meters
        ("Coke_Thickness_Meter_Old", "[Coke_Thickness_Old]/1000"),
        # 2. Coupling indicator
        ("Ranking_Coupled", "if([Coupled_Mode_Identifier]>0,1,0)"),
        # 3. Feed_Old (excluding steam) — Mixed_Feed_Old / (1+SHC_Old)
        ("Feed_Old", "[Mixed_Feed_Old]/(1+[SHC_Old])"),
        # 4. Tube count sanity check
        ("Good_Tubes_Old",
         "if([Feed_Old]/[Good_Tubes_Old]*1000>300,36,[Good_Tubes_Old])"),
        # 5. Per-tube feed
        ("Tube_Flow_Old", "[Feed_Old]/[Good_Tubes_Old]*1000"),
        # 6. Per-tube radiant heat
        ("Tube_Radiant_Heat_Old", "[Radiant_Heat_Absorbed_Old]/[Good_Tubes_Old]"),
        # 7. COT calculated from physics-based correlation
        ("COT_From_Equation_Old",
         "914.719435 + (0.000049446659699003812525261448)*1 "
         "+ (-0.358227690117577657336056518034)*CIT_Old "
         "+ (-14.124434964818018301002666703425)*SHC_Old "
         "+ (3.640788886684934055892881588079*288)*Tube_Radiant_Heat_Old "
         "+ (-0.490532278448592096165015163933)*Tube_Flow_Old "
         "+ (-30.841100624879896230368103715591)*COP_Old"),
        # 8. Bias columns gated by Ranking_Coupled
        ("Feed_Bias",       "[Feed_Bias]*[Ranking_Coupled]"),
        ("Conversion_Bias", "[Conversion_Bias]*[Ranking_Coupled]"),
        ("COT_Bias",        "[COT_Bias]*[Ranking_Coupled]"),
        ("Heat_Bias",       "[Heat_Bias]*[Ranking_Coupled]"),
        # 9. Macro-driven decoke time
        ("Decoke_Time", "%{decoke_time}"),
    ]
    return apply_generate_attributes(df, formulas, macros, keep_all_columns=True)
