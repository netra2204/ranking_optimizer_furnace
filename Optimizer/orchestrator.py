"""
orchestrator.py
===============

Top-level orchestrator that wires the United (2) sub-process end-to-end.
Mirrors the connect graph of the outermost United (2) operator:

    in 1
      │
      ▼
    Fe_inferred (2) ──┐         (writes Tag_for_Optimizer to registry,
      │               │          plus macros TMT_Name, Mixed_Feed_Name, …)
      ▼               │
    Remove Duplicates (16)
      │
      ▼
    Handle Timestamp
      │
      ▼
    Re-formatting
      │
      ▼
    Generate tags (2)
      │
      ▼
    Date to Nominal (7)         (timestamps → string)
      │
      ▼
    Join (2) ◄────── Recall (23) "constraint"
      │
      ▼
    Main_Process
      ├── Extract Macro (88): Furnace_Status, ccp_status_curr, total_optimizer_run_check
      ├── Bias Constants
      ├── Overall_Calcs
      ├── GRID_AND_BIASING          (Branch: only when Furnace_Status==1 &&
      │     ccp_status_curr != 0   ccp_status_curr!=0 && total_optimizer_run_check==1)
      ├── ACT=OPT
      └── COUPLED_CCP_USED_CHECK
      │
      ▼
    Remove Duplicates (37)
      │
      ▼
    Parse Numbers (34)            (sub_model_id, coupled_mode → numeric)
      │
      ▼
    Numerical to Real (57)
      │
      ▼
    out 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from rm_runtime import (
    MacroStore,
    StoreRegistry,
    CoilsimModelProvider,
    apply_generate_attributes,
    evaluate_expression,
    extract_macro_from_dataset,
)
from united_optimizer import (
    _run_fe_inferred,
    _run_handle_timestamp,
    _run_reformatting,
    _run_generate_tags_2,
)
from bias_and_grid import (
    DEFAULT_PIPELINE_PARAMETERS,
    run_bias_constants,
    run_overall_calcs,
    run_inferred_calculations,
    run_feed_grid,
    build_objective_function,
)
from act_opt import (
    run_act_opt,
    run_coupled_ccp_used_check,
)


# ═════════════════════════════════════════════════════════════════════════════
#                              Input container
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class UnitedOptimizerInputs:
    """Inputs the United (2) sub-process needs.

    The parent .rmp passes these in via Recall operators executed before
    United (2) starts; the Python orchestrator accepts them up-front.
    """
    # Main data stream that hits "in 1" — long-format pi data joined with
    # furnace metadata, columns: Timestamp, name, value, sub_model_id, …
    main_data: pd.DataFrame

    # Reference tables (recalled inside Fe_inferred (2))
    tag: pd.DataFrame
    tag_child: pd.DataFrame
    tag_details: pd.DataFrame

    # CCP status / pipeline parameters / constraints (recalled inside the body)
    ccp_status: pd.DataFrame
    pipeline_parameters_opt: pd.DataFrame
    constraint: pd.DataFrame

    # Coilsim model provider (4 pre-trained models + 1 normalisation)
    coilsim: CoilsimModelProvider

    # Initial macros — sub_model_id, decoke_time, Max_Permissible_TMT,
    # post_optimizer_transformation_utd, etc.
    initial_macros: Dict[str, Any]


# ═════════════════════════════════════════════════════════════════════════════
#                          run_united_optimizer
# ═════════════════════════════════════════════════════════════════════════════
def run_united_optimizer(inputs: UnitedOptimizerInputs) -> pd.DataFrame:
    """Run the full United (2) sub-process.

    Returns the final wide-format DataFrame produced by Main_Process →
    ACT=OPT → COUPLED_CCP_USED_CHECK → Remove Duplicates (37) →
    Parse Numbers (34) → Numerical to Real (57).
    """
    # ──────────────────────────────────────────────────────────────────────
    # Initialise the per-process runtime state
    # ──────────────────────────────────────────────────────────────────────
    macros = MacroStore()
    macros.set_many(DEFAULT_PIPELINE_PARAMETERS)
    macros.set_many(inputs.initial_macros or {})

    registry = StoreRegistry()
    registry.remember("ccp_status",              inputs.ccp_status)
    registry.remember("pipeline_parameters_opt", inputs.pipeline_parameters_opt)
    registry.remember("constraint",              inputs.constraint)
    registry.remember("tag",                     inputs.tag)
    registry.remember("tag_child",               inputs.tag_child)
    registry.remember("tag_details",             inputs.tag_details)

    # ──────────────────────────────────────────────────────────────────────
    # Fe_inferred (2): populate macros + store Tag_for_Optimizer
    # ──────────────────────────────────────────────────────────────────────
    tag_for_optimizer = _run_fe_inferred(
        tag=inputs.tag, tag_child=inputs.tag_child, tag_details=inputs.tag_details,
        macros=macros, registry=registry,
    )
    # Expose Tag_for_Optimizer to the Coilsim block via a macro pointer
    # (used inside _apply_coilsim_block to filter by Category).
    macros.values["__inferred_tags__"] = tag_for_optimizer

    # ──────────────────────────────────────────────────────────────────────
    # Remove Duplicates (16) — drop duplicate rows from the main stream
    # ──────────────────────────────────────────────────────────────────────
    main = inputs.main_data.drop_duplicates().reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Handle Timestamp
    # ──────────────────────────────────────────────────────────────────────
    main = _run_handle_timestamp(main)

    # ──────────────────────────────────────────────────────────────────────
    # Re-formatting
    # ──────────────────────────────────────────────────────────────────────
    main = _run_reformatting(main, macros, registry)

    # ──────────────────────────────────────────────────────────────────────
    # Generate tags (2)
    # ──────────────────────────────────────────────────────────────────────
    main = _run_generate_tags_2(main, macros)

    # ──────────────────────────────────────────────────────────────────────
    # Date to Nominal (7): Timestamp → string formatted yyyy-MM-dd HH:mm:ss
    # ──────────────────────────────────────────────────────────────────────
    if "Timestamp" in main.columns:
        try:
            main["Timestamp"] = pd.to_datetime(main["Timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (TypeError, ValueError):
            main["Timestamp"] = main["Timestamp"].astype(str)

    # ──────────────────────────────────────────────────────────────────────
    # Join (2): LEFT join with Recall (23) "constraint" by Timestamp
    # ──────────────────────────────────────────────────────────────────────
    if not inputs.constraint.empty and "Timestamp" in inputs.constraint.columns:
        main = main.merge(inputs.constraint, on="Timestamp", how="left")

    # ──────────────────────────────────────────────────────────────────────
    # Main_Process
    # ──────────────────────────────────────────────────────────────────────
    out = _run_main_process(main, macros, registry, inputs.coilsim)

    # ──────────────────────────────────────────────────────────────────────
    # Remove Duplicates (37)
    # ──────────────────────────────────────────────────────────────────────
    out = out.drop_duplicates().reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Parse Numbers (34): sub_model_id|coupled_mode → numeric
    # ──────────────────────────────────────────────────────────────────────
    for col in ("sub_model_id", "coupled_mode"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # ──────────────────────────────────────────────────────────────────────
    # Numerical to Real (57): all numeric columns → real (no-op in pandas)
    # ──────────────────────────────────────────────────────────────────────
    return out


# ═════════════════════════════════════════════════════════════════════════════
#                              Main_Process
# ═════════════════════════════════════════════════════════════════════════════
def _run_main_process(
    df: pd.DataFrame,
    macros: MacroStore,
    registry: StoreRegistry,
    coilsim: CoilsimModelProvider,
) -> pd.DataFrame:
    """Main_Process subprocess.

    Operators (in order):
      • Extract Macro (88):     Furnace_Status, ccp_status_curr,
                                 total_optimizer_run_check (from the inbound df)
      • Bias Constants
      • Overall_Calcs
      • GRID_AND_BIASING branch:
            expression = "eval(%{Furnace_Status})==1 &&
                          eval(%{ccp_status_curr})!=0 &&
                          eval(%{total_optimizer_run_check})==1"
            true → Inferred calculations + FEED GRID (2) + Generate Attributes (619)
                   + Remember (77) + Generate Macro (25)
            false → Generate Attributes (619): Overall_Opt_Branch_Indicator = -1.0
                    Generate Macro (25):
                       if(total_optimizer_run_check==0, 100.0,
                          if(Furnace_Status==1 && ccp_status_curr==0, -1.1, -1.0))
      • ACT=OPT
      • COUPLED_CCP_USED_CHECK
    """
    # ──────────────────────────────────────────────────────────────────────
    # Extract Macro (88)
    # ──────────────────────────────────────────────────────────────────────
    extract_macro_from_dataset(
        df, macros,
        macro_name="Furnace_Status",
        attribute_name="Furnace_Status",
        additional={
            "ccp_status_curr":           "ccp_status",
            "total_optimizer_run_check": "total_optimizer_run_check",
        },
    )

    # ──────────────────────────────────────────────────────────────────────
    # Bias Constants
    # ──────────────────────────────────────────────────────────────────────
    run_bias_constants(macros, registry)

    # ──────────────────────────────────────────────────────────────────────
    # Overall_Calcs
    # ──────────────────────────────────────────────────────────────────────
    df = run_overall_calcs(df, macros)

    # ──────────────────────────────────────────────────────────────────────
    # GRID_AND_BIASING branch
    # ──────────────────────────────────────────────────────────────────────
    cond = evaluate_expression(
        "eval(%{Furnace_Status})==1 && eval(%{ccp_status_curr})!=0 "
        "&& eval(%{total_optimizer_run_check})==1",
        macros,
    )
    # RapidMiner's Branch returns false when the expression cannot be evaluated
    # (missing macros, parse errors, …). In Python NaN is truthy, so guard
    # explicitly: cond is True only if it's the literal Python True.
    branch_active = cond is True
    if branch_active:
        # Inferred calculations subprocess (produces pass_<n>_feed_bias_min/max)
        run_inferred_calculations(df, macros, registry)
        # FEED GRID (2)
        objective = build_objective_function(coilsim)
        winning_df, _grid_meta = run_feed_grid(df, macros, registry, coilsim, objective)
        if not winning_df.empty:
            df = winning_df
        # Generate Attributes (619): Overall_Opt_Branch_Indicator = -1.0 (placeholder)
        df = apply_generate_attributes(df, [
            ("Overall_Opt_Branch_Indicator", "-1.0"),
        ], macros)
        # Remember (77): store the final pre-ACT=OPT frame
        registry.remember("pre_act_opt", df.copy())
    else:
        # False branch: Generate Macro (25)
        macros.set(
            "Overall_Opt_Branch_Indicator",
            evaluate_expression(
                "if(eval(%{total_optimizer_run_check})==0,100.0,"
                "if(eval(%{Furnace_Status})==1 && eval(%{ccp_status_curr})==0,-1.1,-1.0))",
                macros,
            ),
        )
        macros.set("then_block", 0)

    # ──────────────────────────────────────────────────────────────────────
    # ACT=OPT
    # ──────────────────────────────────────────────────────────────────────
    df = run_act_opt(df, macros, registry)

    # ──────────────────────────────────────────────────────────────────────
    # COUPLED_CCP_USED_CHECK
    # ──────────────────────────────────────────────────────────────────────
    df = run_coupled_ccp_used_check(df, macros, registry)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#                                Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover - integration smoke-test stub
    import sys
    print(
        "united_optimizer.orchestrator: "
        "import run_united_optimizer and feed it a UnitedOptimizerInputs.",
        file=sys.stderr,
    )
