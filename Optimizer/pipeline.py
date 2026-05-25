"""
pipeline.py
===========

End-to-end driver that wires together the three RapidMiner blocks now
ported to Python:

    United (2)               →  united_optimizer.run_united_optimizer
    post-optimizer.rmp       →  post_opt_prepare_output.post_optimizer
    prepare-output.rmp       →  post_opt_prepare_output.prepare_output

In the parent .rmp these three sit back-to-back inside the outer process:

    Input branch
        ↓
    Initialization
        ↓
    Get_PI_Data
        ↓
    Preprocessing
        ↓
    Feature Input Output to Optimizer
        ↓
    Optimizer loop (concurrency:loop)
        │
        ▼
    ┌─────────────────────────────────────────────────────────┐
    │  United (2)                  ← run_united_optimizer     │
    │     ↓ long-format ExampleSet                            │
    │  post_optimizer_transformation                          │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │  post-optimizer.rmp      ← post_optimizer        │   │ ◄── this file
    │  │  prepare-output.rmp      ← prepare_output        │   │
    │  └──────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
        ↓
    Write Database

This file is the *integration glue*. It does **NOT** re-implement any
operator logic — it only:

  1. Calls `run_united_optimizer` with the inputs provided.
  2. Adapts the output schema so it satisfies the post-optimizer's
     contract (only one tiny shim is needed — see `_normalise_for_post_opt`).
  3. Forwards to `post_optimizer` and then `prepare_output`.
  4. Returns the four-tuple `(out_1, out_2, out_3, out_4)` exactly as the
     RapidMiner prepare-output block does.

Where it plugs into the existing `post_opt_prepare_output.py`:

  • `post_opt_prepare_output.run_pipeline(...)` is the *standalone* driver
    when the optimizer output already exists (e.g. loaded from a file).
    This module supersedes it by *producing* that optimizer output via
    United (2) instead of reading it from disk.
  • Both drivers ultimately call the same `post_optimizer(...)` and
    `prepare_output(...)` functions, so no changes to the existing
    file are required.

Usage
-----

    from united_optimizer import UnitedOptimizerInputs, CoilsimModelProvider
    from united_optimizer.pipeline import run_full_pipeline

    out_1, out_2, out_3, out_4 = run_full_pipeline(
        united_inputs    = UnitedOptimizerInputs(...),
        ccp_status_utd   = ...,
        model_alert      = ...,
        entity           = ...,
        furnace_selection= ...,
        opt_last_good    = ...,
        runtime_macros   = {"end_time": "2024-01-01 00:00:00", ...},
        macros_xlsx      = "/path/to/Macros.xlsx",  # optional
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Local imports from this package
from orchestrator import UnitedOptimizerInputs, run_united_optimizer
from rm_runtime import CoilsimModelProvider

# Import the pre-existing post-optimizer + prepare-output module.
# It is expected to sit next to this file (or be importable from PYTHONPATH).
try:
    from post_opt_prepare_output import (
        load_macros,
        post_optimizer,
        prepare_output,
    )
except ImportError:                                          # pragma: no cover
    # Fall back to the uploads directory for development workflows.
    sys.path.insert(0, "/mnt/user-data/uploads")
    from post_opt_prepare_output import (                    # type: ignore
        load_macros,
        post_optimizer,
        prepare_output,
    )


# ═════════════════════════════════════════════════════════════════════════════
#                              Schema adapter
# ═════════════════════════════════════════════════════════════════════════════
def _normalise_for_post_opt(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt United's long-format output to the contract that
    ``post_opt_prepare_output.post_optimizer`` expects.

    Two tiny coercions are needed:

    1.  Timestamp dtype:
            United emits ``Timestamp`` as pandas StringDtype (the result of
            ``Date to Nominal (7)`` inside the .rmp). ``post_optimizer`` then
            runs ``np.issubdtype(df["Timestamp"].dtype, np.datetime64)`` to
            decide whether to call ``pd.to_datetime``; that check raises
            ``TypeError`` on StringDtype.  Casting the column to plain object
            (the dtype a fresh ``pd.read_csv`` would produce) makes the check
            return ``False`` and the subsequent ``pd.to_datetime`` succeed.

    2.  Numeric coupled_mode / sub_model_id:
            United's ``Parse Numbers (34)`` step converts these to floats
            already, so the contract is satisfied. No work needed here, but
            we run a defensive ``pd.to_numeric`` to be safe.

    Everything else (``tag``, ``value`` columns) is already in the schema
    ``post_optimizer`` reads (see lines 61, 91, 162 of
    ``post_opt_prepare_output.py``).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # 1. Timestamp shim
    if "Timestamp" in out.columns:
        # StringDtype → object → datetime
        if isinstance(out["Timestamp"].dtype, pd.StringDtype):
            out["Timestamp"] = out["Timestamp"].astype(object)

    # 2. Defensive numeric coercion for the two columns post_optimizer
    #    casts to float (lines 162 / 174 / 196 of post_opt_prepare_output.py)
    for col in ("coupled_mode", "sub_model_id"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ═════════════════════════════════════════════════════════════════════════════
#                              Macro plumbing
# ═════════════════════════════════════════════════════════════════════════════
def _build_combined_macros(
    united_inputs: UnitedOptimizerInputs,
    macros_xlsx: Optional[str],
    runtime_macros: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Build the macro dict shared by post_optimizer + prepare_output.

    The .rmp has a single global macro registry; in Python that's recreated
    by merging four sources, in this precedence order (later overwrites earlier):

      1.  Macros.xlsx (sheet 'Optimizer')          — base defaults
      2.  united_inputs.initial_macros             — values supplied to United
      3.  runtime_macros                           — caller overrides
      4.  The `post_optimizer_transformation` /
          `post_optimizer_transformation_utd` flags must be 'active' for the
          post-optimizer to actually do anything; we keep whatever the user
          provided.
    """
    macros: Dict[str, str] = {}
    if macros_xlsx and Path(macros_xlsx).exists():
        macros.update(load_macros(macros_xlsx, sheet="Optimizer"))
    if united_inputs.initial_macros:
        macros.update({str(k): str(v) for k, v in united_inputs.initial_macros.items()})
    if runtime_macros:
        macros.update({str(k): str(v) for k, v in runtime_macros.items()})
    return macros


# ═════════════════════════════════════════════════════════════════════════════
#                              Main entry point
# ═════════════════════════════════════════════════════════════════════════════
def run_full_pipeline(
    united_inputs: UnitedOptimizerInputs,
    ccp_status_utd: pd.DataFrame,
    model_alert_output: pd.DataFrame,
    entity: pd.DataFrame,
    furnace_selection: pd.DataFrame,
    opt_last_good_value: pd.DataFrame,
    runtime_macros: Optional[Dict[str, Any]] = None,
    macros_xlsx: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full chain: United (2) → post_optimizer → prepare_output.

    Parameters
    ----------
    united_inputs
        Inputs to feed `run_united_optimizer`. See
        :class:`united_optimizer.orchestrator.UnitedOptimizerInputs`.
    ccp_status_utd
        The `ccp_status_utd` ExampleSet — same shape as `ccp_status` but
        already filtered to the current sub_model_id (stored by
        `Fe_inferred (2)` via Remember "ccp_status_utd_sub_model").
        Pass the appropriate filtered slice, or simply `ccp_status` if the
        downstream join on (sub_model_id, Timestamp) is sufficient.
    model_alert_output, entity, furnace_selection, opt_last_good_value
        The four extra Recall inputs that prepare-output expects.
    runtime_macros
        Per-run macros that the parent .rmp would have set:
        `case_id`, `optimizer_model_id`, `end_time`, `Uptime_opportunity_threshold`,
        and any `Fur<n>_Total_Benefit_Per_Day_Result_actual` overrides.
    macros_xlsx
        Optional path to `Macros.xlsx` (sheet `Optimizer`). When provided,
        its rows are loaded as the base macro registry — same as the
        existing `run_pipeline` helper inside `post_opt_prepare_output.py`.

    Returns
    -------
    (out_1, out_2, out_3, out_4)
        The same four-tuple `prepare_output` returns:
            out_1 — Output_format_check (post-optimizer output + 1-row template)
            out_2 — opt_last_good_value
            out_3 — model status frame
            out_4 — model alert frame (deduped)
    """
    # ── Step 1: United (2) ──────────────────────────────────────────────────
    united_output = run_united_optimizer(united_inputs)

    # ── Step 2: schema adapter ──────────────────────────────────────────────
    united_output = _normalise_for_post_opt(united_output)

    # ── Step 3: macro merge ─────────────────────────────────────────────────
    macros = _build_combined_macros(united_inputs, macros_xlsx, runtime_macros)

    # ── Step 4: post_optimizer ──────────────────────────────────────────────
    post_out = post_optimizer(united_output, ccp_status_utd, macros)

    # ── Step 5: prepare_output ──────────────────────────────────────────────
    out_1, out_2, out_3, out_4 = prepare_output(
        optimizer_output_from_post = post_out,
        model_alert_output         = model_alert_output,
        entity                     = entity,
        furnace_selection          = furnace_selection,
        opt_last_good_value        = opt_last_good_value,
        macros                     = macros,
    )
    return out_1, out_2, out_3, out_4


# ═════════════════════════════════════════════════════════════════════════════
#                              CLI / smoke-test
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":                                  # pragma: no cover
    print(
        "united_optimizer.pipeline: import run_full_pipeline and call it with "
        "a UnitedOptimizerInputs plus the four prepare-output recall tables.",
        file=sys.stderr,
    )
