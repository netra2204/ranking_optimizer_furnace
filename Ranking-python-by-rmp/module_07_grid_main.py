"""
module_07_grid_main.py
======================
Orchestrator for the Grid-Main subprocess.

The Grid-Main logic is split into two layers for readability and isolated
debugging:

    module_07a_grid_feed.py        – outer FEED-grid layer.
                                     Owns: Handle Exception (12), FEED GRID (2),
                                     Loop (49) step-size override,
                                     Feed_Grid_Character signature/dedupe log,
                                     Branch (90)/(113), per-row table build,
                                     Subprocess (132)/Set Macros (3) best-tracker,
                                     Log "Feed_Conversion_merged_log",
                                     Generate Macro (45) exception fallback.

    module_07b_grid_conversion.py  – inner MAIN CONVERSION GRID + GRID -Conversion.
                                     Owns: Subprocess (16) bound update
                                     (Subprocess (102) recycle probe,
                                      Subprocess (17) Loop-While convergence,
                                      Subprocess (20)/(50) topN bound assignment),
                                     inferred_tags_3 evaluation,
                                     Branch (29) grid_condition check,
                                     Generate Macro (21) Conversion_Grid_Success,
                                     GRID -Conversion (Loop (95), GRID (2),
                                      Conversion inside Feed grid (2),
                                      Loop (96) rounding, Loop (145) per-row build,
                                      Inf tags calculations (2) → inferred_tags_2,
                                      Aggregate (84) sums, Generate Macro (46) gate,
                                      Conv_GRID_LOG(feed)-main log),
                                     Subprocess (44)/(131) post-grid finalisation,
                                     Generate Macro (44) Max_Benefit acceptance rule.

Wiring
------
- main.py calls `module_07_grid_main.run(df)` exactly once.
- This file forwards to Chunk A.
- Chunk A imports Chunk B internally and calls `gridB.run(df_per_row)`
  once per surviving feed combo.

Public surface unchanged:  `run(df) -> df`.
"""

import logging
import pandas as pd

import module_07a_grid_feed       as _gridA
import module_07b_grid_conversion as _gridB   # noqa: F401 (visibility / explicit dep)

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute Grid-Main. Returns df unchanged; outputs live in MACROS and STORE.

    Macros consumed (from pre-grid):
        Row_N_lower/upper/step_size_feed, Number_of_rows,
        fresh_feed_change, fresh_feed_quantity, shc_ratio,
        mixed_feed_margin, change_recycle_ethane_upper/lower_limit,
        conversion_upper/lower_limit_expansion_max_limit,
        max_conversion_single_furnace_limit, Extra_Recycle_Ethane,
        biasing_condition, benefit_percent_threshold,
        ranking_improve_energy_consumption,
        ranking_improve_specific_energy_consumption,
        ROPT_all_furnace_for_conversion_biasing

    Macros produced (for post-grid):
        Row_N_feed_delta (best),
        Grid_Row_N_conversion_delta (best),
        sum_del_Feed_flow, sum_del_ethylene,
        Max_Benefit, Max_Benefit_SPC,
        Min_target_sum_feed_bias (updated),
        Conversion_Grid_Success,
        Feed_Grid_Character (of winning combo),
        ranking_cause_indicator (1 / -1 / -5 / -6)
    """
    return _gridA.run(df)
