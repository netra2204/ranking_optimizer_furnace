"""
parameterization.py
================================================================================
Replica of the RapidMiner top-level sub-process "parameterization".

    in 1 -> Branch (38)[cond] -> Numerical to Real (7) -> De-Pivot (7)
         -> Join (2) (right = Recall tag_parameter_mapping) -> Remove Duplicates (9)
         -> Pivot (9) -> Rename by Replacing (12) -> out 1

Run order in the full pipeline:  parameterization  --out 1-->  ranking
================================================================================
"""
from __future__ import annotations

import pandas as pd

from rm_common import (
    Macros, IOStore, _log_block, _to_num,
    op_numerical_to_real, op_de_pivot, op_join, op_remove_duplicates,
    op_pivot, op_rename_by_replacing,
)

def parameterization(example_set: pd.DataFrame,
                     macros: Macros,
                     store: IOStore) -> pd.DataFrame:
    """
    Top-level sub-process "parameterization".

    Data-flow (from the <connect> wiring):
        in 1 -> Branch (38)[cond] -> Numerical to Real (7) -> De-Pivot (7)
             -> Join (2)[left] (right = Recall(3)) -> Remove Duplicates (9)
             -> Pivot (9) -> Rename by Replacing (12) -> out 1
    """
    df = example_set.copy()

    # ----------------------------------------------------------------------
    # [log start (11)] (subprocess)  -- telemetry side-effect, not in flow
    # ----------------------------------------------------------------------
    _log_block("Parameterization", "startedAt", macros)

    # ----------------------------------------------------------------------
    # [Branch (38)] (branch)  condition_type = macro_defined "Ranking_Feed_YSB"
    #   THEN -> [Branch (11)] (branch) expression: %{Ranking_Feed_YSB}=="active"
    #             THEN -> [Loop (6)] 8 iterations recoding furnace mode
    #             ELSE -> pass-through
    #   ELSE -> pass-through
    # ----------------------------------------------------------------------
    if "Ranking_Feed_YSB" in macros:                       # macro_defined
        # [Branch (11)] expression branch
        if macros.get("Ranking_Feed_YSB") == "active":
            # [Loop (6)] (concurrency:loop) number_of_iterations = 8
            #            iteration_macro = "iteration"
            for i in range(1, 9):
                macros["iteration"] = str(i)
                # [Generate Attributes (29)] (blending:generate_columns)
                #   FURN{i}_Furnace_Mode =
                #     if(FURN{i}_Furnace_Mode == 16, 7, FURN{i}_Furnace_Mode)
                col = f"FURN{i}_Furnace_Mode"
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: 7 if _to_num(v) == 16 else v)
        # ELSE (Branch 11): pass-through (no operators)
    # ELSE (Branch 38): pass-through (no operators)

    # ----------------------------------------------------------------------
    # [Numerical to Real (7)] (numerical_to_real)  -- all numeric -> float
    # ----------------------------------------------------------------------
    df = op_numerical_to_real(df)

    # ----------------------------------------------------------------------
    # [De-Pivot (7)] (de_pivot)
    #   index_attribute = tag_name ; value cols = ^(?!Timestamp$).+ (all but Timestamp)
    #   keep_missings = true  -> long table (Timestamp, tag_name, value)
    # ----------------------------------------------------------------------
    df = op_de_pivot(df,
                     value_regex=r"^(?!Timestamp$).+",
                     index_attribute="tag_name",
                     keep_missings=True)

    # ----------------------------------------------------------------------
    # [Recall (3)] (recall)  name = tag_parameter_mapping  -> Join right
    # ----------------------------------------------------------------------
    tag_parameter_mapping = store.recall("tag_parameter_mapping")

    # ----------------------------------------------------------------------
    # [Join (2)] (concurrency:join)  inner ; key tag_name = short_name
    #   left = De-Pivot output ; right = tag_parameter_mapping
    # ----------------------------------------------------------------------
    df = op_join(left=df, right=tag_parameter_mapping,
                 keys={"tag_name": "short_name"},
                 join_type="inner", macros=macros,
                 remove_double_attributes=True)

    # ----------------------------------------------------------------------
    # [Remove Duplicates (9)] (remove_duplicates)  subset = tag_name | Timestamp
    # ----------------------------------------------------------------------
    df = op_remove_duplicates(df, subset=["tag_name", "Timestamp"])

    # ----------------------------------------------------------------------
    # [Pivot (9)] (blending:pivot)
    #   group_by = Timestamp | entity_name ; columns = parameter_name ; first(value)
    # ----------------------------------------------------------------------
    df = op_pivot(df,
                  group_by=["Timestamp", "entity_name"],
                  column_grouping="parameter_name",
                  value_attribute="value", agg="first")

    # ----------------------------------------------------------------------
    # [Rename by Replacing (12)] (rename_by_replacing)
    #   replace_what = first\(value\)_   -> strip the generated "first(value)_" prefix
    # ----------------------------------------------------------------------
    df = op_rename_by_replacing(df, replace_what=r"first\(value\)_", replace_by="")

    # ----------------------------------------------------------------------
    # [log end (6)] (subprocess)  -- telemetry side-effect
    # ----------------------------------------------------------------------
    _log_block("Parameterization", "endedAt", macros)

    return df       # -> out 1

