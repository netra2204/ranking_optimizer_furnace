"""
main.py  --  orchestrator
================================================================================
Python replica of RapidMiner process `ranking-common-process-subpart.rmp`
(RapidMiner Studio 12.1.001). Reproduces the root wiring:

        parameterization  --out 1-->  ranking  --out 1-->  result

Project layout
--------------
    rm_common.py        shared helpers: Macros, IOStore, expression engine,
                        op_* operator emulations, telemetry blocks
    parameterization.py top-level sub-process "parameterization"
    ranking.py          top-level sub-process "ranking" (+ nested sub-processes)
    main.py             this file - orchestration & entry point

Because this is a SUB-PART of a larger process, the eight objects that are
`Recall`-ed inside it (tag_parameter_mapping, text_code_mapping, ccp_status,
entity, parameters, entity_parameter, tag, furnace_ranking_info) are produced
by the PARENT process and must be supplied via the IOStore. Likewise the
runtime macros (feature toggles, column-name drivers) are set upstream;
`build_default_macros()` documents each one.

No automated testing is performed (per request).
================================================================================
"""
from __future__ import annotations

import pandas as pd

from rm_common import Macros, IOStore, LOG
from parameterization import parameterization
from ranking import ranking

def run_process(example_set: pd.DataFrame,
                macros: Macros,
                store: IOStore) -> pd.DataFrame:
    """
    Reproduces the root process wiring:
        parameterization --out 1--> ranking --out 1--> result
    """
    # [parameterization] (subprocess)
    after_param = parameterization(example_set, macros, store)
    # [ranking] (subprocess)
    result = ranking(after_param, macros, store)
    return result


def build_default_macros() -> Macros:
    """
    Macro scope expected by this sub-part. In the real pipeline these are set
    by the PARENT process; defaults below document their meaning. Adjust as
    required before calling `run_process`.
    """
    return Macros({
        # --- telemetry ---
        "case_id": "0",
        "ranking_model_id": "0",
        # --- feature toggles (each '<macro> == "active"' enables a branch) ---
        # "Ranking_Feed_YSB":              "active",   # enable furnace-mode recode
        # "furnace_system_model_skip_filter": "active",
        # "furnace_system_manual_filter":  "active",
        "score_based_ranking":           "inactive",   # else -> rank-based
        # --- column-name drivers ---
        "furnace_status":   "furnace_status",          # code column to decode
        "ranking_splitter": "ranking_splitter",        # splitter code column
        # "sum_parameter_weightage": auto-derived if not provided
    })


def build_io_store(**objects: pd.DataFrame) -> IOStore:
    """
    Assemble the Recall repository. Required keys (produced upstream):
      tag_parameter_mapping : [short_name, parameter_name, entity_name, ...]
      text_code_mapping     : [code, text]
      ccp_status            : [Timestamp, entity_id, ccp_status, ...]
      entity                : [entity_id, entity_name, ...]
      parameters            : [parameter_id, parameter_name, ...]
      entity_parameter      : [parameter_id, entity_name, formula, ...]
      tag                   : [name, short_name, ...]
      furnace_ranking_info  : [parameter_name, sort_type, parameter_weightage, ...]
    """
    return IOStore(objects)