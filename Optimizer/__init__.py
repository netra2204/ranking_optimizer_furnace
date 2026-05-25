"""
united_optimizer — Python replica of the RapidMiner ``United (2)`` block,
plus the integration glue to run the full optimizer chain end-to-end.

Public API:

    # Just the United (2) block:
    from united_optimizer import (
        UnitedOptimizerInputs,
        run_united_optimizer,
        CoilsimModelProvider,
        MacroStore,
        StoreRegistry,
    )

    # Full chain (United → post-optimizer → prepare-output):
    from united_optimizer import run_full_pipeline

Each operator in the source .rmp has a corresponding Python step here, in
the same execution order. See ``orchestrator.py`` for the wiring of
United (2), and ``pipeline.py`` for the end-to-end driver.
"""
from rm_runtime import (
    MacroStore,
    StoreRegistry,
    CoilsimModelProvider,
)
from orchestrator import (
    UnitedOptimizerInputs,
    run_united_optimizer,
)
from pipeline import run_full_pipeline

__all__ = [
    "UnitedOptimizerInputs",
    "run_united_optimizer",
    "run_full_pipeline",
    "CoilsimModelProvider",
    "MacroStore",
    "StoreRegistry",
]
