"""
united_optimizer — Python replica of the RapidMiner ``United (2)`` block.

Public API:

    from united_optimizer import (
        UnitedOptimizerInputs,
        run_united_optimizer,
        CoilsimModelProvider,
        MacroStore,
        StoreRegistry,
    )

Each operator in the source .rmp has a corresponding Python step here, in
the same execution order. See ``orchestrator.py`` for the wiring and the
docstrings of the individual functions for per-operator details.
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

__all__ = [
    "UnitedOptimizerInputs",
    "run_united_optimizer",
    "CoilsimModelProvider",
    "MacroStore",
    "StoreRegistry",
]
