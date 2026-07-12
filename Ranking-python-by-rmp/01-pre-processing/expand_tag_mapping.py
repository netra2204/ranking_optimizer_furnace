"""
Expand the compressed tag-parameter mapping sheet.

The compressed sheet uses placeholders:
    A   -> furnace number   (short_name "Furnace_A...", entity_name "FA")
    B   -> pass number      (short_name "...Pass_B...", parameter_name "...passB...")
    FS  -> furnace-system rows: emitted once, never expanded.

The number of furnaces and passes is read from the ``config_overrides`` sheet
(``variable_name`` / ``value`` columns) so the mapping stays in sync with the
rest of the pipeline. The fully expanded mapping is written back into the same
compiled workbook under a new sheet.

Used by the orchestrator (see :func:`run_pre_processing`); can also be run
standalone against the compiled workbook.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Columns in the config_overrides sheet.
_CFG_VAR_COL = "variable_name"
_CFG_VAL_COL = "value"
NUM_FURNACES_KEY = "NUM_FURNACES"
NUM_PASSES_KEY = "NUM_PASSES"

# Columns expected in the (compressed) tag-parameter mapping sheet.
_TPM_COLS = ["short_name", "entity_name", "parameter_name"]


def read_counts_from_config(path: str, sheet_name) -> tuple[int, int]:
    """Read NUM_FURNACES / NUM_PASSES from a ``config_overrides`` sheet.

    The sheet is a ``variable_name`` / ``value`` table (same format the
    pre-rank stage consumes).
    """
    df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    if _CFG_VAR_COL not in df.columns or _CFG_VAL_COL not in df.columns:
        raise ValueError(
            f"config sheet must contain columns '{_CFG_VAR_COL}' and '{_CFG_VAL_COL}'; "
            f"found {list(df.columns)}"
        )
    raw = {str(k).strip(): v for k, v in zip(df[_CFG_VAR_COL], df[_CFG_VAL_COL])}

    def _as_positive_int(key: str) -> int:
        if key not in raw:
            raise KeyError(f"'{key}' not found in config sheet ({path})")
        try:
            ival = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{key}' must be an integer, got {raw[key]!r}") from exc
        if ival <= 0:
            raise ValueError(f"'{key}' must be positive, got {ival}")
        return ival

    return _as_positive_int(NUM_FURNACES_KEY), _as_positive_int(NUM_PASSES_KEY)


def expand(df: pd.DataFrame, n_furnaces: int, n_passes: int) -> pd.DataFrame:
    """Expand every template row into concrete furnace/pass rows."""
    rows = []
    for _, r in df.iterrows():
        sn = str(r["short_name"])
        en = str(r["entity_name"])
        pn = str(r["parameter_name"])

        has_furnace = ("Furnace_A" in sn) or (en == "FA")
        has_pass = ("Pass_B" in sn) or bool(re.search("passB", pn, re.IGNORECASE))

        # System-level rows (FS): keep as-is.
        if not has_furnace:
            rows.append((sn, en, pn))
            continue

        for f in range(1, n_furnaces + 1):
            s1 = sn.replace("Furnace_A", f"Furnace_{f}")
            e1 = f"F{f}" if en == "FA" else en

            if not has_pass:
                rows.append((s1, e1, pn))
            else:
                for p in range(1, n_passes + 1):
                    s2 = s1.replace("Pass_B", f"Pass_{p}")
                    p2 = re.sub("passB", f"pass{p}", pn, flags=re.IGNORECASE)
                    rows.append((s2, e1, p2))

    return pd.DataFrame(rows, columns=_TPM_COLS)


def run_pre_processing(
    compiled_path: str,
    config_sheet: str = "config_overrides",
    tpm_sheet: str = "tpm-newname-rev3",
    output_sheet: str = "expanded-tpm-newname-rev3",
) -> pd.DataFrame:
    """Expand the tag-parameter mapping and write it back into the workbook.

    Reads the furnace/pass counts from ``config_sheet`` and the compressed
    mapping from ``tpm_sheet`` (both inside ``compiled_path``), expands the
    template, and writes the result to ``output_sheet`` in the same workbook
    (replacing it if it already exists). Returns the expanded DataFrame.
    """
    n_furnaces, n_passes = read_counts_from_config(compiled_path, config_sheet)
    logger.info("Expanding with %d furnaces and %d passes ...", n_furnaces, n_passes)

    tpm = pd.read_excel(compiled_path, sheet_name=tpm_sheet)
    missing = [c for c in _TPM_COLS if c not in tpm.columns]
    if missing:
        raise ValueError(
            f"'{tpm_sheet}' must contain columns {_TPM_COLS}; missing {missing}"
        )
    tpm = tpm[_TPM_COLS]

    expanded = expand(tpm, n_furnaces, n_passes)

    # Append/replace the expanded sheet without disturbing the other sheets.
    with pd.ExcelWriter(
        compiled_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        expanded.to_excel(writer, sheet_name=output_sheet, index=False)

    logger.info(
        "Wrote %d expanded rows -> sheet '%s' in %s",
        len(expanded),
        output_sheet,
        compiled_path,
    )
    return expanded


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    _here = os.path.dirname(os.path.abspath(__file__))
    _compiled = os.path.join(
        _here, os.pardir, "Python-Inputs", "ranking-inputs-compiled.xlsx"
    )
    run_pre_processing(os.path.abspath(_compiled))
