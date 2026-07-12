"""
Build the wide-format pre-rank input from the KPI-calc output database.

This is the JSON counterpart of the reference ``extract_output_db_n_wide_format``
Excel script. Differences from that reference:

  * Input is ``KPI_DB.json`` (a flat list of records) instead of an Excel
    workbook with 3 long-format sheets. Each record carries a ``Sheet`` field
    (``mapping_op`` / ``package_zero`` / ``input_sensor``) so the source sheet
    is still recorded.
  * The value column is ``actual`` (previously ``Stored_Value``).
  * The KPI DB holds several timestamps, so rows are first filtered to a single
    target ``Timestamp`` before anything else.

The set of tags to keep comes from the expanded tag-parameter mapping
(``expanded-tpm-newname-rev3``, produced by ``expand_tag_mapping.py``): only
records whose ``Mapping_Attribute`` equals an expanded ``short_name`` survive.
The kept rows are pivoted to one row per Timestamp / one column per tag and
written back into the compiled workbook as a new sheet.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# JSON record fields.
_NAME_FIELD = "Mapping_Attribute"
_VALUE_FIELD = "actual"          # value column in the JSON (was "Stored_Value")
_TS_FIELD = "Timestamp"
_SHEET_FIELD = "Sheet"

# Only these source sheets are considered (matches the reference script).
_SHEETS_TO_USE = ("mapping_op", "package_zero", "input_sensor")

# Static timestamp to select from the KPI DB. The Jan-12 batch is stored at
# midnight in the JSON (the "1 am" wide sheet is the same batch relabelled).
DEFAULT_TIMESTAMP = "2026-01-12 00:00:00"


def _load_wanted_names(compiled_path: str, expanded_sheet: str) -> set:
    """Return the set of short_names to keep, from the expanded mapping sheet."""
    df = pd.read_excel(compiled_path, sheet_name=expanded_sheet)
    if "short_name" not in df.columns:
        raise ValueError(
            f"'{expanded_sheet}' must contain a 'short_name' column; "
            f"found {list(df.columns)}"
        )
    return {
        str(n).strip()
        for n in df["short_name"].dropna()
        if str(n).strip()
    }


def run_output_db_to_wide(
    json_path: str,
    compiled_path: str,
    expanded_sheet: str = "expanded-tpm-newname-rev3",
    timestamp: str = DEFAULT_TIMESTAMP,
    output_sheet: str = "wide-from-kpi-db",
) -> pd.DataFrame:
    """Filter the KPI DB to one timestamp and the wanted tags, then pivot to wide.

    Parameters
    ----------
    json_path
        Path to ``KPI_DB.json`` (a list of records).
    compiled_path
        Path to the compiled workbook (source of ``expanded_sheet``, and where
        the wide result is written).
    expanded_sheet
        Sheet holding the expanded ``short_name`` list to keep.
    timestamp
        The single Timestamp to select. Compared after parsing so any equivalent
        datetime format matches the JSON's stored value.
    output_sheet
        Sheet name to (re)write with the wide result.

    Returns
    -------
    pd.DataFrame
        The wide frame (one row per Timestamp, one column per matched tag).
    """
    wanted = _load_wanted_names(compiled_path, expanded_sheet)
    logger.info("Wanted tags (expanded short_names): %d", len(wanted))

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("KPI DB records: %d", len(records))

    # 1) Filter to the single target timestamp (the DB holds several).
    target_ts = pd.to_datetime(timestamp)
    at_ts = [r for r in records if pd.to_datetime(r.get(_TS_FIELD)) == target_ts]
    logger.info("Records at %s: %d", timestamp, len(at_ts))
    if not at_ts:
        raise ValueError(
            f"No records found at timestamp {timestamp!r} in {json_path}. "
            f"Available timestamps: "
            f"{sorted({str(r.get(_TS_FIELD)) for r in records})}"
        )

    # 2) Keep only wanted tags from the allowed source sheets.
    long_rows = []
    matched_names = set()
    per_sheet = {}
    for r in at_ts:
        if r.get(_SHEET_FIELD) not in _SHEETS_TO_USE:
            continue
        name = r.get(_NAME_FIELD)
        if name is None:
            continue
        name = str(name).strip()
        if name in wanted:
            long_rows.append({
                _NAME_FIELD: name,
                _VALUE_FIELD: r.get(_VALUE_FIELD),
                _TS_FIELD: r.get(_TS_FIELD),
                "source_sheet": r.get(_SHEET_FIELD),
            })
            matched_names.add(name)
            per_sheet[r[_SHEET_FIELD]] = per_sheet.get(r[_SHEET_FIELD], 0) + 1

    logger.info("Kept %d rows | distinct matched tags: %d | per-sheet: %s",
                len(long_rows), len(matched_names), per_sheet)
    missing = wanted - matched_names
    if missing:
        logger.info(
            "Wanted tags NOT found in KPI DB at %s: %d (e.g. %s)",
            timestamp, len(missing), sorted(missing)[:5],
        )

    # 3) Pivot long -> wide: one row per Timestamp, one column per tag.
    df_long = pd.DataFrame(
        long_rows, columns=[_NAME_FIELD, _VALUE_FIELD, _TS_FIELD, "source_sheet"]
    )
    df_wide = (
        df_long.pivot_table(
            index=_TS_FIELD,
            columns=_NAME_FIELD,
            values=_VALUE_FIELD,
            aggfunc="first",
            dropna=False,
        )
        .reset_index()
    )
    df_wide.columns.name = None

    # 4) Write back into the compiled workbook (replace if present).
    with pd.ExcelWriter(
        compiled_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        df_wide.to_excel(writer, sheet_name=output_sheet, index=False)

    logger.info(
        "Wrote wide output: %d row(s) x %d col(s) (Timestamp + %d tags) "
        "-> sheet '%s' in %s",
        len(df_wide), df_wide.shape[1], df_wide.shape[1] - 1,
        output_sheet, compiled_path,
    )
    return df_wide


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    _here = os.path.dirname(os.path.abspath(__file__))
    _inputs = os.path.abspath(os.path.join(_here, os.pardir, "Python-Inputs"))
    run_output_db_to_wide(
        json_path=os.path.join(_inputs, "KPI_DB.json"),
        compiled_path=os.path.join(_inputs, "ranking-inputs-compiled.xlsx"),
    )
