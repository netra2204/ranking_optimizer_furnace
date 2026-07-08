"""Orchestrate the pre-rank stage end to end."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

import pre_rank_config as con
import evaluation as eva
import expansion as exp

logger = logging.getLogger(__name__)

_SHORT_NAME_COL = "short_name"
_FORMULA_COL = "formula"


@dataclass
class PreRankResult:
    """Everything the stage produces."""

    wide_output: pd.DataFrame          # wide input + appended computed columns
    expanded: List[exp.TemplateRow]        # expanded template (short_name/formula)
    values: pd.DataFrame               # short_name, formula, value(s)


def _read_template(path: str, sheet_name) -> List[Tuple[object, object]]:
    df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
    if _SHORT_NAME_COL not in df.columns or _FORMULA_COL not in df.columns:
        raise ValueError(
            f"template must contain '{_SHORT_NAME_COL}' and '{_FORMULA_COL}' columns; "
            f"found {list(df.columns)}"
        )
    # Keep NaN formulas as None (treated as empty -> 0 downstream).
    return [
        (row[_SHORT_NAME_COL], None if pd.isna(row[_FORMULA_COL]) else row[_FORMULA_COL])
        for _, row in df.iterrows()
    ]


def _formula_tags(formula: object) -> list:
    """Tags referenced by a formula (empty for numeric constants)."""
    if isinstance(formula, (int, float)) and not isinstance(formula, bool):
        return []
    if formula is None:
        return []
    return eva.extract_tags(str(formula))


def _dependency_order(expanded: List[exp.TemplateRow]) -> List[exp.TemplateRow]:
    """Order rows so that any row referencing another row's short_name is
    evaluated *after* that short_name (topological sort).

    Rows involved in a dependency cycle (should not happen in practice) are
    appended at the end and evaluated with whatever is available; a warning is
    logged so the situation is visible.
    """
    short_names = {r.short_name for r in expanded}
    # Preserve original order for rows that share dependencies (stable).
    index = {id(r): i for i, r in enumerate(expanded)}

    # deps[row] = set of short_names it depends on (that are themselves rows).
    deps = {}
    for r in expanded:
        referenced = {t for t in _formula_tags(r.formula) if t in short_names}
        referenced.discard(r.short_name)  # ignore self-reference
        deps[id(r)] = referenced

    # Map short_name -> the row(s) producing it (system tags produce one row).
    producers = {}
    for r in expanded:
        producers.setdefault(r.short_name, []).append(r)

    resolved_names = set()
    ordered: List[exp.TemplateRow] = []
    remaining = list(expanded)

    while remaining:
        progressed = []
        stuck = []
        for r in remaining:
            if deps[id(r)] <= resolved_names:
                progressed.append(r)
            else:
                stuck.append(r)
        if not progressed:
            # Cycle (or dependency on a name no row produces and that will be
            # taken from the wide input instead). Emit the rest in file order.
            logger.warning(
                "Could not fully order %d row(s) by dependency (possible cycle); "
                "evaluating them in file order.",
                len(stuck),
            )
            stuck.sort(key=lambda r: index[id(r)])
            ordered.extend(stuck)
            break
        progressed.sort(key=lambda r: index[id(r)])
        ordered.extend(progressed)
        for r in progressed:
            resolved_names.add(r.short_name)
        remaining = stuck

    return ordered


def run_pre_rank(
    config_path: str,
    common_inferred_path: str,
    wide_input_path: str,
    output_path: Optional[str] = None,
    expanded_output_path: Optional[str] = None,
    overwrite_existing: bool = True,
    ignore_existing_short_name_columns: bool = True,
    config_sheet=None,
    template_sheet=None,
    wide_sheet=None,
) -> PreRankResult:
    """Run the full pre-rank stage.

    Parameters
    ----------
    config_path
        Path to ``config_overrides`` (provides NUM_FURNACES, NUM_PASSES).
    common_inferred_path
        Path to the ``common-inferred`` template (short_name / formula).
    wide_input_path
        Path to the wide-format input to augment.
    output_path
        If given, the augmented wide input is written here (.xlsx).
    expanded_output_path
        If given, the expanded+evaluated template is written here (.xlsx),
        useful for auditing.

    Returns
    -------
    PreRankResult
    """
    # 1) Config.
    cfg = con.read_config(config_path, sheet_name=config_sheet)

    # 2) Expand template.
    template_rows = _read_template(common_inferred_path, template_sheet)
    expanded = exp.expand_template(template_rows, cfg.num_furnaces, cfg.num_passes)

    # 3) Load wide input and evaluate every expanded formula per data row.
    wide_df = pd.read_excel(wide_input_path, sheet_name=wide_sheet if wide_sheet is not None else 0)
    logger.info("Wide input: %d row(s), %d column(s)", wide_df.shape[0], wide_df.shape[1])

    # Column -> value maps, one per data row.
    # Pre-existing columns whose name matches an expanded short_name are
    # placeholder/output columns, not genuine input tags -- exclude them so a
    # formula can never read a stale value from a column that will be removed.
    short_name_set = {row.short_name for row in expanded}
    if ignore_existing_short_name_columns:
        ignored = short_name_set & set(wide_df.columns)
        if ignored:
            logger.info(
                "Ignoring %d pre-existing short_name column(s) as data sources.",
                len(ignored),
            )

    def _tag_map(row_series) -> dict:
        d = row_series.to_dict()
        if ignore_existing_short_name_columns:
            for k in short_name_set:
                d.pop(k, None)
        return d

    row_dicts = [_tag_map(row) for _, row in wide_df.iterrows()]

    # Evaluate in dependency order so a formula referencing another
    # common-inferred short_name sees that short_name's computed value.
    eval_order = _dependency_order(expanded)

    computed = {row.short_name: [] for row in expanded}
    for tag_values in row_dicts:
        # Per wide-data row: start from the wide input tags, then layer in each
        # computed short_name as it is produced. A formula tag is resolvable if
        # it exists in the wide input OR is a common-inferred short_name; only
        # if it is absent from BOTH does it fall back to dummy 0.
        resolved = dict(tag_values)
        for row in eval_order:
            value = eva.evaluate_formula(row.formula, resolved, row.short_name, logger)
            resolved[row.short_name] = value
        for row in expanded:
            computed[row.short_name].append(resolved[row.short_name])

    if not row_dicts:  # no data rows: still record the formulas with no values
        computed = {row.short_name: [] for row in expanded}

    # 4) Transpose + append. short_names become columns on the wide input.
    #    Some short_name columns may already exist in the wide input (e.g.
    #    placeholder columns). ``overwrite_existing`` controls what happens then.
    #    Build all new columns first and concat once (avoids fragmentation).
    existing_cols = set(wide_df.columns)
    to_add = {}
    overwritten = 0
    skipped = 0
    seen = set()
    for row in expanded:
        col = row.short_name
        if col in seen:
            continue
        seen.add(col)
        if col in existing_cols:
            if not overwrite_existing:
                skipped += 1
                continue
            overwritten += 1
        to_add[col] = computed[col]

    wide_output = wide_df.copy()
    if overwritten:
        # Drop the columns we intend to overwrite, then re-add in one block.
        wide_output = wide_output.drop(columns=[c for c in to_add if c in existing_cols])
    if to_add:
        wide_output = pd.concat(
            [wide_output, pd.DataFrame(to_add, index=wide_output.index)], axis=1
        )

    if overwritten:
        logger.warning(
            "Overwrote %d pre-existing column(s) in the wide input with computed values "
            "(set overwrite_existing=False to keep the originals).",
            overwritten,
        )
    if skipped:
        logger.info("Skipped %d pre-existing column(s) (overwrite_existing=False).", skipped)

    # Audit frame: short_name, formula, and the value(s).
    audit_records = []
    for row in expanded:
        rec = {"short_name": row.short_name, "formula": row.formula}
        vals = computed[row.short_name]
        if len(vals) <= 1:
            rec["value"] = vals[0] if vals else None
        else:
            for i, v in enumerate(vals):
                rec[f"value_row_{i}"] = v
        audit_records.append(rec)
    values_df = pd.DataFrame(audit_records)

    # 5) Persist.
    if output_path:
        wide_output.to_excel(output_path, index=False)
        logger.info("Wrote augmented wide input -> %s", output_path)
    if expanded_output_path:
        values_df.to_excel(expanded_output_path, index=False)
        logger.info("Wrote expanded/evaluated template -> %s", expanded_output_path)

    return PreRankResult(wide_output=wide_output, expanded=expanded, values=values_df)
