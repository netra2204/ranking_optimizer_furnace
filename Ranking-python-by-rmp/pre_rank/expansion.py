"""Expand the ``common-inferred`` template.

Placeholder convention
----------------------
* ``A`` is the **furnace** number.
* ``B`` is the **pass** number.

A placeholder is a standalone ``A``/``B`` token, i.e. one that is *not* part of
a larger word. In practice these appear delimited by underscores or brackets
(``Furnace_A_...``, ``[Furnace_A_Pass_B_...]``). The capital ``A`` inside words
such as ``Aramco`` or ``Calculated`` is therefore never treated as a
placeholder.

Expansion rule for each template row (looking at short_name AND formula):

* contains A and B  -> one row per (furnace, pass)  -> NUM_FURNACES * NUM_PASSES
* contains only A   -> one row per furnace          -> NUM_FURNACES
* contains only B   -> one row per pass             -> NUM_PASSES
* contains neither  -> a single, unchanged row      -> 1

Pass-token normalisation
-------------------------
Some short_names carry a literal pass (e.g. ``..._Pass_1_...``) while their
formula uses the ``B`` placeholder. To keep the generated short_names unique
(and per the confirmed requirement) such a literal ``Pass_<n>`` in the
short_name is rewritten to ``Pass_B`` before expansion so the pass number is
substituted consistently on both sides.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List

logger = logging.getLogger(__name__)

# Standalone placeholder tokens. Underscores count as separators here: a token
# is "standalone" when it is not flanked by a letter or digit.
_PLACEHOLDER_A = re.compile(r"(?<![A-Za-z0-9])A(?![A-Za-z0-9])")
_PLACEHOLDER_B = re.compile(r"(?<![A-Za-z0-9])B(?![A-Za-z0-9])")

# A literal pass token inside a short_name, e.g. "Pass_1".
_LITERAL_PASS = re.compile(r"Pass_\d+")


@dataclass
class TemplateRow:
    """A single expanded template row awaiting evaluation."""

    short_name: str
    formula: object  # raw formula cell: number, expression string, or if(...) string


def _has_a(text: str) -> bool:
    return bool(_PLACEHOLDER_A.search(text))


def _has_b(text: str) -> bool:
    return bool(_PLACEHOLDER_B.search(text))


def _sub_a(text: str, furnace: int) -> str:
    return _PLACEHOLDER_A.sub(str(furnace), text)


def _sub_b(text: str, pass_no: int) -> str:
    return _PLACEHOLDER_B.sub(str(pass_no), text)


def _normalise_pass_token(short_name: str, formula_str: str) -> str:
    """If the formula uses the ``B`` pass placeholder but the short_name has a
    literal ``Pass_<n>`` instead, rewrite it to ``Pass_B`` so pass substitution
    happens on both sides."""
    if _has_b(formula_str) and not _has_b(short_name) and _LITERAL_PASS.search(short_name):
        fixed = _LITERAL_PASS.sub("Pass_B", short_name)
        logger.info(
            "Normalised pass token: '%s' -> '%s' (formula uses Pass_B)",
            short_name,
            fixed,
        )
        return fixed
    return short_name


def expand_template(
    rows: Iterable[tuple],
    num_furnaces: int,
    num_passes: int,
) -> List[TemplateRow]:
    """Expand ``(short_name, formula)`` template rows.

    ``rows`` is any iterable of 2-tuples. Returns the fully expanded list of
    :class:`TemplateRow` with all A/B placeholders substituted in both the
    short_name and the formula.
    """
    expanded: List[TemplateRow] = []

    for short_name_raw, formula_raw in rows:
        short_name = str(short_name_raw).strip()
        # Formulas may be numbers (constants); only stringify for placeholder
        # inspection/substitution, but keep numeric constants numeric.
        is_numeric_constant = isinstance(formula_raw, (int, float)) and not isinstance(
            formula_raw, bool
        )
        formula_str = "" if formula_raw is None else str(formula_raw)

        # Reconcile a literal Pass_n in the short_name with a Pass_B in formula.
        short_name = _normalise_pass_token(short_name, formula_str)

        combined = f"{short_name} {formula_str}"
        has_a = _has_a(combined)
        has_b = _has_b(combined)

        def _emit(furnace: int | None, pass_no: int | None) -> None:
            sn = short_name
            fm_str = formula_str
            if furnace is not None:
                sn = _sub_a(sn, furnace)
                fm_str = _sub_a(fm_str, furnace)
            if pass_no is not None:
                sn = _sub_b(sn, pass_no)
                fm_str = _sub_b(fm_str, pass_no)
            # Preserve numeric constants as numbers; otherwise use the
            # (possibly substituted) formula string.
            fm_out = formula_raw if is_numeric_constant else fm_str
            expanded.append(TemplateRow(short_name=sn, formula=fm_out))

        if has_a and has_b:
            for f in range(1, num_furnaces + 1):
                for p in range(1, num_passes + 1):
                    _emit(f, p)
        elif has_a:
            for f in range(1, num_furnaces + 1):
                _emit(f, None)
        elif has_b:
            for p in range(1, num_passes + 1):
                _emit(None, p)
        else:
            _emit(None, None)

    _warn_on_duplicates(expanded)
    logger.info("Expanded template to %d rows", len(expanded))
    return expanded


def _warn_on_duplicates(rows: List[TemplateRow]) -> None:
    seen: dict[str, int] = {}
    for r in rows:
        seen[r.short_name] = seen.get(r.short_name, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        logger.warning("Duplicate short_names after expansion: %s", dups)
