"""Evaluate a single (already placeholder-substituted) formula.

Supported formula shapes
------------------------
1. **Constant**            -> ``0``            (number, returned as-is)
2. **Bare single tag**     -> ``Some_Tag``     (looked up in the wide input)
3. **Arithmetic express.** -> ``[A]*max([A]*[B]+[C],...)`` or ``((X-Y)/Z*100)``
4. **Excel condition**     -> ``if([X]>=[Y],1,0)``

Tags may be written either bracketed (``[Tag_Name]``) or bare (``Tag_Name``).
Before evaluation every tag referenced by the formula must be present in the
wide input; if even one is missing the formula is not evaluated -- the result
is a dummy ``0`` and an INFO log line is emitted:

    ``<tag> -- missing hence <short_name> gets dummy value 0``

``eval`` is only ever run on a numeric expression: every tag has already been
replaced by its numeric value and every ``if(...)`` has been rewritten to a
Python conditional, so a raw ``if(...)`` string is never passed to ``eval``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import List, Mapping

logger = logging.getLogger(__name__)

# Function names that must NOT be mistaken for data tags.
_FUNCTION_NAMES = {"if", "max", "min", "abs", "round", "and", "or", "not"}

# Restricted namespace for eval -- no builtins, only whitelisted helpers.
_SAFE_GLOBALS = {"__builtins__": {}, "max": max, "min": min, "abs": abs, "round": round}

_BRACKETED = re.compile(r"\[([^\]]+)\]")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IF_CALL = re.compile(r"(?<![A-Za-z0-9_])if\s*\(", re.IGNORECASE)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def extract_tags(formula: str) -> List[str]:
    """Return the ordered, de-duplicated list of data tags in ``formula``.

    Handles both bracketed tags and bare identifiers, ignoring function names
    and numeric literals.
    """
    tags: List[str] = []
    seen = set()

    # 1) Bracketed tags.
    for m in _BRACKETED.finditer(formula):
        tag = m.group(1).strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    # 2) Bare identifiers, from the formula with bracketed parts removed.
    without_brackets = _BRACKETED.sub(" ", formula)
    for m in _IDENTIFIER.finditer(without_brackets):
        ident = m.group(0)
        if ident.lower() in _FUNCTION_NAMES:
            continue
        if ident not in seen:
            seen.add(ident)
            tags.append(ident)

    return tags


def _available(tag: str, tag_values: Mapping[str, object]) -> bool:
    """A tag is usable only if present AND holding a real (non-NaN) number."""
    if tag not in tag_values:
        return False
    v = tag_values[tag]
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True


def _substitute_tags(formula: str, tags: List[str], tag_values: Mapping[str, object]) -> str:
    """Replace every tag occurrence (bracketed or bare) with its numeric value.

    Longest tags first, so a shorter tag that is a prefix of a longer one does
    not corrupt the replacement.
    """
    expr = formula
    for tag in sorted(tags, key=len, reverse=True):
        value_repr = repr(tag_values[tag])  # numeric -> safe literal
        # Bracketed form: [tag]
        expr = expr.replace(f"[{tag}]", value_repr)
        # Bare form: tag not flanked by identifier chars.
        bare = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(tag)}(?![A-Za-z0-9_])")
        expr = bare.sub(value_repr, expr)
    return expr


# Sentinels for the inserted conditional keywords, so the freshly written
# Python ``if``/``else`` are not re-matched as Excel ``if(`` calls. Non-alpha
# control characters can never appear in a formula or match _IF_CALL.
_IF_SENTINEL = "\x01"
_ELSE_SENTINEL = "\x02"


def _excel_if_to_python(expr: str) -> str:
    """Rewrite Excel ``if(cond, t, f)`` into ``((t) if (cond) else (f))``.

    Handles nesting and only splits on top-level commas of each if() call.
    """
    expr = _convert_if(expr)
    return expr.replace(_IF_SENTINEL, " if ").replace(_ELSE_SENTINEL, " else ")


def _convert_if(expr: str) -> str:
    while True:
        m = _IF_CALL.search(expr)
        if not m:
            return expr

        start = m.start()
        open_paren = m.end() - 1  # index of '('

        depth = 0
        commas: List[int] = []
        close = None
        for j in range(open_paren, len(expr)):
            c = expr[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    close = j
                    break
            elif c == "," and depth == 1:
                commas.append(j)

        if close is None or len(commas) < 2:
            raise ValueError(f"Malformed if(...) in expression: {expr!r}")

        args_start = open_paren + 1
        cond = expr[args_start : commas[0]]
        true_part = expr[commas[0] + 1 : commas[1]]
        false_part = expr[commas[1] + 1 : close]

        # Recurse into each argument to resolve nested if(...) calls.
        cond = _convert_if(cond)
        true_part = _convert_if(true_part)
        false_part = _convert_if(false_part)

        replacement = (
            f"(({true_part}){_IF_SENTINEL}({cond}){_ELSE_SENTINEL}({false_part}))"
        )
        expr = expr[:start] + replacement + expr[close + 1 :]


def _normalise_operators(expr: str) -> str:
    """Map Excel comparison operators to Python ones."""
    expr = expr.replace("<>", "!=")
    # A lone '=' (not part of <=, >=, ==, !=) becomes '=='.
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    return expr


def evaluate_formula(
    formula: object,
    tag_values: Mapping[str, object],
    short_name: str,
    logger: logging.Logger = logger,
) -> float:
    """Evaluate ``formula`` against ``tag_values``.

    Returns the numeric result, or ``0`` if the formula is empty, references a
    missing tag, or fails to evaluate.
    """
    # 1) Numeric constant.
    if _is_number(formula):
        return formula

    text = "" if formula is None else str(formula).strip()
    if text == "" or text.lower() == "nan":
        return 0

    # A plain numeric string is also a constant.
    try:
        return float(text)
    except ValueError:
        pass

    # 2) Tag presence check.
    tags = extract_tags(text)
    missing = [t for t in tags if not _available(t, tag_values)]
    if missing:
        for tag in missing:
            logger.info("%s -- missing hence %s gets dummy value 0", tag, short_name)
        return 0

    # 3) Substitute values, rewrite Excel constructs, evaluate.
    expr = _substitute_tags(text, tags, tag_values)
    expr = _excel_if_to_python(expr)
    expr = _normalise_operators(expr)

    try:
        result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307 - sanitised numeric expr
    except ZeroDivisionError:
        logger.warning("%s: division by zero -> 0 (expr: %s)", short_name, expr)
        return 0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("%s: failed to evaluate (%s) -> 0 (expr: %s)", short_name, exc, expr)
        return 0

    # Excel-style booleans as 1/0.
    if isinstance(result, bool):
        return int(result)
    return result
