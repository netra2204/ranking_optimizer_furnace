"""
rm_runtime.py
=============
Helpers that emulate RapidMiner runtime primitives so the Python replica
of the United (2) optimizer block can faithfully recreate every operator.

Two things RapidMiner gives you for free that we need to recreate:

  1. A *MacroStore* — a per-process key/value bag accessed in expressions via
     ``%{macro_name}``. Every operator that reads or writes macros (Set Macro,
     Generate Macro, Extract Macro, etc.) goes through this store.

  2. An expression evaluator that understands RapidMiner's `if(...)`,
     `parse(...)`, `eval(...)`, `concat(...)`, `min/max/abs/floor`, etc.,
     and resolves both ``%{macro}`` substitutions and ``[column]`` /
     ``attribute("column")`` references against the current row/dataframe.

The helpers below are intentionally NOT a general-purpose RapidMiner port —
they implement only the subset used by united-optimizer-main-block.rmp.

Author: Optimizer Replica Project
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# MacroStore — emulates RapidMiner's macro bag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroStore:
    """Key/value bag, all values stored as *strings* (RapidMiner semantics).

    `parse(...)` and `eval(...)` are responsible for turning the string back
    into a number when the expression demands it.
    """
    values: Dict[str, str] = field(default_factory=dict)

    # --- core API ----------------------------------------------------------
    def set(self, name: str, value: Any) -> None:
        self.values[name] = "" if value is None else str(value)

    def set_many(self, mapping: Dict[str, Any]) -> None:
        for k, v in mapping.items():
            self.set(k, v)

    def get(self, name: str, default: str = "") -> str:
        return self.values.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.values

    def remove(self, name: str) -> None:
        self.values.pop(name, None)

    def substitute(self, text: str) -> str:
        """Resolve every ``%{name}`` placeholder in `text` against the store.

        Mirrors RapidMiner's standard macro substitution that happens before
        expression evaluation.
        """
        if not isinstance(text, str):
            return text

        def repl(match: re.Match) -> str:
            name = match.group(1)
            return self.values.get(name, "")
        return re.sub(r"%\{([^}]+)\}", repl, text)


# ─────────────────────────────────────────────────────────────────────────────
# Expression evaluator — handles the RM functions used in the .rmp
# ─────────────────────────────────────────────────────────────────────────────

# Functions exposed inside RM expressions:
def _rm_if(cond: Any, a: Any, b: Any):
    """RapidMiner if(condition, then, else)."""
    return a if cond else b


def _rm_parse(value: Any) -> float:
    """RapidMiner parse(str) → number; NaN if not parsable."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "missing_numeric", "missing_nominal", "missing_date"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _rm_eval(value: Any) -> float:
    """RapidMiner eval(str) — also turns a string into a number.

    In RM `eval()` is sometimes used to take a previously-substituted macro
    string and convert to numeric. We treat it identically to parse() for the
    arithmetic operations seen in this .rmp.
    """
    return _rm_parse(value)


def _rm_concat(*parts: Any) -> str:
    return "".join("" if p is None else str(p) for p in parts)


def _rm_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return str(value)


def _rm_contains(haystack: Any, needle: Any) -> bool:
    return str(needle) in str(haystack)


def _rm_starts(haystack: Any, needle: Any) -> bool:
    return str(haystack).startswith(str(needle))


def _rm_index(haystack: Any, needle: Any) -> int:
    return str(haystack).find(str(needle))


def _rm_length(value: Any) -> int:
    return len(str(value))


def _rm_cut(value: Any, start: int, length: int) -> str:
    s = str(value)
    start = int(start)
    length = int(length)
    return s[start:start + length]


def _rm_replace(value: Any, find: Any, repl: Any) -> str:
    return str(value).replace(str(find), str(repl), 1)


def _rm_replace_all(value: Any, pattern: Any, repl: Any) -> str:
    return re.sub(str(pattern), str(repl), str(value))


def _rm_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan",
                                                            "missing_numeric",
                                                            "missing_nominal",
                                                            "missing_date"):
        return True
    return False


def _rm_abs(value: Any) -> float:
    v = _rm_parse(value)
    return abs(v) if not math.isnan(v) else float("nan")


def _rm_min(*values: Any) -> float:
    nums = [_rm_parse(v) for v in values]
    nums = [n for n in nums if not math.isnan(n)]
    return min(nums) if nums else float("nan")


def _rm_max(*values: Any) -> float:
    nums = [_rm_parse(v) for v in values]
    nums = [n for n in nums if not math.isnan(n)]
    return max(nums) if nums else float("nan")


def _rm_floor(value: Any) -> float:
    v = _rm_parse(value)
    return math.floor(v) if not math.isnan(v) else float("nan")


def _rm_round(value: Any) -> float:
    v = _rm_parse(value)
    return round(v) if not math.isnan(v) else float("nan")


# Mapping of the RapidMiner function name → Python callable
RM_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "if": _rm_if,
    "_rm_if_": _rm_if,         # alias used by the evaluator after rewriting
                               #   RM's reserved `if(` → safe `_rm_if_(`
    "parse": _rm_parse,
    "eval": _rm_eval,
    "concat": _rm_concat,
    "str": _rm_str,
    "contains": _rm_contains,
    "starts": _rm_starts,
    "index": _rm_index,
    "length": _rm_length,
    "cut": _rm_cut,
    "replace": _rm_replace,
    "replaceAll": _rm_replace_all,
    "missing": _rm_missing,
    "abs": _rm_abs,
    "min": _rm_min,
    "max": _rm_max,
    "floor": _rm_floor,
    "round": _rm_round,
    # python builtins also accepted inside expressions
    "True": True,
    "False": False,
}


# Regex to recognise RM-style references inside an expression so we can
# rewrite them into safe Python lookups:
_ATTR_BRACKET = re.compile(r'\[([A-Za-z_][A-Za-z0-9_]*)\]')             # [col_name]
_ATTR_FN      = re.compile(r'attribute\("([^"]+)"\)')                    # attribute("col")
_ATTR_HASH    = re.compile(r'#\{([A-Za-z_][A-Za-z0-9_]*)\}')             # #{name}  (Loop Attributes "name" macro)


def evaluate_expression(
    expression: str,
    macros: MacroStore,
    row: Optional[Union[pd.Series, Dict[str, Any]]] = None,
) -> Any:
    """Evaluate a single RapidMiner-style expression.

    Steps performed (in order):
      1. Macro substitution: every ``%{name}`` is replaced with the literal
         macro value taken from `macros`.
      2. Reference rewriting: ``[col]``, ``attribute("col")`` and ``#{col}``
         tokens are rewritten into ``_attr("col")`` calls. ``_attr`` reads
         from `row` (a pandas Series or dict).
      3. Python `eval` with a sandboxed namespace exposing
         :data:`RM_FUNCTIONS` plus the row helper.

    Returns whatever the expression produces — typically a float, bool, or
    string. Exceptions are caught and turned into NaN (RapidMiner's silent
    failure mode for `parse(...)` over a bad value).
    """
    if expression is None:
        return float("nan")
    if not isinstance(expression, str):
        return expression
    expr = expression

    # Step 1: macro substitution
    expr = macros.substitute(expr)

    # Step 2: rewrite attribute references → _attr("name")
    expr = _ATTR_BRACKET.sub(lambda m: f'_attr("{m.group(1)}")', expr)
    expr = _ATTR_FN.sub(lambda m: f'_attr("{m.group(1)}")', expr)
    expr = _ATTR_HASH.sub(lambda m: f'_attr("{m.group(1)}")', expr)

    # Step 3: eval inside a restricted namespace
    row_map: Dict[str, Any] = {}
    if isinstance(row, pd.Series):
        # pandas Series .to_dict() handles NaN, ints, etc.
        row_map = row.to_dict()
    elif isinstance(row, dict):
        row_map = row

    def _attr(name: str) -> Any:
        v = row_map.get(name, float("nan"))
        if isinstance(v, str):
            # numeric strings → float for arithmetic, otherwise leave as string
            try:
                return float(v)
            except ValueError:
                return v
        return v

    # Build the eval namespace. Bare identifiers in RM expressions can refer
    # to columns of the current row (e.g. `Mixed_Feed_Old + Feed_Bias_Feed_Grid`
    # in Generate Attributes (20)), so we expose every row column as a name in
    # the namespace, on top of `_attr("...")` for bracketed references.
    namespace: Dict[str, Any] = {**RM_FUNCTIONS, "_attr": _attr, "math": math}
    for col, val in row_map.items():
        # Only valid Python identifiers
        if col and isinstance(col, str) and col.isidentifier() and col not in namespace:
            if isinstance(val, str):
                try:
                    namespace[col] = float(val)
                except ValueError:
                    namespace[col] = val
            else:
                namespace[col] = val

    # Python uses `**` for power, RM uses `^`. Rewrite top-level `^` to `**`
    # (this is safe because the .rmp doesn't use bitwise XOR).
    expr_py = expr.replace("^", "**")

    # RapidMiner uses `&&` / `||` for logical and/or; Python expects `and` /
    # `or`. We add surrounding spaces to avoid token gluing (e.g. `a&&b` →
    # `a and b`, not `aandb`).
    expr_py = expr_py.replace("&&", " and ").replace("||", " or ")

    # `if` is a Python keyword, so we rewrite RM's `if(` → `_rm_if_(`. The
    # function lives in `namespace` under that name. We use a word-boundary
    # regex so we don't touch `notif(`, identifiers, etc.
    expr_py = re.sub(r"\bif\(", "_rm_if_(", expr_py)

    try:
        return eval(expr_py, {"__builtins__": {}}, namespace)        # noqa: S307
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# ExampleSet store — emulates RapidMiner Remember/Recall objects
# ─────────────────────────────────────────────────────────────────────────────

class StoreRegistry:
    """In-process registry used by every Remember/Recall in the process."""

    def __init__(self) -> None:
        self._objects: Dict[str, Any] = {}

    def remember(self, name: str, obj: Any) -> None:
        self._objects[name] = obj

    def recall(self, name: str, default: Any = None) -> Any:
        if name not in self._objects and default is None:
            raise KeyError(f"Recall: no object stored under name {name!r}")
        return self._objects.get(name, default)

    def has(self, name: str) -> bool:
        return name in self._objects

    def remove(self, name: str) -> None:
        self._objects.pop(name, None)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation primitives — emulate RM Aggregate operator
# ─────────────────────────────────────────────────────────────────────────────

_AGG_FUNCS = {
    "sum":     lambda s: pd.to_numeric(s, errors="coerce").sum(),
    "average": lambda s: pd.to_numeric(s, errors="coerce").mean(),
    "mean":    lambda s: pd.to_numeric(s, errors="coerce").mean(),
    "minimum": lambda s: pd.to_numeric(s, errors="coerce").min(),
    "maximum": lambda s: pd.to_numeric(s, errors="coerce").max(),
    "count":   lambda s: s.count(),
    "first":   lambda s: s.iloc[0] if len(s) else None,
    "last":    lambda s: s.iloc[-1] if len(s) else None,
}


def aggregate(
    df: pd.DataFrame,
    aggregations: List[tuple],            # list of (column, function_name)
    group_by: Optional[List[str]] = None,
    ignore_missings: bool = True,
) -> pd.DataFrame:
    """Replicate the RM Aggregate operator.

    Produces columns named like RapidMiner does (e.g. ``sum(Mixed_Feed_Old)``)
    so downstream Extract Macro operators that read those attributes by name
    find them in the right place.
    """
    if df.empty:
        return df.head(0)

    if group_by:
        groups = df.groupby(group_by, dropna=False)
        out_records: List[Dict[str, Any]] = []
        for key, sub in groups:
            rec: Dict[str, Any] = {}
            if isinstance(key, tuple):
                for k_name, k_val in zip(group_by, key):
                    rec[k_name] = k_val
            else:
                rec[group_by[0]] = key
            for col, func in aggregations:
                fn = _AGG_FUNCS[func]
                rec[f"{func}({col})"] = fn(sub[col]) if col in sub.columns else None
            out_records.append(rec)
        return pd.DataFrame(out_records)

    # No group-by → single-row result
    rec = {}
    for col, func in aggregations:
        fn = _AGG_FUNCS[func]
        rec[f"{func}({col})"] = fn(df[col]) if col in df.columns else None
    return pd.DataFrame([rec])


def extract_macro_from_dataset(
    df: pd.DataFrame,
    macros: MacroStore,
    macro_name: str,
    attribute_name: str,
    macro_type: str = "data_value",
    example_index: int = 1,
    additional: Optional[Dict[str, str]] = None,
) -> None:
    """Replicate the RM Extract Macro operator (data_value / number_of_examples).

    Writes one or more macros to the store from a small ExampleSet.
    """
    if macro_type == "number_of_examples":
        macros.set(macro_name, len(df))
        return

    if df.empty:
        macros.set(macro_name, "")
        if additional:
            for m_name, _attr in additional.items():
                macros.set(m_name, "")
        return

    idx = example_index - 1
    if idx >= len(df):
        idx = 0
    row = df.iloc[idx]

    if attribute_name and attribute_name in df.columns:
        macros.set(macro_name, row[attribute_name])

    if additional:
        for m_name, attr in additional.items():
            if attr and attr in df.columns:
                macros.set(m_name, row[attr])


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: applying a Generate Attributes list to a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def apply_generate_attributes(
    df: pd.DataFrame,
    formulas: List[tuple],          # list of (new_attr_name, expression_str)
    macros: MacroStore,
    keep_all_columns: bool = True,
) -> pd.DataFrame:
    """Replicate the RM Generate Attributes operator.

    Formulas are evaluated **in order**, row-wise. Later formulas can refer
    to earlier ones (RapidMiner semantics).
    """
    if df.empty:
        # Add empty columns so downstream operators don't KeyError
        for name, _expr in formulas:
            if name not in df.columns:
                df[name] = pd.Series(dtype=float)
        return df

    out = df.copy()
    for new_attr, expr in formulas:
        # The new attribute name itself can be a macro-substituted token:
        resolved_name = macros.substitute(new_attr)

        # Evaluate expression for every row
        new_vals: List[Any] = []
        for _, row in out.iterrows():
            new_vals.append(evaluate_expression(expr, macros, row))
        out[resolved_name] = new_vals

    if not keep_all_columns:
        wanted = [macros.substitute(name) for name, _ in formulas]
        out = out[wanted]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: filter_examples replacement
# ─────────────────────────────────────────────────────────────────────────────

# Map of RM custom-filter operators → pandas filter expressions
_FILTER_OP_MAP = {
    "equals": lambda series, val: series.astype(str) == str(val),
    "eq":     lambda series, val: pd.to_numeric(series, errors="coerce") == float(val),
    "contains": lambda series, val: series.astype(str).str.contains(re.escape(str(val))),
    "is_not_missing": lambda series, _: series.notna() & (series.astype(str).str.strip() != ""),
    "is_missing":     lambda series, _: series.isna() | (series.astype(str).str.strip() == ""),
}


def apply_filter_examples(
    df: pd.DataFrame,
    filters: List[str],
    macros: MacroStore,
    invert: bool = False,
    logic_and: bool = True,
) -> pd.DataFrame:
    """Apply RapidMiner ``filter_examples`` custom-filter list.

    Each filter is encoded as a dot-separated string ``<attr>.<op>.<value>``.
    Macros inside the value (``%{name}``) are substituted first.
    """
    if df.empty:
        return df

    mask = None
    for f in filters:
        # `attr.op.value` or `attr.op.`
        parts = f.split(".", 2)
        if len(parts) < 2:
            continue
        attr = parts[0]
        op = parts[1]
        val = parts[2] if len(parts) > 2 else ""
        val = macros.substitute(val)
        if attr not in df.columns:
            sub_mask = pd.Series(False, index=df.index)
        else:
            op_fn = _FILTER_OP_MAP.get(op)
            if op_fn is None:
                sub_mask = pd.Series(False, index=df.index)
            else:
                try:
                    sub_mask = op_fn(df[attr], val).fillna(False)
                except Exception:
                    sub_mask = pd.Series(False, index=df.index)
        if mask is None:
            mask = sub_mask
        else:
            mask = (mask & sub_mask) if logic_and else (mask | sub_mask)

    if mask is None:
        mask = pd.Series(True, index=df.index)
    if invert:
        mask = ~mask
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Coilsim model interface — abstraction over the four pre-trained models
# referenced from the RM repository
#   //ing_manufacturing_furnace_2024_dev/02_Model_Files/163_United_Olf_Furnace_System/
#       482_Main_<Y>                        ← 4 different output models (per Y)
#       482_Main_Fur_Coil_Normalized        ← shared normalization parameters
#
# In RapidMiner these are PreprocessingModel + RegressionModel objects.
# In Python we expose a single callable interface that downstream code
# can plug into without caring about the underlying tech (scikit-learn,
# ONNX, an HTTP call, etc.).
# ─────────────────────────────────────────────────────────────────────────────

class CoilsimModelProvider:
    """Pluggable interface mirroring the RapidMiner Normalized Model block.

    Concrete deployments will subclass this and load the actual models.
    The reference implementation only exposes the formula-based fallback,
    matching RapidMiner's ``model/egn`` branch when ``use_model='inactive'``.
    """

    def predict(self, df: pd.DataFrame, y_name: str) -> pd.Series:
        """Apply ``482_Main_<y_name>`` to ``df``. Override in production."""
        raise NotImplementedError(
            "CoilsimModelProvider.predict must be supplied for live runs. "
            "Fall back to the formula path (model/egn=inactive) for now."
        )

    def available(self, y_name: str) -> bool:
        """Whether a model for the given Y exists."""
        return False
