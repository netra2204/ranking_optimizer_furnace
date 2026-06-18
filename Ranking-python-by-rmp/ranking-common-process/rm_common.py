"""
rm_common.py
================================================================================
Shared building blocks for the Python replica of the RapidMiner process
`ranking-common-process-subpart.rmp`.

Contains everything used by BOTH process scripts:
  * Macros        - RapidMiner %{macro} scope (with token resolution)
  * IOStore       - RapidMiner Remember/Recall repository
  * expression engine        - eval of Generate Attributes / Generate Macro
  * op_*()        - generic emulations of individual RapidMiner operators
  * execute_feature_log / _log_block - telemetry side-effects

Imported by:  parameterization.py, ranking.py, main.py
================================================================================
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("rm_replica")

# RapidMiner uses these constants for "missing"; we map them to NaN.
MISSING_NUMERIC = np.nan
RM_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"   # yyyy-MM-dd HH:mm:ss


# =============================================================================
#  Macro scope  (RapidMiner %{macro} variables)
# =============================================================================
class Macros(dict):
    """RapidMiner macro scope. Values are stored/used as strings, mirroring RM."""

    def resolve(self, text: str) -> str:
        """Substitute every %{macro} token in `text` with its current value."""
        if text is None:
            return text

        def repl(m: "re.Match") -> str:
            key = m.group(1)
            return str(self.get(key, m.group(0)))

        return re.sub(r"%\{([^}]+)\}", repl, str(text))


# =============================================================================
#  IO store  (RapidMiner Remember / Recall repository)
# =============================================================================
class IOStore(dict):
    """Key/value store backing the `recall` operator. Mirrors RM Repository."""

    def recall(self, name: str) -> pd.DataFrame:
        if name not in self:
            raise KeyError(
                f"Recall: object '{name}' is not present in the IOStore. "
                f"It must be Remember-ed by the parent process before this "
                f"sub-part runs."
            )
        return self[name].copy()


# =============================================================================
#  RapidMiner expression / formula engine
# =============================================================================
# RapidMiner "Generate Attributes" and "Generate Macro" use a small expression
# language. We translate the handful of functions actually used in this process
# (if, lower, attribute, eval, concat, date_now) into Python and evaluate them
# per-row (for column generation) or once (for macros).

def _rm_if(cond: Any, a: Any, b: Any) -> Any:
    return a if cond else b


def _rm_lower(s: Any) -> str:
    return str(s).lower()


def _rm_concat(*parts: Any) -> str:
    return "".join("" if p is None else str(p) for p in parts)


def _rm_date_now() -> str:
    return datetime.now().strftime(RM_DATE_FORMAT)


def eval_generate_attribute(df: pd.DataFrame,
                            expr: str,
                            macros: Macros) -> pd.Series:
    """
    Evaluate one RapidMiner Generate-Attributes expression across all rows.

    Supported constructs (the only ones this process uses):
      * if(cond, a, b)            -> _rm_if
      * lower(x)                  -> _rm_lower
      * attribute("name") / attribute(expr)  -> column reference (dynamic)
      * eval("123") / eval(macro) -> numeric coercion of a value
      * concat(a, "_", b)         -> string concatenation
      * column references [Name]  -> df["Name"]
      * arithmetic + - * / and == comparisons
      * MISSING_NUMERIC           -> NaN
    Macros are substituted first, so by the time we eval the string the only
    dynamic parts left are column references.
    """
    expr = macros.resolve(expr)

    # `attribute("X")` and `attribute(X)` -> dynamic column lookup helper.
    # `eval(X)` -> float(X). `concat(...)`-> _rm_concat.
    def make_row_evaluator() -> Callable[[pd.Series], Any]:
        local_pat_attr = re.compile(r'attribute\(\s*"?([^")]+)"?\s*\)')

        def row_eval(row: pd.Series) -> Any:
            scope: Dict[str, Any] = {
                "if": _rm_if,
                "lower": _rm_lower,
                "concat": _rm_concat,
                "eval": lambda x: float(x) if str(x).strip() not in ("", "nan") else np.nan,
                "MISSING_NUMERIC": np.nan,
                "true": True,
                "false": False,
            }
            # expose every column as a bare identifier AND via [Name]
            safe_cols = {}
            for c in row.index:
                scope[_safe_ident(c)] = row[c]
                safe_cols[c] = row[c]

            def attribute(name: str) -> Any:
                return row.get(name, np.nan)

            scope["attribute"] = attribute

            # turn [Col Name] -> __col__('Col Name')
            local_expr = re.sub(r"\[([^\]]+)\]",
                                lambda m: f"__col__({m.group(1)!r})", expr)
            scope["__col__"] = lambda name: row.get(name, np.nan)
            # turn attribute("X") into __col__('X') as well (string-literal form)
            local_expr = local_pat_attr.sub(
                lambda m: f"__col__({m.group(1)!r})", local_expr)
            # bare identifiers that are columns already injected into scope
            local_expr = _rm_to_python(local_expr)
            try:
                return eval(local_expr, {"__builtins__": {}}, scope)  # noqa: S307
            except Exception:               # pragma: no cover - mirror RM "missing"
                return np.nan

        return row_eval

    if len(df) == 0:
        return pd.Series([], dtype="object")
    return df.apply(make_row_evaluator(), axis=1)


def eval_generate_macro(expr: str, macros: Macros) -> str:
    """
    Evaluate a Generate-Macro expression. Result is always stored as a string
    (RapidMiner macros are strings). Handles the string-building accumulator
    used for `overall_ranking_score` and the numeric `sorting_selector`.
    """
    resolved = macros.resolve(expr)
    scope = {
        "if": _rm_if,
        "lower": _rm_lower,
        "concat": _rm_concat,
        "date_now": _rm_date_now,
        "eval": lambda x: float(x),
    }
    try:
        value = eval(_rm_to_python(resolved), {"__builtins__": {}}, scope)  # noqa: S307
    except Exception:
        # The overall_ranking_score accumulator is a pure string expression
        # such as  "0"+"+"+"["+param+"_score"+"]"  -> RM concatenates strings.
        value = resolved
    return str(value)


def _safe_ident(col: str) -> str:
    return re.sub(r"\W", "_", str(col))


def _rm_to_python(expr: str) -> str:
    """Light syntactic translation of RM expression operators to Python."""
    # RM uses == and != already compatible; string concat with + is fine.
    # RM `&&` / `||` -> Python and/or (not used here but safe to translate).
    expr = expr.replace("&&", " and ").replace("||", " or ")
    return expr


# =============================================================================
#  Generic RapidMiner operator emulations (reusable building blocks)
# =============================================================================

def op_create_exampleset_attribute_functions(
        function_descriptions: Dict[str, str],
        macros: Macros,
        number_of_examples: int = 1) -> pd.DataFrame:
    """
    `utility:create_exampleset` (generator_type = attribute functions).
    Builds an N-row table; each attribute is a small expression (date_now(),
    a macro reference, or a string literal). Used by the log-start/log-end
    blocks to build the single-row telemetry record.
    """
    row: Dict[str, Any] = {}
    for col, expr in function_descriptions.items():
        row[col] = eval_generate_macro(expr, macros)
    return pd.DataFrame([row] * int(number_of_examples)).reset_index(drop=True)


def op_create_exampleset_csv(input_csv_text: str,
                             column_separator: str = ",") -> pd.DataFrame:
    """
    `utility:create_exampleset` (generator_type = comma separated text).
    The CSV here is a single column 'suffix' with rows 'score' and 'rank'.
    """
    lines = [ln for ln in input_csv_text.splitlines() if ln != ""]
    header, *rows = lines
    cols = header.split(column_separator)
    data = [r.split(column_separator) for r in rows]
    return pd.DataFrame(data, columns=cols)


def op_numerical_to_real(df: pd.DataFrame) -> pd.DataFrame:
    """`numerical_to_real` (all numeric attributes -> float)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float)
    return out


def op_numerical_to_polynominal(df: pd.DataFrame) -> pd.DataFrame:
    """`numerical_to_polynominal` (all numeric attributes -> nominal/string)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            # RM renders integers without trailing .0 when they are whole.
            out[c] = out[c].map(_num_to_nominal)
    return out


def _num_to_nominal(v: Any) -> Any:
    if pd.isna(v):
        return v
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def op_de_pivot(df: pd.DataFrame,
                value_regex: str,
                index_attribute: str,
                keep_missings: bool = True,
                id_like: Optional[List[str]] = None) -> pd.DataFrame:
    """
    `de_pivot` : un-pivot (wide -> long).
    `value_regex` selects which columns become rows; columns NOT matched are
    kept as identifier columns. `index_attribute` is the name of the new column
    that receives the original column names; the values go into 'value'.
    """
    pattern = re.compile(value_regex)
    value_cols = [c for c in df.columns if pattern.match(str(c))]
    id_cols = [c for c in df.columns if c not in value_cols]
    long = df.melt(id_vars=id_cols,
                   value_vars=value_cols,
                   var_name=index_attribute,
                   value_name="value")
    if not keep_missings:
        long = long.dropna(subset=["value"])
    return long.reset_index(drop=True)


def op_pivot(df: pd.DataFrame,
             group_by: List[str],
             column_grouping: str,
             value_attribute: str = "value",
             agg: str = "first") -> pd.DataFrame:
    """
    `blending:pivot` : long -> wide.
    Cell aggregation here is always `first`. RM names the resulting columns
    `first(value)_<columnvalue>`; that prefix is stripped later by
    `Rename by Replacing`.
    """
    agg_func = {"first": "first"}.get(agg, "first")
    wide = (df
            .pivot_table(index=group_by,
                         columns=column_grouping,
                         values=value_attribute,
                         aggfunc=agg_func)
            .reset_index())
    wide.columns.name = None
    # Re-create RapidMiner's generated column name prefix.
    rename = {c: f"{agg}({value_attribute})_{c}"
              for c in wide.columns if c not in group_by}
    return wide.rename(columns=rename)


def op_rename_by_replacing(df: pd.DataFrame, replace_what: str,
                           replace_by: str = "") -> pd.DataFrame:
    """
    `rename_by_replacing` : regex-rename of attribute names.
    `replace_what` arrives RM-escaped, e.g.  first\\(value\\)_  meaning the
    literal text 'first(value)_'.
    """
    pat = re.compile(replace_what)
    return df.rename(columns=lambda c: pat.sub(replace_by, str(c)))


def op_rename(df: pd.DataFrame, mapping: Dict[str, str],
              macros: Macros) -> pd.DataFrame:
    """`blending:rename` : explicit attribute renames (macro-aware)."""
    resolved = {macros.resolve(k): macros.resolve(v) for k, v in mapping.items()}
    return df.rename(columns=resolved)


def op_select_attributes(df: pd.DataFrame,
                         subset: List[str],
                         include: bool,
                         macros: Macros) -> pd.DataFrame:
    """
    `blending:select_attributes`.
    include=True  -> keep only `subset`.
    include=False -> drop `subset` (exclude).
    """
    subset = [macros.resolve(s) for s in subset]
    if include:
        keep = [c for c in df.columns if c in subset]
        return df[keep].copy()
    return df.drop(columns=[c for c in subset if c in df.columns]).copy()


def op_join(left: pd.DataFrame, right: pd.DataFrame,
            keys: Dict[str, str], join_type: str,
            macros: Macros,
            remove_double_attributes: bool = True) -> pd.DataFrame:
    """
    `concurrency:join`.
    `keys` maps left-key -> right-key (each may be a macro). join_type is one
    of inner/left/right/outer. remove_double_attributes drops duplicated
    non-key columns coming from the right side (RM default behaviour).
    """
    left_keys = [macros.resolve(k) for k in keys.keys()]
    right_keys = [macros.resolve(v) for v in keys.values()]

    how = {"inner": "inner", "left": "left", "right": "right",
           "outer": "outer"}[join_type]

    if remove_double_attributes:
        dup = [c for c in right.columns
               if c in left.columns and c not in right_keys]
        right = right.drop(columns=dup)

    merged = left.merge(right, how=how,
                        left_on=left_keys, right_on=right_keys,
                        suffixes=("", "_from_right"))
    # RM does not keep the right-hand key columns when they are merely keys.
    drop_right_keys = [k for k in right_keys
                       if k not in left_keys and k in merged.columns
                       and k not in left.columns]
    return merged.drop(columns=drop_right_keys, errors="ignore").reset_index(drop=True)


def op_cartesian(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """`cartesian_product` : full cross join."""
    return left.merge(right, how="cross")


def op_remove_duplicates(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """`remove_duplicates` over the given attribute subset (keep first)."""
    subset = [c for c in subset if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def op_append(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """`append` : row-wise union of all input ExampleSets (merge_type=all)."""
    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def op_merge_attributes(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    `operator_toolbox:merge` : column-wise side-by-side merge of ExampleSets
    that share the same rows (duplicate attributes are renamed, first special
    role kept). Implemented as positional column concatenation.
    """
    frames = [f.reset_index(drop=True) for f in frames if f is not None]
    if not frames:
        return pd.DataFrame()
    out = frames[0].copy()
    for f in frames[1:]:
        for c in f.columns:
            new_name = c
            while new_name in out.columns:
                new_name = f"{new_name}_2"     # RM "rename" duplicate handling
            out[new_name] = f[c].values
    return out


# ---- Filter Examples (custom_filters) ---------------------------------------
_FILTER_OP = {
    "equals": lambda s, v: s.astype(str) == str(v),
    "does_not_equal": lambda s, v: s.astype(str) != str(v),
    "ne": lambda s, v: pd.to_numeric(s, errors="coerce") != _to_num(v),
    "eq": lambda s, v: pd.to_numeric(s, errors="coerce") == _to_num(v),
}


def _to_num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _parse_filter_entry(entry: str, macros: Macros):
    """A filter entry is 'attribute.operator.value' (each part macro-resolved)."""
    entry = macros.resolve(entry)
    attr, operator, value = entry.split(".", 2)
    return attr, operator, value


def op_filter_examples(df: pd.DataFrame,
                       filter_entries: List[str],
                       macros: Macros,
                       logic_and: bool = True,
                       invert: bool = False):
    """
    `filter_examples` (custom_filters).
    Returns (matched_df, unmatched_df) so that BOTH the 'example set output'
    and the 'unmatched example set' output ports can be wired, exactly as the
    RapidMiner process does.
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    masks = []
    for entry in filter_entries:
        attr, operator, value = _parse_filter_entry(entry, macros)
        if attr not in df.columns:
            masks.append(pd.Series(False, index=df.index))
            continue
        masks.append(_FILTER_OP[operator](df[attr], value))

    combined = masks[0]
    for m in masks[1:]:
        combined = (combined & m) if logic_and else (combined | m)
    if invert:
        combined = ~combined

    matched = df[combined].reset_index(drop=True)
    unmatched = df[~combined].reset_index(drop=True)
    return matched, unmatched


def op_sort(df: pd.DataFrame, by: str, ascending: bool,
            macros: Macros) -> pd.DataFrame:
    """`blending:sort` on a single (macro-resolved) attribute."""
    by = macros.resolve(by)
    if by not in df.columns:
        return df.copy()
    return df.sort_values(by=by, ascending=ascending,
                          kind="mergesort").reset_index(drop=True)


def op_generate_id(df: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """
    `blending:generate_id` : add a 1-based integer 'id' attribute (special
    role = id). The downstream `Set Role id=regular` demotes it to a normal
    column, which is the behaviour relied on here.
    """
    out = df.copy()
    out.insert(0, "id", range(1 + offset, len(out) + 1 + offset))
    return out


def op_normalize_range(df: pd.DataFrame, lo: float = 0.0,
                       hi: float = 1.0) -> pd.DataFrame:
    """`normalize` (range transformation) of all numeric attributes to [lo,hi]."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            cmin, cmax = out[c].min(), out[c].max()
            if pd.notna(cmin) and pd.notna(cmax) and cmax != cmin:
                out[c] = lo + (out[c] - cmin) * (hi - lo) / (cmax - cmin)
            else:
                out[c] = lo
    return out


def op_parse_numbers(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """`parse_numbers` : parse a nominal attribute into a number (fail mode)."""
    out = df.copy()
    if attribute in out.columns:
        out[attribute] = pd.to_numeric(out[attribute], errors="raise")
    return out


def op_date_to_nominal(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """`date_to_nominal` : format a datetime attribute to a string."""
    out = df.copy()
    if attribute in out.columns:
        out[attribute] = pd.to_datetime(out[attribute]).dt.strftime(RM_DATE_FORMAT)
    return out


def op_nominal_to_date(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """`nominal_to_date` : parse a string attribute into a datetime."""
    out = df.copy()
    if attribute in out.columns:
        out[attribute] = pd.to_datetime(out[attribute], format=RM_DATE_FORMAT,
                                        errors="coerce")
    return out


def op_generate_columns(df: pd.DataFrame,
                        function_descriptions: Dict[str, str],
                        macros: Macros) -> pd.DataFrame:
    """
    `blending:generate_columns` / `Generate Attributes` (keep_all_columns=True).
    Each entry  new_attr = expression  is evaluated row-wise. Attribute names
    themselves may contain macros (e.g. `%{parameter_name}_score`).
    """
    out = df.copy()
    for raw_name, expr in function_descriptions.items():
        name = macros.resolve(raw_name)
        out[name] = eval_generate_attribute(out, expr, macros)
    return out


def op_extract_macro_data_value(df: pd.DataFrame,
                                attribute_name: str,
                                example_index: int,
                                additional: Dict[str, str],
                                macros: Macros) -> None:
    """
    `extract_macro` (macro_type = data_value).
    Reads a single cell (example_index is 1-based) and writes it to a macro,
    plus the `additional_macros` which read other columns of the SAME row.
    Mutates `macros` in place (RM macros are global scope).
    """
    if len(df) == 0:
        return
    idx = min(example_index - 1, len(df) - 1)
    macros["parameter_name"] = str(df.iloc[idx][attribute_name])
    for macro_name, col in additional.items():
        if col in df.columns:
            macros[macro_name] = str(df.iloc[idx][col])


# =============================================================================
#  External / non-mappable operator
# =============================================================================
def execute_feature_log(log_record: pd.DataFrame, macros: Macros) -> None:
    """
    [productivity:execute_process]  "Execute feature_log (12/23/13/24)"
    ---------------------------------------------------------------------------
    RapidMiner runs the external process  ../../01_Common_Processes/feature_log_trg
    purely for telemetry. Its output is never wired back into the data flow.
    No direct Python equivalent exists for an external RM process, so we treat
    it as a logging side-effect (Assumption: logging only, no data contract).
    """
    if len(log_record):
        rec = log_record.iloc[0].to_dict()
        LOG.info("feature_log -> %s", rec)


# =============================================================================
#  LOG-START / LOG-END telemetry sub-blocks (shared shape)
# =============================================================================
def _log_block(process_name: str, info: str, macros: Macros) -> None:
    """
    Reproduces the  "log start (N)" / "log end (N)"  sub-process:
      [37/79/38/80] Create ExampleSet (attribute functions)  ->
      [12/23/13/24] Execute feature_log
    """
    # ---- [Create ExampleSet] (utility:create_exampleset) --------------------
    local = Macros(macros)
    local["process_name"] = f'"{process_name}"'
    record = op_create_exampleset_attribute_functions(
        function_descriptions={
            "timestamp": "date_now()",
            "level": "%{process_name}",
            "process_name": f'"{process_name}"',
            "category": '"RM"',
            "case_id": "%{case_id}",
            "info": f'"{info}"',
            "model_id": "%{ranking_model_id}",
        },
        macros=local,
        number_of_examples=1,
    )
    # ---- [Execute feature_log] (productivity:execute_process) ---------------
    execute_feature_log(record, macros)


