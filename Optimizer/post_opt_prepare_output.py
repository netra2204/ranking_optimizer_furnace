"""
RapidMiner → Python conversion of:
    1. post-optimizer.rmp   (run first)
    2. prepare-output.rmp   (consumes post-optimizer output)

Macros are seeded from Macros.xlsx (sheet 'Optimizer').

Execution order:  post_optimizer(...)  ->  prepare_output(...)
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Macros: load + helpers (RapidMiner macro registry)
# ---------------------------------------------------------------------------

def load_macros(xlsx_path: str, sheet: str = "Optimizer") -> Dict[str, str]:
    """Read Macros.xlsx → dict of name -> str(value). All RM macros are nominal."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return {str(r["name"]): str(r["value"]) for _, r in df.iterrows()}


def _eval_num(macros: Dict[str, str], key: str, default: float = 0.0) -> float:
    """RapidMiner eval(%{x}) — coerce macro value to float."""
    if key not in macros or macros[key] in (None, "", "NaN", "nan"):
        return default
    try:
        return float(macros[key])
    except (ValueError, TypeError):
        return default


def _feature_log(category: str, process_name: str, info: str,
                 case_id: str = "", model_id: str = "") -> None:
    """Stub for RM 'feature_log_trg' Execute Process — replace with real logger."""
    # Original RM step writes a 1-row ExampleSet to a logging endpoint.
    pass


# ---------------------------------------------------------------------------
# POST-OPTIMIZER
# ---------------------------------------------------------------------------

def _benefit_calculations(df_slice: pd.DataFrame, macros: Dict[str, str]) -> None:
    """
    Mirrors subprocess 'Benefit_calculations':
      - Select Attributes (tag, value)
      - Set Macros from ExampleSet (tag -> value)
      - Generate Macro Total_benefit_Per_day_all_furnace =
            sum(Fur1..Fur9 _Total_Benefit_Per_Day_Result_actual)

    Updates `macros` in place. Note: Generate Macro (56) is deactivated in
    the .rmp (activated="false"), so it is intentionally not applied.
    """
    # === Select Attributes (360): keep only [tag, value] ===
    sub = df_slice[["tag", "value"]].copy()

    # === Set Macros from ExampleSet (8): for each row, macros[tag] = value ===
    macros.update({str(r["tag"]): str(r["value"]) for _, r in sub.iterrows()})

    # === Generate Macro (53): Total_benefit_Per_day_all_furnace =
    #     sum_{i=1..9} eval(%{Fur{i}_Total_Benefit_Per_Day_Result_actual}) ===
    total = 0.0
    for i in range(1, 10):
        total += _eval_num(macros, f"Fur{i}_Total_Benefit_Per_Day_Result_actual", 0.0)
    macros["Total_benefit_Per_day_all_furnace"] = str(total)


def _ccp_check(df_slice: pd.DataFrame,
               ccp_status_utd: pd.DataFrame,
               macros: Dict[str, str]) -> None:
    """
    Mirrors subprocess 'ccp_check (2)':
      1. Recall ccp_status_utd
      2. Filter ccp_status == 0
      3. Inner-join LEFT(df_slice) RIGHT(filtered ccp) on (sub_model_id, Timestamp)
      4. Branch min_examples >= 1:
           true  -> ccp_check_fin = 0   (a row matches → CCP is blocked)
           false -> ccp_check_fin = 1
    """
    # === Recall (254): retrieve 'ccp_status_utd' (passed as parameter) ===
    # === Filter Examples (141): ccp_status == 0 (CCP-blocked rows) ===
    right = ccp_status_utd[ccp_status_utd["ccp_status"] == 0]

    # === Join (58): inner join on (sub_model_id, Timestamp) ===
    joined = df_slice.merge(right, on=["sub_model_id", "Timestamp"], how="inner")

    # === Branch (99): if min_examples >= 1
    #     true  → Set Macro (15) ccp_check_fin = 0
    #     false → Set Macro (18) ccp_check_fin = 1 ===
    macros["ccp_check_fin"] = "0" if len(joined) >= 1 else "1"


def _zero_actual_and_make_furnace_optimum(coupled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors 'Subprocess (50)' — runs when the overall opportunity is below
    threshold OR ccp is blocked.

      a. Keep tag containing '_actual'
      b. Among those, split: rows where tag contains 'Total_Benefit'
            -> matched: set value = 0           (Generate Attributes (17))
            -> unmatched: untouched
         Append (59) merges them back.
      c. Multiply -> two copies:
            copy A: keep as-is (now-zeroed) -> append directly
            copy B: rename '_actual' → '_furnace_optimum' in `tag`
         Append (81) merges A + B.
    """
    # === Filter Examples (181): keep tag.contains "_actual" ===
    actual = coupled_df[coupled_df["tag"].astype(str).str.contains("_actual", na=False)].copy()

    # === Filter Examples (58): split on tag.contains "Total_Benefit" ===
    has_tb = actual["tag"].astype(str).str.contains("Total_Benefit", na=False)
    matched = actual[has_tb].copy()
    unmatched = actual[~has_tb].copy()

    # === Generate Attributes (17): value = 0 (on Total_Benefit rows) ===
    matched["value"] = 0

    # === Append (59): matched + unmatched ===
    appended_59 = pd.concat([matched, unmatched], ignore_index=True)

    # === Multiply (66): two copies of the appended set ===
    copy_a = appended_59.copy()
    copy_b = appended_59.copy()

    # === Replace (1): in copy_b, tag '_actual' → '_furnace_optimum' ===
    copy_b["tag"] = copy_b["tag"].astype(str).str.replace("_actual", "_furnace_optimum", regex=False)

    # === Append (81): copy_a + copy_b ===
    return pd.concat([copy_a, copy_b], ignore_index=True)


def _post_opt_transformation_one_timestamp(
    df_ts: pd.DataFrame,
    ccp_status_utd: pd.DataFrame,
    macros: Dict[str, str],
) -> pd.DataFrame:
    """
    Mirrors subprocess 'post_opt_transformation' for ONE timestamp slice.
    df_ts already has Timestamp_nominal dropped (Select Attributes (15)).

    Loop Values has reuse_results="false" → macros set INSIDE the iteration
    body (Set Macros from ExampleSet, Generate Macro, ccp_check_fin) must not
    leak across iterations. We work on a local macro view and merge nothing
    back into the caller's dict.
    """
    iter_macros = dict(macros)  # iteration-local view; outer dict untouched

    # === Multiply (64): branch the input two ways
    #     (left → Parse Numbers, right → Benefit_calculations) ===
    _benefit_calculations(df_ts, iter_macros)          # writes total + tag macros

    # === Parse Numbers (31): coupled_mode nominal → numeric
    #     unparsable_value_handling=skip attribute ⇒ coerce on failure ===
    parsed = df_ts.copy()
    parsed["coupled_mode"] = pd.to_numeric(parsed["coupled_mode"], errors="coerce")

    # === Filter Examples (86): coupled_mode == 1
    #     matched → coupled, unmatched → uncoupled ===
    mask = parsed["coupled_mode"] == 1
    coupled = parsed[mask].copy()
    uncoupled = parsed[~mask].copy()

    # === Multiply (65): branch coupled into ccp_check (2) and the main path ===
    _ccp_check(coupled, ccp_status_utd, iter_macros)

    # === Numerical to Real (11): cast coupled_mode + sub_model_id → float ===
    coupled["coupled_mode"] = coupled["coupled_mode"].astype(float)
    if "sub_model_id" in coupled.columns:
        coupled["sub_model_id"] = pd.to_numeric(coupled["sub_model_id"], errors="coerce").astype(float)

    # === Branch (100): expression
    #     eval(%{Total_benefit_Per_day_all_furnace}) < eval(%{Uptime_opportunity_threshold})
    #         || %{ccp_check_fin} == 0
    #     TRUE  → Subprocess (50): zero actuals & duplicate as _furnace_optimum
    #     FALSE → pass through unchanged ===
    total = _eval_num(iter_macros, "Total_benefit_Per_day_all_furnace", 0.0)
    threshold = _eval_num(iter_macros, "Uptime_opportunity_threshold", 0.0)
    ccp_fin = _eval_num(iter_macros, "ccp_check_fin", 1.0)

    if total < threshold or ccp_fin == 0:
        coupled_out = _zero_actual_and_make_furnace_optimum(coupled)
    else:
        coupled_out = coupled

    # === Numerical to Real (12): cast coupled_mode + sub_model_id → float on unmatched ===
    if not uncoupled.empty:
        uncoupled["coupled_mode"] = uncoupled["coupled_mode"].astype(float)
        if "sub_model_id" in uncoupled.columns:
            uncoupled["sub_model_id"] = pd.to_numeric(uncoupled["sub_model_id"], errors="coerce").astype(float)

    # === Subprocess 'Appending both' ===
    # === Append (89): coupled_out + uncoupled ===
    appended = pd.concat([coupled_out, uncoupled], ignore_index=True)
    # === Select Attributes (361): exclude coupled_mode ===
    if "coupled_mode" in appended.columns:
        appended = appended.drop(columns=["coupled_mode"])
    # === Sort (31): by sub_model_id, ascending (stable) ===
    appended = appended.sort_values(by="sub_model_id", kind="mergesort").reset_index(drop=True)

    return appended


def post_optimizer(
    optimizer_output: pd.DataFrame,
    ccp_status_utd: pd.DataFrame,
    macros: Dict[str, str],
) -> pd.DataFrame:
    """
    Top-level entry point for post-optimizer.rmp.

    Outer guards (in order, exactly as in the .rmp):
      1. macro 'post_optimizer_transformation' is defined?       (macro_defined)
      2. macro 'post_optimizer_transformation'      == 'active'?
      3. macro 'post_optimizer_transformation_utd'  is defined?
      4. macro 'post_optimizer_transformation_utd'  == 'active'?

    If any guard fails the input is returned unchanged
    (true=process, false=pass-through).
    """
    # === Branch 'post_optimizer_transformation (3)' (macro_defined) ===
    if "post_optimizer_transformation" not in macros:
        return optimizer_output
    # === Branch (30) (expression: %{post_optimizer_transformation} == "active") ===
    if macros.get("post_optimizer_transformation") != "active":
        return optimizer_output
    # === Branch 'Post Optimizer Transformation UTD' (macro_defined) ===
    if "post_optimizer_transformation_utd" not in macros:
        return optimizer_output
    # === Branch 'post_optimizer_transformation (4)'
    #     (expression: %{post_optimizer_transformation_utd} == "active") ===
    if macros.get("post_optimizer_transformation_utd") != "active":
        return optimizer_output

    # === Subprocess 'Log_start (12)':
    #       → Create ExampleSet 'started at' (1-row log payload)
    #       → Execute Process 'Execute feature_log (3)' ===
    _feature_log("RM", "Inferred Tags Calculations", "startedAt",
                 macros.get("case_id", ""), macros.get("optimizer_model_id", ""))

    # === Date to Nominal (11): Timestamp → Timestamp_nominal 'yyyy-MM-dd HH:mm:ss'
    #     keep_old_attribute=true ===
    df = optimizer_output.copy()
    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp_nominal"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # === Loop Values (1) on Timestamp_nominal (reuse_results=false)
    #     → per-iteration: Filter Examples (87) on %{loop_value}
    #                      → Select Attributes (15) (exclude Timestamp_nominal)
    #                      → post_opt_transformation (subprocess)
    #     → Append (82) collects iteration outputs ===
    pieces = []
    for ts_val in df["Timestamp_nominal"].dropna().unique():
        # === Filter Examples (55): Timestamp_nominal.equals.%{loop_value} ===
        slice_ = df[df["Timestamp_nominal"] == ts_val].copy()
        # === Select Attributes (15): exclude Timestamp_nominal ===
        slice_ = slice_.drop(columns=["Timestamp_nominal"])
        pieces.append(_post_opt_transformation_one_timestamp(slice_, ccp_status_utd, macros))

    # === Append (82): concatenate all loop iterations ===
    result = pd.concat(pieces, ignore_index=True) if pieces else df.drop(columns=["Timestamp_nominal"])

    # === Subprocess 'Log_end (12)':
    #       → Create ExampleSet 'ended_at (13)' (1-row log payload)
    #       → Execute Process 'Execute feature_log' ===
    _feature_log("RM", "Inferred Tags Calculations", "endedAt",
                 macros.get("case_id", ""), macros.get("optimizer_model_id", ""))
    return result


# ---------------------------------------------------------------------------
# PREPARE-OUTPUT
# ---------------------------------------------------------------------------

def _sub_model_furnace_mapping(
    entity_subset_a: pd.DataFrame,
    model_alert_subset_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mirrors RM operator chain:
        Subprocess 'fetching funace_entity-id (2)' →
        Subprocess 'sub_model_furnace_mapping (2)' →
        Subprocess (185) + Loop (While) (6).

    Walks up the entity tree until we find a row whose entity_type is
    'furnace' or 'system', or until the loop stops (10 iterations or no
    unmatched rows left).
    """
    # === Select Attributes (595): keep only [entity_id, entity_type] ===
    et = entity_subset_a[["entity_id", "entity_type"]].copy()

    # === Join (180): inner join on entity_id (left=model_alert_subset_b, right=et) ===
    joined = model_alert_subset_b.merge(et, on="entity_id", how="inner")

    # === Generate Attributes (425): parent_entity_id = [entity_id]  (alias) ===
    joined["parent_entity_id"] = joined["entity_id"]

    # === Filter Examples (304): lower([entity_type]) == "furnace" || == "system"
    #     matched / unmatched split ===
    et_lower = joined["entity_type"].astype(str).str.lower()
    matched = joined[et_lower.isin(["furnace", "system"])].copy()
    unmatched = joined[~et_lower.isin(["furnace", "system"])].copy()

    # === Branch (252): condition = min_examples >= 1 on unmatched ===
    if len(unmatched) < 1:
        return matched

    # === Extract Macro (466): nEX = number_of_examples(unmatched) ===
    nEX = len(unmatched)
    # === Set Macro (33): iteration = 1 ===
    iteration = 1
    current_unmatched = unmatched
    collected = [matched]

    # === Loop (While) (6): stop_expression = (%{nEX}==0 || %{iteration}>=10) ===
    while not (nEX == 0 or iteration >= 10):
        # === Rename (198): parent_entity_id → child_entity_id ===
        renamed = current_unmatched.rename(columns={"parent_entity_id": "child_entity_id"})

        # === Select Attributes (596): exclude entity_type ===
        if "entity_type" in renamed.columns:
            renamed = renamed.drop(columns=["entity_type"])

        # === Subprocess (186) ===
        # === Select Attributes (597): from entity drop entity_type ===
        ent_no_type = entity_subset_a.drop(columns=["entity_type"], errors="ignore")

        # === Join (181): inner, left.child_entity_id == right.entity_id
        #     remove_double_attributes=true ===
        j1 = renamed.merge(
            ent_no_type, left_on="child_entity_id", right_on="entity_id",
            how="inner", suffixes=("", "__r"),
        )
        j1 = j1.loc[:, ~j1.columns.str.endswith("__r")]

        # === Select Attributes (599): exclude child_entity_id ===
        j1 = j1.drop(columns=["child_entity_id"], errors="ignore")

        # === Select Attributes (598): from entity drop parent_entity_id ===
        ent_no_parent = entity_subset_a.drop(columns=["parent_entity_id"], errors="ignore")

        # === Join (182): inner, left.parent_entity_id == right.entity_id
        #     remove_double_attributes=true ===
        j2 = j1.merge(
            ent_no_parent, left_on="parent_entity_id", right_on="entity_id",
            how="inner", suffixes=("", "__r"),
        )
        j2 = j2.loc[:, ~j2.columns.str.endswith("__r")]

        # === Filter Examples (305): lower([entity_type]) ∈ {furnace, system} ===
        et2 = j2["entity_type"].astype(str).str.lower()
        new_matched = j2[et2.isin(["furnace", "system"])].copy()
        new_unmatched = j2[~et2.isin(["furnace", "system"])].copy()

        # === Extract Macro (467): nEX = number_of_examples(new_unmatched) ===
        nEX = len(new_unmatched)

        # === Multiply (182): branch new_matched (one copy feeds Append (143),
        #     the other is implicit — pandas reference is shared) ===
        # === Append (143): collects each iteration's matched rows ===
        collected.append(new_matched)

        current_unmatched = new_unmatched
        iteration += 1

    return pd.concat(collected, ignore_index=True)


def _empty_model_alert_template() -> pd.DataFrame:
    """Create ExampleSet (160): 0-row template used when Branch (251) false-side fires."""
    return pd.DataFrame({
        "Timestamp":    pd.Series([], dtype="datetime64[ns]"),
        "tag":          pd.Series([], dtype="object"),
        "current":      pd.Series([], dtype="float64"),
        "ccp_info_id":  pd.Series([], dtype="float64"),
        "ccp_id":       pd.Series([], dtype="float64"),
        "parameter":    pd.Series([], dtype="object"),
        "logic":        pd.Series([], dtype="object"),
        "entity_id":    pd.Series([], dtype="float64"),
        "model_id":     pd.Series([], dtype="float64"),
    })


def _empty_output_template() -> pd.DataFrame:
    """Create ExampleSet (161): 1-row template appended on exception."""
    return pd.DataFrame({
        "Timestamp":    [pd.NaT],
        "sub_model_id": [np.nan],
        "tag":          [None],
        "value":        [np.nan],
    })


def _compute_model_status(row) -> int:
    """
    Generate Attributes (426) — nested if expression:
      if furnace_status==0 -> 4
      else if ccp_status==2 -> 2
      else if manual_switch==0 -> 3
      else if ccp_status==0 -> 3
      else 1
    """
    if row.get("furnace_status") == 0:
        return 4
    if row.get("ccp_status") == 2:
        return 2
    if row.get("manual_switch") == 0:
        return 3
    if row.get("ccp_status") == 0:
        return 3
    return 1


def prepare_output(
    optimizer_output_from_post: pd.DataFrame,
    model_alert_output: pd.DataFrame,
    entity: pd.DataFrame,
    furnace_selection: pd.DataFrame,
    opt_last_good_value: pd.DataFrame,
    macros: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mirrors prepare-output.rmp ('Prepare_output' > 'prepare_output (6)').

    Returns (out_1, out_2, out_3, out_4):
        out_1 — Output_format_check  (post-optimizer output, error-guarded)
        out_2 — opt_last_good_value  (Recall)
        out_3 — Subprocess (187)     (model status)
        out_4 — Subprocess (184)     (model alert output, deduped)
    """
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ Subprocess (184) : model_alert_output                                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # === Recall 'model skip status type (2)': retrieve 'model_alert_output' ===
    end_time = macros.get("end_time")
    ma = model_alert_output.copy()
    # === Filter Examples (303): date_diff(%{end_time}, [Timestamp]) >= 0 ===
    if "Timestamp" in ma.columns and end_time:
        if not np.issubdtype(ma["Timestamp"].dtype, np.datetime64):
            ma["Timestamp"] = pd.to_datetime(ma["Timestamp"], errors="coerce")
        end_dt = pd.to_datetime(end_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        ma = ma[ma["Timestamp"] <= end_dt]

    # === Branch (251): condition = min_examples >= 1 on filtered ma ===
    if len(ma) >= 1:
        # === Recall (337): retrieve 'entity' (passed as parameter) ===
        # === Multiply (178) → Select Attributes (593) (Subset A on entity) ===
        cols_a = ["child_entity_id", "entity_id", "entity_type", "parent_entity_id"]
        ent_a = entity[[c for c in cols_a if c in entity.columns]].copy()

        # === Multiply (179) → Select Attributes (594) (Subset B on model_alert_output) ===
        cols_b = ["ccp_id", "ccp_info_id", "entity_id", "logic",
                  "model_id", "parameter", "tag", "Timestamp", "current"]
        ma_b = ma[[c for c in cols_b if c in ma.columns]].copy()

        # === Subprocess (185) + Loop (While) (6) : Sub_model_furnace_mapping ===
        mapped = _sub_model_furnace_mapping(ent_a, ma_b)

        # === Select Attributes (600): keep final output subset ===
        keep_cols = ["ccp_id", "ccp_info_id", "current", "logic", "model_id",
                     "parent_entity_id", "tag", "Timestamp", "parameter"]
        mapped = mapped[[c for c in keep_cols if c in mapped.columns]].copy()

        # === Rename (199): parent_entity_id → entity_id ===
        mapped = mapped.rename(columns={"parent_entity_id": "entity_id"})
    else:
        # === Create ExampleSet (160): 0-row template (false branch) ===
        mapped = _empty_model_alert_template()

    # === Append (144): mapped (single source here) ===
    out_4 = mapped.copy()
    # === Remove Duplicates (62): keys = entity_id, Timestamp, tag
    #     treat_missing_values_as_duplicates=false ===
    dedup_keys = [k for k in ["entity_id", "Timestamp", "tag"] if k in out_4.columns]
    if dedup_keys:
        out_4 = out_4.drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ Subprocess (187) : model status                                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # === Recall (338): retrieve 'furnace_selection' (passed as parameter) ===
    fs = furnace_selection.copy()
    # === Generate Attributes (426): defines BOTH model_status (nested if) AND model_id
    #     keep_all_columns=true, default_time_zone="Asia/Riyadh" ===
    fs["model_status"] = fs.apply(_compute_model_status, axis=1).astype(int)
    fs["model_id"] = macros.get("optimizer_model_id", None)

    # === Select Attributes (601): keep [Timestamp, model_id, entity_id, model_status] ===
    keep_status = ["Timestamp", "model_id", "entity_id", "model_status"]
    out_3 = fs[[c for c in keep_status if c in fs.columns]].copy()

    # === Remove Duplicates (63): keys = entity_id, Timestamp ===
    dedup_status = [k for k in ["entity_id", "Timestamp"] if k in out_3.columns]
    if dedup_status:
        out_3 = out_3.drop_duplicates(subset=dedup_status, keep="first").reset_index(drop=True)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ Output_format_check (Handle Exception)                                ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # === Handle Exception : try block contains ===
    #     in 1 ──┐
    #            ├─► Append (145) ──► out 1
    #     Create ExampleSet (161) ──┘   (always a 1-row MISSING template)
    # === On the happy path RM ALWAYS appends the 1-row template to the input.
    # === Catch block: Throw Exception "Output is not in correct format".
    try:
        if optimizer_output_from_post is None:
            raise ValueError("Output is not in correct format")
        # === Append (145): input + Create ExampleSet (161) template ===
        out_1 = pd.concat(
            [optimizer_output_from_post, _empty_output_template()],
            ignore_index=True,
        )
    except Exception:
        # === Throw Exception (13): "Output is not in correct format"
        #     (swallowed here so downstream outputs remain available) ===
        import warnings
        warnings.warn("Output is not in correct format", RuntimeWarning)
        out_1 = _empty_output_template()

    # === Recall (339): retrieve 'opt_last_good_value' (passed as parameter) ===
    out_2 = opt_last_good_value.copy() if opt_last_good_value is not None else pd.DataFrame()

    # === Subprocess 'Log_end (21)':
    #       → Create ExampleSet 'ended_at (22)' (1-row log payload)
    #       → Execute Process 'Execute feature_log (49)' ===
    _feature_log("RM", "Optimizer_ind", "endedAt",
                 macros.get("case_id", ""), "0")

    return out_1, out_2, out_3, out_4


# ---------------------------------------------------------------------------
# Combined driver
# ---------------------------------------------------------------------------

def run_pipeline(
    optimizer_output: pd.DataFrame,
    ccp_status_utd: pd.DataFrame,
    model_alert_output: pd.DataFrame,
    entity: pd.DataFrame,
    furnace_selection: pd.DataFrame,
    opt_last_good_value: pd.DataFrame,
    macros_xlsx: str = "/mnt/user-data/uploads/Macros.xlsx",
    runtime_macros: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Load macros from xlsx (sheet 'Optimizer')
    2. Merge runtime_macros (e.g. case_id, optimizer_model_id, end_time, Fur*_*)
    3. post_optimizer → produces modified optimizer output
    4. prepare_output → produces 4 outputs
    """
    macros = load_macros(macros_xlsx, sheet="Optimizer")
    if runtime_macros:
        macros.update(runtime_macros)

    post_out = post_optimizer(optimizer_output, ccp_status_utd, macros)
    return prepare_output(
        post_out, model_alert_output, entity,
        furnace_selection, opt_last_good_value, macros,
    )
