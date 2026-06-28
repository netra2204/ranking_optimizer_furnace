"""
ranking.py
================================================================================
Replica of the RapidMiner top-level sub-process "ranking" and all of its
nested sub-processes:

    in 1 -> text_code_mapping (2) -> furnace_selection (2)
    furnace_selection out1 -> ranking_and _scoring -> Append(11)[set 1]
    furnace_selection out2 -> ranking_parameters -> Date to Nominal(2) -> Append(11)[set 2]
    Append(11) -> de-parameterization (2) -> Nominal to Date -> out 1

Nested helpers (same execution hierarchy as RapidMiner):
    sub_text_code_mapping, sub_furnace_selection, _sub_manual_filter_join,
    ranking_and_scoring -> _rank_one_group -> _score_one_parameter / _sorting_selector,
    ranking_parameters -> sub_all_parameters, sub_de_parameterization
================================================================================
"""
from __future__ import annotations

from typing import List
import os
import pandas as pd

from rm_common import (
    Macros, IOStore, LOG, _log_block, _to_num,
    op_create_exampleset_csv, op_cartesian, op_join, op_rename,
    op_select_attributes, op_filter_examples, op_sort, op_generate_id,
    op_generate_columns, op_normalize_range, op_parse_numbers, op_append,
    op_merge_attributes, op_remove_duplicates, op_pivot, op_de_pivot,
    op_rename_by_replacing, op_date_to_nominal, op_nominal_to_date,
    op_numerical_to_polynominal, op_extract_macro_data_value,
    eval_generate_macro,
)

def sub_text_code_mapping(df: pd.DataFrame, macros: Macros,
                          store: IOStore) -> pd.DataFrame:
    """
    [text_code_mapping (2)] (subprocess)

    Adds the decoded text for the furnace-status code and the splitter code
    by left-joining the `text_code_mapping` lookup twice.

    Flow:
      in 1 -> Join (6)[left]
      Recall(5) text_code_mapping -> Rename(4) text->furnace_status_text -> Join(6)[right]
      Join(6) -> Join(7)[left]
      Recall(6) text_code_mapping -> Rename(5) text->%{ranking_splitter}_text -> Join(7)[right]
      Join(7) -> out 1
    """
    # [Recall (5)] text_code_mapping
    map1 = store.recall("text_code_mapping")
    # [Rename (4)] text -> furnace_status_text
    map1 = op_rename(map1, {"text": "furnace_status_text"}, macros)
    # [Join (6)] left ; key %{furnace_status} = code
    df = op_join(df, map1,
                 keys={"%{furnace_status}": "code"},
                 join_type="left", macros=macros)

    # [Recall (6)] text_code_mapping
    map2 = store.recall("text_code_mapping")
    # [Rename (5)] text -> %{ranking_splitter}_text
    map2 = op_rename(map2, {"text": "%{ranking_splitter}_text"}, macros)
    # [Join (7)] left ; key %{ranking_splitter} = code
    df = op_join(df, map2,
                 keys={"%{ranking_splitter}": "code"},
                 join_type="left", macros=macros)
    return df          # -> out 1


# -----------------------------------------------------------------------------
#  ranking  -- nested sub-process: furnace_selection (2)
# -----------------------------------------------------------------------------
def _sub_manual_filter_join(macros: Macros, store: IOStore) -> pd.DataFrame:
    """
    [Subprocess (2)] inside Manual_Filtering_ :
      Recall(13) parameters     -> Join(21)[left]
      Recall(16) entity_parameter-> Join(21)[right]  inner key parameter_id
      Join(21) -> Join(23)[left]
      Recall(17) tag            -> Join(23)[right]    inner key name
      Join(23) -> out 1
    """
    parameters = store.recall("parameters")              # [Recall (13)]
    entity_parameter = store.recall("entity_parameter")  # [Recall (16)]
    joined = op_join(parameters, entity_parameter,       # [Join (21)] inner
                     keys={"parameter_id": "parameter_id"},
                     join_type="inner", macros=macros)
    tag = store.recall("tag")                            # [Recall (17)]
    joined = op_join(joined, tag,                        # [Join (23)] inner
                     keys={"name": "name"},
                     join_type="inner", macros=macros)
    return joined


def sub_furnace_selection(df: pd.DataFrame, macros: Macros,
                          store: IOStore):
    """
    [furnace_selection (2)] (subprocess)  -- TWO outputs.

    out 1 = the SELECTED rows (cracking furnaces running a valid splitter feed)
    out 2 = the appended "rejected/other" rows that bypass ranking

    Flow:
      in 1 -> Model_Skip_Filtering[cond]
      Model_Skip_Filtering:in1 -> Manual_Filtering[cond]
      Model_Skip_Filtering:in2 -> Append[set 3]
      Manual_Filtering:in1 -> Filter Examples (18)
      Manual_Filtering:in2 -> Append[set 4]
      Filter Examples (18) matched   -> Filter Examples (19)
      Filter Examples (18) unmatched -> Append[set 1]
      Filter Examples (19) matched   -> out 1
      Filter Examples (19) unmatched -> Append[set 2]
      Append -> out 2
    """
    # ======================================================================
    # [Model_Skip_Filtering] (branch) macro_defined furnace_system_model_skip_filter
    #   THEN -> [Model_Skip_Filtering_] (branch) expr == "active"
    # ======================================================================
    skip_kept = df          # input 1 (kept furnaces)
    skip_removed = pd.DataFrame()   # input 2 (removed furnaces)

    if "furnace_system_model_skip_filter" in macros:
        if macros.get("furnace_system_model_skip_filter") == "active":
            # ----- model-skip filtering active -----
            # [Recall (8)] ccp_status
            ccp_status = store.recall("ccp_status")
            # [Recall (9)] entity  -> [Select Attributes (3)] include {entity_id, entity_name}
            entity = store.recall("entity")
            entity_sel = op_select_attributes(
                entity, ["entity_id", "entity_name"], include=True, macros=macros)
            # [Join (15)] left ; key entity_id = entity_id
            ccp = op_join(ccp_status, entity_sel,
                          keys={"entity_id": "entity_id"},
                          join_type="left", macros=macros)
            # [Select Attributes (17)] include {Timestamp, ccp_status, entity_name}
            ccp = op_select_attributes(
                ccp, ["Timestamp", "ccp_status", "entity_name"],
                include=True, macros=macros)
            # [Join (17)] inner ; left = subprocess input ; keys entity_name & Timestamp
            joined = op_join(df, ccp,
                             keys={"entity_name": "entity_name",
                                   "Timestamp": "Timestamp"},
                             join_type="inner", macros=macros)
            # [Filter Examples (6)] custom: ccp_status != 0  (numeric ne)
            matched, unmatched = op_filter_examples(
                joined, ["ccp_status.ne.0"], macros, logic_and=True)
            # [Select Attributes (16)] exclude {entity_id, ccp_status} (matched)
            skip_kept = op_select_attributes(
                matched, ["entity_id", "ccp_status"], include=False, macros=macros)
            # [Select Attributes (24)] exclude {entity_id, ccp_status} (unmatched)
            skip_removed = op_select_attributes(
                unmatched, ["entity_id", "ccp_status"], include=False, macros=macros)
        # ELSE (Model_Skip_Filtering_): pass-through (skip_kept = df)
    # ELSE (Model_Skip_Filtering): pass-through

    # ======================================================================
    # [Manual_Filtering] (branch) macro_defined furnace_system_manual_filter
    #   THEN -> [Manual_Filtering_] (branch) expr == "active"
    # ======================================================================
    manual_kept = skip_kept
    manual_removed = pd.DataFrame()

    if "furnace_system_manual_filter" in macros:
        if macros.get("furnace_system_manual_filter") == "active":
            # [Subprocess (2)] build entity/parameter/tag lookup
            lookup = _sub_manual_filter_join(macros, store)
            # [Filter Examples (16)] parameter_name == manual_overall_furnace_selection_switch
            lookup_f, _ = op_filter_examples(
                lookup,
                ["parameter_name.equals.manual_overall_furnace_selection_switch"],
                macros, logic_and=True)
            # [Select Attributes (18)] include {formula, entity_name}
            lookup_sel = op_select_attributes(
                lookup_f, ["formula", "entity_name"], include=True, macros=macros)
            # [Join (18)] inner ; left = skip_kept ; key entity_name
            joined = op_join(skip_kept, lookup_sel,
                             keys={"entity_name": "entity_name"},
                             join_type="inner", macros=macros)
            # [Filter Examples (17)] formula == 1
            matched, unmatched = op_filter_examples(
                joined, ["formula.equals.1"], macros, logic_and=True)
            # [Select Attributes (19)] exclude formula (matched)
            manual_kept = op_select_attributes(
                matched, ["formula"], include=False, macros=macros)
            # [Select Attributes (23)] exclude formula (unmatched)
            manual_removed = op_select_attributes(
                unmatched, ["formula"], include=False, macros=macros)
        # ELSE (Manual_Filtering_): pass-through
    # ELSE (Manual_Filtering): pass-through

    # ======================================================================
    # [Filter Examples (18)] %{furnace_status}_text == "cracking"
    # ======================================================================
    cracking, not_cracking = op_filter_examples(
        manual_kept, ["%{furnace_status}_text.equals.cracking"],
        macros, logic_and=True)

    # ======================================================================
    # [Filter Examples (19)]  OR of %{ranking_splitter}_text in
    #   {ethane, butane, propane, naphtha, a180}   (filters_logic_and = false)
    # ======================================================================
    valid_splitter, invalid_splitter = op_filter_examples(
        cracking,
        [f"%{{ranking_splitter}}_text.equals.{feed}"
         for feed in ("ethane", "butane", "propane", "naphtha", "a180")],
        macros, logic_and=False)

    # ======================================================================
    # [Append] merge_type=all : set1=not_cracking, set2=invalid_splitter,
    #          set3=skip_removed, set4=manual_removed   -> out 2
    # ======================================================================
    rejected = op_append([not_cracking, invalid_splitter,
                          skip_removed, manual_removed])

    out1 = valid_splitter        # selected furnaces  -> ranking_and_scoring
    out2 = rejected              # rejected furnaces  -> ranking_parameters
    return out1, out2


# -----------------------------------------------------------------------------
#  ranking  -- nested: ranking_and _scoring  (innermost per-parameter scoring)
# -----------------------------------------------------------------------------
def _sorting_selector(data: pd.DataFrame, macros: Macros) -> pd.DataFrame:
    """
    [sorting_selector] (select_subprocess)  select_which = %{sorting_selector}
      sorting_selector == 1  -> case "desc"  : Sort desc, Generate ID, Set Role
      sorting_selector == 2  -> case "asc"   : Sort asc,  Generate ID, Set Role
      sorting_selector == 3  -> case "eval"  : id = eval(sort_type) fallback
    In every case the resulting 'id' column is the 1-based rank position and is
    demoted to a regular attribute by Set Role (id = regular).
    """
    selector = str(macros.get("sorting_selector"))
    param = macros.get("parameter_name")

    if selector == "1":
        # ----- desc branch: [Sort (2)] desc -> [Generate ID (2)] -> [Set Role (2)]
        d = op_sort(data, by="%{parameter_name}", ascending=False, macros=macros)
        d = op_generate_id(d, offset=0)                # id role -> regular
        return d
    if selector == "2":
        # ----- asc branch: [Sort (3)] asc -> [Generate ID (7)] -> [Set Role (4)]
        d = op_sort(data, by="%{parameter_name}", ascending=True, macros=macros)
        d = op_generate_id(d, offset=0)
        return d

    # ----- eval branch (selector == 3) ; sort_type is itself an expression -----
    # %{sort_type} expands to e.g. if([fresh_feed_change]!=0,"desc","asc");
    # eval_generate_attribute evaluates it row-wise to "desc" or "asc".
    d = op_generate_columns(data, {"id": "%{sort_type}"}, macros)
    # [Filter Examples (14)] invert: keep rows whose id is NOT 'asc'/'desc'
    numeric_part, text_part = op_filter_examples(
        d, ["id.equals.asc", "id.equals.desc"], macros,
        logic_and=False, invert=True)
    # [Parse Numbers (2)] id -> number (on numeric_part)
    try:
        numeric_part = op_parse_numbers(numeric_part, "id")
    except Exception:
        pass
    # [Filter Examples (15)] id == 'asc' on the text_part
    asc_part, desc_part = op_filter_examples(
        text_part, ["id.equals.asc"], macros, logic_and=False)
    # asc_part -> [Sort (6)] asc -> [Generate ID (8)] -> [Set Role (5)]
    asc_part = op_generate_id(op_sort(asc_part, "%{parameter_name}", True, macros))
    # desc_part-> [Sort (7)] desc-> [Generate ID (9)] -> [Set Role (6)]
    desc_part = op_generate_id(op_sort(desc_part, "%{parameter_name}", False, macros))
    # [Append (8)] numeric_part + asc_part + desc_part
    return op_append([numeric_part, asc_part, desc_part])


def _score_one_parameter(data: pd.DataFrame, macros: Macros) -> pd.DataFrame:
    """
    One iteration of the innermost  [Loop Values] over parameter_name.

    input 2 (data) has already been sorted & id'd by `_sorting_selector`.
    Branch (8) then turns that into the parameter's _rank and _score columns.
    Returns `data` with two new columns: <param>_rank and <param>_score.
    """
    param = macros.get("parameter_name")

    # ----------------------------------------------------------------------
    # [Branch (8)] macro_defined / expression: %{score_based_ranking} == "active"
    # ----------------------------------------------------------------------
    score_based = (macros.get("score_based_ranking") == "active")

    if score_based:
        # --- THEN (score based) -------------------------------------------
        # [Select Attributes (20)] exclude id  -> [Normalize] range 0..1
        no_id = op_select_attributes(data, ["id"], include=False, macros=macros)
        normed = op_normalize_range(no_id, 0.0, 1.0)
        # [Generate Attributes (15)] <param>_score =
        #   if(lower(sort_type)=="asc", val*weight, val*weight*-1)
        scored = op_generate_columns(
            normed,
            {"%{parameter_name}_score":
             'if(lower("%{sort_type}")=="asc",'
             '([%{parameter_name}]*eval(%{weight})),'
             '([%{parameter_name}]*eval(%{weight})*-1))'},
            macros)
        # [Select Attributes (31)] include only <param>_score
        score_col = op_select_attributes(
            scored, ["%{parameter_name}_score"], include=True, macros=macros)
        # [Rename (11)] id -> <param>_rank  (on the original branch input)
        rank_col = op_rename(data, {"id": "%{parameter_name}_rank"}, macros)
        # [Merge Attributes (2)] side-by-side merge
        merged = op_merge_attributes([score_col, rank_col])
    else:
        # --- ELSE (rank based) --------------------------------------------
        # [Rename (12)] id -> <param>_rank
        merged = op_rename(data, {"id": "%{parameter_name}_rank"}, macros)
        # [Generate Attributes (11)] <param>_score = <param>_rank * eval(weight)
        merged = op_generate_columns(
            merged,
            {"%{parameter_name}_score":
             '[%{parameter_name}_rank]*eval(%{weight})'},
            macros)
    return merged


def _rank_one_group(group: pd.DataFrame, ranking_info: pd.DataFrame,
                    macros: Macros) -> pd.DataFrame:
    """
    Body of [Loop Values (3)] for ONE (timestamp, splitter_text) group.

    Iterates over every parameter_name in `furnace_ranking_info`, scores it,
    accumulates the overall ranking score, then computes the final
    overall_ranking via Sort/Generate ID/Set Role/Rename.
    """
    # [Generate Macro] overall_ranking_score = "0"  (accumulator init)
    macros["overall_ranking_score"] = "0"

    scored = group.copy()
    param_score_cols: List[str] = []

    # ----------------------------------------------------------------------
    # [Loop Values] over parameter_name  (iteration_macro = current_parameter_name)
    #   reuse_results = true, sequential -> columns accumulate across iterations
    # ----------------------------------------------------------------------
    parameter_names = list(pd.unique(ranking_info["parameter_name"]))
    for current_parameter_name in parameter_names:
        macros["current_parameter_name"] = str(current_parameter_name)

        # [Filter Examples (11)] parameter_name == %{current_parameter_name}
        info_row, _ = op_filter_examples(
            ranking_info,
            ["parameter_name.equals.%{current_parameter_name}"],
            macros, logic_and=True)

        # [Extract Macro (3)] parameter_name (value), sort_type, weight(parameter_weightage)
        op_extract_macro_data_value(
            info_row, attribute_name="parameter_name", example_index=1,
            additional={"sort_type": "sort_type", "weight": "parameter_weightage"},
            macros=macros)

        # [Generate Macro (2)] sorting_selector + accumulate overall_ranking_score
        macros["sorting_selector"] = eval_generate_macro(
            'if(lower("%{sort_type}")=="desc",1,if(lower("%{sort_type}")=="asc",2,3))',
            macros)
        # overall_ranking_score = prev + "+" + "[" + param + "_score]"
        macros["overall_ranking_score"] = (
            f'{macros["overall_ranking_score"]}+'
            f'[{macros["parameter_name"]}_score]')

        # [sorting_selector] (select_subprocess) -> ranked data with 'id'
        ranked = _sorting_selector(scored, macros)

        # [Branch (8)] -> produce <param>_rank and <param>_score columns
        per_param = _score_one_parameter(ranked, macros)

        # accumulate the two new columns onto the running dataset (keyed by row)
        pcol = f'{macros["parameter_name"]}_score'
        rcol = f'{macros["parameter_name"]}_rank'
        # align back to `scored` by entity_name (furnace identity preserved)
        key = "entity_name" if "entity_name" in scored.columns else None
        if key and key in per_param.columns:
            add = per_param[[key, pcol, rcol]].drop_duplicates(subset=[key])
            scored = scored.merge(add, on=key, how="left")
        else:
            scored[pcol] = per_param[pcol].values
            scored[rcol] = per_param[rcol].values
        param_score_cols.append(pcol)

    # ----------------------------------------------------------------------
    # sum_parameter_weightage : expected as an upstream macro. If absent we
    # derive it from furnace_ranking_info (Assumption documented in module head).
    # ----------------------------------------------------------------------
    if "sum_parameter_weightage" not in macros:
        macros["sum_parameter_weightage"] = str(
            pd.to_numeric(ranking_info.get("parameter_weightage"),
                          errors="coerce").sum())
    sum_w = _to_num(macros["sum_parameter_weightage"])

    # ----------------------------------------------------------------------
    # [Branch (3)] expression: %{score_based_ranking} == "active"
    #   THEN [Generate Attributes (14)] overall = (sum(scores)/sum_w)+1
    #   ELSE [Generate Attributes (9)]  overall =  sum(scores)/sum_w
    # NOTE: RM uses eval(%{overall_ranking_score}); the accumulated macro is the
    #       string "0+[p1_score]+[p2_score]..." -> i.e. the row-wise sum of the
    #       per-parameter score columns. We compute that sum directly.
    # ----------------------------------------------------------------------
    score_sum = scored[param_score_cols].apply(
        pd.to_numeric, errors="coerce").sum(axis=1) if param_score_cols \
        else pd.Series(0.0, index=scored.index)
    if macros.get("score_based_ranking") == "active":
        scored["overall_ranking_score"] = (score_sum / sum_w) + 1
    else:
        scored["overall_ranking_score"] = score_sum / sum_w

    # [Sort (5)] overall_ranking_score ascending
    scored = op_sort(scored, "overall_ranking_score", True, macros)
    # [Generate ID (4)] -> [Set Role (3)] id=regular -> [Rename (8)] id->overall_ranking
    scored = op_generate_id(scored, offset=0)
    scored = op_rename(scored, {"id": "overall_ranking"}, macros)
    return scored


def ranking_and_scoring(selected: pd.DataFrame, macros: Macros,
                        store: IOStore) -> pd.DataFrame:
    """
    [ranking_and _scoring] (branch)  condition_type = min_examples, value = 1.

    THEN (>= 1 selected example): rank & score every furnace per timestamp and
    splitter feed. ELSE: empty pass-through.
    """
    # ---- min_examples condition -----------------------------------------
    if len(selected) < 1:
        return pd.DataFrame()            # ELSE branch: empty

    # [Date to Nominal] Timestamp -> "yyyy-MM-dd HH:mm:ss"
    df = op_date_to_nominal(selected, "Timestamp")

    # [furnace_ranking_info (2)] (recall) used inside the loops
    furnace_ranking_info = store.recall("furnace_ranking_info")

    splitter_text_col = macros.resolve("%{ranking_splitter}_text")

    # ----------------------------------------------------------------------
    # [Loop Values (2)] over Timestamp  (iteration_macro = loop_value)
    # ----------------------------------------------------------------------
    per_timestamp_results: List[pd.DataFrame] = []
    for loop_value in pd.unique(df["Timestamp"]):
        macros["loop_value"] = str(loop_value)
        # [Filter Examples (12)] Timestamp == %{loop_value}
        ts_df, _ = op_filter_examples(
            df, ["Timestamp.equals.%{loop_value}"], macros, logic_and=True)

        # ------------------------------------------------------------------
        # [Loop Values (3)] over %{ranking_splitter}_text
        #                   (iteration_macro = current_ranking_splitter_text)
        # ------------------------------------------------------------------
        per_splitter_results: List[pd.DataFrame] = []
        for current_splitter in pd.unique(ts_df[splitter_text_col]):
            macros["current_ranking_splitter_text"] = str(current_splitter)
            # [Filter Examples (13)] %{ranking_splitter}_text == %{current_...}
            grp, _ = op_filter_examples(
                ts_df,
                ["%{ranking_splitter}_text.equals.%{current_ranking_splitter_text}"],
                macros, logic_and=True)
            ranked = _rank_one_group(grp, furnace_ranking_info, macros)
            per_splitter_results.append(ranked)
        # [Append (9)] over splitter results
        per_timestamp_results.append(op_append(per_splitter_results))
    # [Append (10)] over timestamp results
    return op_append(per_timestamp_results)


# -----------------------------------------------------------------------------
#  ranking  -- nested: ranking_parameters  (blank ranking columns for rejected)
# -----------------------------------------------------------------------------
def sub_all_parameters(macros: Macros, store: IOStore) -> pd.DataFrame:
    """
    [all_parameters] (subprocess)
    Builds a single 1-row frame containing one MISSING column for every
    "<parameter_name>_score" and "<parameter_name>_rank" so that rejected
    furnaces line up column-wise with the ranked ones.

    Flow:
      Recall(14) furnace_ranking_info -> Select Attributes(10) include parameter_name
        -> Cartesian (right = Create ExampleSet(7): suffix in {score,rank})
        -> Generate Attributes(5): all_parameters=concat(param,"_",suffix); value=MISSING; row_id=1
        -> Pivot(5) group_by row_id, columns all_parameters, first(value)
        -> Select Attributes(12) exclude row_id
        -> Rename by Replacing(6) strip "first(value)_"
    """
    # [Recall (14)] furnace_ranking_info
    info = store.recall("furnace_ranking_info")
    # [Select Attributes (10)] include parameter_name
    info = op_select_attributes(info, ["parameter_name"], include=True, macros=macros)
    # [Create ExampleSet (7)] csv "suffix\nscore\nrank"
    suffix = op_create_exampleset_csv("suffix\nscore\nrank", column_separator=",")
    # [Cartesian] product
    cart = op_cartesian(info, suffix)
    # [Generate Attributes (5)] all_parameters / value / row_id
    cart = op_generate_columns(
        cart,
        {"all_parameters": 'concat([parameter_name],"_",[suffix])',
         "value": "MISSING_NUMERIC",
         "row_id": "1"},
        macros)
    # [Pivot (5)] group_by row_id ; columns all_parameters ; first(value)
    wide = op_pivot(cart, group_by=["row_id"],
                    column_grouping="all_parameters",
                    value_attribute="value", agg="first", dropna=False)
    # [Select Attributes (12)] exclude row_id
    wide = op_select_attributes(wide, ["row_id"], include=False, macros=macros)
    # [Rename by Replacing (6)] strip "first(value)_"
    wide = op_rename_by_replacing(wide, replace_what=r"first\(value\)_", replace_by="")
    return wide


def ranking_parameters(rejected: pd.DataFrame, macros: Macros,
                       store: IOStore) -> pd.DataFrame:
    """
    [ranking_parameters] (subprocess)
    Gives the rejected furnaces empty overall ranking columns plus a blank
    column for every parameter score/rank, so Append(11) is schema-compatible.

    Flow:
      in 1 -> Generate Attributes (10): overall_ranking_score / overall_ranking = MISSING
            -> Merge Attributes (4)[set 1]
      all_parameters -> Merge Attributes (4)[set 2]
      -> out 1
    """
    # [Generate Attributes (10)] overall_ranking_score / overall_ranking = MISSING
    base = op_generate_columns(
        rejected,
        {"overall_ranking_score": "MISSING_NUMERIC",
         "overall_ranking": "MISSING_NUMERIC"},
        macros)
    # [all_parameters] (subprocess) -> the blank per-parameter columns
    blanks = sub_all_parameters(macros, store)
    # broadcast the single blank row to every rejected row
    if len(base) > 0 and len(blanks) > 0:
        blanks = pd.concat([blanks] * len(base), ignore_index=True)
    # [Merge Attributes (4)] side-by-side
    return op_merge_attributes([base.reset_index(drop=True), blanks])


# -----------------------------------------------------------------------------
#  ranking  -- nested: de-parameterization (2)  (wide param table -> tag table)
# -----------------------------------------------------------------------------
def sub_de_parameterization(df: pd.DataFrame, macros: Macros,
                            store: IOStore) -> pd.DataFrame:
    """
    [de-parameterization (2)] (subprocess)
    Un-pivots the parameter-level wide table back into a tag-level wide table,
    mapping parameter_name -> short_name via tag_parameter_mapping.

    Flow:
      in 1 -> Numerical to Polynominal (6) -> De-Pivot (8) (index parameter_name)
           -> Join (8) (right = Recall(7) tag_parameter_mapping; inner on
              parameter_name & entity_name)
           -> Filter Examples (5) short_name != Timestamp
           -> Remove Duplicates (10) [Timestamp, short_name]
           -> Pivot (group_by Timestamp; columns short_name; first(value))
           -> Rename by Replacing strip "first(value)_"
           -> out 1
    """
    # [Numerical to Polynominal (6)] numeric -> nominal
    work = op_numerical_to_polynominal(df)
    # [De-Pivot (8)] index parameter_name ; values = all but Timestamp & entity_name
    work = op_de_pivot(work,
                       value_regex=r"^(?!Timestamp$|entity_name$).+",
                       index_attribute="parameter_name",
                       keep_missings=True)
    # [Recall (7)] tag_parameter_mapping
    mapping = store.recall("tag_parameter_mapping")
    # [Join (8)] inner ; keys parameter_name & entity_name
    work = op_join(work, mapping,
                   keys={"parameter_name": "parameter_name",
                         "entity_name": "entity_name"},
                   join_type="inner", macros=macros)
    # [Filter Examples (5)] short_name != Timestamp
    work, _ = op_filter_examples(
        work, ["short_name.does_not_equal.Timestamp"], macros, logic_and=True)
    # [Remove Duplicates (10)] subset Timestamp | short_name
    work = op_remove_duplicates(work, subset=["Timestamp", "short_name"])
    # [Pivot] group_by Timestamp ; columns short_name ; first(value)
    work = op_pivot(work, group_by=["Timestamp"],
                    column_grouping="short_name",
                    value_attribute="value", agg="first", dropna=False)
    # [Rename by Replacing] strip "first(value)_"
    work = op_rename_by_replacing(work, replace_what=r"first\(value\)_", replace_by="")
    return work


# =============================================================================
#  SUB-PROCESS 2 :  ranking  (orchestrator)
# =============================================================================
def ranking(example_set: pd.DataFrame, macros: Macros,
            store: IOStore) -> pd.DataFrame:
    """
    Top-level sub-process "ranking".

    Flow:
      in 1 -> text_code_mapping (2) -> furnace_selection (2)
      furnace_selection out1 -> ranking_and _scoring -> Append(11)[set 1]
      furnace_selection out2 -> ranking_parameters -> Date to Nominal(2)
                                                    -> Append(11)[set 2]
      Append(11) -> de-parameterization (2) -> Nominal to Date -> out 1
    """
    df = example_set.copy()

    # [log start (12)] telemetry
    _log_block("Ranking", "startedAt", macros)

    # [text_code_mapping (2)] (subprocess)
    df = sub_text_code_mapping(df, macros, store)

    # [furnace_selection (2)] (subprocess) -> out1 (selected), out2 (rejected)
    selected, rejected = sub_furnace_selection(df, macros, store)

    # [ranking_and _scoring] (branch min_examples=1) on the SELECTED set
    ranked = ranking_and_scoring(selected, macros, store)

    # [ranking_parameters] (subprocess) on the REJECTED set
    rej = ranking_parameters(rejected, macros, store)
    # [Date to Nominal (2)] Timestamp -> string
    rej = op_date_to_nominal(rej, "Timestamp")

    # [Append (11)] set1 = ranked, set2 = rejected-with-blank-columns
    combined = op_append([ranked, rej])

    # [de-parameterization (2)] (subprocess)
    combined = sub_de_parameterization(combined, macros, store)

    # [Nominal to Date] Timestamp string -> datetime
    combined = op_nominal_to_date(combined, "Timestamp")

    # [log end (7)] telemetry
    _log_block("Ranking", "endedAt", macros)

    return combined         # -> out 1

