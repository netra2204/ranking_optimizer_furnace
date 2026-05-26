"""
module_09_generate_output.py
============================
Replicates the "Generate Output" subprocess (RMP lines 8852–9339).

The RMP block produces a 4-column LONG dataframe:
    Timestamp | sub_model_id | tag | value

This is the final shape that gets written to the Furnace_Output DB table.
The wide-to-long transformation, the canonical-tag-name renaming, the
NonBiasing furnace union, and the per-tag zero-fill rules all happen
inside THIS module — not module 10. Module 10 is a pure schema validator.

Operator flow (in execution order)
----------------------------------
    1.  Date to Nominal (7)             Timestamp → string
    2.  exclude (7)                     drop mixed_feed_margin, sum(del_Feed_flow)
    3.  Extract Macro (133)             current_loop_time = row[0]["Timestamp"]
    4.  Set Role (35)                   overall_ranking = regular (no-op in pandas)
    5.  Branch (134)  expr: final_run_optimizer_check == 1
         ├─ THEN: Generate Attributes (131) — *_bias columns
         └─ ELSE: Branch (131) expr: deviation_exists == 0
                   ├─ THEN: pass-through
                   └─ ELSE: Generate Attributes (137) — *_system_optimum columns
    6.  Subprocess (69)                 Union with NonBiasing_furnaces
    7.  Generate Attributes (138)       conditional ranking_opportunity
    8.  Generate Attributes (336)       Change_in_*, change_in_furnace (0/1/2/3),
                                        furnace_condition (encoded), total_optimizer_run_check
    9.  exclude (4)                     drop helper cols
    10. Rename (56)                     to canonical tag names
    11. de-parameterization subprocess  (Numerical→Polynominal + De-Pivot +
                                         Recall tag_parameter_mapping + Join +
                                         Filter + Remove Duplicates + Pivot + Rename)
    12. Parse Numbers (11)              numeric coercion
    13. req tags  Generate Attributes   stamp 4 furnace-level macros
    14. Loop (52) ×9                    rename *_system_optimum → bare names
    15. Subprocess (71)                 Transpose + Rename + Generate(Timestamp, sub_model_id=710)
                                        + Nominal-to-Date + Set Role + Filter (drop Timestamp tag)
    16. Filter Examples (201)           tag.contains "rank"
    17. Filter Examples (202)           tag.contains "ranking_opportunity"
    18. Replace Missing Values (7)      value → 0 (unmatched of 201 = non-rank tags)
    19. Replace Missing Values (2)      value → 0 (matched of 202 = ranking_opportunity)
    20. final 4 col output (append)     three-stream concat:
                                        (Filter 202 unmatched) +
                                        (Replace 2 output) +
                                        (Replace 7 output)
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

import numpy as np
import pandas as pd

from config import MACROS, STORE, NUM_FURNACES

logger = logging.getLogger(__name__)

# RMP hard-codes sub_model_id = 710 inside Generate Attributes (337).
# This overrides DB_CONFIG["model_id"]; the two are different tables.
SUB_MODEL_ID = 710

# Encoding map from Generate Attributes (336)
FURNACE_CONDITION_ENCODING = {
    "Good":            1,
    "Bad":             2,
    "Semi Good":       3,
    "SOR":             4,
    "EOR":             5,
    "No Optimization": -1,
}

# Columns the *output dataset* should not carry forward (Gen Attr 138/336)
HELPER_COLS_TO_DROP = [
    "Change_in_conversion",
    "Change_in_feed",
    "percent_above_threshold",
    "percent_above_threshold_rank",
]

# Rename (56) — canonical tag names that tag_parameter_mapping expects
CANONICAL_RENAMES = {
    "percent_above_threshold_rank": "Forecasted_runlength_rank_org",
    "percent_above_threshold":      "runtime",
    "Overall_conversion":           "overall_conversion",
    "Feed_flow":                    "wet_feed_total_flow",
}

# Per-furnace pairs that the optimizer-OFF arm writes with _system_optimum
# suffix and Loop (52) renames back. Order matters because the suffix path
# uses these names verbatim.
SYSTEM_OPTIMUM_FIELDS = [
    "COT_bias", "CIT_bias", "Feed_bias", "Conversion_bias",
    "SHC_bias", "Heat_bias", "Change_in_furnace",
]


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
def _m(key: str, default=0):
    """Resolve a macro to float (parse() semantics)."""
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _mi(key: str, default: int = 0) -> int:
    try:
        return int(round(_m(key, default)))
    except Exception:
        return int(default)


def _ms(key: str, default: str = "") -> str:
    return str(MACROS.get(key, default))


def _recall(name: str) -> pd.DataFrame:
    return STORE.get(name, pd.DataFrame())


def _remember(name: str, df: pd.DataFrame) -> None:
    STORE[name] = df.copy() if isinstance(df, pd.DataFrame) else df


def _safe_series(df: pd.DataFrame, col: str, fill=0.0) -> pd.Series:
    """Return df[col] coerced to float, or a fill-valued series if absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(fill)
    return pd.Series([fill] * len(df), index=df.index, dtype=float)


# ===========================================================================
# Op 1 — Date to Nominal (7)
# ===========================================================================
def _date_to_nominal_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Timestamp column to canonical string "yyyy-MM-dd HH:mm:ss".
    keep_old_attribute=false in RMP → we overwrite.
    """
    if "Timestamp" not in df.columns:
        return df
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["Timestamp"]):
        out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce") \
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


# ===========================================================================
# Op 2 — exclude (7): drop mixed_feed_margin, sum(del_Feed_flow)
# ===========================================================================
def _exclude_grid_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["mixed_feed_margin", "sum(del_Feed_flow)"], errors="ignore")


# ===========================================================================
# Op 3 — Extract Macro (133): current_loop_time = row[0]["Timestamp"]
# ===========================================================================
def _extract_current_loop_time(df: pd.DataFrame) -> None:
    if df.empty or "Timestamp" not in df.columns:
        MACROS["current_loop_time"] = MACROS.get("end_time")
        return
    MACROS["current_loop_time"] = str(df["Timestamp"].iloc[0])
    logger.info("current_loop_time = %s", MACROS["current_loop_time"])


# ===========================================================================
# Ops 5–8 — Branch (134) + Branch (131) bias generation
# ===========================================================================
def _generate_bias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch (134): final_run_optimizer_check == 1
      THEN  → Generate Attributes (131)
              cot_bias  = COT_new - weighted_average_cot_calculated
              cit_bias  = CIT_new - CIT
              heat_bias = Heat_new - Heat
              shc_bias  = 0
              feed_bias and conversion_bias pass-through (already on df)
      ELSE  → Branch (131): deviation_exists == 0
                THEN → pass-through (no new columns)
                ELSE → Generate Attributes (137)
                       Feed_bias_system_optimum       = feed_bias
                       Conversion_bias_system_optimum = conversion_bias
                       COT_bias_system_optimum        = 0
                       CIT_bias_system_optimum        = 0
                       Heat_bias_system_optimum       = 0
                       SHC_bias_system_optimum        = 0
    """
    out = df.copy()
    final_run = _mi("final_run_optimizer_check", 0)

    if final_run == 1:
        # ── Generate Attributes (131) — THEN arm ──────────────────────────
        cot_new = _safe_series(out, "COT_new")
        wa_cot  = _safe_series(out, "weighted_average_cot_calculated")
        cit_new = _safe_series(out, "CIT_new")
        cit     = _safe_series(out, "CIT")
        heat_new = _safe_series(out, "Heat_new")
        heat     = _safe_series(out, "Heat")

        # feed_bias/conversion_bias pass through unchanged (created by post-grid)
        if "feed_bias" not in out.columns:
            out["feed_bias"] = 0.0
        if "conversion_bias" not in out.columns:
            out["conversion_bias"] = 0.0

        out["cot_bias"]  = cot_new - wa_cot
        out["cit_bias"]  = cit_new - cit
        out["heat_bias"] = heat_new - heat
        out["shc_bias"]  = 0.0
        logger.info("Generated bias columns (final_run_optimizer_check=1 path)")
        return out

    # ── ELSE arm: Branch (131) ────────────────────────────────────────────
    deviation_exists = _mi("deviation_exists", 1)
    if deviation_exists == 0:
        # THEN of Branch (131) — pure pass-through
        logger.info("Bias generation skipped (deviation_exists=0)")
        return out

    # ── Generate Attributes (137) — ELSE-ELSE arm ─────────────────────────
    if "feed_bias" not in out.columns:
        out["feed_bias"] = 0.0
    if "conversion_bias" not in out.columns:
        out["conversion_bias"] = 0.0

    out["Feed_bias_system_optimum"]       = out["feed_bias"]
    out["Conversion_bias_system_optimum"] = out["conversion_bias"]
    out["COT_bias_system_optimum"]        = 0.0
    out["CIT_bias_system_optimum"]        = 0.0
    out["Heat_bias_system_optimum"]       = 0.0
    out["SHC_bias_system_optimum"]        = 0.0
    logger.info("Generated *_system_optimum bias columns (optimizer-off path)")
    return out


# ===========================================================================
# Op 9 — Subprocess (69): Union with NonBiasing_furnaces
# ===========================================================================
def _union_nonbiasing_furnaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recall STORE["NonBiasing_furnaces"] (populated by module 04 — furnaces
    whose Furnace_condition is NOT in GOOD_CONDITIONS, e.g. "No Optimization")
    and append them to the optimized rows.

    These rows don't have bias columns, so missing fields stay NaN; later
    Replace Missing Values steps zero them out.

    RMP's Union operator does a positional row append; pandas concat (axis=0)
    with sort=False is the closest analogue.
    """
    df_nb = _recall("NonBiasing_furnaces")
    if df_nb is None or df_nb.empty:
        logger.info("NonBiasing_furnaces empty – nothing to union.")
        return df

    # Align on Timestamp string format (RMP's Set Role just changes role, not value)
    if "Timestamp" in df_nb.columns and pd.api.types.is_datetime64_any_dtype(df_nb["Timestamp"]):
        df_nb = df_nb.copy()
        df_nb["Timestamp"] = df_nb["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    merged = pd.concat([df, df_nb], axis=0, ignore_index=True, sort=False)
    logger.info("Unioned NonBiasing_furnaces: +%d rows → %d total", len(df_nb), len(merged))
    return merged


# ===========================================================================
# Op 10 — Generate Attributes (138): conditional ranking_opportunity
# ===========================================================================
def _generate_ranking_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    """
    ranking_opportunity =
        if(deviation_exists == 0,  ranking_opportunity (keep),
           if(entity_name == "FS", sum_del_ethylene_final * 24,
              ranking_opportunity))
    """
    out = df.copy()
    deviation_exists = _mi("deviation_exists", 1)
    sum_eth_final    = _m("sum_del_ethylene_final", 0.0)

    if "ranking_opportunity" not in out.columns:
        out["ranking_opportunity"] = 0.0
    base = pd.to_numeric(out["ranking_opportunity"], errors="coerce").fillna(0.0)

    if deviation_exists == 0:
        out["ranking_opportunity"] = base
        return out

    # ELSE: entity_name == "FS" gets the system-level total
    if "entity_name" in out.columns:
        is_fs = out["entity_name"].astype(str) == "FS"
        out["ranking_opportunity"] = np.where(is_fs, sum_eth_final * 24.0, base)
    else:
        out["ranking_opportunity"] = base
    return out


# ===========================================================================
# Op 11 — Generate Attributes (336)
# ===========================================================================
def _generate_change_and_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change_in_conversion = if(abs(conversion_bias) > 0, 1, 0)
    Change_in_feed       = if(abs(feed_bias) > 0, 1, 0)
    change_in_furnace    = if(deviation_exists == 0, keep,
                              if(cnv==0 AND feed==0, 0,
                                 if(cnv==0 AND feed==1, 1,
                                    if(cnv==1 AND feed==0, 2, 3))))
                          ← (0/1/2/3 encoding, NOT 0/1)
    furnace_condition    = integer encoding of Furnace_condition
    total_optimizer_run_check = if(deviation_exists == 0, keep,
                                  (1 - steam_water_deoke_status) *
                                  total_optimizer_run_check)
    """
    out = df.copy()
    deviation_exists = _mi("deviation_exists", 1)
    total_run_macro  = _m("total_optimizer_run_check", 0.0)

    fb = _safe_series(out, "feed_bias").abs()
    cb = _safe_series(out, "conversion_bias").abs()

    out["Change_in_conversion"] = (cb > 0).astype(int)
    out["Change_in_feed"]       = (fb > 0).astype(int)

    # change_in_furnace (0/1/2/3 encoding)
    if deviation_exists == 0 and "change_in_furnace" in out.columns:
        # Keep existing value
        existing_change_in_furnace = pd.to_numeric(out["change_in_furnace"], errors="coerce").fillna(0).astype(int)
        out["change_in_furnace"] = existing_change_in_furnace
    else:
        cnv = out["Change_in_conversion"]
        fd  = out["Change_in_feed"]
        # Truth table from RMP expression
        out["change_in_furnace"] = np.select(
            [(cnv == 0) & (fd == 0),
             (cnv == 0) & (fd == 1),
             (cnv == 1) & (fd == 0)],
            [0, 1, 2],
            default=3,
        ).astype(int)

    # furnace_condition encoding
    if "Furnace_condition" in out.columns:
        out["furnace_condition"] = out["Furnace_condition"].astype(str).map(
            FURNACE_CONDITION_ENCODING
        ).fillna(0).astype(int)
    else:
        out["furnace_condition"] = 0

    # total_optimizer_run_check
    if deviation_exists == 0 and "total_optimizer_run_check" in out.columns:
        # keep existing
        pass
    else:
        steam_deoke = _safe_series(out, "steam_water_deoke_status")
        out["total_optimizer_run_check"] = (1.0 - steam_deoke) * total_run_macro

    return out


# ===========================================================================
# Op 12 — exclude (4): drop helper columns
# ===========================================================================
def _drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=HELPER_COLS_TO_DROP, errors="ignore")


# ===========================================================================
# Op 13 — Rename (56): canonical tag names
# ===========================================================================
def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    RMP Rename (56) renames internal column names to the canonical tag-mapping
    names. The mapping below mirrors the RMP exactly, with one subtlety:
    in RMP `rename_attributes` the list is `(new_name, old_name)` pairs even
    though it reads `key="A" value="B"`. So `key=Forecasted_runlength_rank_org
    value=percent_above_threshold_rank` means rename A→B (old→new).

    Our CANONICAL_RENAMES dict has been written as {old: new}.
    """
    # If the old name doesn't exist, pandas rename silently skips — OK.
    return df.rename(columns=CANONICAL_RENAMES)


# ===========================================================================
# Op 11 — de-parameterization subprocess
# ===========================================================================
def _de_parameterize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide → long → tag-name re-key → wide-on-short_name flow.

    Steps:
      14. Numerical to Polynominal (4)   — stringify numerics
      15. De-Pivot (6)                   — id_vars = Timestamp, entity_name
                                          value_var regex = ^(?!Timestamp$|entity_name$).+
                                          → cols: Timestamp, entity_name,
                                                  parameter_name, value
      16. Recall (69)                    — tag_parameter_mapping
      17. Join (62)                      — inner on (parameter_name, entity_name)
      18. Filter Examples (109)          — short_name != "Timestamp"
      19. Remove Duplicates (29)         — on (Timestamp, short_name)
      20. Pivot (7)                      — group=Timestamp, col=short_name,
                                          agg=first(value)
      21. Rename by Replacing (2)        — strip "first(value)_" prefix
    """
    df_tpm = _recall("tag_parameter_mapping")
    if df_tpm is None or df_tpm.empty:
        logger.warning("tag_parameter_mapping empty – skipping de-parameterization. "
                       "Output will use internal column names, not canonical tag names.")
        return df

    # ---- 14. Numerical to Polynominal (no-op for our melt) ----
    work = df.copy()

    # ---- 15. De-Pivot (6) ----
    id_vars = [c for c in ["Timestamp", "entity_name"] if c in work.columns]
    value_vars = [c for c in work.columns if c not in id_vars]
    long_df = work.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="parameter_name",
        value_name="value",
    )

    # ---- 16. + 17. Join with tag_parameter_mapping ----
    # tag_parameter_mapping is expected to have at least these columns:
    #   parameter_name, entity_name, short_name
    join_keys = [k for k in ["parameter_name", "entity_name"] if k in df_tpm.columns]
    if not join_keys:
        logger.warning("tag_parameter_mapping missing join keys; skipping de-parameterization.")
        return df

    merged = pd.merge(long_df, df_tpm, on=join_keys, how="inner")

    # ---- 18. Filter Examples (109): short_name != "Timestamp" ----
    if "short_name" in merged.columns:
        merged = merged[merged["short_name"].astype(str) != "Timestamp"]

    # ---- 19. Remove Duplicates (29) on (Timestamp, short_name) ----
    if {"Timestamp", "short_name"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["Timestamp", "short_name"], keep="first")

    # ---- 20. Pivot (7) ----
    if "short_name" not in merged.columns:
        logger.warning("'short_name' missing after join; returning long form.")
        return merged

    wide = merged.pivot_table(
        index=["Timestamp"],
        columns="short_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # ---- 21. Rename by Replacing (2): strip "first(value)_" ----
    wide.columns = [re.sub(r"^first\(value\)_", "", str(c)) for c in wide.columns]

    logger.info("de-parameterization: long-form %d rows → wide %d × %d",
                len(merged), len(wide), len(wide.columns))
    return wide


# ===========================================================================
# Op 12 — Parse Numbers (11)
# ===========================================================================
def _parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "Timestamp":
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        # If coercion turned everything to NaN, the column wasn't numeric;
        # keep the original (mirrors RMP's "skip" behaviour for non-parseable).
        if coerced.notna().any() or len(out) == 0:
            out[col] = coerced
    return out


# ===========================================================================
# Op 13 — req tags: 4 furnace-level macro stamps
# ===========================================================================
def _generate_req_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stamp 4 columns derived from global macros:
      Fur_Next_Decoking_Furnace               = parse(%{Fur_Next_Decoking_Furnace})
      Fur_biasing_condition                   = encoded biasing_condition
      Fur_ranking_cause_indicator             = parse(%{ranking_cause_indicator})
      Fur_all_furnace_for_conversion_biasing  = parse(%{all_furnace_for_conversion_biasing})

    Encoding for Fur_biasing_condition (from the RMP nested if):
        biasing_condition == 1:
            fresh_feed_change != 0          → 11
            ROPT_..._biasing == "inactive"  → 12
            else                            → 13
        biasing_condition == 2:
            ROPT_..._biasing == "inactive"  → 21
            else                            → 22
        biasing_condition == 3:
            fresh_feed_change != 0          → 31
            else                            → 32
        else                                → 0
    """
    out = df.copy()

    # Fur_Next_Decoking_Furnace: parse(%{macro})
    fndf = MACROS.get("Fur_Next_Decoking_Furnace", "")
    try:
        out["Fur_Next_Decoking_Furnace"] = float(fndf)
    except (TypeError, ValueError):
        # Non-numeric (e.g. "F3") → keep as string
        out["Fur_Next_Decoking_Furnace"] = fndf

    # Encoded biasing_condition
    bcond  = _mi("biasing_condition", 0)
    ffc    = _mi("fresh_feed_change", 0)
    bias_mode = _ms("ROPT_all_furnace_for_conversion_biasing", "active")

    if bcond == 1:
        if ffc != 0:
            code = 11
        elif bias_mode == "inactive":
            code = 12
        else:
            code = 13
    elif bcond == 2:
        code = 21 if bias_mode == "inactive" else 22
    elif bcond == 3:
        code = 31 if ffc != 0 else 32
    else:
        code = 0
    out["Fur_biasing_condition"] = code

    out["Fur_ranking_cause_indicator"]            = _mi("ranking_cause_indicator", 0)
    out["Fur_all_furnace_for_conversion_biasing"] = _m("all_furnace_for_conversion_biasing", 0.0)

    return out


# ===========================================================================
# Op 14 — Loop (52) ×9: rename *_system_optimum back to bare names
# ===========================================================================
def _loop_rename_system_optimum(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each i in 1..9 and each field in SYSTEM_OPTIMUM_FIELDS:
        rename Fur{i}_{field}_system_optimum  →  Fur{i}_{field}

    RMP's rename_attributes list is {key=new_name, value=old_name}, so the
    RMP is saying "Fur{i}_COT_bias_system_optimum is renamed to Fur{i}_COT_bias"
    (which makes more sense if you read it as "the new column will be the
    plain name, and we're sourcing it from the _system_optimum one").

    If the source column doesn't exist (i.e. final_run_optimizer_check == 1
    path was taken and no _system_optimum columns were created), the rename
    silently skips that column — which is exactly what RMP does.
    """
    rename_map = {}
    for i in range(1, NUM_FURNACES + 1):
        for field in SYSTEM_OPTIMUM_FIELDS:
            old = f"Fur{i}_{field}_system_optimum"
            new = f"Fur{i}_{field}"
            rename_map[old] = new
    return df.rename(columns=rename_map)


# ===========================================================================
# Op 15 — Subprocess (71): wide → long with Timestamp/sub_model_id stamp
# ===========================================================================
def _final_transpose_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subprocess (71):
      25. Transpose (48)         columns→rows; new (id, att_1) pair
      26. Select Attributes (128) keep only att_1 + id
      27. Rename (108)            id→tag, att_1→value
      28. Generate Attributes (337) Timestamp = %{current_loop_time},
                                    sub_model_id = 710
      29. Nominal to Date (9)     Timestamp string → datetime
      30. Set Role (60)           tag = regular (no-op in pandas)
      31. Filter Examples (66)    INVERT tag.equals.Timestamp
                                  (i.e. KEEP tag != "Timestamp")

    Because Subprocess (71) runs *after* aggregation/repivot — meaning the
    input is already a single-row wide frame — the transpose flattens it
    into (column_name → tag, cell_value → value) rows.

    If the input has multiple rows (it shouldn't after Pivot 7, but the
    NonBiasing union can produce multi-row outputs), the RMP transposes
    the *first* row only because Transpose works on the whole table but
    Pivot ahead of it has already collapsed by Timestamp. We do the same
    by melting on Timestamp.
    """
    out = df.copy()
    current_loop_time = MACROS.get("current_loop_time", out.get("Timestamp", pd.Series([""])).iloc[0])

    # Use Timestamp as the index so the melt produces one row per (Timestamp, col).
    id_vars = [c for c in ["Timestamp"] if c in out.columns]
    value_vars = [c for c in out.columns if c not in id_vars]

    long = out.melt(id_vars=id_vars, value_vars=value_vars, var_name="tag", value_name="value")

    # Generate Attributes (337)
    if "Timestamp" not in long.columns:
        long["Timestamp"] = current_loop_time
    long["sub_model_id"] = SUB_MODEL_ID

    # Nominal to Date (9): Timestamp string → datetime
    long["Timestamp"] = pd.to_datetime(long["Timestamp"], errors="coerce")

    # Filter Examples (66): KEEP rows where tag != "Timestamp"
    long = long[long["tag"].astype(str) != "Timestamp"].copy()

    # Reorder to the final 4-column schema
    long = long[["Timestamp", "sub_model_id", "tag", "value"]]
    return long


# ===========================================================================
# Ops 16–20 — three-stream zero-fill split-and-append
# ===========================================================================
def _split_zero_fill_concat(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Examples (201): tag.contains("rank")
        matched   → fed to RMV (7)
        unmatched → fed straight to append

    But the matched output of 201 is itself piped through Filter Examples (202):
        Filter (202) on tag.contains("ranking_opportunity"):
            matched   → RMV (2) (zero-fill values)
            unmatched → fed straight to append

    Replace Missing Values (7) — operates on the *unmatched* of 201
       (i.e. tags that do NOT contain "rank") — fills NaN value with 0.

    NOTE the careful semantics:
      Filter 201 unmatched (= tags not containing "rank")     → RMV (7) → stream A
      Filter 202 unmatched (= tags containing "rank" but NOT "ranking_opportunity")
                                                              → straight  → stream B
      Filter 202 matched   (= tags containing "ranking_opportunity")
                                                              → RMV (2)  → stream C

    Final append = A + B + C  (RMP "all" merge type with three inputs)
    """
    if df_long.empty:
        return df_long.copy()

    out = df_long.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    tag_str = out["tag"].astype(str)
    contains_rank   = tag_str.str.contains("rank",                regex=False, na=False)
    contains_rkopp  = tag_str.str.contains("ranking_opportunity", regex=False, na=False)

    # Stream A: NOT containing "rank" — zero-fill missing values  (RMV 7)
    stream_a = out[~contains_rank].copy()
    stream_a["value"] = stream_a["value"].fillna(0.0)

    # Stream B: contains "rank" but NOT "ranking_opportunity" — pass-through
    stream_b = out[contains_rank & ~contains_rkopp].copy()

    # Stream C: contains "ranking_opportunity" — zero-fill missing values  (RMV 2)
    stream_c = out[contains_rkopp].copy()
    stream_c["value"] = stream_c["value"].fillna(0.0)

    final = pd.concat([stream_a, stream_b, stream_c], axis=0, ignore_index=True)
    logger.info("Three-stream append: A=%d (non-rank, zero-filled), "
                "B=%d (rank pass-through), C=%d (ranking_opportunity, zero-filled) → %d total",
                len(stream_a), len(stream_b), len(stream_c), len(final))
    return final


# ===========================================================================
# Public entry point
# ===========================================================================
def run(df_pre_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Generate Output subprocess.

    Parameters
    ----------
    df_pre_grid : pd.DataFrame
        The data flowing into Generate Output. In main.py this is whatever
        the MAIN branch outputs:
          - df_post_grid           (if final_run_optimizer_check == 1)
          - df_post_grid (else arm with deviation_exists==0 reuse)
          - df_else                (else arm with deviation_exists!=0)

    Returns
    -------
    df_long : pd.DataFrame
        4-column dataframe: Timestamp | sub_model_id | tag | value
        Ready for module 10's schema check & DB insert.
    """
    logger.info("=== MODULE 09 – GENERATE OUTPUT ===")

    if df_pre_grid is None or df_pre_grid.empty:
        logger.warning("Generate Output: input is empty – returning empty 4-col frame.")
        return pd.DataFrame(columns=["Timestamp", "sub_model_id", "tag", "value"])

    # ── Op 1: Date to Nominal (7) ────────────────────────────────────────
    df = _date_to_nominal_timestamp(df_pre_grid)

    # ── Op 2: exclude (7) ────────────────────────────────────────────────
    df = _exclude_grid_cols(df)

    # ── Op 3: Extract Macro (133) — current_loop_time ────────────────────
    _extract_current_loop_time(df)

    # ── Op 4: Set Role (35) — no-op in pandas ────────────────────────────
    # (overall_ranking was a "label" in RMP; pandas has no equivalent concept)

    # ── Ops 5–8: Branch (134) + (131) bias generation ───────────────────
    df = _generate_bias_columns(df)

    # ── Op 9: Subprocess (69) — Union with NonBiasing_furnaces ──────────
    df = _union_nonbiasing_furnaces(df)

    # ── Op 10: Generate Attributes (138) — ranking_opportunity ──────────
    df = _generate_ranking_opportunity(df)

    # ── Op 11: Generate Attributes (336) ────────────────────────────────
    df = _generate_change_and_condition(df)

    # ── Op 12: exclude (4) ──────────────────────────────────────────────
    df = _drop_helper_columns(df)

    # ── Op 13: Rename (56) ──────────────────────────────────────────────
    df = _rename_to_canonical(df)

    # ── Op 11 (subprocess): de-parameterization ─────────────────────────
    df = _de_parameterize(df)

    # ── Op 12: Parse Numbers (11) ───────────────────────────────────────
    df = _parse_numbers(df)

    # ── Op 13: req tags ─────────────────────────────────────────────────
    df = _generate_req_tags(df)

    # ── Op 14: Loop (52) ×9 rename ──────────────────────────────────────
    df = _loop_rename_system_optimum(df)

    # ── Op 15: Subprocess (71) — final wide → long ──────────────────────
    df_long = _final_transpose_to_long(df)

    # ── Ops 16–20: three-stream zero-fill append ────────────────────────
    df_final = _split_zero_fill_concat(df_long)

    _remember("df_final_output", df_final)
    logger.info("GENERATE OUTPUT complete – %d rows × %d cols (long format)",
                len(df_final), len(df_final.columns))
    return df_final
