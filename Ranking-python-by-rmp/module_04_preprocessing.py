"""
module_04_preprocessing.py
==========================
Replicates the "Preprocess" subprocess inside "United OLF".

Responsibilities
----------------
1.  Extract limit values for each ROPT parameter (from ROPT_extract_macro_value)
    and push them into MACROS (mirrors the Loop over extract_value_count).
2.  Compute margin flags (FG, Steam, Damper, Quench, CGC, C2-splitter, ERC,
    PRC, NOx, Saturator, SHC) using the extracted limits.
3.  Handle external constraints (reorder ranking based on furnace_external_constraint).
4.  Rename wet_feed_total_flow → Feed_flow and overall_conversion → Overall_conversion.
5.  Evaluate inferred tags (formula expressions for inferred_tags_4).
6.  Compute Forecasted_runlength_rank_org.
7.  Determine next decoking furnace (min days_remaining, max coke_thickness).
8.  Apply furnace-coupling logic (flag coupled furnaces as "No Optimization").
9.  Run the 'calculations' sub-step: split Good / No-Good furnaces, store counts.

Inputs  (STORE + args)
------
    df_param      – output of module_03_parameterization
    "ROPT_extract_macro_value"
    "inferred_tags_4"
    "deviation_check_furnace_system"

Outputs  (STORE + return)
-------
    "good_fur_data"       – rows with Furnace_condition == 'Good'
    "NonBiasing_furnaces" – all rows with a recognised Furnace_condition
    MACROS["count_of_good_fur"]
    MACROS["count_of_no_good_fur"]
    Returns df_param with all new columns appended.
"""

import pandas as pd
import numpy as np
import logging
import re

from config import MACROS, STORE

logger = logging.getLogger(__name__)

GOOD_CONDITIONS = {"Good", "Bad", "SOR", "Semi Good"}


def _recall(name: str) -> pd.DataFrame:
    df = STORE.get(name, pd.DataFrame())
    if df.empty:
        logger.debug("recall('%s') → empty", name)
    return df


def _remember(name: str, df: pd.DataFrame):
    STORE[name] = df.copy()


def _m(key, default=None):
    return MACROS.get(key, default)


# ---------------------------------------------------------------------------
# Step 1 – Extract limit values from ROPT_extract_macro_value into MACROS
# Mirrors: Loop (28) over extract_value_count
# ---------------------------------------------------------------------------
def extract_limit_macros(df_param: pd.DataFrame):
    """
    For each (entity_name, parameter_name) pair in ROPT_extract_macro_value,
    find the matching value in df_param and push it into MACROS as
    <parameter_name> (the value for entity_index row).

    RapidMiner loops over the ROPT table extracting:
       macro = %{parameter_name}  from attribute %{parameter_name}
               at row %{entity_index}.
    We replicate this by iterating the mapping table.
    """
    df_ropt = _recall("ROPT_extract_macro_value")
    if df_ropt.empty or df_param.empty:
        logger.info("Skipping limit extraction – ROPT or param data empty.")
        return

    # For each row in the ROPT mapping table
    for _, row in df_ropt.iterrows():
        param_name  = row.get("parameter_name", "")
        entity_name = row.get("entity_name", "")

        # Find matching row in df_param
        mask = df_param["entity_name"] == entity_name if "entity_name" in df_param.columns else pd.Series([False]*len(df_param))
        sub = df_param[mask]

        if param_name in sub.columns and len(sub) > 0:
            val = sub.iloc[0][param_name]
            try:
                MACROS[param_name] = float(val)
            except (TypeError, ValueError):
                MACROS[param_name] = val
            logger.debug("MACRO[%s] = %s  (entity=%s)", param_name, MACROS[param_name], entity_name)
        else:
            # Try to find the column ignoring entity filter
            if param_name in df_param.columns and len(df_param) > 0:
                MACROS[param_name] = float(df_param.iloc[0][param_name]) if pd.notna(df_param.iloc[0][param_name]) else 0

    logger.info("Limit macros extracted – %d parameters", len(df_ropt))

    # ── New: _limit filter + Transpose (12) – push all _limit cols from FS row ──
    # Mirrors: Select Attributes regex (.+)_limit → Transpose → MACROS
    df_fs = df_param[df_param["entity_name"] == "FS"] if "entity_name" in df_param.columns else pd.DataFrame()
    if df_fs.empty:
        logger.warning("No FS row found – skipping _limit extraction.")
        return

    fs_row = df_fs.iloc[0]
    limit_cols = [col for col in fs_row.index if col.endswith("_limit")]

    for col in limit_cols:
        val = fs_row[col]
        try:
            MACROS[col] = float(val)
        except (TypeError, ValueError):
            MACROS[col] = val
        logger.debug("MACRO[%s] = %s  (from FS _limit)", col, MACROS[col])

    logger.info("_limit macros extracted – %d _limit params pushed to MACROS", len(limit_cols))

# ---------------------------------------------------------------------------
# Step 2 – Compute margin flags
# Mirrors: Generate Attributes (265) inside 'extract values and calc margin'
# ---------------------------------------------------------------------------
def compute_margins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean/integer margin columns using the limit macros.
    Returns df with new columns added.
    """
    df = df.copy()

    def col(name, default=0.0):
        return df[name].fillna(default).astype(float) if name in df.columns else pd.Series(default, index=df.index)

    def macro(name, default=0.0):
        return float(_m(name, default))

    # Margin in Fuel Gas (pressure control valve and pressure)
    df["Margin_in_FG"] = np.where(
        (col("fuel_gas_pressure_controlvalve_opening") < macro("fuel_gas_pressure_controlvalve_opening_limit")) &
        (col("fuel_gas_pressure") < macro("fuel_gas_pressure_limit")),
        1, 0
    )

    # Margin in Steam – always 0 per formula
    df["Margin_in_Steam"] = 0

    pass_cols = [
    "pass1_mixed_feed_flow_controller_opening",
    "pass2_mixed_feed_flow_controller_opening",
    "pass3_mixed_feed_flow_controller_opening",
    "pass4_mixed_feed_flow_controller_opening",
    "pass5_mixed_feed_flow_controller_opening",
    "pass6_mixed_feed_flow_controller_opening",
    "pass7_mixed_feed_flow_controller_opening",
    "pass8_mixed_feed_flow_controller_opening",
    ]
    feed_limit = macro("margin_value_feed_limit")
    df["Margin_in_Feed"] = np.where(
        np.all([col(p) < feed_limit for p in pass_cols], axis=0),
        1, 0
    )

    # Margin in Damper
    df["Margin_in_Damper"] = np.where(
        col("damper_opening") < macro("damper_opening_limit"),
        1, 0
    )

    # Quench overhead gas temp margin (always 1)
    df["Quench_OD_Gas_temp_margin"] = 1

    # CGC suction pressure margin (always 1)
    df["CGC_suction_pressure_margin"] = 1

    # C2 splitter DP
    df["C2_splitter_dp_margin"] = np.where(
        col("c2_splitter_dp") < macro("c2_splitter_dp_limit"),
        1, 0
    )

    # C2 splitter bottom ethylene mol percent (always 1)
    df["C2_splitter_btm_c2h4_mol_percent_margin"] = 1

    # C2 splitter reflux pump suction temp (always 1)
    df["C2_splitter_reflux_pump_suction_temp_margin"] = 1

    # ERC governor + ethylene compressor speed
    df["ERC_margin"] = np.where(
        (col("erc_governor_opening") < macro("erc_governor_opening_limit")) &
        (col("ethylene_compressure_suction_speed") < macro("ethylene_compressure_suction_speed_limit")),
        1, 0
    )

    # PRC governor + propylene compressor speed
    df["PRC_margin"] = np.where(
        (col("prc_governor_opening") < macro("prc_governor_opening_limit")) &
        (col("propylene_compressure_suction_speed") < macro("propylene_compressure_suction_speed_limit")),
        1, 0
    )

    lower_limit = macro("margin_in_feed_lower_check_limit")
    df["Margin_in_Feed_lower_check"] = np.where(
        np.all([col(p) < lower_limit for p in pass_cols], axis=0),
        1, 0
    )

    # NOx emission
    df["nox_margin"] = np.where(
        col("net_nox_emission") >= macro("nox_emission_permissible_limit"),
        1, 0
    )

    # Saturator margin
    saturator_pres = macro("ethane_feed_saturator_drum_overhead_pressure")
    df["saturator_margin"] = np.where(
        saturator_pres > macro("saturator_drum_pressure_margin_limit"),
        1, 0
    )

    # SHC margin
    df["shc_margin"] = np.where(
        col("shc_ratio_calculated") > macro("shc_margin_limit"),
        1, 0
    )

    margin_cols = [
    "Margin_in_FG", "Margin_in_Steam", "Margin_in_Feed", "Margin_in_Damper",
    "Quench_OD_Gas_temp_margin", "CGC_suction_pressure_margin",
    "C2_splitter_dp_margin", "C2_splitter_btm_c2h4_mol_percent_margin",
    "C2_splitter_reflux_pump_suction_temp_margin", "ERC_margin", "PRC_margin"
    ]
    df["Total_margin"] = sum(df[c] for c in margin_cols if c in df.columns)

    logger.info("Margin columns computed.")
    return df


# ---------------------------------------------------------------------------
# Step 3 – External constraint reordering
# Mirrors: 'external constraint' Branch (ROPT_external_constraint == "active")
# ---------------------------------------------------------------------------
def apply_external_constraint(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ROPT_external_constraint == 'active', re-rank rows that have
    furnace_external_constraint == 1 to the top of overall_ranking.
    """
    if _m("ROPT_external_constraint") != "active":
        return df

    if "overall_ranking" not in df.columns:
        return df

    df = df.copy()
    # Filter rows that have overall_ranking (not missing)
    df_ranked = df[df["overall_ranking"].notna()].copy()
    if df_ranked.empty:
        return df

    # Separate constrained furnaces
    if "furnace_external_constraint" in df.columns:
        df_constrained = df_ranked[df_ranked["furnace_external_constraint"] == 1].copy()
        df_rest        = df_ranked[df_ranked["furnace_external_constraint"] != 1].copy()
    else:
        return df

    if df_constrained.empty:
        return df

    # Sort remaining by overall_ranking ascending
    df_rest = df_rest.sort_values("overall_ranking")

    # Append constrained rows after rest, then re-assign IDs as new ranking
    combined = pd.concat([df_rest, df_constrained], ignore_index=True)
    combined["overall_ranking"] = range(1, len(combined) + 1)

    # Merge back non-ranked rows
    df_not_ranked = df[df["overall_ranking"].isna()].copy()
    result = pd.concat([combined, df_not_ranked], ignore_index=True)
    logger.info("External constraint applied – re-ranked %d rows.", len(df_constrained))
    return result


# ---------------------------------------------------------------------------
# Step 4 – Rename columns
# Mirrors: Rename (21)
# ---------------------------------------------------------------------------
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "wet_feed_total_flow" in df.columns:
        rename_map["wet_feed_total_flow"] = "Feed_flow"
    if "overall_conversion" in df.columns:
        rename_map["overall_conversion"] = "Overall_conversion"
    if "good _tubes_calculated" in df.columns:
        rename_map["good _tubes_calculated"] = "good_tubes_calculated"
    df = df.rename(columns=rename_map)
    logger.info("Renaming done")
    return df


# ---------------------------------------------------------------------------
# Step 5 – Evaluate inferred tags (inferred_tags_4)
# Mirrors: 'inf tag final (2)' Loop over inferred_tags_4
# ---------------------------------------------------------------------------
def get_optimization_status(row):
    if pd.isna(row['overall_ranking']) or row['steam_water_deoke_status'] == 1:
        return "No Optimization"
    
    if row['furnace_status'] == 1 and row['cracking_cycle_runlength_calculated'] > 1:
        if 1 < row['cracking_cycle_runlength_calculated'] <= 2:
            return "SOR"
        if row['days_remaining'] < 2 or row['max_coke_thickness'] >= _m("max_coke_thickness_limit"):
            return "EOR"
        if (row['percent_above_threshold'] < -100 or
                row['Margin_in_Feed'] + row['Margin_in_Damper'] + row['Margin_in_FG'] < 3):
            if row['Margin_in_Damper'] == 1 and row['Margin_in_FG'] == 1:
                return "Semi Good"
            return "Bad"
        return "Good"
    
    return "No cracking"

def evaluate_inferred_tags(df: pd.DataFrame, tag_store_name: str = "inferred_tags_4") -> pd.DataFrame:
    """
    For each (Inferred_tag, Inferred_tag_formula) row, evaluate the formula
    expression against each row of df and add the result as a new column.
    Uses eval() mirroring RapidMiner's eval(%{Inferred_tag_formula}).
    """
    df_tags = _recall(tag_store_name)
    if df_tags.empty:
        return df

    df = df.copy()
    FUNCTION_FORMULA_MAP = {
        "get_optimization_status": get_optimization_status,
    }
    for _, tag_row in df_tags.iterrows():
        tag_name    = tag_row.get("Inferred_tag", "")
        formula_str = tag_row.get("Inferred_tag_formula", "")
        if not tag_name or not formula_str:
            continue
        try:
            # Allow formulas to reference df columns by name
            matched_fn = next(
                (fn for key, fn in FUNCTION_FORMULA_MAP.items() if key in formula_str),
                None
            )
            if matched_fn:
                df[tag_name] = df.apply(matched_fn, axis=1)
            else:
                df[tag_name] = df.eval(formula_str)
        except Exception as e:
            logger.warning("Inferred tag '%s' eval failed: %s", tag_name, e)
    logger.info("eval inf tag done")
    logger.info("Furnace_condition values after calculating inf:\n%s", df[["entity_name", "Furnace_condition"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Step 6 – Forecasted runlength rank org
# Mirrors: Generate Attributes (269)
# ---------------------------------------------------------------------------
def add_forecasted_runlength_rank_org(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Furnace_condition_code — Generate Attributes (269) formula 1
    cond_map = {"Good": 1, "Bad": 2, "Semi Good": 3, "SOR": 4, "EOR": 5, "No Optimization": -1}
    df["Furnace_condition_code"] = df["Furnace_condition"].map(cond_map).fillna(0).astype(int) \
                                   if "Furnace_condition" in df.columns else 0

    # Forecasted_runlength_rank_org — Generate Attributes (269) formula 2
    if "percent_above_threshold_rank" in df.columns:
        df["Forecasted_runlength_rank_org"] = df["percent_above_threshold_rank"]
    elif "Forecasted_runlength_rank_org" not in df.columns:
        df["Forecasted_runlength_rank_org"] = np.nan

    # Forecasted_runlength_rank — Generate Attributes (269) formula 3
    def _frl_rank(row):
        org = row.get("Forecasted_runlength_rank_org", np.nan)
        if pd.isna(org):
            return np.nan
        if (float(row.get("cracking_cycle_runlength_calculated", 0)) < 2 or
                float(row.get("days_remaining", 0)) < 2):
            return 50
        if float(row.get("percent_above_threshold", 0)) < 0:
            return 100
        return org

    df["Forecasted_runlength_rank"] = df.apply(_frl_rank, axis=1)

    logger.info("forecasted runlength rank org and rank done")
    return df

# ---------------------------------------------------------------------------
# Step 7 – Determine next decoking furnace
# Mirrors: 'decoke' subprocess
# ---------------------------------------------------------------------------
def determine_decoking_furnace(df: pd.DataFrame):
    """
    Find the furnace with the minimum days_remaining; among ties pick max
    coke_thickness. Store in MACROS["Fur_Next_Decoking_Furnace"].
    """
    if "days_remaining" not in df.columns:
        return

    min_days = df["days_remaining"].min()
    MACROS["min_days_remaining"] = float(min_days) if pd.notna(min_days) else 0.0

    df_min = df[df["days_remaining"] == min_days].copy()
    if "max_coke_thickness" in df_min.columns:
        df_min = df_min.sort_values("max_coke_thickness", ascending=False)

    if "entity_name" in df_min.columns and len(df_min) > 0:
        MACROS["Fur_Next_Decoking_Furnace"] = str(df_min.iloc[0]["entity_name"])
        logger.info("Next decoking furnace: %s", MACROS["Fur_Next_Decoking_Furnace"])
    # Generate Macro (10): map furnace name to entity numeric ID
    fur_id_map = {
        "F1": 3796, "F2": 3797, "F3": 3798, "F4": 3799, "F5": 3800,
        "F6": 3801, "F7": 3802, "F8": 3803, "F9": 3804
    }
    MACROS["Fur_Next_Decoking_Furnace_ID"] = fur_id_map.get(
        MACROS["Fur_Next_Decoking_Furnace"], 0
    )
    logger.info("Fur_Next_Decoking_Furnace_ID=%d", MACROS["Fur_Next_Decoking_Furnace_ID"])


# ---------------------------------------------------------------------------
# Step 8 – Furnace coupling logic
# Mirrors: Branch (176) – ROPT_furnace_coupling == 'active'
# ---------------------------------------------------------------------------
def apply_furnace_coupling(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a furnace is in coupled mode AND NOT externally constrained,
    override its Furnace_condition to 'No Optimization'.
    """
    if _m("ROPT_furnace_coupling") != "active":
        return df

    df = df.copy()
    if "furnace_coupled_mode" not in df.columns or "furnace_external_constraint" not in df.columns:
        return df
    if "Furnace_condition" not in df.columns:
        return df

    # mask = (df["furnace_coupled_mode"] == 1) & (df["furnace_external_constraint"] == 0)
    mask = ~((df["furnace_coupled_mode"] == 1) & (df["furnace_external_constraint"] == 0))
    df.loc[mask, "Furnace_condition"] = "No Optimization"
    logger.info("Furnace coupling applied: %d furnaces set to 'No Optimization'.", mask.sum())
    logger.info("Furnace_condition values after coupling:\n%s", df[["entity_name", "Furnace_condition"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Step 9 – Calculations sub-step (Good / No-Good split + counts)
# Mirrors: 'calculations' subprocess
# ---------------------------------------------------------------------------
def run_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split by Furnace_condition, store counts in MACROS.
    Returns df filtered to recognised conditions only.
    """
    if "Furnace_condition" not in df.columns:
        logger.warning("'Furnace_condition' missing – skipping calculations.")
        MACROS["count_of_good_fur"]    = 0
        MACROS["count_of_no_good_fur"] = 0
        return df

    df_fs = df[df["entity_name"] == "FS"] if "entity_name" in df.columns else pd.DataFrame()
    num_cracking = float(df_fs["num_of_furnace_cracking"].iloc[0]) if not df_fs.empty and "num_of_furnace_cracking" in df_fs.columns else 0
    min_limit    = float(df_fs["minimum_fur_for_optimization_limit"].iloc[0]) if not df_fs.empty and "minimum_fur_for_optimization_limit" in df_fs.columns else 1
    logger.info("num cracking: %d , min limit: %d", num_cracking, min_limit)
    df_non_biasing = df[~df["Furnace_condition"].isin(GOOD_CONDITIONS)].copy()
    _remember("NonBiasing_furnaces", df_non_biasing)

    # Keep rows with a recognised condition
    df_recogn = df[df["Furnace_condition"].isin(GOOD_CONDITIONS)].copy()
    # _remember("NonBiasing_furnaces", df_recogn)

    # Good furnaces
    df_good = df_recogn[df_recogn["Furnace_condition"] == "Good"].copy()
    df_good = df_good.sort_values("overall_ranking") if "overall_ranking" in df_good.columns else df_good
    MACROS["count_of_good_fur"] = len(df_good)
    _remember("good_fur_data", df_good)

    # No-good = everything else in recognised set
    df_no_good = df_recogn[df_recogn["Furnace_condition"] != "Good"].copy()
    MACROS["count_of_no_good_fur"] = len(df_no_good)
    _remember("no_good_fur_data", df_no_good)

    MACROS["total_fur_available_for_bias"] = len(df_recogn)
    MACROS["all_furnace_for_conversion_biasing"] = (
        1 if _m("ROPT_all_furnace_for_conversion_biasing") == "active" else 0
    )
    MACROS["minimum_cracking_furnace_available_check"] = 0 if num_cracking < min_limit else 1

    logger.info("Calculations: good=%d, no_good=%d, total=%d, min_crack_check=%d, all_fur_conv_bias=%d",
                MACROS["count_of_good_fur"], MACROS["count_of_no_good_fur"],
                MACROS["total_fur_available_for_bias"],
                MACROS["minimum_cracking_furnace_available_check"],
                MACROS["all_furnace_for_conversion_biasing"])
    return df_recogn

def run_branch5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates Branch (5) of the Preprocess subprocess.
    Condition: minimum_cracking_furnace_available_check == 1

    If True (inner path):
      - Fetch ethane_feed_flow_from_aramco_past_time from DB or set to 0
      - Compute bias_skip_checks macros:
          want_to_change_recycle_feed_flow
          recycle_ethane_controller_margin
          recycle_change_possible
          biasing_condition
          fresh_feed_quantity
          deviation_in_aramco
          fresh_feed_deviation_check
          total_optimizer_run_check
          min_fur_skip_check_bias2
      - Compute Current_Recycle_Ethane_Feed_overall
      - Run biasing_condition == 2 sub-checks (constraint_limit_bias2)
      - Compute final_run_optimizer_check_init

    If False (else path):
      - Set final_run_optimizer_check_init = 0
      - Set constraint_limit_bias2 = 99
      - Set total_optimizer_run_check = 99
    """
    if int(_m("minimum_cracking_furnace_available_check", 0)) != 1:
        MACROS["final_run_optimizer_check_init"] = 0
        MACROS["constraint_limit_bias2"]         = 99
        MACROS["total_optimizer_run_check"]      = 99
        logger.info("Branch 5 ELSE: minimum_cracking_furnace_available_check != 1")
        return df

    # ── Subprocess (15): fetch ethane_feed_flow_from_aramco_past_time ────────
    # In RMP: queries DB for prev-hour aramco flow value
    # We read it from MACROS if already set (by past-hour logic), else default 0
    if "ethane_feed_flow_from_aramco_past_time" not in MACROS:
        MACROS["ethane_feed_flow_from_aramco_past_time"] = 0

    # ── bias skip checks (Generate Macro (11)) ────────────────────────────────
    fresh_feed_change   = float(_m("fresh_feed_change", 0))
    want_recycle        = float(_m("Fur_Want_To_Change_Recycle_Feed_Flow_2", 0))
    recycle_opening     = float(_m("recycle_ethane_flow_controller_opening", 0))
    recycle_limit       = float(_m("recycle_ethane_flow_controller_opening_limit", 90))
    aramco_flow         = float(_m("ethane_feed_flow_from_aramco", 0))
    aramco_past         = float(_m("ethane_feed_flow_from_aramco_past_time", 0))
    aramco_limit        = float(_m("ethane_feed_flow_aramco_limit", 5))
    expected_ff         = float(_m("Fur_Expected_Fresh_Feed", 110))
    count_good          = int(_m("count_of_good_fur", 0))
    total_fur           = int(_m("total_fur_available_for_bias", 0))

    want_to_change_recycle = (
        1 if fresh_feed_change != 0 else want_recycle
    )

    recycle_ethane_controller_margin = (
        1 if fresh_feed_change != 0
        else (1 if recycle_opening < recycle_limit else 0)
    )

    recycle_change_possible = (
        1 if (want_to_change_recycle == 1 and recycle_ethane_controller_margin == 1)
        else 0
    )

    biasing_condition = (
        3 if count_good == 0
        else (1 if recycle_change_possible == 1 else 2)
    )

    fresh_feed_quantity = (
        fresh_feed_change * abs(expected_ff - aramco_flow) / 0.6618
    )

    deviation_in_aramco = (
        0 if aramco_past == 0
        else aramco_flow - aramco_past
    )

    ff_dev_check_1 = (
        0 if (fresh_feed_change == 0 and abs(deviation_in_aramco) > aramco_limit)
        else 1
    )
    ff_dev_check_2 = (
        0 if (fresh_feed_change != 0 and deviation_in_aramco < -1)
        else 1
    )
    fresh_feed_deviation_check = 1 if (ff_dev_check_1 == 1 and ff_dev_check_2 == 1) else 0

    total_optimizer_run_check = (
        1 if (int(_m("minimum_cracking_furnace_available_check", 0)) == 1
              and fresh_feed_deviation_check == 1)
        else 0
    )

    min_fur_skip_check_bias2 = (
        0 if (biasing_condition == 2 and total_fur == 1)
        else 1
    )

    # Push to MACROS
    MACROS["want_to_change_recycle_feed_flow"]   = want_to_change_recycle
    MACROS["recycle_ethane_controller_margin"]    = recycle_ethane_controller_margin
    MACROS["recycle_change_possible"]             = recycle_change_possible
    MACROS["biasing_condition"]                   = biasing_condition
    MACROS["fresh_feed_quantity"]                 = fresh_feed_quantity
    MACROS["deviation_in_aramco"]                 = deviation_in_aramco
    MACROS["fresh_feed_deviation_check"]          = fresh_feed_deviation_check
    MACROS["total_optimizer_run_check"]           = total_optimizer_run_check
    MACROS["min_fur_skip_check_bias2"]            = min_fur_skip_check_bias2

    logger.info("bias_skip_checks: biasing_condition=%d, recycle_change_possible=%d, "
                "fresh_feed_deviation_check=%d, total_optimizer_run_check=%d",
                biasing_condition, recycle_change_possible,
                fresh_feed_deviation_check, total_optimizer_run_check)

    # ── Subprocess (18): Current_Recycle_Ethane_Feed_overall ─────────────────
    # Filter to furnace rows (not FS) with furnace_status == 1; sum Feed_flow
    df_fur = df[
        (df["entity_name"] != "FS") &
        (df["furnace_status"].astype(float) == 1)
    ] if "furnace_status" in df.columns else df[df["entity_name"] != "FS"]

    sum_feed_overall = float(df_fur["Feed_flow"].sum()) if "Feed_flow" in df_fur.columns else 0.0
    MACROS["sum_Feed_flow_overall"] = sum_feed_overall

    shc_calc = float(_m("shc_ratio_calculated", _m("shc_ratio", 0)))
    sys_conv = float(_m("system_overall_conversion", 0))

    current_recycle_overall = (
        sum_feed_overall / (1 + shc_calc) * (100 - sys_conv) / 100
        if (1 + shc_calc) != 0 else 0.0
    )
    MACROS["Current_Recycle_Ethane_Feed_overall"] = current_recycle_overall
    logger.info("Current_Recycle_Ethane_Feed_overall=%.4f", current_recycle_overall)

    # ── Branch (14): biasing_condition == 2 → constraint_limit_bias2 ─────────
    constraint_limit_bias2 = 1   # default (else path of Branch 14)

    if biasing_condition == 2:
        df_good = _recall("good_fur_data")
        min_conv_good = float(df_good["Overall_conversion"].min()) if not df_good.empty else 0.0
        MACROS["min_Overall_conversion_good"] = min_conv_good

        # Check if any good furnace has percent_above_threshold > 0
        if "percent_above_threshold" in df_good.columns:
            df_pos = df_good[df_good["percent_above_threshold"].astype(float) > 0]
            good_fur_with_pos = 1 if len(df_pos) > 0 else 0
        else:
            good_fur_with_pos = 0
        MACROS["good_fur_with_positive_per_thres_available"] = good_fur_with_pos

        if good_fur_with_pos == 0:
            # Check no-good furnaces with conversion below min good conversion
            df_no_good = _recall("no_good_fur_data")
            if not df_no_good.empty and "Overall_conversion" in df_no_good.columns:
                df_no_good = df_no_good.copy()
                df_no_good["min_Overall_conversion_no_good"] = (
                    df_no_good["Overall_conversion"].astype(float)
                    .apply(lambda v: 1 if v < min_conv_good else 0)
                )
                sum_min_conv_no_good = int(df_no_good["min_Overall_conversion_no_good"].sum())
                MACROS["sum_min_Overall_conversion_no_good"] = sum_min_conv_no_good
                constraint_limit_bias2 = 0 if sum_min_conv_no_good == 0 else 1
            else:
                constraint_limit_bias2 = 1
        else:
            constraint_limit_bias2 = 1

    MACROS["constraint_limit_bias2"] = constraint_limit_bias2
    logger.info("constraint_limit_bias2=%d", constraint_limit_bias2)

    # ── Generate Macro (11): final_run_optimizer_check_init ──────────────────
    final_run_optimizer_check_init = (
        1 if (
            total_optimizer_run_check == 1 and
            total_fur > 0 and
            min_fur_skip_check_bias2 == 1 and
            constraint_limit_bias2 == 1
        )
        else 0
    )
    MACROS["final_run_optimizer_check_init"] = final_run_optimizer_check_init
    
    # Subprocess (13)
    # Multiply (11): keep original df intact, work on a copy
    df_full = df.copy()

    # Select Attributes (pass wise): keep only _runlength_remaining cols
    pass_rl_cols = [c for c in df.columns if c.endswith("_runlength_remaining")]
    if pass_rl_cols:
        df_rl = df[pass_rl_cols].copy()

        # Generate Attributes (35): min across all pass runlength_remaining
        df_rl["min_days_pass"] = df_rl.min(axis=1)

        # Generate Attributes (36): passN_feed_red_potential_on_min_days
        for n in range(1, 9):
            rl_col = f"pass{n}_runlength_remaining"
            df_rl[f"pass{n}_feed_red_potential_on_min_days"] = (
                np.where(df_rl[rl_col] == df_rl["min_days_pass"], 0, 1)
                if rl_col in df_rl.columns else 0
            )
        df_rl.drop(columns=["min_days_pass"], inplace=True)

        # Select Attributes (42): keep only potential_on_min_days cols
        pot_cols = [c for c in df_rl.columns if c.endswith("potential_on_min_days")]
        df_pot = df_rl[pot_cols].copy()

        # Join (22): join back onto original full df
        df = df_full.join(df_pot)
        logger.info("Subprocess (13): %d passN_feed_red_potential_on_min_days cols joined.", len(pot_cols))

    logger.info("final_run_optimizer_check_init=%d", final_run_optimizer_check_init)

    return df

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_param: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the Preprocess subprocess.

    Parameters
    ----------
    df_param : pd.DataFrame
        Output of module_03_parameterization.

    Returns
    -------
    df : pd.DataFrame
        Enriched DataFrame with margin flags, renamed columns, and inferred tags.
    """
    logger.info("=== MODULE 04 – PRE-PROCESSING ===")

    # Step 1: extract limit values into MACROS
    extract_limit_macros(df_param)

    # Step 2: compute margin flags
    df = compute_margins(df_param)

    # Step 3: external constraint re-ranking
    df = apply_external_constraint(df)

    # Step 4: rename columns
    df = rename_columns(df)

    # Step 5: evaluate inferred tags
    df = evaluate_inferred_tags(df)

    # Step 6: forecasted runlength rank
    df = add_forecasted_runlength_rank_org(df)

    # Step 7: decoking furnace
    determine_decoking_furnace(df)

    # Step 8: furnace coupling
    df = apply_furnace_coupling(df)

    # Step 9: Good/No-Good split + counts
    df = run_calculations(df)
    # Branch 5
    df = run_branch5(df)
    
    STORE["df_preprocessed"] = df.copy()
    logger.info("PRE-PROCESSING complete – %d rows × %d cols",
                len(df), len(df.columns))
    return df
