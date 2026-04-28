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
    df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Step 5 – Evaluate inferred tags (inferred_tags_4)
# Mirrors: 'inf tag final (2)' Loop over inferred_tags_4
# ---------------------------------------------------------------------------
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
    for _, tag_row in df_tags.iterrows():
        tag_name    = tag_row.get("Inferred_tag", "")
        formula_str = tag_row.get("Inferred_tag_formula", "")
        if not tag_name or not formula_str:
            continue
        try:
            # Allow formulas to reference df columns by name
            df[tag_name] = df.eval(formula_str)
        except Exception as e:
            logger.warning("Inferred tag '%s' eval failed: %s", tag_name, e)

    return df


# ---------------------------------------------------------------------------
# Step 6 – Forecasted runlength rank org
# Mirrors: Generate Attributes (269)
# ---------------------------------------------------------------------------
def add_forecasted_runlength_rank_org(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "percent_above_threshold_rank" in df.columns:
        df["Forecasted_runlength_rank_org"] = df["percent_above_threshold_rank"]
    elif "Forecasted_runlength_rank_org" not in df.columns:
        df["Forecasted_runlength_rank_org"] = np.nan
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

    mask = (df["furnace_coupled_mode"] == 1) & (df["furnace_external_constraint"] == 0)
    df.loc[mask, "Furnace_condition"] = "No Optimization"
    logger.info("Furnace coupling applied: %d furnaces set to 'No Optimization'.", mask.sum())
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

    # Keep rows with a recognised condition
    df_recogn = df[df["Furnace_condition"].isin(GOOD_CONDITIONS)].copy()
    _remember("NonBiasing_furnaces", df_recogn)

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

    logger.info("Calculations: good=%d, no_good=%d",
                MACROS["count_of_good_fur"], MACROS["count_of_no_good_fur"])
    return df_recogn


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

    STORE["df_preprocessed"] = df.copy()
    logger.info("PRE-PROCESSING complete – %d rows × %d cols",
                len(df), len(df.columns))
    return df
