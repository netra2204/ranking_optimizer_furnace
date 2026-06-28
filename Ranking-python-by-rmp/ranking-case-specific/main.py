"""
main.py
=======
Furnace Ranking Optimisation Pipeline – Python replica of the RapidMiner
process (ranking_overall_optimization_14th_april.rmp).

Execution order mirrors the RapidMiner operator chain exactly:

  1.  INPUTS (2)                  →  module_01_inputs
  2.  Initialization (3)           →  module_02_initialization
  3.  Parameterization             →  module_03_parameterization
  4.  Preprocess                   →  module_04_preprocessing
  5.  Branch (6) / Past Hour Logic →  module_05_past_hour_logic
  6.  [United OLF → Ranking optimizaion main → main]
        Pre Grid                   →  module_06_pre_grid
        Grid-Main                  →  module_07_grid_main
        Post Grid                  →  module_08_post_grid
  7.  Generate Output              →  module_09_generate_output
  8.  Output_Format_Check          →  module_10_output_format_check

Usage
-----
    python main.py --csv path/to/join_data.csv [--prev-csv path/to/prev_output.csv]

    Or import and call `run_pipeline(csv_path=...)` from another script.
"""
print("START OF SCRIPT")

import argparse
import logging
import sys
import pandas as pd
from datetime import datetime
import os
from config import MACROS, STORE
# ── Modules ───────────────────────────────────────────────────────────────────
import module_01_inputs
import module_02_initialization
import module_03_parameterization
import module_04_preprocessing
import module_05_past_hour_logic
import module_06_pre_grid
import h09_module_07_grid_main
import module_08_post_grid
import module_09_generate_output
import module_10_output_format_check

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")
print("LOGGING IMPORTED")

try:
    from pandas._libs.parsers import STR_NA_VALUES as _PD_NA
    _NA_KEEP_NULL = list(_PD_NA - {"null", "NULL", "Null"})
except Exception:                                   # pragma: no cover - version drift
    _NA_KEEP_NULL = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                     "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
                     "NaN", "None", "n/a", "nan"]
def load_store_data(
    tag_parameter_mapping_csv: str = None,
    ropt_extract_macro_values_csv: str = None,
    inferred_tags_1_csv: str = None,
    inferred_tags_2_csv: str = None,
    inferred_tags_3_csv: str = None,
    inferred_tags_4_csv: str = None,
):
    """
    Pre-populate STORE with all DataFrames required by modules 05 onwards.
    Call this before run_pipeline() when bypassing modules 02 and 03.
    """
    loaders = {
        "tag_parameter_mapping":         tag_parameter_mapping_csv,
        "ROPT_extract_macro_value":      ropt_extract_macro_values_csv,
        "inferred_tags_1":               inferred_tags_1_csv,
        "inferred_tags_2":               inferred_tags_2_csv,
        "inferred_tags_3":               inferred_tags_3_csv,
        "inferred_tags_4":               inferred_tags_4_csv,
    }

    for store_key, path in loaders.items():
        if path is None:
            # Store empty DataFrame as safe fallback
            STORE[store_key] = pd.DataFrame()
            logger.info("STORE['%s'] → empty (no path provided)", store_key)
        else:
            # Preserve the literal "null"/"NULL" text in parameter_name for the
            # tag_parameter_mapping file only; all other files use pandas defaults.
            na_kwargs = ({"keep_default_na": False, "na_values": _NA_KEEP_NULL}
                         if store_key == "tag_parameter_mapping" else {})
            if path.endswith(".xlsx") or path.endswith(".xls"):
                STORE[store_key] = pd.read_excel(path, **na_kwargs)
            else:
                STORE[store_key] = pd.read_csv(path, **na_kwargs)
            logger.info("STORE['%s'] → loaded from '%s' (%d rows)",
                        store_key, path, len(STORE[store_key]))
            
# =============================================================================
# Pipeline orchestrator
# =============================================================================
def run_pipeline(
    csv_path: str = None,
    prev_hour_csv_path: str = None,
    return_wide: bool = False,
    input_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Run the complete furnace-ranking optimisation pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the main input CSV (join-data).
        If None, falls back to DB_CONFIG["repository_entry"] + ".csv".
    input_df : pd.DataFrame, optional
        In-memory input DataFrame. When provided, it is used directly and the
        csv_path / DB read is bypassed (e.g. the common-process result handed
        off from the orchestrator).
    prev_hour_csv_path : str, optional
        Path to the previous hour's ranking output CSV.
        Used by the past-hour logic for deviation detection.
    return_wide : bool
        If True, return the wide (per-furnace) final DataFrame instead
        of the long (tag/value) format.

    Returns
    -------
    pd.DataFrame
        Long-format output (Timestamp | sub_model_id | tag | value) by default,
        or wide format if return_wide=True.
    """
    start_ts = datetime.now()
    logger.info("=" * 70)
    logger.info("FURNACE RANKING OPTIMISATION PIPELINE – START")
    logger.info("=" * 70)

    # ── Step 1: INPUTS ────────────────────────────────────────────────────────
    df_main = module_01_inputs.run(csv_path=csv_path, input_df=input_df)

    # ── Step 2: INITIALIZATION ────────────────────────────────────────────────
    # df_main = module_02_initialization.run(df_main)

    # ── Step 3: PARAMETERIZATION ──────────────────────────────────────────────
    df_param = module_03_parameterization.run(df_main)
    df_param.to_excel(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "Results",
        "united-parameterization.xlsx"
    ))
    # ── Step 4: PRE-PROCESSING ────────────────────────────────────────────────
    # Bypassing initialization & parameterization — data already in wide format
    # df_param = df_main.copy()
    df_preprocessed = module_04_preprocessing.run(df_param)
    # df_preprocessed.to_excel(r"C:\Users\User\Documents\POC\prepro-output.xlsx", index=False)
    # logger.info("pre-pro output written")

    # ── Step 5: PAST HOUR LOGIC ───────────────────────────────────────────────
    # Checks deviation_exists; sets MACROS["deviation_exists"] = 0 or 1
    # df_preprocessed = module_05_past_hour_logic.run(
    #     df_preprocessed, prev_hour_csv_path=prev_hour_csv_path
    # )

    # ── bias_final_decider  (Generate Macro before MAIN branch) ───────────────
    # final_run_optimizer_check = deviation_exists AND final_run_optimizer_check_init
    MACROS["deviation_exists"] = 1
    dev = int(MACROS.get("deviation_exists", 0))
    init_check = int(MACROS.get("final_run_optimizer_check_init", 1))
    MACROS["final_run_optimizer_check"] = 1 if (dev == 1 and init_check == 1) else 0
    MACROS["sum_del_ethylene_final"]    = 0

    # ── MAIN branch: only run the optimizer if final_run_optimizer_check == 1 ─
    if MACROS["final_run_optimizer_check"] == 1:
        logger.info("MAIN branch: optimizer ACTIVE (deviation_exists=%d)", dev)

        # ── Step 6: PRE-GRID ──────────────────────────────────────────────────
        df_pre_grid = module_06_pre_grid.run(df_preprocessed)
        # df_pre_grid.to_excel(r"C:\Users\User\Documents\POC\pre-grid-output.xlsx", index=False)
        # logger.info("pre-grid output written")
        logger.info(f"pre-grid result shape: {df_pre_grid.shape}")
        # logger.info(f"IMP VALUES TO CROSS CHECK:  {MACROS["mixed_feed_margin"]}, {MACROS["Extra_Recycle_Ethane"]},{MACROS["recycle_change_possible"]}")
        # logger.info(f"pre-grid result columns: {df_pre_grid.columns.tolist()}")
       
        # ── Step 7: GRID MAIN ─────────────────────────────────────────────────
        df_pre_grid = h09_module_07_grid_main.run(df_pre_grid)
        logger.info(f"grid main result shape: {df_pre_grid.shape}")
        # df_pre_grid.to_excel(r"C:\Users\netra.joshi\Documents\POC\Results\grid-main-result-2.xlsx", index=False)
        # logger.info("grid main output written")


        # ── Step 8: POST GRID ─────────────────────────────────────────────────
        df_post_grid = module_08_post_grid.run(df_pre_grid)
        logger.info(f"post grid result shape: {df_post_grid.shape}")


    else:
        logger.info("MAIN branch: optimizer SKIPPED (no deviation / check=0).")

        # Branch (114): deviation_exists == 0 → recall prev output and join
        if int(MACROS.get("deviation_exists", 1)) == 0:
            # Then side of Branch (114): reuse prev hour output
            df_prev = STORE.get("prev_timestamp_ranking_output", pd.DataFrame())
            if not df_prev.empty:
                # Exclude bias columns from prev output
                exclude_cols = ["change_in_furnace", "cit_bias", "conversion_bias",
                                "cot_bias", "feed_bias", "heat_bias", "shc_bias",
                                "total_optimizer_run_check"]
                df_prev = df_prev.drop(columns=[c for c in exclude_cols if c in df_prev.columns], errors="ignore")
                # Inner join on entity_name with current data
                curr_cols = [c for c in df_preprocessed.columns if c not in df_prev.columns or c == "entity_name"]
                df_joined = pd.merge(df_prev, df_preprocessed[curr_cols], on="entity_name", how="inner")
                MACROS["ranking_cause_indicator"] = 99
                df_joined["ranking_opportunity"] = 0
                STORE["df_post_grid"] = df_joined.copy()
                df_post_grid = df_joined.copy()
            else:
                STORE["df_post_grid"] = df_preprocessed.copy()
                df_post_grid = df_preprocessed.copy()
        else:
            # Else side of Branch (114): deviation_exists != 0, optimizer check failed
            # Set ranking_cause_indicator based on RMP Generate Macro (49)
            min_fur_check      = int(MACROS.get("minimum_cracking_furnace_available_check", 0))
            total_opt_check    = int(MACROS.get("total_optimizer_run_check", 0))
            constraint_bias2   = int(MACROS.get("constraint_limit_bias2", 1))

            MACROS["ranking_cause_indicator"] = (
                0  if min_fur_check == 0
                else (2  if total_opt_check == 0
                else (7  if constraint_bias2 == 1
                else 11))
            )

            # Set bias columns to 0 and New_ columns = current values
            df_else = df_preprocessed.copy()
            df_else["conversion_bias"]        = 0
            df_else["feed_bias"]              = 0
            df_else["New_Overall_conversion"] = df_else["Overall_conversion"] if "Overall_conversion" in df_else.columns else 0
            df_else["New_Feed_flow"]          = df_else["Feed_flow"] if "Feed_flow" in df_else.columns else 0
            df_else["ranking_opportunity"]    = 0
            MACROS["biasing_condition"]       = 0
            STORE["df_post_grid"] = df_else.copy()
            df_post_grid = df_else.copy()


    # ── Step 9: GENERATE OUTPUT ───────────────────────────────────────────────
    df_output = module_09_generate_output.run(df_post_grid)
    logger.info(f"generate output result shape: {df_output.shape}")

    # ── Step 10: OUTPUT FORMAT CHECK ──────────────────────────────────────────
    try:
        df_long = module_10_output_format_check.run(df_output)
        logger.info(f"output format check result shape: {df_long.shape}")

    except ValueError as exc:
        logger.error("Output format check failed: %s", exc)
        raise

    _log_summary(df_output, start_ts)
    return df_output if return_wide else df_long


# =============================================================================
# Summary logger
# =============================================================================
def _log_summary(df_output: pd.DataFrame, start_ts: datetime):
    elapsed = (datetime.now() - start_ts).total_seconds()
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE  (%.1f s)", elapsed)
    logger.info("  deviation_exists        = %s", MACROS.get("deviation_exists"))
    logger.info("  ranking_cause_indicator = %s", MACROS.get("ranking_cause_indicator"))
    logger.info("  sum_del_ethylene_final  = %.4f t/h", MACROS.get("sum_del_ethylene_final", 0))
    # logger.info("  Furnaces in output      = %d", len(df_output))
    if "overall_ranking" in df_output.columns:
        logger.info("  Ranking:")
        for _, row in df_output.sort_values("overall_ranking").iterrows():
            logger.info(
                "    Rank %2s  %-8s  cond=%-12s  ΔFeed=%+.2f  Δconv=%+.3f  Δeth=%.4f t/h",
                int(row.get("overall_ranking", 0)),
                row.get("entity_name", "?"),
                row.get("Furnace_condition", "?"),
                float(row.get("feed_bias", 0)),
                float(row.get("conversion_bias", 0)),
                float(row.get("del_ethylene", 0)),
            )
    logger.info("=" * 70)