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

import argparse
import logging
import sys
import pandas as pd
from datetime import datetime

# ── Global state ──────────────────────────────────────────────────────────────
from config import MACROS, STORE

# ── Modules ───────────────────────────────────────────────────────────────────
import module_01_inputs
import module_02_initialization
import module_03_parameterization
import module_04_preprocessing
import module_05_past_hour_logic
import module_06_pre_grid
import module_07_grid_main
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


# =============================================================================
# Pipeline orchestrator
# =============================================================================
def run_pipeline(
    csv_path: str = None,
    prev_hour_csv_path: str = None,
    return_wide: bool = False,
) -> pd.DataFrame:
    """
    Run the complete furnace-ranking optimisation pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the main input CSV (join-data).
        If None, falls back to DB_CONFIG["repository_entry"] + ".csv".
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
    df_main = module_01_inputs.run(csv_path=csv_path)

    # ── Step 2: INITIALIZATION ────────────────────────────────────────────────
    df_main = module_02_initialization.run(df_main)

    # ── Step 3: PARAMETERIZATION ──────────────────────────────────────────────
    df_param = module_03_parameterization.run(df_main)

    # ── Step 4: PRE-PROCESSING ────────────────────────────────────────────────
    df_preprocessed = module_04_preprocessing.run(df_param)

    # ── Step 5: PAST HOUR LOGIC ───────────────────────────────────────────────
    # Checks deviation_exists; sets MACROS["deviation_exists"] = 0 or 1
    df_preprocessed = module_05_past_hour_logic.run(
        df_preprocessed, prev_hour_csv_path=prev_hour_csv_path
    )

    # ── Guard: minimum_cracking_furnace check ─────────────────────────────────
    # The RapidMiner process has Branch (6) guarding the optimizer path.
    # If no furnaces are available, we skip to output generation.
    if MACROS.get("minimum_cracking_furnace_available_check", 0) != 1:
        logger.warning(
            "minimum_cracking_furnace_available_check != 1 – skipping optimizer."
        )
        MACROS["deviation_exists"]    = 1
        MACROS["ranking_cause_indicator"] = -99
        df_output = module_09_generate_output.run(df_preprocessed)
        df_long   = module_10_output_format_check.run(df_output)
        _log_summary(df_output, start_ts)
        return df_output if return_wide else df_long

    # ── bias_final_decider  (Generate Macro before MAIN branch) ───────────────
    # final_run_optimizer_check = deviation_exists AND final_run_optimizer_check_init
    dev = int(MACROS.get("deviation_exists", 0))
    init_check = int(MACROS.get("final_run_optimizer_check_init", 1))
    MACROS["final_run_optimizer_check"] = 1 if (dev == 1 and init_check == 1) else 0
    MACROS["sum_del_ethylene_final"]    = 0

    # ── MAIN branch: only run the optimizer if final_run_optimizer_check == 1 ─
    if MACROS["final_run_optimizer_check"] == 1:
        logger.info("MAIN branch: optimizer ACTIVE (deviation_exists=%d)", dev)

        # ── Step 6: PRE-GRID ──────────────────────────────────────────────────
        df_pre_grid = module_06_pre_grid.run(df_preprocessed)

        # ── Step 7: GRID MAIN ─────────────────────────────────────────────────
        df_pre_grid = module_07_grid_main.run(df_pre_grid)

        # ── Step 8: POST GRID ─────────────────────────────────────────────────
        df_post_grid = module_08_post_grid.run(df_pre_grid)

    else:
        logger.info("MAIN branch: optimizer SKIPPED (no deviation / check=0).")
        # Ensure df_post_grid exists in STORE for generate_output
        STORE["df_post_grid"] = df_preprocessed.copy()

    # ── Step 9: GENERATE OUTPUT ───────────────────────────────────────────────
    df_output = module_09_generate_output.run(df_preprocessed)

    # ── Step 10: OUTPUT FORMAT CHECK ──────────────────────────────────────────
    try:
        df_long = module_10_output_format_check.run(df_output)
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
    logger.info("  Furnaces in output      = %d", len(df_output))
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


# =============================================================================
# CLI entry point
# =============================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Furnace Ranking Optimisation Pipeline"
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Path to the main input CSV (join-data). "
             "Defaults to DB_CONFIG['repository_entry'].csv",
    )
    parser.add_argument(
        "--prev-csv",
        metavar="PATH",
        default=None,
        help="Path to the previous-hour ranking output CSV "
             "(used for deviation detection).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write the long-format output to this CSV path.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Return / write wide per-furnace output instead of long format.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Re-configure log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        result = run_pipeline(
            csv_path=args.csv,
            prev_hour_csv_path=args.prev_csv,
            return_wide=args.wide,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    if args.output:
        result.to_csv(args.output, index=False)
        logger.info("Output written to '%s'  (%d rows).", args.output, len(result))
    else:
        # Print first few rows to stdout
        print("\n── Pipeline result (first 20 rows) ──")
        print(result.head(20).to_string(index=False))

    return result


if __name__ == "__main__":
    main()
