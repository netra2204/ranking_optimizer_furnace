"""
module_05_past_hour_logic.py
============================
Replicates the "Branch (6)" / "deviation check" / "past time" logic inside
"United OLF".

Flow (mirrors the nested Branch operators in the .rmp)
------
    Branch (6)  condition: minimum_cracking_furnace_available_check == 1
      └─ deviation check  Branch (macro_defined: ROPT_use_past_time_output)
           └─ Branch (10)  condition: ROPT_use_past_time_output == "active"
                └─ past time subprocess
                     ├─ Compute 24-hrs-ago and prev-hour timestamps
                     ├─ Query the DB for past-24h output data
                     ├─ Filter to the prev-hour timestamp
                     ├─ Branch (16): min_examples >= 1 → past_time_bypass_check
                     └─ Branch (17): past_time_bypass == 0 → deviation check

Responsibilities
----------------
1.  Check minimum_cracking_furnace_available_check; if != 1, set
    deviation_exists = 1 and skip.
2.  Compute timestamp macros:
      24hrs_Timestamp_final_output  (24 h before end_time)
      prev_Timestamp_final_output   (1 h before end_time, MM/dd/yyyy format)
      prev_Timestamp_final_output2  (1 h before end_time, yyyy-MM-dd format)
3.  Query (or load) previous-hour ranking output from DB / local store.
4.  If records found, run "past_time_bypass_check" sub-logic:
      – Pull the prev-hour dataset (prev_timestamp_ranking_output).
      – Compare with current data; if deviation detected set deviation_exists = 1.
5.  If no records found, set past_time_bypass = 1.
6.  If ROPT_use_past_time_output != "active" or macro not defined,
    set deviation_exists = 1.

Outputs  (MACROS + STORE)
-------
    MACROS["deviation_exists"]           – 0 = no change; 1 = run optimizer
    MACROS["past_time_bypass"]           – 1 = skip past-hour comparison
    STORE["prev_timestamp_ranking_output"] – previous hour DataFrame
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from config import MACROS, STORE, DB_CONFIG

logger = logging.getLogger(__name__)


def _m(key, default=None):
    return MACROS.get(key, default)


def _remember(name, df):
    STORE[name] = df.copy()


def _recall(name):
    return STORE.get(name, pd.DataFrame())


# ---------------------------------------------------------------------------
# Step 1 – Compute timestamp macros
# Mirrors: past 24hrs (adjust_date -24h) + past _hour (adjust_date -1h)
#          + Date to Nominal (three format variants)
# ---------------------------------------------------------------------------
def compute_timestamp_macros(end_time_str: str):
    """
    Given end_time as a string (yyyy-MM-dd HH:mm:ss), compute:
      24hrs_Timestamp_final_output  → yyyy-MM-dd HH:mm:ss
      prev_Timestamp_final_output   → MM/dd/yyyy h:mm:ss a
      prev_Timestamp_final_output2  → yyyy-MM-dd HH:mm:ss
    """
    try:
        end_dt = pd.to_datetime(end_time_str)
    except Exception:
        logger.warning("Cannot parse end_time '%s' – setting deviation_exists=1.", end_time_str)
        MACROS["deviation_exists"] = 1
        return

    dt_24h = end_dt - timedelta(hours=24)
    dt_1h  = end_dt - timedelta(hours=1)

    MACROS["24hrs_Timestamp_final_output"]  = dt_24h.strftime("%Y-%m-%d %H:%M:%S")
    MACROS["prev_Timestamp_final_output"]   = dt_1h.strftime("%m/%d/%Y %-I:%M:%S %p")  # e.g. 04/14/2026 3:00:00 AM
    MACROS["prev_Timestamp_final_output2"]  = dt_1h.strftime("%Y-%m-%d %H:%M:%S")

    logger.info("24h timestamp : %s", MACROS["24hrs_Timestamp_final_output"])
    logger.info("prev timestamp: %s", MACROS["prev_Timestamp_final_output"])


# ---------------------------------------------------------------------------
# Step 2 – Load past-24h output data
# Mirrors: Read Database (11) inside 'past time' subprocess
# ---------------------------------------------------------------------------
def load_past_output(prev_hour_csv_path: str = None) -> pd.DataFrame:
    """
    If DB is enabled, query Furnace_Output for the past 24 h.
    Otherwise try loading from a local CSV / STORE.
    """
    if _m("pull_tables_from_db", 0) == 1:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(MACROS.get("db_connection_string", ""))
            ts_24h  = _m("24hrs_Timestamp_final_output")
            ts_end  = _m("end_time")
            model_id = DB_CONFIG["model_id"]
            query = f"""
                SELECT fo.*, t.name
                FROM {DB_CONFIG['output_table']} fo WITH(NOLOCK)
                LEFT JOIN {DB_CONFIG['tag_table']} t ON fo.tag_id = t.tag_id
                WHERE fo.model_id = {model_id}
                  AND fo.time_stamp >= '{ts_24h}'
                  AND fo.time_stamp <= '{ts_end}'
                  AND t.name LIKE 'un.olf%'
            """
            df = pd.read_sql(query, engine)
            logger.info("Past 24h data loaded from DB: %d rows", len(df))
            return df
        except Exception as e:
            logger.error("DB load for past output failed: %s", e)
            return pd.DataFrame()
    else:
        # Fallback: check STORE, then try local CSV
        df_prev = _recall("prev_timestamp_ranking_output")
        if not df_prev.empty:
            return df_prev
        if prev_hour_csv_path:
            try:
                df = pd.read_csv(prev_hour_csv_path)
                logger.info("Past output loaded from CSV: %d rows", len(df))
                return df
            except Exception as e:
                logger.warning("CSV load for past output failed: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3 – Filter to prev-hour timestamp
# Mirrors: 'past hour' Filter Examples (time_stamp.eq.%{prev_Timestamp_final_output})
# ---------------------------------------------------------------------------
def filter_prev_hour(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = "time_stamp" if "time_stamp" in df.columns else "Timestamp"
    prev_ts = _m("prev_Timestamp_final_output")

    if ts_col not in df.columns or not prev_ts:
        return pd.DataFrame()

    df_filtered = df[df[ts_col].astype(str) == str(prev_ts)].copy()
    logger.info("Prev-hour filter: %d rows match timestamp '%s'", len(df_filtered), prev_ts)
    return df_filtered


# ---------------------------------------------------------------------------
# Step 4 – Past-time bypass check
# Mirrors: 'past time bypass check' subprocess + Branch (16) + Branch (17)
# ---------------------------------------------------------------------------
def past_time_bypass_check(df_prev_hour: pd.DataFrame, df_current: pd.DataFrame):
    """
    If prev-hour records exist:
      – Store as prev_timestamp_ranking_output
      – Compare current vs prev data; set deviation_exists based on diff
    If no records: set past_time_bypass = 1.
    """
    if len(df_prev_hour) == 0:
        MACROS["past_time_bypass"] = 1
        MACROS["deviation_exists"] = 1
        logger.info("No prev-hour records found → past_time_bypass=1, deviation_exists=1")
        return

    _remember("prev_timestamp_ranking_output", df_prev_hour)

    if _m("past_time_bypass", 0) == 0:
        # Deviation check: compare key metrics between current and prev-hour
        deviation_detected = _detect_deviation(df_current, df_prev_hour)
        if deviation_detected:
            MACROS["deviation_exists"] = 1
            logger.info("Deviation detected between current and prev-hour data.")
        else:
            MACROS["deviation_exists"] = 0
            logger.info("No deviation detected – will reuse previous output.")
    else:
        MACROS["deviation_exists"] = 1


def _detect_deviation(df_curr: pd.DataFrame, df_prev: pd.DataFrame) -> bool:
    """
    Simple deviation check: compare entity_name rows on key numeric columns.
    Returns True if any difference exceeds a small tolerance.
    Mirrors: Branch (39) with min_examples >= 1 check.
    """
    if df_curr.empty or df_prev.empty:
        return True

    key_cols = [c for c in ["Feed_flow", "Overall_conversion", "overall_ranking"]
                if c in df_curr.columns and c in df_prev.columns]
    if not key_cols:
        return True   # cannot compare → assume deviation

    merge_key = "entity_name"
    if merge_key not in df_curr.columns or merge_key not in df_prev.columns:
        return True

    merged = pd.merge(
        df_curr[[merge_key] + key_cols].rename(columns={c: c + "_curr" for c in key_cols}),
        df_prev[[merge_key] + key_cols].rename(columns={c: c + "_prev" for c in key_cols}),
        on=merge_key, how="inner"
    )

    if merged.empty:
        return True

    tol = 1e-6
    for col in key_cols:
        diff = (merged[col + "_curr"] - merged[col + "_prev"]).abs()
        if (diff > tol).any():
            return True

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(df_current: pd.DataFrame, prev_hour_csv_path: str = None) -> pd.DataFrame:
    """
    Execute the past-hour logic.

    Parameters
    ----------
    df_current : pd.DataFrame
        Current preprocessed furnace data.
    prev_hour_csv_path : str, optional
        Path to a CSV containing the previous hour's ranking output.

    Returns
    -------
    df_current : pd.DataFrame
        Unchanged; outputs are MACROS and STORE entries.
    """
    logger.info("=== MODULE 05 – PAST HOUR LOGIC ===")

    # Check minimum_cracking_furnace_available_check
    if _m("minimum_cracking_furnace_available_check", 0) != 1:
        MACROS["deviation_exists"] = 1
        logger.info("minimum_cracking_furnace_available_check != 1 → deviation_exists=1")
        return df_current

    # Check ROPT_use_past_time_output macro defined and active
    if "ROPT_use_past_time_output" not in MACROS:
        MACROS["deviation_exists"] = 1
        logger.info("ROPT_use_past_time_output not defined → deviation_exists=1")
        return df_current

    if _m("ROPT_use_past_time_output") != "active":
        MACROS["deviation_exists"] = 1
        logger.info("ROPT_use_past_time_output != 'active' → deviation_exists=1")
        return df_current

    # Compute timestamp macros
    end_time = _m("end_time")
    if end_time is None:
        MACROS["deviation_exists"] = 1
        logger.warning("end_time macro not set → deviation_exists=1")
        return df_current

    compute_timestamp_macros(end_time)

    # Load past 24h output
    df_past_24h = load_past_output(prev_hour_csv_path)

    # Filter to prev-hour
    df_prev_hour = filter_prev_hour(df_past_24h)

    # Past-time bypass check / deviation detection
    past_time_bypass_check(df_prev_hour, df_current)

    logger.info("PAST HOUR LOGIC complete – deviation_exists=%s, past_time_bypass=%s",
                _m("deviation_exists"), _m("past_time_bypass"))
    return df_current
