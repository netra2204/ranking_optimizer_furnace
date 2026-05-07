"""
module_01_inputs.py
===================
Replicates the "INPUTS (2)" subprocess from the RapidMiner process.

Responsibilities
----------------
1.  Load the raw join-data from a local CSV (or DB if configured).
2.  Derive temp columns needed by downstream modules
    (Fur_Want_To_Change_Recycle_Feed_Flow_2, fresh-feed flags, limits, etc.).
3.  Convert the Timestamp column to a consistent string format and extract
    the end_time macro.
4.  Load the PIPELINE MACROS table and propagate all flag/param macros.
5.  Optionally fetch tag / coilsim / sensor data from a SQL database.
6.  Store the prepared DataFrames in STORE for downstream modules.

Returns
-------
df_main : pd.DataFrame   – one row per furnace, current-state columns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from config import MACROS, STORE, INPUTS, PIPELINE_MACROS, DB_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: macro resolution  (mirrors RapidMiner's %{macro_name} syntax)
# ---------------------------------------------------------------------------
def _m(key, default=None):
    """Resolve a macro value from the global MACROS dict."""
    return MACROS.get(key, default)


# ---------------------------------------------------------------------------
# Step 1 – Load raw data
# ---------------------------------------------------------------------------
def load_raw_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the join-data dataset.
    Falls back to a local CSV when pull_tables_from_db == 0.

    The CSV is expected to contain at least:
      Timestamp, entity_name (furnace ID), and all sensor/tag columns.
    """
    if _m("pull_tables_from_db", 0) == 1:
        # --- DB path (requires sqlalchemy / pyodbc in your environment) ---
        try:
            from sqlalchemy import create_engine
            conn_str = MACROS.get("db_connection_string", "")
            engine = create_engine(conn_str)
            query = f"""
                SELECT fo.*, t.name
                FROM {DB_CONFIG['output_table']} fo WITH(NOLOCK)
                LEFT JOIN {DB_CONFIG['tag_table']} t
                    ON fo.tag_id = t.tag_id
                WHERE fo.model_id = {DB_CONFIG['model_id']}
                  AND t.name LIKE '{DB_CONFIG['tag_prefix']}'
            """
            df = pd.read_sql(query, engine)
            logger.info("Data loaded from database: %d rows", len(df))
        except Exception as e:
            logger.error("DB load failed (%s); falling back to CSV.", e)
            df = _load_csv(csv_path)
    else:
        df = _load_csv(csv_path)

    return df


def _load_csv(path: str) -> pd.DataFrame:
    if path is None:
        print('db config path')
        path = DB_CONFIG.get("repository_entry", "tag-in-parameter-format-preprocess-input.xlsx")
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    logger.info("Data loaded from '%s': %d rows, %d cols", path, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Step 2 – Derive temp / macro columns  (TEMP CHANGE IN DATA operator)
# ---------------------------------------------------------------------------
def apply_temp_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror of the 'TEMP CHANGE IN DATA' Generate-Columns operator.
    Adds / overwrites columns that carry the input macros into the dataset
    so that downstream Generate-Attributes expressions can reference them.
    """
    lim = float(INPUTS["Fur_change_recycle_ethane_limit"])

    df = df.copy()
    df["Fur_Want_To_Change_Recycle_Feed_Flow_2"] = float(INPUTS["want_to_change_recycle_feed_flow_set"])
    df["Fur_Fresh_Feed_Change"]                  = float(INPUTS["fresh_feed_change_set"])
    df["Fur_change_recycle_ethane_upper_limit"]  = lim
    df["Fur_change_recycle_ethane_lower_limit"]  = -lim

    for i in range(1, 10):
        df[f"Fur{i}_Fresh_Feed_Change"] = df["Fur_Fresh_Feed_Change"]

    df["Fur_Maximum_Conversion_Single_furnace_limit"] = float(INPUTS["single_fur_limit"])
    df["Fur_Expected_Fresh_Feed"]                     = float(INPUTS["fresh_feed_input"])
    return df


# ---------------------------------------------------------------------------
# Step 3 – Timestamp handling
# ---------------------------------------------------------------------------
def process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Timestamp to a canonical string (yyyy-MM-dd HH:mm:ss) and
    extract the end_time macro (value of the first row).
    Mirrors: Date to Nominal (2) + Extract Macro (110).
    """
    df = df.copy()

    # Try to parse whatever format the raw data uses
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], infer_datetime_format=True)

    # Store datetime for later arithmetic, then convert to canonical string
    df["_Timestamp_dt"] = df["Timestamp"]
    df["Timestamp"] = df["_Timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Extract end_time macro from the first row
    end_time = df["Timestamp"].iloc[0] if len(df) > 0 else None
    MACROS["end_time"] = end_time
    logger.info("end_time macro set to: %s", end_time)

    return df


# ---------------------------------------------------------------------------
# Step 4 – Pipeline macros exampleset  (PIPELINE MACROS + Set Macros from ES)
# ---------------------------------------------------------------------------
def load_pipeline_macros():
    """
    Replicates the 'PIPELINE MACROS' Create-ExampleSet operator followed by
    'Set Macros from ExampleSet (6)'.  Pushes every key/value pair in
    PIPELINE_MACROS into the global MACROS dict.
    """
    for key, val in PIPELINE_MACROS.items():
        MACROS[key] = val
    logger.info("Pipeline macros loaded: %s", list(PIPELINE_MACROS.keys()))


# ---------------------------------------------------------------------------
# Step 5 – DB-sourced tag/coilsim data  (Subprocess (9) / Branch (11))
# ---------------------------------------------------------------------------
def load_tag_data(df_main: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mirrors the optional DB read inside 'Subprocess (9)' which is guarded by
    the 'pull_tables_from_db' macro.

    Returns (df_tag, df_tag_details).
    When DB is off, returns placeholder DataFrames that are already merged
    into df_main by the caller.
    """
    if _m("pull_tables_from_db", 0) == 0:
        logger.info("pull_tables_from_db=0 – using pre-merged data from CSV.")
        # Return empty frames; initialization will use whatever is already in df_main
        df_tag         = pd.DataFrame()
        df_tag_details = pd.DataFrame()
        return df_tag, df_tag_details

    # ---- DB path ---------------------------------------------------------
    try:
        from sqlalchemy import create_engine
        engine = create_engine(MACROS.get("db_connection_string", ""))

        # Tag master table
        df_tag = pd.read_sql(
            f"SELECT * FROM {DB_CONFIG['tag_table']}",
            engine
        )

        # Tag details / coilsim equation table
        df_tag_details = pd.read_sql(
            "SELECT * FROM dbo.tag_details",  # adjust table name as needed
            engine
        )
        logger.info("Tag data loaded from DB: tag=%d rows, details=%d rows",
                    len(df_tag), len(df_tag_details))
    except Exception as e:
        logger.error("Tag DB load failed: %s", e)
        df_tag, df_tag_details = pd.DataFrame(), pd.DataFrame()

    return df_tag, df_tag_details


# ---------------------------------------------------------------------------
# Store helper
# ---------------------------------------------------------------------------
def _remember(name: str, df: pd.DataFrame):
    STORE[name] = df.copy()
    logger.debug("Stored '%s' (%d rows × %d cols)", name, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(csv_path: str = None) -> pd.DataFrame:
    """
    Execute the full INPUTS (2) subprocess and return the prepared main
    DataFrame.  Also populates STORE["tag"], STORE["tag_details"] and all
    relevant MACROS entries.

    Parameters
    ----------
    csv_path : str, optional
        Path to the input CSV.  Falls back to DB_CONFIG["repository_entry"].

    Returns
    -------
    df_main : pd.DataFrame
        The main per-furnace dataset, ready for the Initialization module.
    """
    logger.info("=== MODULE 01 – INPUTS ===")

    # 1. Load raw data
    df_main = load_raw_data(csv_path)

    # 2. Apply temp changes (derive macro columns)
    df_main = apply_temp_changes(df_main)

    # 3. Process timestamp; extract end_time macro
    df_main = process_timestamp(df_main)

    # 4. Push pipeline macros into global MACROS dict
    load_pipeline_macros()

    # 5. (Optional) load tag/coilsim data from DB
    df_tag, df_tag_details = load_tag_data(df_main)
    if not df_tag.empty:
        _remember("tag", df_tag)
    if not df_tag_details.empty:
        _remember("tag_details", df_tag_details)

    # Keep df_main in store for other modules
    _remember("df_main_inputs", df_main)

    logger.info("INPUTS complete – %d furnace rows, %d columns", len(df_main), len(df_main.columns))
    return df_main
