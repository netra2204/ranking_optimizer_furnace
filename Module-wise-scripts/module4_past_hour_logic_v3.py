"""
=============================================================================
MODULE 4: PAST HOUR LOGIC
=============================================================================
Inputs (from Module 3 preprocessing):
    1. eligible_furnaces_df  → passed through UNCHANGED to output
    2. all_furnaces_df       → past hour logic runs on this (includes FS row)

Set macro (from RapidMiner):
    furnace_step_adjust_feed_grid_limit = 6

Gate macro:
    ropt_use_past_time_output
        - NOT in current input → comes from DB when deployed
        - Must exist AND have value 'active' to proceed

Check order:
    1. minimum_cracking_furnace_available_check == 1 (else deviation_exists = 1)
    2. ropt_use_past_time_output exists AND == 'active' (else deviation_exists = 1)
    3. Past hour consecutive use check (last N=6 hours)
    4. 6 individual deviation signals + further_deviation_check

Outputs:
    1. eligible_furnaces_df  → UNCHANGED
    2. all_furnaces_df       → with new columns: 'deviation_exists', 'deviation_signals'
    3. deviation_exists      → scalar int (0 or 1)

Dependencies: pip install pandas numpy openpyxl
=============================================================================
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

FURNACE_STEP_ADJUST_FEED_GRID_LIMIT = 6
PAST_HOUR_WINDOW                    = 6
HISTORY_HOURS                       = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(v, default=None):
    if v is None:
        return default
    try:
        if math.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _is_missing(v):
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# CHECK 1: minimum_cracking_furnace_available_check
# Formula (from FS row of all_furnaces_df):
#   count of furnaces whose Furnace_condition in
#       {'Good','Semi Good','Bad','SOR','EOR'}
#   If count < minimum_fur_for_optimization_limit → 0 else → 1
# ---------------------------------------------------------------------------

def compute_min_cracking_check(all_df: pd.DataFrame) -> int:
    fs_row = all_df[all_df['entity_name'] == 'FS']
    fs     = fs_row.iloc[0] if not fs_row.empty else pd.Series(dtype=float)

    min_fur_limit      = int(_val(fs.get('minimum_fur_for_optimization_limit'), 5))
    num_fur_cracking   = _val(fs.get('num_of_furnace_cracking'))

    if _is_missing(num_fur_cracking):
        return 0

    return 0 if int(float(num_fur_cracking)) < min_fur_limit else 1


# ---------------------------------------------------------------------------
# Extract current signal values from all_furnaces_df
# ---------------------------------------------------------------------------

def extract_current_values(all_df: pd.DataFrame) -> Dict[str, Any]:
    fs_row       = all_df[all_df['entity_name'] == 'FS']
    furnace_rows = all_df[all_df['entity_name'] != 'FS']
    fs           = fs_row.iloc[0] if not fs_row.empty else pd.Series(dtype=float)

    conv_delta_threshold = _val(fs.get('past_time_conversion_delta_threshold'), 0.3)
    feed_delta_threshold = _val(fs.get('past_time_feed_delta_threshold'), 0.5)
    run_freq_threshold   = _val(fs.get('past_time_run_frequency_threshold'), 6.0)

    fresh_feed_change = _val(fs.get('fresh_feed_change'), 0)

    # biasing_condition — prefer FS row, fall back to furnace rows
    biasing_condition = _val(fs.get('biasing_condition'))
    if _is_missing(biasing_condition) and 'biasing_condition' in furnace_rows.columns:
        vals = furnace_rows['biasing_condition'].dropna()
        biasing_condition = _val(vals.iloc[0]) if not vals.empty else None

    saturated_pressure = pd.to_numeric(
        furnace_rows['ethane_feed_saturator_drum_overhead_pressure'], errors='coerce'
    ).mean()

    consider_for_conversion = _val(fs.get('all_furnace_for_conversion_biasing'), 0)

    specific_energy = pd.to_numeric(
        furnace_rows['specific_energy_consumption'], errors='coerce'
    ).sum()

    energy = pd.to_numeric(
        furnace_rows['total_fired_duty'], errors='coerce'
    ).sum()

    return {
        'fresh_feed_change':       fresh_feed_change,
        'biasing_condition':       biasing_condition,
        'saturated_pressure':      saturated_pressure,
        'consider_for_conversion': consider_for_conversion,
        'specific_energy':         specific_energy,
        'energy':                  energy,
        'conv_delta_threshold':    conv_delta_threshold,
        'feed_delta_threshold':    feed_delta_threshold,
        'run_freq_threshold':      run_freq_threshold,
        'current_timestamp':       all_df['Timestamp'].iloc[0] if 'Timestamp' in all_df.columns else None,
    }


# ---------------------------------------------------------------------------
# CHECK 3: consecutive past-hour use
# ---------------------------------------------------------------------------

def was_past_hour_used_consecutively(history: list, window: int = PAST_HOUR_WINDOW) -> bool:
    if len(history) < window:
        return False
    return all(h.get('used_past_hour_data', 0) == 1 for h in history[:window])


# ---------------------------------------------------------------------------
# Deviation signal functions (CHECK 4)
# ---------------------------------------------------------------------------

def check_deviation_fresh_feed(current: Any, prev: Any) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return float(current) != float(prev)


def check_deviation_biasing_condition(current: Any, prev: Any) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return int(float(current)) != int(float(prev))


def check_deviation_saturated_pressure(current: Any, prev: Any, threshold: float = 0.05) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return abs(float(current) - float(prev)) > threshold


def check_deviation_consider_for_conversion(current: Any, prev: Any) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return float(current) != float(prev)


def check_deviation_specific_energy(current: Any, prev: Any, threshold: float = 0.1) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return abs(float(current) - float(prev)) > threshold


def check_deviation_energy(current: Any, prev: Any, threshold: float = 0.5) -> bool:
    if _is_missing(current) or _is_missing(prev):
        return True
    return abs(float(current) - float(prev)) > threshold


# ---------------------------------------------------------------------------
# MAIN MODULE 4 FUNCTION
# ---------------------------------------------------------------------------

def run_past_hour_logic(
    eligible_furnaces_df: pd.DataFrame,
    all_furnaces_df: pd.DataFrame,
    ropt_use_past_time_output: Optional[str] = None,   # expects 'active' to proceed
    past_hour_data: Optional[Dict[str, Any]] = None,
    history_records: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Returns:
        eligible_furnaces_df  — unchanged
        all_furnaces_df       — with 'deviation_exists' and 'deviation_signals' columns
        deviation_exists      — scalar int (0 or 1)
    """
    result_df = all_furnaces_df.copy()

    def _set_deviation(reason: str, value: int = 1):
        result_df['deviation_exists']  = value
        result_df['deviation_signals'] = str({'reason': reason})
        print(f'[Module 4] deviation_exists = {value} | reason: {reason}')
        return eligible_furnaces_df, result_df, value

    # ------------------------------------------------------------------
    # CHECK 1: minimum_cracking_furnace_available_check
    # ------------------------------------------------------------------
    min_check = compute_min_cracking_check(result_df)
    print(f'[Module 4] minimum_cracking_furnace_available_check = {min_check}')

    if min_check != 1:
        return _set_deviation(f'minimum_cracking_furnace_available_check = {min_check}')

    # ------------------------------------------------------------------
    # CHECK 2: ropt_use_past_time_output must exist AND == 'active'
    # ------------------------------------------------------------------
    print(f'[Module 4] ropt_use_past_time_output = {ropt_use_past_time_output!r}')

    if ropt_use_past_time_output is None or str(ropt_use_past_time_output).strip().lower() != 'active':
        return _set_deviation('ropt_use_past_time_output not defined or not active')

    # ------------------------------------------------------------------
    # CHECK 3: Fetch 24h history — block if last 6 consecutive used past-hour
    # ------------------------------------------------------------------
    current    = extract_current_values(result_df)
    current_ts = current['current_timestamp']
    past_hour_dt = pd.to_datetime(current_ts) - timedelta(hours=1) if current_ts is not None else None
    print(f'[Module 4] Current timestamp : {current_ts}')
    print(f'[Module 4] Past hour datetime: {past_hour_dt}')

    if history_records is not None:
        window = int(current['run_freq_threshold'])
        if was_past_hour_used_consecutively(history_records, window=window):
            return _set_deviation(f'{window} consecutive past-hour uses — forcing deviation')
    else:
        print('[Module 4] No history records provided — skipping consecutive check (DB needed in production)')

    # ------------------------------------------------------------------
    # CHECK 4: Individual deviation signals
    # ------------------------------------------------------------------
    if past_hour_data is None:
        return _set_deviation('No past hour data available — DB needed in production')

    prev = past_hour_data

    sig1 = check_deviation_fresh_feed(current['fresh_feed_change'], prev.get('fresh_feed_change'))
    sig2 = check_deviation_biasing_condition(current['biasing_condition'], prev.get('biasing_condition'))
    sig3 = check_deviation_saturated_pressure(
        current['saturated_pressure'], prev.get('saturated_pressure'),
        current['conv_delta_threshold']
    )
    sig4 = check_deviation_consider_for_conversion(
        current['consider_for_conversion'], prev.get('consider_for_conversion')
    )
    sig5 = check_deviation_specific_energy(
        current['specific_energy'], prev.get('specific_energy'),
        current['conv_delta_threshold']
    )
    sig6 = check_deviation_energy(
        current['energy'], prev.get('energy'),
        current['feed_delta_threshold']
    )

    # Signal 7: further_deviation_check — evaluated SEPARATELY to avoid self-reference
    signal_flags = {
        'deviation_fresh_feed':              sig1,
        'deviation_biasing_condition':       sig2,
        'deviation_saturated_pressure':      sig3,
        'deviation_consider_for_conversion': sig4,
        'deviation_specific_energy':         sig5,
        'deviation_energy':                  sig6,
    }
    sig7 = any(signal_flags.values())
    signal_flags['further_deviation_check'] = sig7

    deviation_exists = 1 if sig7 else 0

    result_df['deviation_exists']  = deviation_exists
    result_df['deviation_signals'] = str(signal_flags)

    print(f'\n[Module 4] Deviation Signal Results:')
    for k, v in signal_flags.items():
        print(f'  {k:<44s} = {v}')
    print(f'\n[Module 4] deviation_exists = {deviation_exists}')
    if deviation_exists == 1:
        triggered = [k for k, v in signal_flags.items() if v and k != 'further_deviation_check']
        print(f'[Module 4] Triggered by: {triggered}')
    else:
        print('[Module 4] No deviation → Optimizer can proceed')

    return eligible_furnaces_df, result_df, deviation_exists


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # FILE  = '/mnt/user-data/uploads/Ranking_req_info.xlsx'
    # SHEET = 'tags in parameter format'

    # import sys
    # sys.path.insert(0, '/mnt/user-data/outputs')
    # from module3_preprocessing_v5 import run_preprocessing

    # df = pd.read_excel(FILE, sheet_name=SHEET)
    # all_results_all, all_results_eligible = [], []
    # for ts, group in df.groupby('Timestamp'):
    #     all_df_ts, eligible_df_ts = run_preprocessing(group)
    #     all_results_all.append(all_df_ts)
    #     all_results_eligible.append(eligible_df_ts)

    all_df      = pd.read_csv(r"C:\Users\User\Documents\POC\FURNACE_PRODUCT\module3_v5_all_furnaces.csv")
    eligible_df = pd.read_csv(r"C:\Users\User\Documents\POC\FURNACE_PRODUCT\module3_v5_eligible_furnaces.csv")
    print('=' * 70)
    print('  MODULE 4: PAST HOUR LOGIC')
    print('=' * 70)
    print(f'Input 1 — eligible_furnaces_df : {eligible_df.shape[0]} rows x {eligible_df.shape[1]} cols')
    print(f'Input 2 — all_furnaces_df      : {all_df.shape[0]} rows x {all_df.shape[1]} cols')
    print()
    print("ropt_use_past_time_output : simulating as 'active' for testing")
    print('past_hour_data            : Mock values (replace with DB query in production)')
    print()

    mock_past_hour = {
        'fresh_feed_change':       0,
        'biasing_condition':       1,
        'saturated_pressure':      5.90,
        'consider_for_conversion': 0.0,
        'specific_energy':         4.90,
        'energy':                  230.0,
        'used_past_hour_data':     0,
    }
    mock_history = [{'used_past_hour_data': 0}] * 6

    out_eligible, out_all, dev_exists = run_past_hour_logic(
        eligible_furnaces_df      = eligible_df,
        all_furnaces_df           = all_df,
        ropt_use_past_time_output = 'active',
        past_hour_data            = mock_past_hour,
        history_records           = mock_history,
    )

    print('\n' + '=' * 70)
    print('  OUTPUT SUMMARY')
    print('=' * 70)
    print(f'\nOutput 1 — eligible_furnaces_df (UNCHANGED): {out_eligible.shape[0]} rows x {out_eligible.shape[1]} cols')
    print(f'Output 2 — all_furnaces_df (with deviation): {out_all.shape[0]} rows x {out_all.shape[1]} cols')
    print(f'Output 3 — deviation_exists (scalar)       : {dev_exists}')
    new_cols = [c for c in out_all.columns if c not in all_df.columns]
    print(f'New columns added to all_df: {new_cols}')
    print()
    print(out_all[['entity_name', 'deviation_exists']].to_string(index=False))

    # out_eligible.to_csv('/mnt/user-data/outputs/module4_eligible_furnaces.csv', index=False)
    # out_all.to_csv('/mnt/user-data/outputs/module4_all_furnaces.csv', index=False)
    print('\nSaved:')
    # print('  /mnt/user-data/outputs/module4_eligible_furnaces.csv')
    # print('  /mnt/user-data/outputs/module4_all_furnaces.csv')
    print(f'\nSet macro: furnace_step_adjust_feed_grid_limit = {FURNACE_STEP_ADJUST_FEED_GRID_LIMIT}')
