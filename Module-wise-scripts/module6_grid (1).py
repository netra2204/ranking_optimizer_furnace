"""
=============================================================================
MODULE 6: GRID — Feed & Conversion Optimisation Grid
=============================================================================
Source: Grid_Process_Documentation.docx  (authoritative reference)

Position in pipeline:
    Runs immediately after Module 5 (Pre_Grid).
    Guarded implicitly — only runs if Module 5 returned a non-empty grid_df.

Inputs:
    grid_df         — DataFrame from Module 5 run_pre_grid()
    macros          — dict from Module 5 (all Row_N_* + park-level macros)
    inferred_tags   — list of dicts: [{'tag': str, 'formula': str}, ...]
                      In production: recalled from DB.
                      In testing: mock list provided in __main__.

Key macro inputs (from Module 5):
    Row_N_lower_limit_feed, Row_N_upper_limit_feed, Row_N_step_size_feed   (N=1..9)
    Row_N_lower_limit_conversion, Row_N_upper_limit_conversion             (N=1..9)
    Row_N_step_size_conversion, Row_N_part_override                        (N=1..9)
    mixed_feed_margin, Extra_Recycle_Ethane
    upper_limit_change_in_recycle_ethane, lower_limit_change_in_recycle_ethane
    shc_ratio, fresh_feed_change, fresh_feed_quantity, biasing_condition
    Number_of_rows, max_conversion_single_furnace_limit
    conversion_upper_limit_expansion_max_limit
    conversion_lower_limit_expansion_max_limit
    benefit_percent_threshold
    ranking_improve_energy_consumption
    ranking_improve_specific_energy_consumption
    ROPT_all_furnace_for_conversion_biasing

Outputs:
    Grid_Row_N_conversion_delta_best  (N=1..9)
    Row_N_feed_delta                  (N=1..9)
    Max_Benefit, sum_del_ethylene
    sum_Change_in_Recycle_Ethane_Feed
    ranking_cause_indicator, Conversion_Grid_Success
    result_df — grid_df with best New_Feed_flow + New_Overall_conversion

Dependencies: pip install pandas numpy openpyxl
=============================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(v, default=0):
    if v is None:
        return default
    try:
        if math.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _floor025(x):
    return math.floor(float(x) / 0.25) * 0.25


def _floor05(x):
    return math.floor(float(x) / 0.5) * 0.5


def _frange(start, stop, step):
    """Inclusive float range. Returns list. step=0 returns [start]."""
    if step == 0:
        return [float(start)]
    vals, v = [], float(start)
    while v <= float(stop) + 1e-9:
        vals.append(round(v, 6))
        v += float(step)
    return vals if vals else [float(start)]


# ---------------------------------------------------------------------------
# STEP 3 — Initialise key macros
# ---------------------------------------------------------------------------

def _step3_init_macros(macros):
    fresh_feed_change = int(_val(macros.get('fresh_feed_change'), 0))
    fresh_feed_qty    = _val(macros.get('fresh_feed_quantity'), 0)
    shc               = _val(macros.get('shc_ratio'), 0)
    macros['Min_target_sum_feed_bias'] = fresh_feed_qty * (1 + shc)
    macros['Max_Benefit']              = -1000.0 if fresh_feed_change == -1 else 0.0
    macros['Max_Benefit_SPC']          = 1000
    macros['ranking_cause_indicator']  = 1
    return macros


# ---------------------------------------------------------------------------
# STEP 8 — Step-zero / fixed-position feed delta correction
# ---------------------------------------------------------------------------

def _step8_correct_feed_deltas(feed_deltas, macros):
    corrected = {}
    for n in range(1, 10):
        step = _val(macros.get(f'Row_{n}_step_size_feed'), 0)
        ll   = _val(macros.get(f'Row_{n}_lower_limit_feed'), 0)
        d    = _val(feed_deltas.get(n), 0)
        if step == 0:
            corrected[n] = 0.0
        else:
            corrected[n] = float(ll) if d == 0 else float(d)
    return corrected


# ---------------------------------------------------------------------------
# STEP 15 — Apply feed deltas → New_Feed_flow
# ---------------------------------------------------------------------------

def _step15_apply_feed_deltas(df, feed_deltas):
    df = df.copy().reset_index(drop=True)
    df['del_Feed_flow'] = 0.0
    df['New_Feed_flow'] = df['Feed_flow'].astype(float)
    for i, row in df.iterrows():
        n = int(row['id']) + 1
        delta = _val(feed_deltas.get(n), 0)
        df.loc[i, 'del_Feed_flow'] = delta
        df.loc[i, 'New_Feed_flow'] = float(row['Feed_flow']) + delta
    return df


# ---------------------------------------------------------------------------
# STEP 17B — Update recycle ethane limits when feed is moving
# ---------------------------------------------------------------------------

def _step17b_update_conversion_limits(df, macros):
    fresh_feed_change = int(_val(macros.get('fresh_feed_change'), 0))
    shc               = _val(macros.get('shc_ratio'), 0)
    extra_re          = _val(macros.get('Extra_Recycle_Ethane'), 0)
    re_upper_buf      = _val(macros.get('change_recycle_ethane_upper_limit'), 0)
    re_lower_buf      = _val(macros.get('change_recycle_ethane_lower_limit'), 0)
    sum_del_feed      = _val(macros.get('sum_del_Feed_flow'), 0)
    denom             = (1 + shc) if shc != -1 else 1
    df = df.copy()
    if fresh_feed_change != 0:
        if fresh_feed_change == -1:
            extra_re = sum_del_feed / denom
            macros['Extra_Recycle_Ethane'] = extra_re
        macros['upper_limit_change_in_recycle_ethane'] = extra_re + re_upper_buf
        macros['lower_limit_change_in_recycle_ethane'] = extra_re + re_lower_buf
    return df, macros


# ---------------------------------------------------------------------------
# STEP 17C — Conversion feasibility check (biasing_condition == 3)
# ---------------------------------------------------------------------------

def _step17c_conversion_feasibility(df, macros):
    shc          = _val(macros.get('shc_ratio'), 0)
    extra_re     = _val(macros.get('Extra_Recycle_Ethane'), 0)
    conv_exp_max = _val(macros.get('conversion_upper_limit_expansion_max_limit'), 65.0)
    max_conv_sf  = _val(macros.get('max_conversion_single_furnace_limit'), 70.0)
    denom        = (1 + shc) if shc != -1 else 1

    def _projected_nere(row):
        max_oc  = min(conv_exp_max, max_conv_sf,
                      row['overall_conversion'] + row['conversion_upper_limit_in_grid'])
        new_ref = (_val(row['New_Feed_flow']) / denom) * (100 - max_oc) / 100
        return new_ref - _val(row['Current_Recycle_Ethane_Feed'])

    nere_col   = df.apply(_projected_nere, axis=1)
    upper_elig = df['conversion_upper_limit_in_grid'] > 0
    sum_upper  = nere_col[upper_elig].sum()
    sum_all    = nere_col.sum()

    macros['sum_New_Extra_Recycle_Ethane_max_check_upper'] = sum_upper
    macros['sum_New_Extra_Recycle_Ethane_max_check']       = sum_all
    macros['conversion_max_check_enough']                  = 1 if sum_all   >= extra_re else 0
    macros['conversion_max_check_enough_for_upper_lower']  = 1 if sum_upper >= extra_re else 0
    return macros


# ---------------------------------------------------------------------------
# STEP 17D — Conversion expansion while-loop engine
# ---------------------------------------------------------------------------

def _step17d_expansion_engine(df, macros):
    shc               = _val(macros.get('shc_ratio'), 0)
    extra_re          = _val(macros.get('Extra_Recycle_Ethane'), 0)
    fresh_feed_change = int(_val(macros.get('fresh_feed_change'), 0))
    conv_exp_max      = _val(macros.get('conversion_upper_limit_expansion_max_limit'), 65.0)
    denom             = (1 + shc) if shc != -1 else 1
    step_conv         = 0.5
    enough_upper      = _val(macros.get('conversion_max_check_enough_for_upper_lower'), 0)

    df = df.copy()
    df['Considered_for_conversion_expansion'] = df.apply(
        lambda r: 1 if enough_upper == 1 and r['conversion_upper_limit_in_grid'] > 0 else 0, axis=1
    )
    df['New_Overall_conversion_limit'] = df.apply(
        lambda r: min(conv_exp_max, r['overall_conversion'] + r['conversion_upper_limit_in_grid']), axis=1
    )
    df['New_Overall_conversion'] = df['overall_conversion'].astype(float).copy()

    def _nere(row):
        ref = (_val(row['New_Feed_flow']) / denom) * (100 - row['New_Overall_conversion']) / 100
        return ref - _val(row['Current_Recycle_Ethane_Feed'])

    df['New_Extra_Recycle_Ethane'] = df.apply(_nere, axis=1)

    condition_satisfy = 0
    for _ in range(500):
        if df['New_Extra_Recycle_Ethane'].sum() >= extra_re:
            condition_satisfy = 1
            break
        if fresh_feed_change >= 0:
            eligible = df[
                (df['Considered_for_conversion_expansion'] == 0) &
                (df['New_Overall_conversion'] < df['New_Overall_conversion_limit'])
            ]
            if eligible.empty:
                break
            idx = eligible['New_Overall_conversion'].idxmin()
            df.loc[idx, 'New_Overall_conversion'] = min(
                df.loc[idx, 'New_Overall_conversion'] + step_conv,
                df.loc[idx, 'New_Overall_conversion_limit']
            )
        else:
            lower_lims = df['overall_conversion'] + df['conversion_lower_limit_in_grid']
            eligible   = df[
                (df['Considered_for_conversion_expansion'] == 0) &
                (df['New_Overall_conversion'] > lower_lims)
            ]
            if eligible.empty:
                break
            idx = eligible['New_Overall_conversion'].idxmax()
            df.loc[idx, 'New_Overall_conversion'] = max(
                df.loc[idx, 'New_Overall_conversion'] - step_conv,
                lower_lims.loc[idx]
            )
        df['New_Extra_Recycle_Ethane'] = df.apply(_nere, axis=1)

    macros['condition_satisfy']                 = condition_satisfy
    macros['sum_New_Extra_Recycle_Ethane_curr'] = df['New_Extra_Recycle_Ethane'].sum()

    df['Considered_for_conversion_expansion'] = (
        (df['New_Overall_conversion'] - df['overall_conversion']).abs() > 1e-6
    ).astype(int)

    df['conversion_lower_limit_in_grid'] = df.apply(
        lambda r: _floor05(r['New_Overall_conversion'] - r['overall_conversion'])
                  if r['Considered_for_conversion_expansion'] == 1
                  else r['conversion_lower_limit_in_grid'], axis=1
    )
    df['conversion_upper_limit_in_grid'] = df.apply(
        lambda r: _floor05(r['New_Overall_conversion'] - r['overall_conversion'])
                  if r['Considered_for_conversion_expansion'] == 1
                  else r['conversion_upper_limit_in_grid'], axis=1
    )

    count_no_fixed = int((df['Considered_for_conversion_expansion'] == 0).sum())
    macros['count_no_fixed_grid_fur'] = count_no_fixed

    if count_no_fixed == 0:
        df_s = df.sort_values('percent_above_threshold', ascending=False).reset_index(drop=True)
        df_s['step_size_conversion'] = 0.0
        for i in range(min(2, len(df_s))):
            df_s.loc[i, 'step_size_conversion'] = df_s.loc[i, 'conversion_upper_limit_in_grid'] * 2
        df = df_s
    else:
        step_c = (5.0 if count_no_fixed < 6 else
                  3.0 if count_no_fixed == 6 else
                  1.0 if count_no_fixed == 9 else 2.0)
        df['step_size_conversion'] = step_c

    return df, macros


# ---------------------------------------------------------------------------
# STEP 17E — Top-N furnace selection (biasing_condition != 3, fresh_feed == 0)
# ---------------------------------------------------------------------------

def _step17e_topn_conversion(df, macros):
    shc      = _val(macros.get('shc_ratio'), 0)
    extra_re = _val(macros.get('Extra_Recycle_Ethane'), 0)
    ropt_all = str(macros.get('ROPT_all_furnace_for_conversion_biasing', 'inactive')).lower()
    denom    = (1 + shc) if shc != -1 else 1
    n_rows   = len(df)

    df = df.copy()
    df['_sort1'] = ((df['Furnace_condition'] == 'Good') & (df['percent_above_threshold'] < 0)).astype(int)
    df = df.sort_values(['_sort1', 'overall_ranking'], ascending=[False, True]).reset_index(drop=True)
    df.drop(columns=['_sort1'], inplace=True)
    df['id'] = range(len(df))

    count_of_top_rows       = 2
    conversion_limits_given = 0

    for _ in range(n_rows + 1):
        top_ids = list(range(count_of_top_rows))
        new_oc  = df['overall_conversion'].copy().astype(float)
        for i in top_ids:
            if i < len(df):
                new_oc.iloc[i] = (df.iloc[i]['overall_conversion']
                                  + df.iloc[i]['conversion_lower_limit_in_grid'])

        def _nere(row):
            ref = (_val(row['New_Feed_flow']) / denom) * (100 - new_oc.iloc[row.name]) / 100
            return ref - _val(row['Current_Recycle_Ethane_Feed'])

        if df.apply(_nere, axis=1).sum() > extra_re:
            conversion_limits_given = 1
            break
        count_of_top_rows += 1

    macros['count_of_top_rows']       = count_of_top_rows
    macros['Conversion_Limits_given'] = conversion_limits_given

    effective_top = max(0, count_of_top_rows - 1) if ropt_all == 'inactive' else count_of_top_rows
    for i in range(len(df)):
        if i >= effective_top:
            df.loc[i, 'conversion_upper_limit_in_grid'] = df.loc[i, 'conversion_lower_limit_in_grid']

    step_c = (5.0 if count_of_top_rows < 5 else
              3.0 if count_of_top_rows == 5 else
              2.0 if count_of_top_rows == 6 else 1.0)

    df['step_size_conversion'] = df.apply(
        lambda r: 0.0 if r['conversion_lower_limit_in_grid'] == r['conversion_upper_limit_in_grid']
                  else step_c, axis=1
    )
    return df, macros


# ---------------------------------------------------------------------------
# INFERRED TAGS — evaluate formula string per row
# ---------------------------------------------------------------------------

def _apply_inferred_tags(df, inferred_tags):
    df = df.copy()
    safe = {"__builtins__": {}}
    for i, row in df.iterrows():
        ns = {col: _val(row[col]) for col in df.columns}
        ns.update({'math': math, 'abs': abs, 'max': max, 'min': min,
                   'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp})
        for tag in inferred_tags:
            try:
                result = eval(tag['formula'], safe, ns)
                df.loc[i, tag['tag']] = result
                ns[tag['tag']] = result
            except Exception:
                df.loc[i, tag['tag']] = 0.0
    return df


# ---------------------------------------------------------------------------
# APPLY CONVERSION DELTAS per furnace row
# ---------------------------------------------------------------------------

def _apply_conversion_deltas(df, conv_deltas):
    df = df.copy()
    for i, row in df.iterrows():
        n = int(row['id']) + 1
        delta = _val(conv_deltas.get(n), 0)
        df.loc[i, 'conversion_delta']       = delta
        df.loc[i, 'New_Overall_conversion'] = float(row['overall_conversion']) + delta
    return df


# ---------------------------------------------------------------------------
# ROUND conversion delta — floor025 + part_override suppression
# ---------------------------------------------------------------------------

def _round_conv_delta(delta, part_override):
    floor_num = _floor025(delta)
    if part_override != 0:
        if abs(floor_num) <= 0.5:
            return 0.0
    else:
        if abs(floor_num) < 0.5:
            return 0.0
    return floor_num


# ---------------------------------------------------------------------------
# SCORE a candidate conversion combination
# Returns: (skip_flag, sum_del_eth, sum_re, flag_energy, flag_sec)
# skip_flag=0 → valid new best;  1 → skip
# ---------------------------------------------------------------------------

def _score_conversion(df, macros):
    re_upper    = _val(macros.get('upper_limit_change_in_recycle_ethane'),  1e9)
    re_lower    = _val(macros.get('lower_limit_change_in_recycle_ethane'), -1e9)
    max_benefit = _val(macros.get('Max_Benefit'), 0)
    rank_energy = int(_val(macros.get('ranking_improve_energy_consumption'), 0))
    rank_sec    = int(_val(macros.get('ranking_improve_specific_energy_consumption'), 0))

    sum_re  = df['Change_in_Recycle_Ethane_Feed'].sum() if 'Change_in_Recycle_Ethane_Feed' in df.columns else 0.0
    sum_eth = df['del_ethylene'].sum()                  if 'del_ethylene'                  in df.columns else 0.0

    if not (re_lower <= sum_re <= re_upper):
        return 1, sum_eth, sum_re, 0, 0
    if sum_eth <= max_benefit:
        return 1, sum_eth, sum_re, 0, 0

    flag_energy = 1
    if rank_energy == 1 and 'Heat' in df.columns and 'Heat_new' in df.columns:
        if df['Heat_new'].sum() > df['Heat'].sum():
            flag_energy = 0

    flag_sec = 1
    if rank_sec == 1 and all(c in df.columns for c in
                              ['Heat', 'Temp_Ethylene_Production', 'Heat_new', 'New_Ethylene_Production']):
        curr_sec = df['Heat'].sum() / max(df['Temp_Ethylene_Production'].sum(), 1e-9)
        new_sec  = df['Heat_new'].sum() / max(df['New_Ethylene_Production'].sum(), 1e-9)
        if new_sec > curr_sec:
            flag_sec = 0

    if flag_energy == 0 or flag_sec == 0:
        return 1, sum_eth, sum_re, flag_energy, flag_sec
    return 0, sum_eth, sum_re, flag_energy, flag_sec


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_grid(grid_df, macros, inferred_tags=None):
    """
    Parameters
    ----------
    grid_df       : DataFrame from Module 5. Empty → guard exits immediately.
    macros        : dict from Module 5 (all Row_N_* + park-level macros).
    inferred_tags : list of {'tag': str, 'formula': str}.

    Returns
    -------
    result_df : grid_df with best New_Feed_flow + New_Overall_conversion
    macros    : updated with all output macros
    """

    # ── GUARD: only run if Module 5 produced valid output ────────────────
    if grid_df is None or grid_df.empty:
        print('[Module 6] grid_df is empty — Module 5 produced no results. Skipping.')
        macros['ranking_cause_indicator'] = -1
        macros['Max_Benefit']             = 0
        macros['sum_del_ethylene']        = 0
        return (grid_df if grid_df is not None else pd.DataFrame()), macros

    if inferred_tags is None:
        inferred_tags = []

    macros = dict(macros)

    # ── STEP 1: Assign sequential 0-based IDs ────────────────────────────
    df = grid_df.copy().sort_values('overall_ranking', ascending=True).reset_index(drop=True)
    df['id'] = range(len(df))
    print(f'[Step 1] {len(df)} furnace rows assigned IDs')

    # ── STEP 3: Initialise key macros ────────────────────────────────────
    macros             = _step3_init_macros(macros)
    fresh_feed_change  = int(_val(macros.get('fresh_feed_change'), 0))
    mixed_feed_margin  = _val(macros.get('mixed_feed_margin'), 0)
    min_target         = _val(macros.get('Min_target_sum_feed_bias'), 0)
    biasing            = int(_val(macros.get('biasing_condition'), 1))
    print(f'[Step 3] Max_Benefit={macros["Max_Benefit"]} | Min_target={min_target:.3f}')

    # ── Build per-row feed ranges (9 rows) ───────────────────────────────
    fr = []                         # fr[0..8] = value list for Row_1..Row_9
    for n in range(1, 10):
        ll   = _val(macros.get(f'Row_{n}_lower_limit_feed'), 0)
        ul   = _val(macros.get(f'Row_{n}_upper_limit_feed'), 0)
        step = _val(macros.get(f'Row_{n}_step_size_feed'), 0)
        fr.append(_frange(ll, ul, step))

    # Best-result tracker
    best = {
        'feed_deltas': {n: 0.0 for n in range(1, 10)},
        'conv_deltas': {n: 0.0 for n in range(1, 10)},
        'sum_del_eth': macros['Max_Benefit'],
        'sum_re':      0.0,
        'df':          df.copy(),
    }
    evaluated_feed_chars = set()

    try:
        # ════════════════════════════════════════════════════════════════
        # STEP 6: OUTER FEED GRID — 9 nested for loops (one per furnace)
        # ════════════════════════════════════════════════════════════════
        for f1 in fr[0]:
         for f2 in fr[1]:
          for f3 in fr[2]:
           for f4 in fr[3]:
            for f5 in fr[4]:
             for f6 in fr[5]:
              for f7 in fr[6]:
               for f8 in fr[7]:
                for f9 in fr[8]:

                    feed_combo = {1:f1, 2:f2, 3:f3, 4:f4, 5:f5,
                                  6:f6, 7:f7, 8:f8, 9:f9}

                    # ── STEP 7: set Row_N_feed_delta ─────────────────
                    for n in range(1, 10):
                        macros[f'Row_{n}_feed_delta'] = feed_combo[n]

                    # ── STEP 8: step-zero correction ─────────────────
                    corrected_feed = _step8_correct_feed_deltas(feed_combo, macros)
                    for n, v in corrected_feed.items():
                        macros[f'Row_{n}_feed_delta'] = v

                    # ── STEP 9: sum_del_Feed_flow ────────────────────
                    sum_del_feed = sum(_val(corrected_feed.get(n), 0) for n in range(1, 10))
                    macros['sum_del_Feed_flow']       = sum_del_feed
                    macros['Conversion_Grid_Success'] = 0

                    # ── STEP 11: Feed_Grid_Character fingerprint ─────
                    feed_char = '#'.join(
                        str(_val(corrected_feed.get(n), 0)) for n in range(1, 10)
                    ) + '#'

                    # ── STEPS 12-13: Duplicate / minimum feed check ──
                    if fresh_feed_change == -1:
                        compare_flag = 0 if sum_del_feed >= min_target else 1
                    else:
                        if sum_del_feed == mixed_feed_margin:
                            compare_flag = 0
                        elif feed_char in evaluated_feed_chars:
                            compare_flag = 1
                        else:
                            compare_flag = 0

                    # ── STEP 14: Skip duplicate / invalid ────────────
                    if compare_flag != 0:
                        continue

                    evaluated_feed_chars.add(feed_char)

                    # ── STEP 15: Apply feed deltas → New_Feed_flow ───
                    df_feed = _step15_apply_feed_deltas(df, corrected_feed)

                    # ── STEP 17B: Update recycle ethane limits ────────
                    df_conv, macros = _step17b_update_conversion_limits(df_feed, macros)

                    # ── STEP 17C/D/E: Conversion limit engine ─────────
                    if biasing == 3:
                        macros = _step17c_conversion_feasibility(df_conv, macros)
                        if _val(macros.get('conversion_max_check_enough'), 0) == 1:
                            df_conv, macros = _step17d_expansion_engine(df_conv, macros)
                    else:
                        if fresh_feed_change == 0:
                            df_conv, macros = _step17e_topn_conversion(df_conv, macros)

                    # ── STEP 17F: Check upper conversion limits ───────
                    count_upper = int((df_conv['conversion_upper_limit_in_grid'] > 0).sum())
                    macros['count_upper_limit_available_fur'] = count_upper

                    if count_upper == 0:
                        macros['Conversion_Limits_given']  = 0
                        macros['ranking_cause_indicator']  = 6
                        continue

                    macros['Conversion_Limits_given'] = 1

                    # ── STEP 19A: Zero-init Grid_Row_N conv macros ────
                    for n in range(1, 10):
                        macros[f'Grid_Row_{n}_lower_conversion_limit'] = 0
                        macros[f'Grid_Row_{n}_upper_conversion_limit'] = 0
                        macros[f'Grid_Row_{n}_step_size_conversion']   = 0
                        macros[f'Grid_Row_{n}_part_override']          = 0

                    # ── STEP 19B: Extract Grid_Row_N macros ───────────
                    df_sc = df_conv.sort_values('overall_ranking', ascending=True).reset_index(drop=True)
                    for i, row in df_sc.iterrows():
                        n = i + 1
                        macros[f'Grid_Row_{n}_Feed_flow']              = _val(row.get('New_Feed_flow'), 0)
                        macros[f'Grid_Row_{n}_Furnace']                = row.get('entity_name', 0)
                        macros[f'Grid_Row_{n}_Conversion']             = _val(row.get('overall_conversion'), 0)
                        macros[f'Grid_Row_{n}_lower_conversion_limit'] = _val(row.get('conversion_lower_limit_in_grid'), 0)
                        macros[f'Grid_Row_{n}_upper_conversion_limit'] = _val(row.get('conversion_upper_limit_in_grid'), 0)
                        macros[f'Grid_Row_{n}_step_size_conversion']   = _val(row.get('step_size_conversion'), 0)
                        macros[f'Grid_Row_{n}_part_override']          = _val(row.get('flag_conversion_part_override'), 0)

                    # ── Build per-row conversion ranges (9 rows) ──────
                    cr = []     # cr[0..8] = value list for Grid_Row_1..Grid_Row_9
                    for n in range(1, 10):
                        ll   = _val(macros.get(f'Grid_Row_{n}_lower_conversion_limit'), 0)
                        ul   = _val(macros.get(f'Grid_Row_{n}_upper_conversion_limit'), 0)
                        step = _val(macros.get(f'Grid_Row_{n}_step_size_conversion'), 0)
                        cr.append(_frange(ll, ul, step))

                    conv_log             = []
                    evaluated_conv_chars = set()

                    try:
                        # ════════════════════════════════════════════
                        # STEP 20: INNER CONVERSION GRID — 9 nested loops
                        # ════════════════════════════════════════════
                        for c1 in cr[0]:
                         for c2 in cr[1]:
                          for c3 in cr[2]:
                           for c4 in cr[3]:
                            for c5 in cr[4]:
                             for c6 in cr[5]:
                              for c7 in cr[6]:
                               for c8 in cr[7]:
                                for c9 in cr[8]:

                                    raw_conv = {1:c1, 2:c2, 3:c3, 4:c4, 5:c5,
                                                6:c6, 7:c7, 8:c8, 9:c9}

                                    # Round + suppress each delta
                                    rounded_conv = {}
                                    for n in range(1, 10):
                                        part_ov = _val(macros.get(f'Grid_Row_{n}_part_override'), 0)
                                        rounded_conv[n] = _round_conv_delta(raw_conv[n], part_ov)

                                    # Duplicate conv combination check
                                    conv_char = '#'.join(
                                        str(rounded_conv[n]) for n in range(1, 10)
                                    ) + '#'
                                    if conv_char in evaluated_conv_chars:
                                        continue
                                    evaluated_conv_chars.add(conv_char)

                                    # STEP 20B: apply conversion deltas
                                    df_iter = _apply_conversion_deltas(df_conv, rounded_conv)

                                    # STEP 20C: apply inferred tags
                                    df_iter = _apply_inferred_tags(df_iter, inferred_tags)

                                    # STEP 20D: score
                                    skip, sum_eth, sum_re, fe, fs = _score_conversion(df_iter, macros)
                                    if skip != 0:
                                        continue

                                    # STEP 20E: store valid best
                                    macros['Max_Benefit']             = sum_eth
                                    macros['Conversion_Grid_Success'] = 1
                                    conv_log.append({
                                        'conv_deltas': dict(rounded_conv),
                                        'sum_del_ethylene':                  sum_eth,
                                        'sum_Change_in_Recycle_Ethane_Feed': sum_re,
                                        'df': df_iter.copy(),
                                    })

                    except Exception as e_inner:
                        print(f'[Module 6] Inner conversion loop exception: {e_inner}')

                    # ── STEP 21: Pick best from conv_log ─────────────
                    if conv_log:
                        conv_log.sort(key=lambda x: x['sum_del_ethylene'], reverse=True)
                        best_entry       = conv_log[0]
                        best_conv_deltas = best_entry['conv_deltas']
                        df_best          = best_entry['df']

                        macros['sum_del_ethylene']                  = best_entry['sum_del_ethylene']
                        macros['sum_Change_in_Recycle_Ethane_Feed'] = best_entry['sum_Change_in_Recycle_Ethane_Feed']
                        macros['Conversion_Grid_Success']           = 1

                        # Final Max_Benefit gate (Generate Macro 44)
                        skip_f, eth_f, re_f, fe_f, fs_f = _score_conversion(df_best, macros)
                        if skip_f == 0:
                            macros['Max_Benefit'] = eth_f
                    else:
                        # STEP 22: no valid conversion → sentinel
                        macros['sum_del_ethylene'] = -10000.0
                        best_conv_deltas           = {n: 0.0 for n in range(1, 10)}
                        df_best                    = df_conv.copy()

                    # ── STEP 23: Store if new overall best ───────────
                    sum_eth_curr = _val(macros.get('sum_del_ethylene'), -1e9)
                    max_ben      = _val(macros.get('Max_Benefit'), 0)

                    if abs(sum_eth_curr - max_ben) < 1e-6 and sum_eth_curr > best['sum_del_eth']:
                        best['feed_deltas'] = dict(corrected_feed)
                        best['conv_deltas'] = dict(best_conv_deltas)
                        best['sum_del_eth'] = sum_eth_curr
                        best['sum_re']      = _val(macros.get('sum_Change_in_Recycle_Ethane_Feed'), 0)
                        best['df']          = df_best.copy()

    except Exception as e_outer:
        print(f'[Module 6] Outer grid exception: {e_outer}')
        macros['ranking_cause_indicator'] = -1

    # ── FINALISE OUTPUT MACROS ────────────────────────────────────────────
    for n in range(1, 10):
        macros[f'Row_{n}_feed_delta'] = best['feed_deltas'].get(n, 0.0)

    for n in range(1, 10):
        try:
            macros[f'Grid_Row_{n}_conversion_delta_best'] = best['conv_deltas'].get(n, 0.0)
        except Exception:
            macros[f'Grid_Row_{n}_conversion_delta_best'] = 0.0

    macros['sum_del_ethylene']                  = best['sum_del_eth']
    macros['sum_Change_in_Recycle_Ethane_Feed'] = best['sum_re']

    result_df = best['df'].copy()

    print('\n[Module 6] ── OUTPUT SUMMARY ──')
    print(f'  Max_Benefit                       = {macros["Max_Benefit"]}')
    print(f'  sum_del_ethylene                  = {macros["sum_del_ethylene"]}')
    print(f'  sum_Change_in_Recycle_Ethane_Feed = {macros["sum_Change_in_Recycle_Ethane_Feed"]}')
    print(f'  ranking_cause_indicator           = {macros["ranking_cause_indicator"]}')
    print(f'  Conversion_Grid_Success           = {macros.get("Conversion_Grid_Success", 0)}')
    print('\n  Best feed deltas:')
    for n in range(1, 10):
        v = macros.get(f'Row_{n}_feed_delta', 0)
        if v != 0:
            print(f'    Row_{n}: {v}')
    print('\n  Best conversion deltas:')
    for n in range(1, 10):
        v = macros.get(f'Grid_Row_{n}_conversion_delta_best', 0)
        if v != 0:
            print(f'    Row_{n}: {v}')

    return result_df, macros


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')

    FILE  = '/mnt/user-data/uploads/Ranking_req_info.xlsx'
    SHEET = 'tags in parameter format'

    from module3_preprocessing_v5 import run_preprocessing
    from module5_pre_grid import run_pre_grid

    df_raw = pd.read_excel(FILE, sheet_name=SHEET)
    all_results_all, all_results_eligible = [], []
    for ts, group in df_raw.groupby('Timestamp'):
        all_df_ts, eligible_df_ts = run_preprocessing(group)
        all_results_all.append(all_df_ts)
        all_results_eligible.append(eligible_df_ts)

    eligible_df = pd.concat(all_results_eligible, ignore_index=True)

    mock_macros_m5 = {
        'pass_step_change': 0.25, 'pass_feed_min_limit': 6.5,
        'threshold_pass_mixed_feed_limit': 8.5, 'biasing_condition': 1,
        'fresh_feed_change': 0, 'total_fur_available_for_bias': len(eligible_df),
        'conversion_bias_threshold_lower_limit': -2.5,
        'conversion_bias_threshold_upper_limit': 1.5,
        'conversion_lower_limit_expansion_max_limit': 47.0,
        'conversion_upper_limit_expansion_max_limit': 65.0,
        'change_recycle_ethane_upper_limit': 0.5,
        'change_recycle_ethane_lower_limit': -0.5,
        'fresh_feed_quantity': 0.0,
        'furnace_step_adjust_feed_grid_limit': 6,
        'deviation_exists': 1,
        'final_run_optimizer_check_init': 1,
    }

    grid_df, m5_macros = run_pre_grid(eligible_df, mock_macros_m5)

    if grid_df is None or grid_df.empty:
        print('[Main] Module 5 returned no data — Module 6 will not run.')
    else:
        m5_macros.update({
            'max_conversion_single_furnace_limit':        70.0,
            'benefit_percent_threshold':                   0.02,
            'ranking_improve_energy_consumption':          0,
            'ranking_improve_specific_energy_consumption': 0,
            'ROPT_all_furnace_for_conversion_biasing':    'inactive',
        })

        mock_inferred_tags = [
            {'tag': 'New_Recycle_Ethane_Feed',
             'formula': '(New_Feed_flow / (1 + shc_ratio)) * (100 - New_Overall_conversion) / 100'},
            {'tag': 'Change_in_Recycle_Ethane_Feed',
             'formula': 'New_Recycle_Ethane_Feed - Current_Recycle_Ethane_Feed'},
            {'tag': 'del_ethylene',
             'formula': '(New_Feed_flow - Feed_flow) * overall_conversion / 100 * 0.5'},
        ]

        result_df, m6_macros = run_grid(grid_df, m5_macros, inferred_tags=mock_inferred_tags)

        result_df.to_csv('/mnt/user-data/outputs/module6_result.csv', index=False)
        print('\nSaved: /mnt/user-data/outputs/module6_result.csv')
