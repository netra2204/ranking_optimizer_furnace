"""
=============================================================================
MODULE 5: PRE-GRID (Feed Limit Computation & Grid Preparation)
=============================================================================
Source: Pre_Grid_Process_Documentation_1.docx  (authoritative reference)

Input:
    eligible_furnaces_df   — from Module 3 / Module 4 pass-through
                             One row per furnace (F1–F9 subset that are eligible)

Input macros (must be supplied by caller — come from DB / upstream):
    pass_step_change                        float   step size per pass (t/h)
    pass_feed_min_limit                     float   min allowable mixed feed per pass
    threshold_pass_mixed_feed_limit         float   max allowable mixed feed per pass
    biasing_condition                       int     1=increase, 2=ext-reduce, 3=conv-only
    fresh_feed_change                       int     0=no change, -1=cut, other=adding
    total_fur_available_for_bias            int     eligible furnaces count (loop stop)
    conversion_bias_threshold_lower_limit   float   lower bound for conv step (negative)
    conversion_bias_threshold_upper_limit   float   upper bound for conv step (positive)
    conversion_lower_limit_expansion_max_limit  float   max lower expansion for conv
    conversion_upper_limit_expansion_max_limit  float   max upper expansion for conv
    change_recycle_ethane_upper_limit       float   upper buffer for recycle ethane limit
    change_recycle_ethane_lower_limit       float   lower buffer for recycle ethane limit
    fresh_feed_quantity                     float   current fresh feed rate (t/h)
    count_of_good_fur                       int     count of Good furnaces
    count_of_no_good_fur                    int     count of Not-Good furnaces
    furnace_step_adjust_feed_grid_limit     int     furnace index limit for biasing

Outputs:
    1. grid_df      — per-furnace DataFrame with all computed feed/conversion limits
    2. macros       — dict of all output macros (Row_N_*, park-level recycle macros)

Steps mapped to doc:
    Step 1   → _step1_generate_initial_attributes
    Step 2   → Number_of_rows macro
    Step 3   → shc_ratio macro
    Step 4   → split Good vs Not-Good
    Step 5   → _step5_good_upper_limit
    Step 6   → drop intermediate cols from Good
    Step 7,8 → sum_upper_limit_feed (provisional)
    Step 9   → _step9_no_good_overrides
    Step 10,11 → sum_feed_reduction_potential
    Step 12  → biasing_condition != 2 gate
    Step 13  → merge Good + Not-Good
    Step 14  → fresh_feed_change == -1 gate
    Step 15  → post-branch cleanup
    Step 16  → _subprocess94_feed_distribution (While-loop + 3-step correction)
    Step 17  → sort by overall_ranking
    Step 18  → drop subprocess temp cols
    Step 19  → _step19_reconcile_feed_limits
    Step 20  → drop lower_limit_feed_org, Furnace_condition2
    Step 21,22 → sum_upper_limit_feed (definitive)
    Step 23  → _step23_recycle_ethane_macros
    Step 24  → stamp Extra_Recycle_Ethane column
    Step 25  → _step25_lock_lower_to_upper
    Step 26  → _step26_conversion_limits (Bias 3 vs standard, Subprocess 14)
    Step 27  → zero-initialise Row_N_* macros
    Step 28  → sort by overall_ranking
    Step 29  → extract Row_N_* macros

Dependencies: pip install pandas numpy openpyxl
=============================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


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


def _floor_half(x):
    """Round down to nearest 0.5 — used for all conversion limit calculations."""
    return math.floor(float(x) / 0.5) * 0.5


# ---------------------------------------------------------------------------
# STEP 1 — Generate Initial Attributes
# Columns created: lower_limit_feed, upper_limit_feed, flag_conversion_part_override,
#                  step_size_conversion, Ethane_Feed, Current_Recycle_Ethane_Feed,
#                  factor, feed_reduction_potential
# ---------------------------------------------------------------------------

def _step1_generate_initial_attributes(df: pd.DataFrame, macros: Dict) -> pd.DataFrame:
    df = df.copy()

    pass_step   = _val(macros.get('pass_step_change'), 0.25)
    pass_min    = _val(macros.get('pass_feed_min_limit'), 6.5)

    # Placeholders — overwritten downstream
    df['lower_limit_feed']            = 0.0
    df['upper_limit_feed']            = 0.0
    df['flag_conversion_part_override'] = 0
    df['step_size_conversion']        = 0.0

    # Ethane_Feed = Feed_flow / (1 + shc_ratio)
    df['Ethane_Feed'] = df['Feed_flow'] / (1 + df['shc_ratio'])

    # Current_Recycle_Ethane_Feed = Ethane_Feed × (100 − Overall_conversion) / 100
    df['Current_Recycle_Ethane_Feed'] = df['Ethane_Feed'] * (100 - df['overall_conversion']) / 100

    # factor: positive when above threshold → small 1/conversion; else large negative
    df['factor'] = df.apply(
        lambda r: (1 / r['overall_conversion']) if r['percent_above_threshold'] > 0
                  else (-1 * r['overall_conversion']),
        axis=1
    )

    # feed_reduction_potential per furnace
    # For each pass n (1–8):
    #   floor((pass_n_mixedfeed_flow − pass_min) / pass_step) × pass_n_feed_red_potential_on_min_days
    # Sum all 8, multiply by pass_step, floor(min(result, 4))
    # Replace 3 with 2 (no-3-steps rule)
    def _frp(row):
        total = 0.0
        for n in range(1, 9):
            flow    = _val(row.get(f'pass{n}_mixedfeed_flow'), 0)
            red_pot = _val(row.get(f'pass{n}_feed_red_potential_on_min_days'), 0)
            steps   = math.floor((flow - pass_min) / pass_step) if (flow - pass_min) >= 0 else 0
            total  += steps * red_pot
        total = math.floor(min(total * pass_step, 4))
        if total == 3:
            total = 2
        return float(total)

    df['feed_reduction_potential'] = df.apply(_frp, axis=1)

    return df


# ---------------------------------------------------------------------------
# STEP 5 — Good Furnace: Upper Limit Computation
# New columns: Margin_condition_type, upper_limit_feed (updated), max_potential_total_feed,
#              step_size_feed, Furnace_condition2
# ---------------------------------------------------------------------------

def _step5_good_upper_limit(df: pd.DataFrame, macros: Dict) -> pd.DataFrame:
    df = df.copy()
    pass_step = _val(macros.get('pass_step_change'), 0.25)
    pass_max  = _val(macros.get('threshold_pass_mixed_feed_limit'), 8.5)

    def _margin_cond_type(row):
        mf  = _val(row.get('Margin_in_Feed'), 0)
        mlc = _val(row.get('Margin_in_Feed_lower_check'), 0)
        if mf == 1 and mlc == 0:
            return 0
        if mf == 1 and mlc == 1:
            return 1
        return 1000   # no margin in feed

    def _upper_limit_from_margin(row, mct):
        sat = _val(row.get('saturator_margin'), 0)
        if sat == 1:
            if mct == 1:   return 4.0
            if mct == 0:   return 2.0
            return 0.0
        else:
            if mct == 1:   return 2.0
            if mct == 0:   return 1.0
            return 0.0

    def _max_potential_total_feed(row):
        total = 0.0
        for n in range(1, 9):
            flow    = _val(row.get(f'pass{n}_mixedfeed_flow'), 0)
            uf_cond = _val(row.get(f'upper_feed_condition_pass{n}'), 0)
            headroom = (pass_max - flow)
            steps    = math.floor(headroom / pass_step) if headroom >= 0 else 0
            total   += steps * uf_cond
        total = math.floor(min(total * pass_step, 4))
        if total == 3:
            total = 2
        return float(total)

    df['Margin_condition_type']    = df.apply(_margin_cond_type, axis=1)
    df['upper_limit_feed']         = df.apply(lambda r: _upper_limit_from_margin(r, r['Margin_condition_type']), axis=1)
    df['max_potential_total_feed'] = df.apply(_max_potential_total_feed, axis=1)
    df['upper_limit_feed']         = df.apply(lambda r: min(r['upper_limit_feed'], r['max_potential_total_feed']), axis=1)
    df['step_size_feed']           = df['upper_limit_feed'].apply(lambda u: 2.0 if u > 1 else u)
    df['Furnace_condition2']       = df['Furnace_condition']

    return df


# ---------------------------------------------------------------------------
# STEP 9 — Not-Good Furnace Overrides
# ---------------------------------------------------------------------------

def _step9_no_good_overrides(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Furnace_condition2'] = 'No Good'
    df['step_size_feed']     = 0.0
    return df


# ---------------------------------------------------------------------------
# STEP 16 — Subprocess 94: Feed Distribution Engine
# Phases: 1=sort+ID, 2=while-loop distribution, 3=3-step correction, 4=lower_limit finalisation
# ---------------------------------------------------------------------------

def _subprocess94_feed_distribution(
    good_df: pd.DataFrame,
    no_good_df: pd.DataFrame,
    macros: Dict
) -> pd.DataFrame:

    sum_feed_red = _val(macros.get('sum_feed_reduction_potential'), 0)
    total_fur    = int(_val(macros.get('total_fur_available_for_bias'), len(good_df) + len(no_good_df)))

    # Phase 1: Sort and ID assignment
    # Good: sort ascending factor (most room first)
    # Not-Good: zero out feed_reduction_potential, sort descending factor
    good_s    = good_df.copy().sort_values('factor', ascending=True).reset_index(drop=True)
    no_good_s = no_good_df.copy()
    no_good_s['feed_reduction_potential'] = 0.0
    no_good_s = no_good_s.sort_values('factor', ascending=False).reset_index(drop=True)

    # Append: Not-Good on top, Good below
    combined = pd.concat([no_good_s, good_s], ignore_index=True)
    combined['id']            = range(len(combined))
    combined['upper_limit_feed_org'] = combined['upper_limit_feed'].copy()

    # Phase 2: While-loop distribution — one furnace at a time
    def _distribute(df_in, balance_start):
        df_work = df_in.copy()
        df_work['taken']        = 0.0
        df_work['balance_taken'] = 0.0
        balance_feed = balance_start

        for i in range(total_fur):
            if i >= len(df_work):
                break
            ul = _val(df_work.loc[i, 'upper_limit_feed'], 0)
            taken = min(balance_feed, ul)
            balance_feed = balance_feed - taken
            df_work.loc[i, 'taken']         = taken
            df_work.loc[i, 'balance_taken'] = balance_feed

        return df_work

    distributed = _distribute(combined, sum_feed_red)

    # Phase 3: Three-Step Correction
    # If any row has taken == 3 → forbidden; correct by:
    #   a) Find Good furnace with upper_limit_feed_org == 1 → zero its upper_limit_feed
    #   b) Find Not-Good furnace with feed_reduction_potential_orig == 1 → zero its feed_reduction_potential
    # Then re-distribute
    if (distributed['taken'] == 3).any():
        corrected = combined.copy()

        # Correct on Good side: find first Good furnace (factor ascending) with upper_limit_feed == 1
        good_mask = corrected['Furnace_condition2'] != 'No Good'
        eligible_good = corrected[good_mask & (corrected['upper_limit_feed_org'] == 1)].sort_values('factor', ascending=True)
        if not eligible_good.empty:
            idx = eligible_good.index[0]
            corrected.loc[idx, 'upper_limit_feed']     = 0.0
            corrected.loc[idx, 'upper_limit_feed_org'] = 0.0

        # Correct on Not-Good side: find first Not-Good furnace (factor descending) with feed_reduction_potential == 1
        # Note: feed_reduction_potential was zeroed in Phase 1 for sort, use upper_limit_feed_org for No Good is 0
        # Use original no_good_df to find eligible
        orig_no_good_mask = corrected['Furnace_condition2'] == 'No Good'
        # Rebuild reduction potential from original no_good_df using entity_name key
        orig_frp = no_good_df.set_index('entity_name')['feed_reduction_potential'].to_dict()
        corrected['_orig_frp'] = corrected['entity_name'].map(orig_frp).fillna(0)
        eligible_ng = corrected[orig_no_good_mask & (corrected['_orig_frp'] == 1)].sort_values('factor', ascending=False)
        if not eligible_ng.empty:
            idx = eligible_ng.index[0]
            corrected.loc[idx, '_orig_frp'] = 0.0

        # Recompute sum_feed_reduction_potential from corrected no-good
        corrected_ng_mask = corrected['Furnace_condition2'] == 'No Good'
        new_sum_red = corrected.loc[corrected_ng_mask, '_orig_frp'].sum()

        distributed = _distribute(corrected, new_sum_red)
        distributed.drop(columns=['_orig_frp'], inplace=True, errors='ignore')

    # Phase 4: lower_limit_feed = -(taken)
    distributed['given']           = distributed['taken'].copy()
    distributed['lower_limit_feed'] = -distributed['given']

    # Drop working columns
    drop_cols = ['feed_reduction_potential', 'factor', 'taken', 'balance_taken',
                 'upper_limit_feed_org', 'given', 'balance_given', 'id']
    distributed.drop(columns=[c for c in drop_cols if c in distributed.columns], inplace=True, errors='ignore')

    return distributed


# ---------------------------------------------------------------------------
# STEP 19 — Final Feed Limit Reconciliation
# ---------------------------------------------------------------------------

def _step19_reconcile_feed_limits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['lower_limit_feed_org'] = df['lower_limit_feed'].copy()

    def _reconcile(row):
        ul = row['upper_limit_feed']
        ll = row['lower_limit_feed']
        ll_org = row['lower_limit_feed_org']

        # If upper == lower (no range) → zero both
        if ul == ll_org:
            ul, ll = 0.0, 0.0
        else:
            # If upper is non-zero → set lower = upper (makes lower positive — handled downstream)
            if ul != 0:
                ll = ul
            # If lower is negative → set upper = lower (pure reduction scenario)
            if ll_org < 0:
                ul = ll_org
                ll = ll_org

        # step_size_feed
        step = 0.0 if (ll + ul == 0) else 1.0

        return pd.Series({'upper_limit_feed': ul, 'lower_limit_feed': ll, 'step_size_feed': step})

    reconciled = df.apply(_reconcile, axis=1)
    df[['upper_limit_feed', 'lower_limit_feed', 'step_size_feed']] = reconciled
    return df


# ---------------------------------------------------------------------------
# STEP 23 — Recycle Ethane & Margin Macros (macro-to-macro computation)
# ---------------------------------------------------------------------------

def _step23_recycle_ethane_macros(macros: Dict) -> Dict:
    sum_ul         = _val(macros.get('sum_upper_limit_feed'), 0)
    shc            = _val(macros.get('shc_ratio'), 0)
    fresh_feed_chg = int(_val(macros.get('fresh_feed_change'), 0))
    count_good     = int(_val(macros.get('count_of_good_fur'), 0))
    fresh_feed_qty = _val(macros.get('fresh_feed_quantity'), 0)
    re_upper       = _val(macros.get('change_recycle_ethane_upper_limit'), 0)
    re_lower       = _val(macros.get('change_recycle_ethane_lower_limit'), 0)

    # mixed_feed_margin: cap by good furnace count only when fresh_feed_change == 0
    if fresh_feed_chg != 0:
        mixed_feed_margin = round(sum_ul)
    else:
        mixed_feed_margin = round(min(sum_ul, count_good * 2))

    # Extra_Recycle_Ethane = mixed_feed_margin / (1 + shc) − fresh_feed_quantity
    denom = (1 + shc) if shc != -1 else 1
    extra_recycle_ethane = (mixed_feed_margin / denom) - fresh_feed_qty

    upper_limit_change_in_recycle_ethane = extra_recycle_ethane + re_upper
    lower_limit_change_in_recycle_ethane = extra_recycle_ethane + re_lower

    macros['mixed_feed_margin']                     = mixed_feed_margin
    macros['Extra_Recycle_Ethane']                  = extra_recycle_ethane
    macros['upper_limit_change_in_recycle_ethane']  = upper_limit_change_in_recycle_ethane
    macros['lower_limit_change_in_recycle_ethane']  = lower_limit_change_in_recycle_ethane
    return macros


# ---------------------------------------------------------------------------
# STEP 25 — Lock lower_limit_feed = upper_limit_feed when all margin allocated
# Only when fresh_feed_change != -1 AND biasing_condition == 1
# ---------------------------------------------------------------------------

def _step25_lock_lower_to_upper(df: pd.DataFrame, macros: Dict) -> pd.DataFrame:
    fresh_feed_chg   = int(_val(macros.get('fresh_feed_change'), 0))
    biasing          = int(_val(macros.get('biasing_condition'), 1))
    mixed_feed_margin = _val(macros.get('mixed_feed_margin'), 0)
    sum_ul           = _val(macros.get('sum_upper_limit_feed'), 0)

    if fresh_feed_chg == -1:
        return df  # branch false path — skip

    df = df.copy()
    if biasing == 1 and mixed_feed_margin == sum_ul:
        def _lock(row):
            ul = row['upper_limit_feed']
            step = row['step_size_feed']
            ll = ul   # lock lower = upper
            if step != 0:
                step = 1.0
            if ll == ul:   # after lock — no range
                step = 0.0
            return pd.Series({'lower_limit_feed': ll, 'step_size_feed': step})
        df[['lower_limit_feed', 'step_size_feed']] = df.apply(_lock, axis=1)

    return df


# ---------------------------------------------------------------------------
# STEP 26 — Conversion Limits
# True path (biasing_condition == 3): conversion-only mode
# False path (biasing_condition != 3): standard mode + optional Subprocess 14
# ---------------------------------------------------------------------------

def _step26_conversion_limits(df: pd.DataFrame, macros: Dict) -> Tuple[pd.DataFrame, Dict]:
    biasing          = int(_val(macros.get('biasing_condition'), 1))
    fresh_feed_chg   = int(_val(macros.get('fresh_feed_change'), 0))
    conv_lower_thresh = _val(macros.get('conversion_bias_threshold_lower_limit'), -2.5)
    conv_upper_thresh = _val(macros.get('conversion_bias_threshold_upper_limit'), 1.5)
    conv_lower_exp   = _val(macros.get('conversion_lower_limit_expansion_max_limit'), 47.0)
    conv_upper_exp   = _val(macros.get('conversion_upper_limit_expansion_max_limit'), 65.0)
    count_no_good    = int(_val(macros.get('count_of_no_good_fur'), 0))
    mixed_feed_margin = _val(macros.get('mixed_feed_margin'), 0)
    shc              = _val(macros.get('shc_ratio'), 0)

    df = df.copy()

    # Conversion lower limit formula (common to both paths):
    # min(0, max(conv_lower_thresh, -floor_half(OC - conv_lower_exp)))
    def _conv_lower(oc):
        raw = -_floor_half(oc - conv_lower_exp)
        return min(0.0, max(conv_lower_thresh, raw))

    if biasing == 3:
        # ── TRUE PATH: Conversion-Only Mode ──────────────────────────────
        df['conversion_lower_limit_in_grid'] = df['overall_conversion'].apply(_conv_lower)

        # Upper limit: only for Semi Good with positive percent_above_threshold
        def _conv_upper_bias3(row):
            if row['Furnace_condition'] == 'Semi Good' and row['percent_above_threshold'] > 0:
                raw = _floor_half(conv_upper_exp - row['overall_conversion'])
                return max(0.0, min(conv_upper_thresh, raw))
            return 0.0

        df['conversion_upper_limit_in_grid'] = df.apply(_conv_upper_bias3, axis=1)

        # step_size_conversion driven by count_of_no_good_fur
        if count_no_good < 6:
            step_conv = 5.0
        elif count_no_good == 6:
            step_conv = 3.0
        elif count_no_good == 9:
            step_conv = 1.0
        else:
            step_conv = 2.0
        df['step_size_conversion'] = step_conv
        df['New_Feed_flow']        = df['Feed_flow']   # no feed change in this mode

    else:
        # ── FALSE PATH: Standard Conversion Mode ─────────────────────────
        df['conversion_lower_limit_in_grid'] = df['overall_conversion'].apply(_conv_lower)

        # Upper limit: Good or Semi Good AND (fresh_feed_change != 0 OR Forecasted_runlength_rank != 100)
        def _conv_upper_std_pass1(row):
            cond = row['Furnace_condition'] in ('Good', 'Semi Good')
            run_ok = (fresh_feed_chg != 0) or (_val(row.get('Forecasted_runlength_rank'), 0) != 100)
            if cond and run_ok:
                return conv_upper_thresh
            return 0.0

        def _conv_upper_std_pass2(row):
            p1 = row['_conv_upper_p1']
            if p1 == 0:
                return 0.0
            raw = _floor_half(conv_upper_exp - row['overall_conversion'])
            return max(0.0, min(p1, raw))

        df['_conv_upper_p1']                 = df.apply(_conv_upper_std_pass1, axis=1)
        df['conversion_upper_limit_in_grid'] = df.apply(_conv_upper_std_pass2, axis=1)
        df.drop(columns=['_conv_upper_p1'], inplace=True)

        df['step_size_conversion'] = 1.0   # standard mode default

        # ── Subprocess 14 — only when fresh_feed_change == 0 ─────────────
        if fresh_feed_chg == 0:
            denom = (1 + shc) if shc != -1 else 1

            df['New_Overall_conversion'] = df['overall_conversion'] + df['conversion_lower_limit_in_grid']
            df['New_Recycle_Ethane_Feed'] = (df['Feed_flow'] / denom) * (100 - df['New_Overall_conversion']) / 100
            df['New_Extra_Recycle_Ethane'] = df['New_Recycle_Ethane_Feed'] - df['Current_Recycle_Ethane_Feed']
            df['New_mixed_feed_margin']    = df['New_Extra_Recycle_Ethane'] * denom

            sum_new_mfm = df['New_mixed_feed_margin'].sum()
            floor_sum   = math.floor(sum_new_mfm)

            if mixed_feed_margin > floor_sum:
                # Parity check: if mixed_feed_margin and floor_sum differ in parity → decrement floor by 1
                if (int(mixed_feed_margin) % 2) != (floor_sum % 2):
                    floor_sum -= 1

                # Revise mixed_feed_margin and downstream macros
                mixed_feed_margin = float(floor_sum)
                macros['mixed_feed_margin'] = mixed_feed_margin

                extra_recycle_ethane = (mixed_feed_margin / denom) - _val(macros.get('fresh_feed_quantity'), 0)
                macros['Extra_Recycle_Ethane'] = extra_recycle_ethane
                macros['upper_limit_change_in_recycle_ethane'] = extra_recycle_ethane + _val(macros.get('change_recycle_ethane_upper_limit'), 0)
                macros['lower_limit_change_in_recycle_ethane'] = extra_recycle_ethane + _val(macros.get('change_recycle_ethane_lower_limit'), 0)

            # Drop subprocess temp columns
            df.drop(columns=['New_Overall_conversion', 'New_Recycle_Ethane_Feed',
                              'New_Extra_Recycle_Ethane', 'New_mixed_feed_margin'],
                    inplace=True, errors='ignore')

    return df, macros


# ---------------------------------------------------------------------------
# STEP 27 — Zero-Initialise Row_N_* macros (9 iterations)
# ---------------------------------------------------------------------------

def _step27_zero_init_row_macros(macros: Dict) -> Dict:
    for n in range(1, 10):
        macros[f'Row_{n}_upper_limit_feed']       = 0
        macros[f'Row_{n}_lower_limit_feed']       = 0
        macros[f'Row_{n}_step_size_feed']         = 0
        macros[f'Grid_Row_{n}_conversion_delta']  = 0
        macros[f'Row_{n}_step_size_conversion']   = 0
        macros[f'Row_{n}_upper_limit_conversion'] = 0
        macros[f'Row_{n}_lower_limit_conversion'] = 0
        macros[f'Row_{n}_Furnace']                = 0
        macros[f'Row_{n}_part_override']          = 0
    return macros


# ---------------------------------------------------------------------------
# STEP 29 — Extract Row_N_* macros from sorted final df
# ---------------------------------------------------------------------------

def _step29_extract_row_macros(df: pd.DataFrame, macros: Dict) -> Dict:
    df_sorted = df.sort_values('overall_ranking', ascending=True).reset_index(drop=True)
    for i, row in df_sorted.iterrows():
        n = i + 1  # 1-based
        macros[f'Row_{n}_Feed_flow']                   = _val(row.get('Feed_flow'), 0)
        macros[f'Row_{n}_Furnace']                     = row.get('entity_name', 0)
        macros[f'Row_{n}_Specific_Energy_consumption'] = _val(row.get('specific_energy_consumption'), 0)
        macros[f'Row_{n}_Conversion']                  = _val(row.get('overall_conversion'), 0)
        macros[f'Row_{n}_Furnace_condition']           = row.get('Furnace_condition', 0)
        macros[f'Row_{n}_Ethylene_Production']         = _val(row.get('ethylene_production'), 0)
        macros[f'Row_{n}_Current_Recycle_Ethane_Feed'] = _val(row.get('Current_Recycle_Ethane_Feed'), 0)
        macros[f'Row_{n}_lower_limit_feed']            = _val(row.get('lower_limit_feed'), 0)
        macros[f'Row_{n}_upper_limit_feed']            = _val(row.get('upper_limit_feed'), 0)
        macros[f'Row_{n}_step_size_feed']              = _val(row.get('step_size_feed'), 0)
        macros[f'Row_{n}_lower_limit_conversion']      = _val(row.get('conversion_lower_limit_in_grid'), 0)
        macros[f'Row_{n}_upper_limit_conversion']      = _val(row.get('conversion_upper_limit_in_grid'), 0)
        macros[f'Row_{n}_step_size_conversion']        = _val(row.get('step_size_conversion'), 0)
        macros[f'Row_{n}_part_override']               = _val(row.get('flag_conversion_part_override'), 0)
    return macros


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_pre_grid(
    eligible_furnaces_df: pd.DataFrame,
    input_macros: Dict[str, Any],
    deviation_exists: int = 1,
    previous_ranking_output: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Parameters
    ----------
    eligible_furnaces_df : DataFrame from Module 3 / Module 4 pass-through
    input_macros         : dict of all required input macros (see module docstring)

    Returns
    -------
    grid_df  : per-furnace DataFrame with all limit columns
    macros   : dict with all Row_N_* output macros and park-level macros
    """

    macros = dict(input_macros)   # working copy — will be updated throughout
    # Gate: compute final_run_optimizer_check
    deviation_exists             = int(_val(input_macros.get('deviation_exists'), 0))
    final_run_optimizer_check_init = int(_val(input_macros.get('final_run_optimizer_check_init'), 0))
    final_run_optimizer_check    = 1 if (deviation_exists == 1 and final_run_optimizer_check_init == 1) else 0
    macros['final_run_optimizer_check'] = final_run_optimizer_check

    if final_run_optimizer_check != 1:
        if deviation_exists == 0:
            # Pass previous timestamp ranking output forward (caller must supply it)
            print('[Module 5] deviation_exists=0 → passing previous ranking output forward')
            return input_macros.get('previous_ranking_output', pd.DataFrame()), macros
        else:
            # Zero-substitute feed and conversion bias, pass forward
            print('[Module 5] final_run_optimizer_check=0, deviation=1 → zeroing feed/conversion bias')
            eligible_furnaces_df = eligible_furnaces_df.copy()
            eligible_furnaces_df['upper_limit_feed']              = 0
            eligible_furnaces_df['lower_limit_feed']              = 0
            eligible_furnaces_df['conversion_upper_limit_in_grid'] = 0
            eligible_furnaces_df['conversion_lower_limit_in_grid'] = 0
            return eligible_furnaces_df, macros
    df     = eligible_furnaces_df.copy()

    # ── STEP 1: Initial attribute generation ────────────────────────────
    df = _step1_generate_initial_attributes(df, macros)
    print(f'[Step 1] Columns after init: {list(df.columns)}')

    # ── STEP 2: Number_of_rows macro ────────────────────────────────────
    macros['Number_of_rows'] = len(df)
    print(f'[Step 2] Number_of_rows = {macros["Number_of_rows"]}')

    # ── STEP 3: shc_ratio macro from first row ───────────────────────────
    macros['shc_ratio'] = _val(df['shc_ratio'].iloc[0]) if 'shc_ratio' in df.columns else _val(macros.get('shc_ratio'), 0)
    print(f'[Step 3] shc_ratio = {macros["shc_ratio"]}')

    # ── STEP 4: Split Good vs Not-Good ───────────────────────────────────
    good_df    = df[df['Furnace_condition'] == 'Good'].copy()
    no_good_df = df[df['Furnace_condition'] != 'Good'].copy()
    print(f'[Step 4] Good furnaces: {len(good_df)}  |  Not-Good furnaces: {len(no_good_df)}')

    # ── STEP 5: Good furnace upper limit computation ──────────────────────
    if not good_df.empty:
        good_df = _step5_good_upper_limit(good_df, macros)

    # ── STEP 6: Drop intermediate columns from Good ───────────────────────
    drop_s6 = ['lower_limit_feed_org', 'count_pass_mixed_feed', 'Margin_condition_type', 'max_potential_total_feed']
    good_df.drop(columns=[c for c in drop_s6 if c in good_df.columns], inplace=True, errors='ignore')

    # ── STEPS 7-8: sum_upper_limit_feed provisional ───────────────────────
    macros['sum_upper_limit_feed'] = float(good_df['upper_limit_feed'].sum()) if not good_df.empty else 0.0
    print(f'[Step 8] sum_upper_limit_feed (provisional) = {macros["sum_upper_limit_feed"]}')

    # ── STEP 9: Not-Good overrides ────────────────────────────────────────
    if not no_good_df.empty:
        no_good_df = _step9_no_good_overrides(no_good_df)

    # ── STEPS 10-11: sum_feed_reduction_potential ─────────────────────────
    macros['sum_feed_reduction_potential'] = float(no_good_df['feed_reduction_potential'].sum()) if not no_good_df.empty else 0.0
    macros['count_of_good_fur']    = len(good_df)
    macros['count_of_no_good_fur'] = len(no_good_df)
    print(f'[Step 11] sum_feed_reduction_potential = {macros["sum_feed_reduction_potential"]}')

    # ── STEP 12: Branch — biasing_condition != 2 ─────────────────────────
    biasing = int(_val(macros.get('biasing_condition'), 1))
    print(f'[Step 12] biasing_condition = {biasing}')

    if biasing != 2:
        # ── STEP 13: Merge Good + Not-Good ───────────────────────────────
        merged_df = pd.concat([no_good_df, good_df], ignore_index=True)

        # ── STEP 14: Branch — fresh_feed_change == -1 ────────────────────
        fresh_feed_chg = int(_val(macros.get('fresh_feed_change'), 0))
        print(f'[Step 14] fresh_feed_change = {fresh_feed_chg}')

        if fresh_feed_chg == -1:
            # Feed decrease path
            merged_df['upper_limit_feed'] = 0.0
            merged_df['lower_limit_feed'] = -merged_df['feed_reduction_potential']
            merged_df['step_size_feed']   = 1.0
        else:
            # Feed increase path — recalculate step_size_feed
            merged_df['step_size_feed'] = merged_df['upper_limit_feed'].apply(
                lambda u: 2.0 if u > 1 else u
            )

        # ── STEP 15: Drop residual columns ───────────────────────────────
        drop_s15 = ['lower_limit_feed_org', 'Furnace_condition2']
        merged_df.drop(columns=[c for c in drop_s15 if c in merged_df.columns], inplace=True, errors='ignore')

        # ── STEP 16: Subprocess 94 — feed distribution ───────────────────
        grid_df = _subprocess94_feed_distribution(good_df, no_good_df, macros)
        print(f'[Step 16] Feed distribution complete. Rows: {len(grid_df)}')

    else:
        # Bias 2: pass-through — merge and skip distribution
        merged_df = pd.concat([no_good_df, good_df], ignore_index=True)
        grid_df   = merged_df.copy()
        print('[Step 12] biasing_condition == 2 → pass-through, skipping feed distribution')

    # ── STEP 17: Sort by overall_ranking ─────────────────────────────────
    grid_df = grid_df.sort_values('overall_ranking', ascending=True).reset_index(drop=True)

    # ── STEP 18: Drop subprocess temp columns ────────────────────────────
    drop_s18 = ['factor', 'feed_reduction_potential', 'feed_reduction_potential_1', 'Furnace_condition2', 'id']
    grid_df.drop(columns=[c for c in drop_s18 if c in grid_df.columns], inplace=True, errors='ignore')

    # ── STEP 19: Reconcile feed limits ───────────────────────────────────
    grid_df = _step19_reconcile_feed_limits(grid_df)

    # ── STEP 20: Drop lower_limit_feed_org, Furnace_condition2 ───────────
    drop_s20 = ['lower_limit_feed_org', 'Furnace_condition2']
    grid_df.drop(columns=[c for c in drop_s20 if c in grid_df.columns], inplace=True, errors='ignore')

    # ── STEPS 21-22: sum_upper_limit_feed definitive ─────────────────────
    macros['sum_upper_limit_feed'] = float(grid_df['upper_limit_feed'].sum())
    print(f'[Step 22] sum_upper_limit_feed (definitive) = {macros["sum_upper_limit_feed"]}')

    # ── STEP 23: Recycle ethane macros ────────────────────────────────────
    macros = _step23_recycle_ethane_macros(macros)
    print(f'[Step 23] mixed_feed_margin = {macros["mixed_feed_margin"]}')
    print(f'[Step 23] Extra_Recycle_Ethane = {macros["Extra_Recycle_Ethane"]:.4f}')

    # ── STEP 24: Stamp Extra_Recycle_Ethane as column ────────────────────
    grid_df['Extra_Recycle_Ethane'] = macros['Extra_Recycle_Ethane']

    # ── STEP 25: Lock lower = upper when all margin allocated ─────────────
    grid_df = _step25_lock_lower_to_upper(grid_df, macros)

    # ── STEP 26: Conversion limits ────────────────────────────────────────
    grid_df, macros = _step26_conversion_limits(grid_df, macros)
    print(f'[Step 26] Conversion limits computed. biasing_condition = {biasing}')

    # ── STEP 27: Zero-initialise Row_N_* macros ───────────────────────────
    macros = _step27_zero_init_row_macros(macros)

    # ── STEP 28: Final sort by overall_ranking ────────────────────────────
    grid_df = grid_df.sort_values('overall_ranking', ascending=True).reset_index(drop=True)

    # ── STEP 29: Extract Row_N_* macros ───────────────────────────────────
    macros = _step29_extract_row_macros(grid_df, macros)
    print(f'[Step 29] Row macros extracted for {macros["Number_of_rows"]} furnaces')

    return grid_df, macros


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')

    FILE  = '/mnt/user-data/uploads/Ranking_req_info.xlsx'
    SHEET = 'tags in parameter format'

    from module3_preprocessing_v5 import run_preprocessing

    df = pd.read_excel(FILE, sheet_name=SHEET)
    all_results_all, all_results_eligible = [], []
    for ts, group in df.groupby('Timestamp'):
        all_df_ts, eligible_df_ts = run_preprocessing(group)
        all_results_all.append(all_df_ts)
        all_results_eligible.append(eligible_df_ts)

    eligible_df = pd.concat(all_results_eligible, ignore_index=True)

    print('=' * 70)
    print('  MODULE 5: PRE-GRID')
    print('=' * 70)
    print(f'Input — eligible_furnaces_df: {eligible_df.shape[0]} rows x {eligible_df.shape[1]} cols')
    print()

    # Mock input macros — replace with DB values in production
    mock_macros = {
        'pass_step_change':                          0.25,
        'pass_feed_min_limit':                       6.5,
        'threshold_pass_mixed_feed_limit':           8.5,
        'biasing_condition':                         1,       # 1=increase, 2=ext-reduce, 3=conv-only
        'fresh_feed_change':                         0,       # 0=none, -1=cut
        'total_fur_available_for_bias':              len(eligible_df),
        'conversion_bias_threshold_lower_limit':    -2.5,
        'conversion_bias_threshold_upper_limit':     1.5,
        'conversion_lower_limit_expansion_max_limit': 47.0,
        'conversion_upper_limit_expansion_max_limit': 65.0,
        'change_recycle_ethane_upper_limit':         0.5,
        'change_recycle_ethane_lower_limit':        -0.5,
        'fresh_feed_quantity':                       0.0,
        'furnace_step_adjust_feed_grid_limit':       6,
    }

    grid_df, out_macros = run_pre_grid(eligible_df, mock_macros)

    print('\n' + '=' * 70)
    print('  OUTPUT SUMMARY')
    print('=' * 70)
    print(f'\ngrid_df: {grid_df.shape[0]} rows x {grid_df.shape[1]} cols')
    feed_cols = ['entity_name', 'Furnace_condition', 'upper_limit_feed', 'lower_limit_feed',
                 'step_size_feed', 'conversion_lower_limit_in_grid',
                 'conversion_upper_limit_in_grid', 'step_size_conversion']
    print('\nPer-furnace grid limits:')
    print(grid_df[[c for c in feed_cols if c in grid_df.columns]].to_string(index=False))

    print('\nPark-level macros:')
    park_keys = ['mixed_feed_margin', 'Extra_Recycle_Ethane',
                 'upper_limit_change_in_recycle_ethane', 'lower_limit_change_in_recycle_ethane',
                 'sum_upper_limit_feed', 'sum_feed_reduction_potential']
    for k in park_keys:
        if k in out_macros:
            print(f'  {k:<45s} = {out_macros[k]}')

    print('\nRow macros (sample — first 3 furnaces):')
    for n in range(1, min(4, int(out_macros.get('Number_of_rows', 0)) + 1)):
        keys = [f'Row_{n}_Furnace', f'Row_{n}_upper_limit_feed', f'Row_{n}_lower_limit_feed',
                f'Row_{n}_step_size_feed', f'Row_{n}_lower_limit_conversion',
                f'Row_{n}_upper_limit_conversion', f'Row_{n}_step_size_conversion']
        vals = {k: out_macros.get(k, '-') for k in keys}
        print(f'  Row {n}: {vals}')

    grid_df.to_csv('/mnt/user-data/outputs/module5_grid_df.csv', index=False)
    print('\nSaved: /mnt/user-data/outputs/module5_grid_df.csv')
