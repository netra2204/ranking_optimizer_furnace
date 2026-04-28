"""
=============================================================================
MODULE 3: PRE-PROCESSING  v5
=============================================================================
Changes from v4 → v5 (all corrections applied):

  1.  Margin_in_Feed        → max_feed_valve_opening < margin_value_feed_limit (87%)
                               Margin_in_Feed_lower_check is separate (all passes < 70%)
  2.  Margin_in_Steam       → e-1503_level_controller_opening < lp_steam_flow_controller_opening_limit
  3.  C2_splitter_BTM_C2H4 → ethylene_in_ethane_recycle (FS row) < c2_splitter_btm_c2h4_mol_percent_limit
  4.  C2_splitter_reflux    → quench_tower_top_temperature < c2_splitter_reflux_pump_suction_temp_limit
  5.  Total_margin          → sum of all 11 individual margins (excl. Margin_in_Feed_lower_check)
  6.  days_remaining        → _is_missing() guard before EOR check (no false 0 default)
  7.  fresh_feed_deviation  → OR logic (either flag true → hold), not AND
  8.  Recycle calc          → per-furnace individual calculation then summed
  9.  pass{n}_feed_red_pot  → binary flag: 1 = this pass has min runlength, 0 otherwise
  10. Forecasted_rank       → compress overall_ranking sequentially (not re-rank from scratch)
  11. Condition code        → severity order: No Optimization=0, No cracking=1, SOR=2,
                               EOR=3, Semi Good=4, Bad=5, Good=6
  12. Count filter          → counts all furnaces by condition (no coupling/constraint filter)

Returns TWO DataFrames:
  all_furnaces_df      → 10 rows (F1-F9 + FS), original order, 18 new computed cols
  eligible_furnaces_df → valid-ranked furnaces only, sorted by overall_ranking,
                          26 new computed cols (incl. pass{n}_feed_red_potential_on_min_days)

⚠️  Columns not in input → added as NaN (needs future data source):
    benefit_percent_threshold, furnace_step_adjust_feed_grid_limit,
    net_nox_emission, nox_emission_permissible_limit, nox_margin

Dependencies: pip install pandas numpy openpyxl
=============================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

PASS_NUMBERS  = list(range(1, 9))
PASS_MIN_FLOW = 6.5    # tph — minimum permissible pass flow

# Condition → severity code (v5 ordering)
CONDITION_CODE_MAP = {
    'No Optimization': 0,
    'No cracking':     1,
    'SOR':             2,
    'EOR':             3,
    'Semi Good':       4,
    'Bad':             5,
    'Good':            6,
}

GOOD_CONDITIONS     = {'Good'}
NO_GOOD_CONDITIONS  = {'Bad', 'Semi Good', 'SOR'}
CRACKING_CONDITIONS = {'Good', 'Semi Good', 'Bad', 'SOR', 'EOR'}


# =============================================================================
# Helpers
# =============================================================================

def _val(v, default=None):
    """Return v if not NaN/None, else default."""
    if v is None:
        return default
    try:
        if math.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _is_missing(v):
    """True if v is None or NaN."""
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# =============================================================================
# Step 1A – Extract system config from FS row
# =============================================================================

def extract_system_config(fs_row: pd.Series) -> Dict[str, Any]:
    cfg = {
        # ── Limits ────────────────────────────────────────────────────────────
        'minimum_fur_for_optimization_limit':           _val(fs_row.get('minimum_fur_for_optimization_limit'), 5),
        'margin_in_feed_lower_check_limit':             _val(fs_row.get('margin_in_feed_lower_check_limit'), 70.0),
        'margin_value_feed_limit':                      _val(fs_row.get('margin_value_feed_limit'), 87.0),
        'damper_opening_limit':                         _val(fs_row.get('damper_opening_limit'), 60.0),
        'fuel_gas_pressure_controlvalve_opening_limit': _val(fs_row.get('fuel_gas_pressure_controlvalve_opening_limit'), 85.0),
        'lp_steam_flow_controller_opening_limit':       _val(fs_row.get('lp_steam_flow_controller_opening_limit'), 90.0),
        'cgc_suction_pressure_limit':                   _val(fs_row.get('cgc_suction_pressure_limit'), 0.55),
        'c2_splitter_dp_limit':                         _val(fs_row.get('c2_splitter_dp_limit'), 537.0),
        'c2_splitter_btm_c2h4_mol_percent_limit':       _val(fs_row.get('c2_splitter_btm_c2h4_mol_percent_limit'), 1.0),
        'c2_splitter_reflux_pump_suction_temp_limit':   _val(fs_row.get('c2_splitter_reflux_pump_suction_temp_limit'), 10.0),
        'erc_governor_opening_limit':                   _val(fs_row.get('erc_governor_opening_limit'), 95.0),
        'prc_governor_opening_limit':                   _val(fs_row.get('prc_governor_opening_limit'), 95.0),
        'recycle_ethane_flow_controller_opening_limit': _val(fs_row.get('recycle_ethane_flow_controller_opening_limit'), 90.0),
        'saturator_drum_pressure_margin_limit':         _val(fs_row.get('saturator_drum_pressure_margin_limit'), 5.9),
        'max_coke_thickness_limit':                     _val(fs_row.get('max_coke_thickness_limit'), 10.0),
        'ethane_feed_flow_aramco_limit':                _val(fs_row.get('ethane_feed_flow_aramco_limit'), 5.0),
        'change_recycle_ethane_lower_limit':            _val(fs_row.get('change_recycle_ethane_lower_limit'), -0.3),
        'change_recycle_ethane_upper_limit':            _val(fs_row.get('change_recycle_ethane_upper_limit'), 0.3),
        'conversion_lower_limit_expansion_max_limit':   _val(fs_row.get('conversion_lower_limit_expansion_max_limit'), 47.0),
        'conversion_upper_limit_expansion_max_limit':   _val(fs_row.get('conversion_upper_limit_expansion_max_limit'), 65.0),
        'quench_ovhd_temp_limit':                       _val(fs_row.get('quench_ovhd_temp_limit'), 45.0),

        # ── System-level live measurements ────────────────────────────────────
        'cgc_suction_pressure':                    _val(fs_row.get('cgc_suction_pressure')),
        'c2_splitter_dp':                          _val(fs_row.get('c2_splitter_dp')),
        'c2_splitter_bottom_level':                _val(fs_row.get('c2_splitter_bottom_level')),
        'quench_tower_top_temperature':            _val(fs_row.get('quench_tower_top_temperature')),
        'erc_governor_opening':                    _val(fs_row.get('erc_governor_opening')),
        'prc_governor_opening':                    _val(fs_row.get('prc_governor_opening')),
        'recycle_ethane_flow_controller_opening':  _val(fs_row.get('recycle_ethane_flow_controller_opening')),
        'ethane_feed_flow_from_aramco':            _val(fs_row.get('ethane_feed_flow_from_aramco'), 0.0),
        'want_to_change_recycle_feed_flow':        _val(fs_row.get('want_to_change_recycle_feed_flow'), 0),
        'fresh_feed_change':                       _val(fs_row.get('fresh_feed_change'), 0),
        'expected_fresh_feed':                     _val(fs_row.get('expected_fresh_feed'), 0.0),
        'system_overall_conversion':               _val(fs_row.get('system_overall_conversion')),

        # ── FIX 2: Steam margin tag (IS in input, FS row) ─────────────────────
        'e_1503_level_controller_opening':         _val(fs_row.get('e-1503_level_controller_opening')),

        # ── FIX 3: C2 BTM C2H4 margin tag (IS in input, FS row) ──────────────
        'ethylene_in_ethane_recycle':              _val(fs_row.get('ethylene_in_ethane_recycle')),
    }
    return cfg


# =============================================================================
# Step 1B – Compute all 12 margins  (v5 corrected)
# =============================================================================

def compute_margins(furnace: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = {}

    # ── 1. Margin_in_Feed_lower_check (separate flag — all passes < 70%) ─────
    pass_openings = [
        _val(furnace.get(f'pass{i}_mixed_feed_flow_controller_opening'))
        for i in PASS_NUMBERS
    ]
    valid_openings = [op for op in pass_openings if op is not None]
    lower_limit    = cfg['margin_in_feed_lower_check_limit']   # 70%
    m['Margin_in_Feed_lower_check'] = (
        int(all(op < lower_limit for op in valid_openings)) if valid_openings else 0
    )

    # ── 2. Margin_in_Feed (FIX 1: max_feed_valve_opening < margin_value_feed_limit 87%) ──
    max_valve = _val(furnace.get('max_feed_valve_opening'))
    upper_limit = cfg['margin_value_feed_limit']               # 87%
    m['Margin_in_Feed'] = (
        int(max_valve < upper_limit) if max_valve is not None else 0
    )

    # ── 3. Margin_in_Damper ───────────────────────────────────────────────────
    damper = _val(furnace.get('damper_opening'))
    m['Margin_in_Damper'] = (
        int(damper < cfg['damper_opening_limit']) if damper is not None else 0
    )

    # ── 4. Margin_in_FG ───────────────────────────────────────────────────────
    fg = _val(furnace.get('fuel_gas_pressure_controlvalve_opening'))
    m['Margin_in_FG'] = (
        int(fg < cfg['fuel_gas_pressure_controlvalve_opening_limit']) if fg is not None else 0
    )

    # ── 5. Margin_in_Steam (FIX 2: e-1503_level_controller_opening < lp_steam_limit) ──
    steam_ctrl = cfg.get('e_1503_level_controller_opening')
    steam_lim  = cfg.get('lp_steam_flow_controller_opening_limit')
    m['Margin_in_Steam'] = (
        int(steam_ctrl < steam_lim)
        if (steam_ctrl is not None and steam_lim is not None) else 0
    )

    # ── 6. Quench_OD_Gas_temp_margin ──────────────────────────────────────────
    qt  = cfg.get('quench_tower_top_temperature')
    qtl = cfg.get('quench_ovhd_temp_limit')
    m['Quench_OD_Gas_temp_margin'] = (
        int(qt < qtl) if (qt is not None and qtl is not None) else 1
    )

    # ── 7. CGC_suction_pressure_margin ────────────────────────────────────────
    cgc  = cfg.get('cgc_suction_pressure')
    cgcl = cfg.get('cgc_suction_pressure_limit')
    m['CGC_suction_pressure_margin'] = (
        int(cgc < cgcl) if (cgc is not None and cgcl is not None) else 1
    )

    # ── 8. C2_splitter_dp_margin ──────────────────────────────────────────────
    c2dp  = cfg.get('c2_splitter_dp')
    c2dpl = cfg.get('c2_splitter_dp_limit')
    m['C2_splitter_dp_margin'] = (
        int(c2dp < c2dpl) if (c2dp is not None and c2dpl is not None) else 1
    )

    # ── 9. C2_splitter_btm_c2h4_mol_percent_margin ────────────────────────────
    # FIX 3: use ethylene_in_ethane_recycle (FS row) vs c2_splitter_btm_c2h4_mol_percent_limit
    eth_recycle = cfg.get('ethylene_in_ethane_recycle')
    c2h4_lim    = cfg.get('c2_splitter_btm_c2h4_mol_percent_limit')
    m['C2_splitter_btm_c2h4_mol_percent_margin'] = (
        int(eth_recycle < c2h4_lim)
        if (eth_recycle is not None and c2h4_lim is not None) else 1
    )

    # ── 10. C2_splitter_reflux_pump_suction_temp_margin ───────────────────────
    # FIX 4: quench_tower_top_temperature (FS) vs c2_splitter_reflux_pump_suction_temp_limit
    qtt  = cfg.get('quench_tower_top_temperature')
    rptl = cfg.get('c2_splitter_reflux_pump_suction_temp_limit')
    m['C2_splitter_reflux_pump_suction_temp_margin'] = (
        int(qtt < rptl) if (qtt is not None and rptl is not None) else 1
    )

    # ── 11. ERC_margin ────────────────────────────────────────────────────────
    erc  = cfg.get('erc_governor_opening')
    ercl = cfg.get('erc_governor_opening_limit')
    m['ERC_margin'] = (
        int(erc < ercl) if (erc is not None and ercl is not None) else 1
    )

    # ── 12. PRC_margin ────────────────────────────────────────────────────────
    prc  = cfg.get('prc_governor_opening')
    prcl = cfg.get('prc_governor_opening_limit')
    m['PRC_margin'] = (
        int(prc < prcl) if (prc is not None and prcl is not None) else 1
    )

    # ── Total_margin (FIX 5: sum all 11 margins, exclude Margin_in_Feed_lower_check) ──
    m['Total_margin'] = (
        m['Margin_in_Feed'] + m['Margin_in_Damper'] + m['Margin_in_FG']
        + m['Margin_in_Steam'] + m['Quench_OD_Gas_temp_margin']
        + m['CGC_suction_pressure_margin'] + m['C2_splitter_dp_margin']
        + m['C2_splitter_btm_c2h4_mol_percent_margin']
        + m['C2_splitter_reflux_pump_suction_temp_margin']
        + m['ERC_margin'] + m['PRC_margin']
    )

    return m


# =============================================================================
# Step 2 – External constraint + rank re-shuffle
# =============================================================================

def apply_external_constraints(furnace_df: pd.DataFrame) -> pd.DataFrame:
    df        = furnace_df.copy()
    active    = df['furnace_external_constraint'].fillna(0) == 0
    df['excluded_by_constraint'] = ~active
    active_df = df[active].copy().sort_values('overall_ranking')
    active_df['overall_ranking'] = range(1, len(active_df) + 1)
    df.update(active_df[['overall_ranking']])
    return df


# =============================================================================
# Step 3 – Furnace condition  (FIX 6: safe days_remaining NaN guard)
# =============================================================================

def compute_furnace_condition(row: pd.Series, margins: Dict, cfg: Dict) -> str:
    overall_ranking                     = _val(row.get('overall_ranking'))
    steam_water_deoke_status            = _val(row.get('steam_water_deoke_status'), 0)
    furnace_status                      = _val(row.get('furnace_status'), 0)
    cracking_cycle_runlength_calculated = _val(row.get('cracking_cycle_runlength_calculated'), 0)
    max_coke_thickness                  = _val(row.get('max_coke_thickness'), 0)
    percent_above_threshold             = _val(row.get('percent_above_threshold'), 0)
    max_coke_thickness_limit            = cfg['max_coke_thickness_limit']

    Margin_in_Feed   = margins['Margin_in_Feed']
    Margin_in_Damper = margins['Margin_in_Damper']
    Margin_in_FG     = margins['Margin_in_FG']

    if _is_missing(overall_ranking) or steam_water_deoke_status == 1:
        return 'No Optimization'

    if furnace_status == 1 and cracking_cycle_runlength_calculated > 1:
        if 1 < cracking_cycle_runlength_calculated <= 2:
            return 'SOR'

        # FIX 6: only trigger EOR on days_remaining if value is actually present
        days_remaining = row.get('days_remaining')
        days_eor = (
            not _is_missing(days_remaining) and float(days_remaining) < 2
        )
        coke_eor = (max_coke_thickness >= max_coke_thickness_limit)

        if days_eor or coke_eor:
            return 'EOR'

        margin_sum = Margin_in_Feed + Margin_in_Damper + Margin_in_FG
        if percent_above_threshold < -100 or margin_sum < 3:
            if Margin_in_Damper == 1 and Margin_in_FG == 1:
                return 'Semi Good'
            return 'Bad'
        return 'Good'

    return 'No cracking'


# =============================================================================
# Step 4 – Next decoke identification
# =============================================================================

def identify_next_decoke(furnace_df: pd.DataFrame) -> Optional[str]:
    cracking = furnace_df[furnace_df['furnace_status'].fillna(0) == 1].copy()
    if cracking.empty:
        return None
    cracking['days_remaining']     = pd.to_numeric(cracking['days_remaining'], errors='coerce')
    cracking['max_coke_thickness'] = pd.to_numeric(cracking['max_coke_thickness'], errors='coerce')
    min_days   = cracking['days_remaining'].min()
    candidates = cracking[cracking['days_remaining'] == min_days]
    return candidates.sort_values('max_coke_thickness', ascending=False).iloc[0]['entity_name']


# =============================================================================
# Step 5 – Coupling
# =============================================================================

def apply_coupling(furnace_df: pd.DataFrame) -> pd.DataFrame:
    df = furnace_df.copy()
    df['is_coupled'] = df['furnace_coupled_mode'].fillna(1) == 1
    return df


# =============================================================================
# Step 6 – Furnace counts  (FIX 12: count all by condition, no coupling filter)
# =============================================================================

def compute_furnace_counts(furnace_df: pd.DataFrame, cfg: Dict) -> Dict[str, Any]:
    count_good     = int(furnace_df['Furnace_condition'].isin(GOOD_CONDITIONS).sum())
    count_no_good  = int(furnace_df['Furnace_condition'].isin(NO_GOOD_CONDITIONS).sum())
    count_cracking = int(furnace_df['Furnace_condition'].isin(CRACKING_CONDITIONS).sum())
    min_check      = 1 if count_cracking >= int(cfg['minimum_fur_for_optimization_limit']) else 0
    return {
        'count_good':                               count_good,
        'count_no_good':                            count_no_good,
        'count_cracking':                           count_cracking,
        'total_available_for_bias':                 count_good + count_no_good,
        'minimum_cracking_furnace_available_check': min_check,
    }


# =============================================================================
# Step 7 – Additional constraints
# =============================================================================

def compute_additional_constraints(
    furnace_df: pd.DataFrame, cfg: Dict, counts: Dict
) -> Dict[str, Any]:

    if counts['minimum_cracking_furnace_available_check'] != 1:
        return {
            'biasing_condition':                   None,
            'Total_optimizer_run_check':           99,
            'Constraint_limit_bias2':              99,
            'Final_run_optimizer_check_init':      0,
            'current_recycle_ethane_feed_overall': None,
        }

    want_to_change   = int(_val(cfg.get('want_to_change_recycle_feed_flow'), 0))
    fresh_feed_chg   = float(_val(cfg.get('fresh_feed_change'), 0))
    aramco_feed      = float(_val(cfg.get('ethane_feed_flow_from_aramco'), 0))
    aramco_limit     = float(cfg['ethane_feed_flow_aramco_limit'])
    expected_ff      = float(_val(cfg.get('expected_fresh_feed'), 0))
    recycle_ctrl_op  = float(_val(cfg.get('recycle_ethane_flow_controller_opening'), 0))
    recycle_ctrl_lim = float(cfg['recycle_ethane_flow_controller_opening_limit'])

    recycle_ctrl_margin = (
        1 if fresh_feed_chg != 0 else int(recycle_ctrl_op < recycle_ctrl_lim)
    )
    recycle_change_possible = int(want_to_change == 1 and recycle_ctrl_margin == 1)

    if counts['count_good'] == 0:
        biasing_condition = 3
    elif recycle_change_possible == 1:
        biasing_condition = 1
    else:
        biasing_condition = 2

    # FIX 7: OR logic — if EITHER flag is true → hold (deviation_check = 0)
    deviation_in_aramco = 0.0   # filled by Module 4 (Past Hour Logic)
    flag_a = int(fresh_feed_chg == 0 and abs(deviation_in_aramco) > aramco_limit)
    flag_b = int(fresh_feed_chg != 0 and deviation_in_aramco < -1)
    fresh_feed_deviation_check = 0 if (flag_a == 1 or flag_b == 1) else 1

    total_optimizer_run_check = (
        1 if (counts['minimum_cracking_furnace_available_check'] == 1
              and fresh_feed_deviation_check == 1) else 99
    )
    min_fur_skip_check_bias2 = (
        0 if (biasing_condition == 2 and counts['total_available_for_bias'] == 1) else 1
    )

    # FIX 8: per-furnace recycle calculation, then sum
    cracking_df  = furnace_df[furnace_df['Furnace_condition'].isin(CRACKING_CONDITIONS)].copy()
    current_recycle = 0.0
    for _, frow in cracking_df.iterrows():
        feed = _val(frow.get('total_mixedfeed_flow'))
        shc  = _val(frow.get('shc_ratio'))
        oc   = _val(frow.get('overall_conversion'))
        if all(v is not None for v in [feed, shc, oc]):
            current_recycle += (float(feed) / (1 + float(shc))) * (100 - float(oc)) / 100

    # Constraint_limit_bias2
    if biasing_condition == 2:
        good_df    = furnace_df[furnace_df['Furnace_condition'].isin(GOOD_CONDITIONS)]
        no_good_df = furnace_df[furnace_df['Furnace_condition'].isin(NO_GOOD_CONDITIONS)]
        good_pos   = good_df[
            pd.to_numeric(good_df['percent_above_threshold'], errors='coerce') > 0
        ]
        if good_pos.empty:
            constraint_limit_bias2 = 1
        else:
            sum_min_conv = pd.to_numeric(
                no_good_df['overall_conversion'], errors='coerce'
            ).sum()
            constraint_limit_bias2 = 0 if sum_min_conv == 0 else 1
    else:
        constraint_limit_bias2 = 1

    final_check = int(
        total_optimizer_run_check == 1
        and counts['total_available_for_bias'] > 0
        and min_fur_skip_check_bias2 == 1
        and constraint_limit_bias2 == 1
    )

    return {
        'biasing_condition':                   biasing_condition,
        'Total_optimizer_run_check':           total_optimizer_run_check,
        'Constraint_limit_bias2':              constraint_limit_bias2,
        'Final_run_optimizer_check_init':      final_check,
        'current_recycle_ethane_feed_overall': current_recycle,
    }


# =============================================================================
# Forecasted runlength rank  (FIX 10: compress overall_ranking sequentially)
# =============================================================================

def compute_forecasted_runlength_ranks(furnace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the relative order of overall_ranking but re-numbers sequentially
    after removing constrained/decoupled furnaces — does NOT re-rank from scratch
    by days_remaining.
    """
    df = furnace_df.copy()

    # Only cracking furnaces participate in runlength ranking
    cracking = df[df['furnace_status'].fillna(0) == 1].copy()
    cracking['overall_ranking'] = pd.to_numeric(cracking['overall_ranking'], errors='coerce')
    cracking = cracking.sort_values('overall_ranking', ascending=True)

    # Compress: re-number 1…N preserving existing relative order
    cracking['Forecasted_runlength_rank_org'] = range(1, len(cracking) + 1)
    cracking['Forecasted_runlength_rank']     = range(1, len(cracking) + 1)

    df = df.merge(
        cracking[['entity_name', 'Forecasted_runlength_rank_org',
                  'Forecasted_runlength_rank']],
        on='entity_name', how='left'
    )
    return df


# =============================================================================
# pass{n}_feed_red_potential_on_min_days  (FIX 9: binary flag)
# =============================================================================

def compute_pass_feed_red_potential(furnace_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each furnace, identifies which pass has the MINIMUM runlength_remaining
    and sets that pass's flag to 1, all others to 0.
    If multiple passes tie on minimum, all tied passes get 1.
    """
    df = furnace_df.copy()
    for p in PASS_NUMBERS:
        df[f'pass{p}_feed_red_potential_on_min_days'] = 0

    for idx, row in df.iterrows():
        rl_values = {}
        for p in PASS_NUMBERS:
            v = _val(row.get(f'pass{p}_runlength_remaining'))
            if v is not None:
                rl_values[p] = float(v)

        if not rl_values:
            continue

        min_rl = min(rl_values.values())
        for p, v in rl_values.items():
            df.at[idx, f'pass{p}_feed_red_potential_on_min_days'] = int(v == min_rl)

    return df


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def run_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all preprocessing steps and return two DataFrames.

    Returns
    -------
    (all_furnaces_df, eligible_furnaces_df)

    all_furnaces_df:
        10 rows (F1-F9 + FS), original order
        220 original cols + 18 new computed cols

    eligible_furnaces_df:
        Valid-ranked furnaces only, sorted by overall_ranking ascending
        220 original cols + 26 new computed cols (incl. pass{n} potentials)
    """

    # ── Split FS / furnace rows ───────────────────────────────────────────────
    fs_mask  = df['entity_name'] == 'FS'
    if fs_mask.sum() == 0:
        raise ValueError("No 'FS' row found in input DataFrame.")
    fs_row     = df[fs_mask].iloc[0]
    furnace_df = df[~fs_mask].copy().reset_index(drop=True)

    # ── Step 1A ───────────────────────────────────────────────────────────────
    cfg = extract_system_config(fs_row)

    # ── Step 1B ───────────────────────────────────────────────────────────────
    margin_records = [compute_margins(row, cfg) for _, row in furnace_df.iterrows()]
    margin_df      = pd.DataFrame(margin_records, index=furnace_df.index)
    furnace_df     = pd.concat([furnace_df, margin_df], axis=1)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    furnace_df = apply_external_constraints(furnace_df)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    conditions = []
    for idx, frow in furnace_df.iterrows():
        m    = {c: furnace_df.at[idx, c] for c in margin_df.columns}
        cond = compute_furnace_condition(frow, m, cfg)
        conditions.append(cond)
    furnace_df['Furnace_condition']      = conditions
    furnace_df['Furnace_condition_code'] = furnace_df['Furnace_condition'].map(CONDITION_CODE_MAP)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    next_decoke = identify_next_decoke(furnace_df)
    furnace_df['is_next_decoke'] = furnace_df['entity_name'] == next_decoke

    # ── Step 5 ────────────────────────────────────────────────────────────────
    furnace_df = apply_coupling(furnace_df)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    counts = compute_furnace_counts(furnace_df, cfg)

    # ── Step 7 ────────────────────────────────────────────────────────────────
    constraints = compute_additional_constraints(furnace_df, cfg, counts)

    # ── Forecasted runlength rank ─────────────────────────────────────────────
    furnace_df = compute_forecasted_runlength_ranks(furnace_df)

    # ── Feed_flow (= total_mixedfeed_flow) ────────────────────────────────────
    furnace_df['Feed_flow'] = pd.to_numeric(
        furnace_df['total_mixedfeed_flow'], errors='coerce'
    )

    # ── Missing input columns → NaN ───────────────────────────────────────────
    for col in ['benefit_percent_threshold', 'furnace_step_adjust_feed_grid_limit',
                'net_nox_emission', 'nox_emission_permissible_limit', 'nox_margin']:
        if col not in furnace_df.columns:
            furnace_df[col] = np.nan

    # ── Broadcast system-level flags ──────────────────────────────────────────
    furnace_df['minimum_cracking_furnace_available_check'] = counts['minimum_cracking_furnace_available_check']
    furnace_df['biasing_condition']               = constraints.get('biasing_condition')
    furnace_df['Total_optimizer_run_check']       = constraints.get('Total_optimizer_run_check')
    furnace_df['Constraint_limit_bias2']          = constraints.get('Constraint_limit_bias2')
    furnace_df['Final_run_optimizer_check_init']  = constraints.get('Final_run_optimizer_check_init')
    furnace_df['current_recycle_ethane_feed_overall'] = constraints.get('current_recycle_ethane_feed_overall')
    furnace_df['next_decoke_furnace']             = next_decoke

    # ── Add FS row back ───────────────────────────────────────────────────────
    fs_df = df[fs_mask].copy()
    furnace_only_cols = [c for c in furnace_df.columns if c not in df.columns]
    for col in furnace_only_cols:
        fs_df[col] = np.nan
    for col, val in {
        'minimum_cracking_furnace_available_check': counts['minimum_cracking_furnace_available_check'],
        'biasing_condition':               constraints.get('biasing_condition'),
        'Total_optimizer_run_check':       constraints.get('Total_optimizer_run_check'),
        'Constraint_limit_bias2':          constraints.get('Constraint_limit_bias2'),
        'Final_run_optimizer_check_init':  constraints.get('Final_run_optimizer_check_init'),
        'current_recycle_ethane_feed_overall': constraints.get('current_recycle_ethane_feed_overall'),
        'next_decoke_furnace':             next_decoke,
    }.items():
        fs_df[col] = val

    # ── Define ordered computed columns ───────────────────────────────────────
    computed_cols_all = [
        'Feed_flow',
        'Margin_in_FG', 'Margin_in_Steam', 'Margin_in_Feed', 'Margin_in_Damper',
        'Quench_OD_Gas_temp_margin', 'CGC_suction_pressure_margin',
        'C2_splitter_dp_margin', 'C2_splitter_btm_c2h4_mol_percent_margin',
        'C2_splitter_reflux_pump_suction_temp_margin',
        'ERC_margin', 'PRC_margin', 'Total_margin', 'Margin_in_Feed_lower_check',
        'Furnace_condition', 'Furnace_condition_code',
        'Forecasted_runlength_rank_org', 'Forecasted_runlength_rank',
    ]
    pass_potential_cols = [f'pass{p}_feed_red_potential_on_min_days' for p in PASS_NUMBERS]
    original_cols       = list(df.columns)

    # ── OUTPUT 1: all_furnaces_df ─────────────────────────────────────────────
    all_df          = pd.concat([furnace_df, fs_df], ignore_index=True)
    all_furnaces_df = all_df[original_cols + computed_cols_all].copy()

    # ── OUTPUT 2: eligible_furnaces_df ────────────────────────────────────────
    ELIGIBLE_CONDITIONS = {'Good', 'Bad', 'Semi Good', 'SOR'}
    eligible_raw = furnace_df[
        pd.to_numeric(furnace_df['overall_ranking'], errors='coerce').notna() &
        furnace_df['Furnace_condition'].isin(ELIGIBLE_CONDITIONS)
    ].copy().reset_index(drop=True)
    
    eligible_raw             = compute_pass_feed_red_potential(eligible_raw)
    eligible_furnaces_df     = eligible_raw[original_cols + computed_cols_all + pass_potential_cols].copy()
    eligible_furnaces_df     = eligible_furnaces_df.sort_values(
        'overall_ranking', ascending=True
    ).reset_index(drop=True)

    return all_furnaces_df, eligible_furnaces_df


# =============================================================================
# Run on actual data
# =============================================================================

if __name__ == '__main__':
    FILE  = r"C:\Users\User\Downloads\Ranking_req_info.xlsx"
    SHEET = 'tags in parameter format'

    df = pd.read_excel(FILE, sheet_name=SHEET)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 280)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')

    all_results_all, all_results_eligible = [], []
    for ts, group in df.groupby('Timestamp'):
        a, e = run_preprocessing(group)
        all_results_all.append(a)
        all_results_eligible.append(e)

    final_all      = pd.concat(all_results_all,      ignore_index=True)
    final_eligible = pd.concat(all_results_eligible, ignore_index=True)

    # ── Display key columns ───────────────────────────────────────────────────
    preview = [
        'overall_ranking', 'entity_name', 'Furnace_condition', 'Furnace_condition_code',
        'Feed_flow', 'Margin_in_Feed', 'Margin_in_Damper', 'Margin_in_FG',
        'Margin_in_Steam', 'Margin_in_Feed_lower_check', 'Total_margin',
        'Quench_OD_Gas_temp_margin', 'CGC_suction_pressure_margin',
        'C2_splitter_dp_margin', 'C2_splitter_btm_c2h4_mol_percent_margin',
        'C2_splitter_reflux_pump_suction_temp_margin', 'ERC_margin', 'PRC_margin',
        'Forecasted_runlength_rank_org', 'Forecasted_runlength_rank',
    ]

    print('\n' + '='*120)
    print('  OUTPUT 1: all_furnaces_df')
    print('='*120)
    print(final_all[preview].to_string(index=False))

    print('\n' + '='*120)
    print('  OUTPUT 2: eligible_furnaces_df  (sorted by overall_ranking)')
    print('='*120)
    elig_preview = preview + [f'pass{p}_feed_red_potential_on_min_days' for p in PASS_NUMBERS]
    print(final_eligible[elig_preview].to_string(index=False))

    # ── Shape summary ─────────────────────────────────────────────────────────
    print('\n' + '='*120)
    print('  SHAPE SUMMARY')
    print('='*120)
    orig = len(df.columns)
    print(f'\n  OUTPUT 1 — all_furnaces_df     : {final_all.shape[0]} rows × {final_all.shape[1]} cols  ({orig} original + {final_all.shape[1]-orig} new)')
    print(f'  OUTPUT 2 — eligible_furnaces_df: {final_eligible.shape[0]} rows × {final_eligible.shape[1]} cols  ({orig} original + {final_eligible.shape[1]-orig} new)')   

    final_all.to_csv(r'C:\Users\User\Documents\POC\FURNACE_PRODUCT\module3_v5_all_furnaces.csv', index=False)
    final_eligible.to_csv(r'C:\Users\User\Documents\POC\FURNACE_PRODUCT\module3_v5_eligible_furnaces.csv', index=False)
    print('\n  Saved:')
    # print('    /mnt/user-data/outputs/module3_v5_all_furnaces.csv')
    # print('    /mnt/user-data/outputs/module3_v5_eligible_furnaces.csv')
