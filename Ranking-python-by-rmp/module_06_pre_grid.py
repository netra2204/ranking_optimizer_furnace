"""
module_06_pre_grid.py
=====================
Replicates the "Pre Grid" subprocess (child of "main" → "Ranking optimisation main").

Full RMP operator chain replicated:
  Generate Attributes (906)   – pre calcs incl. feed_reduction_potential (initial)
  Number_of_rows (11)         – extract row count macro
  Extract Macro (434)         – extract shc_ratio macro
  Filter Examples (346)       – split Good vs unmatched (non-Good)
  ├─ GOOD path:
  │    Generate Attributes (833)  – Margin_condition_type, upper_limit_feed,
  │                                  max_potential_total_feed, Furnace_condition2, step_size_feed
  │    exclude (79)               – drop count_pass_mixed_feed, Margin_condition_type,
  │                                  max_potential_total_feed, lower_limit_feed_org
  │    Good (19) aggregate        – sum(upper_limit_feed)
  │    sum_upper_limit_feed (14)  – extract macro
  └─ UNMATCHED (non-Good) path:
       Generate Attributes (177)  – Furnace_condition2="No Good", step_size_feed=0
       No good (7) aggregate      – sum(feed_reduction_potential)
       sum_feed_reduction_potential (19) – extract macro

  Branch (112) [biasing_condition != 2]:
    THEN path (biasing_condition != 2):
      Append (70)              – merge Good + No-Good into one df
      Branch (18) [fresh_feed_change == -1]:
        THEN: Generate Attributes (29)  – upper=0, lower=-frp, step=1
        ELSE: Generate Attributes (324) – step_size_feed = if(ulf>1,2,ulf)
      exclude (55)             – drop Furnace_condition2
      Subprocess (94):
        Sort (87) desc factor + Generate Attributes (38) frp=0  → Append (54)
        Sort (88) asc factor                                     → Append (54)
        Generate ID (16)
        Generate Macro (79): iteration=0, balance_feed=sum_frp
        Loop (While) (33): allocate reduction to no-good furnaces row by row
        Append (73): merge loop output
        Filter Examples (183): taken==3
        Branch (129) [min_examples>=1]:
          THEN: Feed-adjustment sub-logic (Branch 130, Sort 89/90/91/92,
                Append 74/75, Extract Macro 82/7, Generate Attributes 264/267)
        Generate Attributes (325): upper_limit_feed_org=ulf, upper_limit_feed=taken
        Good (18) aggregate: sum(upper_limit_feed)
        sum_feed_reduction_potential (18): extract macro → sum_upper_limit_feed
        Generate Macro (145): iteration=0, balance=sum_upper_limit_feed
        Loop (While) (35): distribute upper budget to no-good furnaces
        Append (81): merge loop output
        Generate Attributes (327): lower_limit_feed = -given
        Select Attributes (109): drop temp cols
      Sort (93): overall_ranking asc
      Select Attributes (117): drop factor, feed_reduction_potential, Furnace_condition2, id
      Generate Attributes (328): final lower/upper/step_size_feed
      exclude (74): drop lower_limit_feed_org, Furnace_condition2

    ELSE path (biasing_condition == 2):
      Subprocess (94) is skipped → df passes through unchanged

  Aggregate (77): sum(upper_limit_feed) after Branch (112)
  sum_upper_limit_feed (23): extract macro
  Generate Macro (149): mixed_feed_margin, Extra_Recycle_Ethane, recycle UL/LL
  Generate Attributes (329): stamp Extra_Recycle_Ethane on df rows

  Branch (19) [fresh_feed_change != -1]:
    THEN: Generate Attributes (330) – update lower_limit_feed and step_size_feed
          for biasing_condition==1 when mixed_feed_margin==sum_upper_limit_feed
    ELSE: pass-through

  Branch (66) [biasing_condition == 3]:
    THEN: Generate Attributes (1039) – conversion limits + step_size_conversion
          + New_Feed_flow = Feed_flow
    ELSE: Generate Attributes (43)   – conversion limits (different UL formula)
          Branch (70) [fresh_feed_change == 0]:
            THEN: Subprocess (14) – update mixed_feed_margin from sum_New_mixed_feed_margin
          Branch (biasing_condition == 1):
            THEN: Sort (2) + Generate ID + Loop (While)(36) + biasing_condition=1
                  feed limit update via taken/id_for_balance logic

  Loop (48) 9 iters: reset Row_N_* macros
  Sort (106): overall_ranking asc
  extract row loop (11): extract Row_N_* macros from sorted df
"""

import math
import logging
import pandas as pd
import numpy as np

from config import MACROS, STORE

logger = logging.getLogger(__name__)

MAX_FURNACES = 9

# ── helpers ──────────────────────────────────────────────────────────────────

def _m(key, default=0):
    val = MACROS.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _remember(name, df):
    STORE[name] = df.copy()


def _recall(name):
    return STORE.get(name, pd.DataFrame())


def _col(df, name, default=0.0):
    return df[name].fillna(default).astype(float) if name in df.columns else pd.Series(default, index=df.index)


# =============================================================================
# Generate Attributes (906) – pre calcs
# =============================================================================
def compute_initial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors Generate Attributes (906): initialise grid limit cols,
    Ethane_Feed, Current_Recycle_Ethane_Feed, factor, and the INITIAL
    feed_reduction_potential formula (before the cap is applied).
    """
    df = df.copy()

    shc = _col(df, "shc_ratio")

    # lower/upper/flag/step initialised to 0
    df["lower_limit_feed"]             = 0.0
    df["upper_limit_feed"]             = 0.0
    df["flag_conversion_part_override"] = 0
    df["step_size_conversion"]          = 0.0

    # Ethane_Feed = Feed_flow / (1 + shc_ratio)
    feed = _col(df, "Feed_flow")
    df["Ethane_Feed"] = feed / (1.0 + shc)

    # Current_Recycle_Ethane_Feed
    conv = _col(df, "Overall_conversion")
    df["Current_Recycle_Ethane_Feed"] = df["Ethane_Feed"] * (100.0 - conv) / 100.0

    # factor
    pat = _col(df, "percent_above_threshold")
    df["factor"] = np.where(
        pat > 0,
        1.0 / conv.replace(0, np.nan).fillna(1),
        -1.0 * conv
    )

    # ── feed_reduction_potential INITIAL formula (Generate Attributes 906) ────
    # floor(min(4, pass_step_change * sum(
    #   floor(max(0, passN_mixedfeed_flow - pass_feed_min_limit) / pass_step_change)
    #   * passN_feed_red_potential_on_min_days)))
    pass_step   = float(_m("pass_step_change", 1))
    pass_min    = float(_m("pass_feed_min_limit", 0))

    def _frp_row(row):
        total = 0.0
        for n in range(1, 9):
            flow_col  = f"pass{n}_mixedfeed_flow"
            pot_col   = f"pass{n}_feed_red_potential_on_min_days"
            flow = float(row.get(flow_col, 0) or 0)
            pot  = float(row.get(pot_col,  0) or 0)
            steps = math.floor(max(0.0, flow - pass_min) / pass_step) if pass_step != 0 else 0
            total += steps * pot
        return math.floor(min(4.0, pass_step * total))

    df["feed_reduction_potential"] = df.apply(_frp_row, axis=1).astype(float)

    # cap: if frp == 3 → 2  (second Generate Attributes (906) row)
    df["feed_reduction_potential"] = df["feed_reduction_potential"].apply(
        lambda v: 2.0 if v == 3.0 else v
    )

    logger.info("compute_initial_columns done.")
    return df


# =============================================================================
# Number_of_rows (11) + Extract Macro (434)
# =============================================================================
def extract_row_count_and_shc(df: pd.DataFrame):
    MACROS["Number_of_rows"] = len(df)
    if "shc_ratio" in df.columns and len(df) > 0:
        MACROS["shc_ratio"] = float(df.iloc[0]["shc_ratio"])
    logger.info("Number_of_rows=%d  shc_ratio=%.4f",
                int(MACROS["Number_of_rows"]), _m("shc_ratio"))


# =============================================================================
# Filter Examples (346) → Good path: Generate Attributes (833) + exclude (79)
#                       → Unmatched path: Generate Attributes (177)
# =============================================================================
def split_good_nongood_and_compute_limits(df: pd.DataFrame):
    """
    Returns (df_good_processed, df_nongood_processed).
    df_good_processed  has Margin_condition_type, upper_limit_feed,
                           max_potential_total_feed, Furnace_condition2, step_size_feed
                           with temp cols dropped (exclude 79).
    df_nongood_processed has Furnace_condition2="No Good", step_size_feed=0.
    """
    if "Furnace_condition" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)

    df_good    = df[df["Furnace_condition"] == "Good"].copy()
    df_nongood = df[df["Furnace_condition"] != "Good"].copy()

    # ── GOOD path: Generate Attributes (833) ─────────────────────────────────
    if not df_good.empty:
        margin_feed       = _col(df_good, "Margin_in_Feed")
        margin_lower      = _col(df_good, "Margin_in_Feed_lower_check")
        saturator         = _col(df_good, "saturator_margin")
        pass_step         = float(_m("pass_step_change", 1))
        thresh_pass       = float(_m("threshold_pass_mixed_feed_limit", 0))

        # Margin_condition_type
        df_good["Margin_condition_type"] = np.where(
            (margin_feed == 1) & (margin_lower == 0), 0,
            np.where((margin_feed == 1) & (margin_lower == 1), 1, 1000)
        )

        # upper_limit_feed (initial before cap)
        mct = df_good["Margin_condition_type"]
        df_good["upper_limit_feed"] = np.where(
            saturator == 1,
            np.where(mct == 1, 4, np.where(mct == 0, 2, 0)),
            np.where(mct == 1, 2, np.where(mct == 0, 1, 0))
        ).astype(float)

        # max_potential_total_feed (initial)
        # floor(min(4, pass_step * sum over passes of
        #   floor(max(0, threshold - passN_flow) / pass_step) * upper_feed_condition_passN))
        def _max_pot_row(row):
            total = 0.0
            for n in range(1, 9):
                flow_col = f"pass{n}_mixedfeed_flow"
                cond_col = f"upper_feed_condition_pass{n}"
                flow = float(row.get(flow_col, 0) or 0)
                cond = float(row.get(cond_col,  0) or 0)
                steps = math.floor(max(0.0, thresh_pass - flow) / pass_step) if pass_step != 0 else 0
                total += steps * cond
            return math.floor(min(4.0, pass_step * total))

        df_good["max_potential_total_feed"] = df_good.apply(_max_pot_row, axis=1).astype(float)

        # cap max_potential_total_feed: if ==3 → 2
        df_good["max_potential_total_feed"] = df_good["max_potential_total_feed"].apply(
            lambda v: 2.0 if v == 3.0 else v
        )

        # upper_limit_feed = min(upper_limit_feed, max_potential_total_feed)
        df_good["upper_limit_feed"] = np.minimum(
            df_good["upper_limit_feed"], df_good["max_potential_total_feed"]
        )

        # step_size_feed
        df_good["step_size_feed"] = np.where(df_good["upper_limit_feed"] > 1, 2.0, df_good["upper_limit_feed"])

        # Furnace_condition2 = Furnace_condition
        df_good["Furnace_condition2"] = df_good["Furnace_condition"]

        # exclude (79): drop count_pass_mixed_feed, Margin_condition_type,
        #               max_potential_total_feed, lower_limit_feed_org
        drop_cols = ["count_pass_mixed_feed", "Margin_condition_type",
                     "max_potential_total_feed", "lower_limit_feed_org"]
        df_good.drop(columns=[c for c in drop_cols if c in df_good.columns], inplace=True)

    # ── UNMATCHED path: Generate Attributes (177) ─────────────────────────────
    if not df_nongood.empty:
        df_nongood["Furnace_condition2"] = "No Good"
        df_nongood["step_size_feed"]     = 0.0

    return df_good, df_nongood


# =============================================================================
# Good (19) aggregate + sum_upper_limit_feed (14)
# No good (7) aggregate + sum_feed_reduction_potential (19)
# =============================================================================
def aggregate_and_extract_sums(df_good: pd.DataFrame, df_nongood: pd.DataFrame):
    sum_ulf = float(df_good["upper_limit_feed"].sum()) if "upper_limit_feed" in df_good.columns else 0.0
    sum_frp = float(df_nongood["feed_reduction_potential"].sum()) if "feed_reduction_potential" in df_nongood.columns else 0.0
    MACROS["sum_upper_limit_feed"]        = sum_ulf
    MACROS["sum_feed_reduction_potential"] = sum_frp
    logger.info("sum_upper_limit_feed=%.2f  sum_feed_reduction_potential=%.2f", sum_ulf, sum_frp)


# =============================================================================
# Branch (112): biasing_condition != 2
# =============================================================================
def branch_112(df_good: pd.DataFrame, df_nongood: pd.DataFrame) -> pd.DataFrame:
    biasing = int(_m("biasing_condition", 0))

    if biasing != 2:
        # ── THEN path ─────────────────────────────────────────────────────────
        # Append (70): merge good + no-good
        df_all = pd.concat([df_good, df_nongood], ignore_index=True)

        # Branch (18): fresh_feed_change == -1
        fresh_feed_change = int(_m("fresh_feed_change", 0))
        if fresh_feed_change == -1:
            # Generate Attributes (29)
            frp = _col(df_all, "feed_reduction_potential")
            df_all["upper_limit_feed"] = 0.0
            df_all["lower_limit_feed"] = -frp
            df_all["step_size_feed"]   = 1.0
        else:
            # Generate Attributes (324)
            ulf = _col(df_all, "upper_limit_feed")
            df_all["step_size_feed"] = np.where(ulf > 1, 2.0, ulf)

        # exclude (55): drop Furnace_condition2, lower_limit_feed_org
        drop55 = ["Furnace_condition2", "lower_limit_feed_org"]
        df_all.drop(columns=[c for c in drop55 if c in df_all.columns], inplace=True)

        return df_all

    else:
        # ── ELSE path ─────────────────────────────────────────────────────────
        # Subprocess (94) takes Good df and No-Good df as separate inputs
        df_all = subprocess_94(df_good, df_nongood)

        # Sort (93): overall_ranking asc
        if "overall_ranking" in df_all.columns:
            df_all = df_all.sort_values("overall_ranking").reset_index(drop=True)

        # Select Attributes (117): drop temp cols
        drop117 = ["factor", "feed_reduction_potential", "feed_reduction_potential_1",
                   "Furnace_condition2", "id"]
        df_all.drop(columns=[c for c in drop117 if c in df_all.columns], inplace=True)

        # Generate Attributes (328): final lower/upper/step
        df_all = generate_attrs_328(df_all)

        # exclude (74): drop lower_limit_feed_org, Furnace_condition2
        drop74 = ["lower_limit_feed_org", "Furnace_condition2"]
        df_all.drop(columns=[c for c in drop74 if c in df_all.columns], inplace=True)

        return df_all

# =============================================================================
# Subprocess (94) – balance feed allocation
# =============================================================================
def subprocess_94(df_good: pd.DataFrame, df_nongood: pd.DataFrame) -> pd.DataFrame:
    # Good gets feed_reduction_potential = 0 (Generate Attributes 38)
    df_good = df_good.copy()
    df_good["feed_reduction_potential"] = 0.0
    
    # Sort good desc factor, nongood asc factor → Append(54) → Generate ID
    df_good    = df_good.sort_values("factor", ascending=False).reset_index(drop=True) \
                 if "factor" in df_good.columns else df_good.reset_index(drop=True)
    df_nongood = df_nongood.sort_values("factor", ascending=True).reset_index(drop=True) \
                 if "factor" in df_nongood.columns else df_nongood.reset_index(drop=True)

    df_merged = pd.concat([df_good, df_nongood], ignore_index=True)
    df_merged["id"] = range(len(df_merged))
    total_fur = int(_m("total_fur_available_for_bias", len(df_merged)))

    # ── Loop (While) (33): allocate reduction potential row by row ────────────
    # stop when iteration == total_fur_available_for_bias
    balance_feed = float(_m("sum_feed_reduction_potential", 0))
    df_merged["taken"]        = 0.0
    df_merged["balance_taken"] = balance_feed

    for i in range(len(df_merged)):
        if i >= total_fur:
            break
        ulf = float(df_merged.loc[i, "upper_limit_feed"]) \
              if "upper_limit_feed" in df_merged.columns else 0.0
        taken = min(balance_feed, ulf)
        df_merged.loc[i, "taken"]         = taken
        df_merged.loc[i, "balance_taken"] = balance_feed - taken
        balance_feed -= taken
        MACROS["balance_feed"] = balance_feed

    # Append (73): loop output already in df_merged (single df, loop updates in place)

    # ── Filter Examples (183): taken == 3 ────────────────────────────────────
    df_taken3 = df_merged[df_merged["taken"] == 3.0].copy()

    # ── Branch (129): min_examples >= 1 (i.e. any taken==3 rows exist) ───────
    if len(df_taken3) >= 1:
        df_merged = branch_129(df_merged, df_taken3, total_fur)

    # ── Generate Attributes (325): upper_limit_feed_org = ulf, ulf = taken ───
    df_merged["upper_limit_feed_org"] = df_merged["upper_limit_feed"].copy()
    df_merged["upper_limit_feed"]     = df_merged["taken"]

    # ── Good (18) aggregate: sum(upper_limit_feed) → sum_upper_limit_feed ────
    sum_ulf2 = float(df_merged["upper_limit_feed"].sum())
    MACROS["sum_upper_limit_feed"] = sum_ulf2

    # ── Loop (While) (35): distribute upper_limit budget to no-good furnaces ──
    balance2 = sum_ulf2
    df_merged["given"]        = 0.0
    df_merged["balance_given"] = balance2

    for i in range(len(df_merged)):
        if i >= total_fur:
            break
        frp = float(df_merged.loc[i, "feed_reduction_potential"]) \
              if "feed_reduction_potential" in df_merged.columns else 0.0
        given = min(balance2, frp)
        df_merged.loc[i, "given"]         = given
        df_merged.loc[i, "balance_given"] = balance2 - given
        balance2 -= given
        MACROS["balance_feed"] = balance2

    # ── Generate Attributes (327): lower_limit_feed = -given ─────────────────
    df_merged["lower_limit_feed"] = -df_merged["given"]

    # ── Select Attributes (109): drop temp cols ───────────────────────────────
    drop109 = ["Furnace_condition2", "feed_reduction_potential", "factor",
               "taken", "balance_taken", "upper_limit_feed_org",
               "given", "balance_given"]
    df_merged.drop(columns=[c for c in drop109 if c in df_merged.columns], inplace=True)

    return df_merged


# =============================================================================
# Branch (129) / Branch (130) – fix taken==3 furnaces
# =============================================================================
def branch_129(df: pd.DataFrame, df_taken3: pd.DataFrame, total_fur: int) -> pd.DataFrame:
    """
    If any row has taken==3:
      Filter to Furnace_condition2==Good & upper_limit_feed==1
      Branch (130) [min_examples>=1]:
        THEN: sort asc factor, get id_upper_1, zero that good furnace's ulf,
              sort desc, re-append with no-good having frp==1 also adjusted.
        ELSE: pass-through
    Then re-run Loop (While) (34) with updated sum_feed_reduction_potential.
    """
    df = df.copy()

    # Filter (184): Furnace_condition2 == Good
    # NOTE: Furnace_condition2 may already be dropped; fall back to Furnace_condition
    cond_col = "Furnace_condition2" if "Furnace_condition2" in df.columns else "Furnace_condition"
    df_good_sub    = df[df.get(cond_col, pd.Series(dtype=str)) == "Good"].copy() \
                     if cond_col in df.columns else pd.DataFrame(columns=df.columns)
    df_nongood_sub = df[df.get(cond_col, pd.Series(dtype=str)) != "Good"].copy() \
                     if cond_col in df.columns else df.copy()

    # Filter (240): upper_limit_feed == 1
    df_ulf1 = df_good_sub[df_good_sub["upper_limit_feed"] == 1.0].copy() \
              if "upper_limit_feed" in df_good_sub.columns else pd.DataFrame(columns=df.columns)

    # Branch (130): min_examples >= 1
    if len(df_ulf1) >= 1:
        # THEN path
        # Sort (89) asc factor → Extract Macro (82): id_upper_1
        df_ulf1_sorted = df_ulf1.sort_values("factor") if "factor" in df_ulf1.columns else df_ulf1
        id_upper_1 = int(df_ulf1_sorted.iloc[0]["id"]) if "id" in df_ulf1_sorted.columns else -1
        MACROS["id_upper_1"] = id_upper_1

        # Generate Attributes (264): zero upper_limit_feed for that id
        df_good_sub.loc[df_good_sub["id"] == id_upper_1, "upper_limit_feed"] = 0.0

        # Sort (90) desc factor + Append (74) with nongood
        df_good_sub = df_good_sub.sort_values("factor", ascending=False) \
                      if "factor" in df_good_sub.columns else df_good_sub

    else:
        # ELSE path for no-good: Filter (242) frp==1 → sort desc → get id_feed_reduce_1
        df_frp1 = df_nongood_sub[df_nongood_sub["feed_reduction_potential"] == 1.0].copy() \
                  if "feed_reduction_potential" in df_nongood_sub.columns else pd.DataFrame(columns=df.columns)
        if not df_frp1.empty:
            df_frp1_sorted = df_frp1.sort_values("factor", ascending=False) \
                             if "factor" in df_frp1.columns else df_frp1
            id_frp1 = int(df_frp1_sorted.iloc[0]["id"]) if "id" in df_frp1_sorted.columns else -1
            MACROS["id_feed_reduce_1"] = id_frp1
            # Generate Attributes (267): zero frp for that id
            df_nongood_sub.loc[df_nongood_sub["id"] == id_frp1, "feed_reduction_potential"] = 0.0
            df_nongood_sub = df_nongood_sub.sort_values("factor", ascending=True) \
                             if "factor" in df_nongood_sub.columns else df_nongood_sub

    # Re-merge
    df = pd.concat([df_good_sub, df_nongood_sub], ignore_index=True)

    # Good (17) aggregate → new sum_feed_reduction_potential
    # NOTE: in RMP this sums feed_reduction_potential on good furnaces (which is 0)
    # to reset balance; effectively sum = 0 for good, full for no-good
    new_sum_frp = float(df["feed_reduction_potential"].sum()) \
                  if "feed_reduction_potential" in df.columns else 0.0
    MACROS["sum_feed_reduction_potential"] = new_sum_frp

    # Loop (While) (34): re-allocate with updated frp sum
    balance = new_sum_frp
    for i in range(len(df)):
        if i >= total_fur:
            break
        ulf = float(df.loc[i, "upper_limit_feed"]) if "upper_limit_feed" in df.columns else 0.0
        taken = min(balance, ulf)
        df.loc[i, "taken"]         = taken
        df.loc[i, "balance_taken"] = balance - taken
        balance -= taken
        MACROS["balance_feed"] = balance

    return df


# =============================================================================
# Generate Attributes (328) – final lower/upper/step_size_feed
# =============================================================================
def generate_attrs_328(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ulf = _col(df, "upper_limit_feed")
    llf = _col(df, "lower_limit_feed")

    df["lower_limit_feed_org"] = llf.copy()
    llf_org = df["lower_limit_feed_org"]

    df["lower_limit_feed"] = np.where(
        ulf == llf_org, 0.0,
        np.where(ulf != 0, ulf, llf)
    )
    df["upper_limit_feed"] = np.where(
        ulf == llf_org, 0.0,
        np.where(llf < 0, llf, ulf)
    )
    df["step_size_feed"] = np.where(
        (df["lower_limit_feed"] + df["upper_limit_feed"]) == 0, 0.0, 1.0
    )
    return df


# =============================================================================
# Aggregate (77) + sum_upper_limit_feed (23)
# =============================================================================
def aggregate_post_branch112(df: pd.DataFrame):
    sum_ulf = float(df["upper_limit_feed"].sum()) if "upper_limit_feed" in df.columns else 0.0
    MACROS["sum_upper_limit_feed"] = sum_ulf
    logger.info("Post-Branch112 sum_upper_limit_feed=%.2f", sum_ulf)


# =============================================================================
# Generate Macro (149) – mixed_feed_margin, Extra_Recycle_Ethane, recycle UL/LL
# =============================================================================
def compute_recycle_ethane_macros():
    fresh_feed_change = int(_m("fresh_feed_change", 0))
    sum_ulf           = _m("sum_upper_limit_feed", 0)
    good_fur_count    = _m("count_of_good_fur", 0)
    shc               = _m("shc_ratio", 0)
    ff_qty            = _m("fresh_feed_quantity", 0)

    if fresh_feed_change != 0:
        mixed_feed_margin = round(sum_ulf)
    else:
        mixed_feed_margin = round(min(sum_ulf, good_fur_count * 2))

    MACROS["mixed_feed_margin"] = mixed_feed_margin

    denom = 1.0 + shc
    extra_re = (mixed_feed_margin / denom - ff_qty) if denom != 0 else 0.0
    MACROS["Extra_Recycle_Ethane"] = extra_re

    re_upper = extra_re + _m("change_recycle_ethane_upper_limit", 0)
    re_lower = extra_re + _m("change_recycle_ethane_lower_limit", 0)
    MACROS["upper_limit_change_in_recycle_ethane"] = re_upper
    MACROS["lower_limit_change_in_recycle_ethane"] = re_lower

    logger.info("mixed_feed_margin=%d  Extra_Recycle_Ethane=%.4f", mixed_feed_margin, extra_re)


# =============================================================================
# Generate Attributes (329) – stamp Extra_Recycle_Ethane on df rows
# =============================================================================
def stamp_extra_recycle_ethane(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Extra_Recycle_Ethane"] = float(_m("Extra_Recycle_Ethane", 0))
    return df


# =============================================================================
# Branch (19): fresh_feed_change != -1
# Generate Attributes (330) – update lower_limit_feed / step_size_feed
# for biasing_condition==1 when mixed_feed_margin == sum_upper_limit_feed
# =============================================================================
def branch_19(df: pd.DataFrame) -> pd.DataFrame:
    fresh_feed_change = int(_m("fresh_feed_change", 0))
    if fresh_feed_change == -1:
        # ELSE path: pass-through
        return df

    # THEN path: Generate Attributes (330)
    df = df.copy()
    biasing          = int(_m("biasing_condition", 0))
    mixed_fm         = float(_m("mixed_feed_margin", 0))
    sum_ulf          = float(_m("sum_upper_limit_feed", 0))

    cond_bias1_full = (biasing == 1) and (mixed_fm == sum_ulf)

    ulf = _col(df, "upper_limit_feed")
    ssf = _col(df, "step_size_feed")
    llf = _col(df, "lower_limit_feed")

    if cond_bias1_full:
        # lower_limit_feed = upper_limit_feed
        df["lower_limit_feed"] = ulf
        # step_size_feed = if(step_size_feed==0, 0, 1)
        df["step_size_feed"]   = np.where(ssf == 0, 0.0, 1.0)

    # Final: if lower == upper → step = 0
    new_llf = _col(df, "lower_limit_feed")
    new_ulf = _col(df, "upper_limit_feed")
    df["step_size_feed"] = np.where(new_llf == new_ulf, 0.0, _col(df, "step_size_feed"))

    return df


# =============================================================================
# Branch (66): biasing_condition == 3
# =============================================================================
def branch_66(df: pd.DataFrame) -> pd.DataFrame:
    biasing = int(_m("biasing_condition", 0))

    if biasing == 3:
        # Generate Attributes (1039)
        df = _gen_attrs_1039(df)
    else:
        # Generate Attributes (43) → Branch (70) → Branch (biasing==1)
        df = _gen_attrs_43(df)
        df = branch_70(df)
        df = branch_biasing1(df)

    return df


def _gen_attrs_1039(df: pd.DataFrame) -> pd.DataFrame:
    """biasing_condition==3: conversion limits + step_size_conversion + New_Feed_flow"""
    df = df.copy()
    cv_lo_thresh = _m("conversion_bias_threshold_lower_limit", -1)
    cv_up_thresh = _m("conversion_bias_threshold_upper_limit",  1)
    cv_lo_exp    = _m("conversion_lower_limit_expansion_max_limit", 0)
    cv_up_exp    = _m("conversion_upper_limit_expansion_max_limit", 0)
    no_good_cnt  = int(_m("count_of_no_good_fur", 0))

    def _lo(row):
        oc = float(row.get("Overall_conversion", 0) or 0)
        return min(0.0, max(cv_lo_thresh,
                            -math.floor((oc - cv_lo_exp) / 0.5) * 0.5))

    def _up(row):
        cond = str(row.get("Furnace_condition", ""))
        pat  = float(row.get("percent_above_threshold", 0) or 0)
        oc   = float(row.get("Overall_conversion", 0) or 0)
        if cond == "Semi Good" and pat > 0:
            return max(0.0, min(cv_up_thresh,
                                math.floor((cv_up_exp - oc) / 0.5) * 0.5))
        return 0.0

    step = (5 if no_good_cnt < 6
            else 3 if no_good_cnt == 6
            else 1 if no_good_cnt == 9
            else 2)

    df["conversion_lower_limit_in_grid"] = df.apply(_lo, axis=1)
    df["conversion_upper_limit_in_grid"] = df.apply(_up, axis=1)
    df["step_size_conversion"]           = float(step)
    df["New_Feed_flow"]                  = _col(df, "Feed_flow")
    return df


def _gen_attrs_43(df: pd.DataFrame) -> pd.DataFrame:
    """Non-biasing-3 path: conversion lower & upper limits"""
    df = df.copy()
    cv_lo_thresh = _m("conversion_bias_threshold_lower_limit", -1)
    cv_up_thresh = _m("conversion_bias_threshold_upper_limit",  1)
    cv_lo_exp    = _m("conversion_lower_limit_expansion_max_limit", 0)
    cv_up_exp    = _m("conversion_upper_limit_expansion_max_limit", 0)
    fresh_fc     = int(_m("fresh_feed_change", 0))

    def _lo(row):
        oc = float(row.get("Overall_conversion", 0) or 0)
        return min(0.0, max(cv_lo_thresh,
                            -math.floor((oc - cv_lo_exp) / 0.5) * 0.5))

    def _up(row):
        cond = str(row.get("Furnace_condition", ""))
        frl  = float(row.get("Forecasted_runlength_rank", 100) or 100)
        oc   = float(row.get("Overall_conversion", 0) or 0)
        eligible = (cond in {"Good", "Semi Good"}) and (
            fresh_fc != 0 or frl != 100
        )
        if not eligible:
            return 0.0
        # First pass: threshold upper limit
        ul = cv_up_thresh
        # Second pass: clamp
        return max(0.0, min(cv_up_thresh,
                            math.floor((cv_up_exp - oc) / 0.5) * 0.5)) if ul != 0 else 0.0

    df["conversion_lower_limit_in_grid"] = df.apply(_lo, axis=1)
    df["conversion_upper_limit_in_grid"] = df.apply(_up, axis=1)
    return df


def branch_70(df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch (70): fresh_feed_change == 0
    THEN: Subprocess (14) – recompute mixed_feed_margin from sum_New_mixed_feed_margin
    ELSE: pass-through
    """
    if int(_m("fresh_feed_change", 0)) != 0:
        return df

    # Subprocess (14): Generate Attributes (44)
    df = df.copy()
    shc = _m("shc_ratio", 0)

    conv_ll  = _col(df, "conversion_lower_limit_in_grid")
    oc       = _col(df, "Overall_conversion")
    feed     = _col(df, "Feed_flow")
    curr_re  = _col(df, "Current_Recycle_Ethane_Feed")

    df["New_Overall_conversion"]  = oc + conv_ll
    df["New_Recycle_Ethane_Feed"] = (feed / (1.0 + shc)) * (100.0 - df["New_Overall_conversion"]) / 100.0
    df["New_Extra_Recycle_Ethane"] = df["New_Recycle_Ethane_Feed"] - curr_re
    df["New_mixed_feed_margin"]    = df["New_Extra_Recycle_Ethane"] * (1.0 + shc)

    # Aggregate (2): sum(New_mixed_feed_margin)
    sum_nmfm = float(df["New_mixed_feed_margin"].sum())
    MACROS["sum_New_mixed_feed_margin"] = sum_nmfm

    # Generate Macro (160): update mixed_feed_margin and Extra_Recycle_Ethane
    mfm = float(_m("mixed_feed_margin", 0))
    if mfm > math.floor(sum_nmfm):
        floor_sum = math.floor(sum_nmfm)
        # parity correction
        if mfm % 2 == 0 and int(floor_sum) % 2 == 1:
            mfm = floor_sum - 1
        else:
            mfm = floor_sum
        MACROS["Target_updated"] = 1
    else:
        MACROS["Target_updated"] = 0

    MACROS["mixed_feed_margin"] = mfm

    denom = 1.0 + _m("shc_ratio", 0)
    extra_re = mfm / denom - _m("fresh_feed_quantity", 0) if denom != 0 else 0.0
    MACROS["Extra_Recycle_Ethane"] = extra_re
    MACROS["upper_limit_change_in_recycle_ethane"] = extra_re + _m("change_recycle_ethane_upper_limit", 0)
    MACROS["lower_limit_change_in_recycle_ethane"] = extra_re + _m("change_recycle_ethane_lower_limit", 0)

    # Drop temp cols (exclude new temp)
    drop_tmp = ["New_mixed_feed_margin", "New_Extra_Recycle_Ethane",
                "New_Recycle_Ethane_Feed", "New_Overall_conversion"]
    df.drop(columns=[c for c in drop_tmp if c in df.columns], inplace=True)

    # Generate Attributes (45): stamp Extra_Recycle_Ethane
    df["Extra_Recycle_Ethane"] = float(MACROS["Extra_Recycle_Ethane"])

    return df


def branch_biasing1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch (biasing_condition == 1):
    THEN: Sort (2) asc factor → Generate ID → Loop (While)(36) using mixed_feed_margin
          → filter balance_taken==0 → Extract id_for_balance
          → Generate Attributes (2): update upper/step/lower_limit_feed
          → Select Attributes (2): drop taken, balance_taken, id
    ELSE: pass-through
    """
    if int(_m("biasing_condition", 0)) != 1:
        return df

    df = df.copy()
    total_fur = int(_m("total_fur_available_for_bias", len(df)))
    fur_step  = float(_m("furnace_step_adjust_feed_grid_limit", 0))

    # Sort (2) asc factor + Generate ID
    if "factor" in df.columns:
        df = df.sort_values("factor", ascending=True).reset_index(drop=True)
    df["id"] = range(len(df))

    # Loop (While) (36): balance_feed = mixed_feed_margin
    balance = float(_m("mixed_feed_margin", 0))
    df["taken"]         = 0.0
    df["balance_taken"] = balance

    for i in range(len(df)):
        if i >= total_fur:
            break
        ulf   = float(df.loc[i, "upper_limit_feed"]) if "upper_limit_feed" in df.columns else 0.0
        taken = min(balance, ulf)
        df.loc[i, "taken"]         = taken
        df.loc[i, "balance_taken"] = balance - taken
        balance -= taken
        MACROS["balance_feed"] = balance

    # Append (76): loop output is already in df

    # Filter (2): balance_taken == 0 → Extract Macro (2): id_for_balance
    df_bal0 = df[df["balance_taken"] == 0.0]
    id_for_balance = int(df_bal0.iloc[0]["id"]) if not df_bal0.empty else -1
    MACROS["id_for_balance"] = id_for_balance

    # Generate Attributes (2):
    # upper_limit_feed = if(fur_step > id_for_balance,
    #                        if(id <= fur_step, ulf, 0), taken)
    # step_size_feed   = if(id_for_balance >= fur_step, 0, 1) * if(ulf>1, 1, 0)
    # lower_limit_feed = if(step_size_feed > 0, 0, ulf)
    def _up2(row):
        row_id = int(row["id"])
        ulf    = float(row["upper_limit_feed"])
        taken  = float(row["taken"])
        if fur_step > id_for_balance:
            return ulf if row_id <= fur_step else 0.0
        return taken

    def _step2(row):
        ulf = float(row["upper_limit_feed"])
        if id_for_balance >= fur_step:
            return 0.0
        return 1.0 if ulf > 1 else 0.0

    df["upper_limit_feed"] = df.apply(_up2, axis=1)
    df["step_size_feed"]   = df.apply(_step2, axis=1)
    df["lower_limit_feed"] = np.where(df["step_size_feed"] > 0, 0.0, df["upper_limit_feed"])

    # Select Attributes (2): drop taken, balance_taken, id
    df.drop(columns=[c for c in ["taken", "balance_taken", "id"] if c in df.columns], inplace=True)

    return df


# =============================================================================
# Reset Row macros – Loop (48)
# =============================================================================
def reset_row_macros():
    for i in range(1, MAX_FURNACES + 1):
        MACROS[f"Row_{i}_upper_limit_feed"]       = 0
        MACROS[f"Row_{i}_lower_limit_feed"]       = 0
        MACROS[f"Row_{i}_step_size_feed"]         = 0
        MACROS[f"Grid_Row_{i}_conversion_delta"]  = 0
        MACROS[f"Row_{i}_step_size_conversion"]   = 0
        MACROS[f"Row_{i}_upper_limit_conversion"] = 0
        MACROS[f"Row_{i}_lower_limit_conversion"] = 0
        MACROS[f"Row_{i}_Furnace"]                = 0
        MACROS[f"Row_{i}_part_override"]          = 0


# =============================================================================
# Sort (106) + extract row loop (11) + Extract Macro (435)
# =============================================================================
def extract_row_macros_from_df(df: pd.DataFrame):
    if "overall_ranking" in df.columns:
        df = df.sort_values("overall_ranking").reset_index(drop=True)

    n = min(len(df), MAX_FURNACES)
    MACROS["Number_of_rows"] = n

    col_map = {
        "Feed_flow":                          "Feed_flow",
        "entity_name":                        "Furnace",
        "specific_energy_consumption":        "Specific_Energy_consumption",
        "Overall_conversion":                 "Conversion",
        "Furnace_condition":                  "Furnace_condition",
        "ethylene_production":                "Ethylene_Production",
        "lower_limit_feed":                   "lower_limit_feed",
        "upper_limit_feed":                   "upper_limit_feed",
        "step_size_feed":                     "step_size_feed",
        "Current_Recycle_Ethane_Feed":        "Current_Recycle_Ethane_Feed",
        "conversion_lower_limit_in_grid":     "lower_limit_conversion",
        "conversion_upper_limit_in_grid":     "upper_limit_conversion",
        "step_size_conversion":               "step_size_conversion",
        "flag_conversion_part_override":      "part_override",
    }

    for i in range(n):
        row     = df.iloc[i]
        row_num = i + 1
        for df_col, macro_suffix in col_map.items():
            val = row.get(df_col, 0)
            MACROS[f"Row_{row_num}_{macro_suffix}"] = val

    logger.info("Row macros extracted for %d furnaces.", n)


# =============================================================================
# Main entry point
# =============================================================================
def run(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== MODULE 06 – PRE-GRID ===")

    # Generate Attributes (906): pre calcs incl. initial feed_reduction_potential
    df = compute_initial_columns(df)

    # Number_of_rows (11) + Extract Macro (434)
    extract_row_count_and_shc(df)

    # Filter (346) → Good: Generate Attrs (833) + exclude (79)
    #              → Unmatched: Generate Attrs (177)
    df_good, df_nongood = split_good_nongood_and_compute_limits(df)

    # Aggregates: sum_upper_limit_feed + sum_feed_reduction_potential
    aggregate_and_extract_sums(df_good, df_nongood)

    # Branch (112): biasing_condition != 2
    df = branch_112(df_good, df_nongood)

    # Aggregate (77) + sum_upper_limit_feed (23)
    aggregate_post_branch112(df)

    # Generate Macro (149): mixed_feed_margin, Extra_Recycle_Ethane, recycle UL/LL
    compute_recycle_ethane_macros()

    # Generate Attributes (329): stamp Extra_Recycle_Ethane on df
    df = stamp_extra_recycle_ethane(df)

    # Branch (19): fresh_feed_change != -1
    df = branch_19(df)

    # Branch (66): biasing_condition == 3
    df = branch_66(df)

    # Loop (48): reset Row_N_* macros
    reset_row_macros()

    # Sort (106) + extract row loop (11)
    extract_row_macros_from_df(df)

    STORE["df_pre_grid"] = df.copy()
    logger.info("PRE-GRID complete – %d rows, Row macros set for %d furnaces.",
                len(df), int(_m("Number_of_rows")))
    return df
