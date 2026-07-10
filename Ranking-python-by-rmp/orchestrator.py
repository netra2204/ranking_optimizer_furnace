import sys, os, importlib.util, logging
from datetime import datetime
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

_BASE   = os.path.dirname(os.path.abspath(__file__))
_INPUTS = os.path.join(_BASE, "Python-Inputs")
_RESULTS = os.path.join(_BASE, "Results")

# Single compiled workbook holding every input as its own sheet.
_COMPILED = os.path.join(_INPUTS, "ranking-inputs-compiled.xlsx")

def _load(alias: str, rel_path: str):
    path = os.path.join(os.path.dirname(__file__), rel_path)
    sys.path.insert(0, os.path.dirname(path))   # let the module resolve its own sibling imports
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod          # needed so dataclasses/typing can resolve the module
    spec.loader.exec_module(mod)
    return mod

# ── 0. Run pre_rank ───────────────────────────────────────────────────────────
pre_rank = _load("pre_rank_pipeline", "pre_rank/pipeline.py")

logger.info("----------------PRE-RANK STARTED---------------")
pre_rank_result = pre_rank.run_pre_rank(
    config_path          = _COMPILED,
    common_inferred_path = _COMPILED,
    wide_input_path      = _COMPILED,
    config_sheet         = "config_overrides",
    template_sheet       = "common-inferred",
    wide_sheet           = "wide-12jan-1am-original",
    overwrite_existing   = True,
)
logger.info(f"PRE-RANK RESULT SHAPE: {pre_rank_result.wide_output.shape}")
logger.info("----------------PRE-RANK COMPLETED---------------")

# ── 1. Run ranking-common-process ─────────────────────────────────────────────
common_main = _load("common_main", "ranking-common-process/main.py")

macros = common_main.build_default_macros()
store  = common_main.build_io_store(
    # Excel sources given as (path, sheet); special handling (na-values for
    # tag_parameter_mapping, Timestamp parse for ccp_status) happens inside
    # build_io_store so this orchestrator never calls read_excel directly.
    tag_parameter_mapping = (_COMPILED, "tpm-newname-rev3"),
    text_code_mapping     = (_COMPILED, "text-code-mapping"),
    ccp_status            = (_COMPILED, "ccp-status"),
    entity                = (_COMPILED, "entity"),
    parameters            = pd.DataFrame(),
    entity_parameter      = pd.DataFrame(),
    tag                   = pd.DataFrame(),
    furnace_ranking_info  = (_COMPILED, "furnace-ranking-info"),
)
input_df = pre_rank_result.wide_output
input_df["Timestamp"] = pd.to_datetime(input_df["Timestamp"])

logger.info("----------------COMMON PIPELINE STARTED---------------")

common_result = common_main.run_process(input_df, macros, store)

_join_key = "Timestamp"
_dup_cols = [c for c in common_result.columns
             if c in input_df.columns and c != _join_key]
common_result = input_df.merge(
    common_result.drop(columns=_dup_cols),
    on=_join_key, how="left")

# # ── Handoff: save to a temp file ──────────────────────────────────────────────
# handoff_path = os.path.join(_RESULTS, "common-result-12jan-26-1am-renamed.xlsx")
# common_result.to_excel(handoff_path, index=False)
logger.info(f"COMMON PIPELINIE RESULT SHAPE: {common_result.shape}")
logger.info("----------------COMMON PIPELINE COMPLETED---------------")

# ── 2. Run ranking-case-specific ──────────────────────────────────────────────
try:
    case_main = _load("case_main", "ranking-case-specific/main.py")

    case_main.load_store_data(
        tag_parameter_mapping_csv       = _COMPILED,
        ropt_extract_macro_values_csv   = _COMPILED,
        inferred_tags_1_csv             = _COMPILED,
        inferred_tags_2_csv             = _COMPILED,
        inferred_tags_3_csv             = _COMPILED,
        inferred_tags_4_csv             = _COMPILED,
        tag_parameter_mapping_sheet     = "tpm-newname-rev3",
        ropt_extract_macro_values_sheet = "parameters",
        inferred_tags_1_sheet           = "inferred_tags_1",
        inferred_tags_2_sheet           = "inferred_tags_2",
        inferred_tags_3_sheet           = "inferred_tags_3",
        inferred_tags_4_sheet           = "inferred_tags_4",
    )

    final_result = case_main.run_pipeline(input_df=common_result)
    _run_stamp  = datetime.now().strftime("%H-%M-%S")                 # current run time
    _data_ts    = pd.to_datetime(input_df["Timestamp"].iloc[0])       # input data timestamp
    _data_stamp = _data_ts.strftime("%d-%m-%Y-%I%p")                  # date-month-year-hour AM/PM
    _out_name   = f"{_run_stamp}_ranking-final-result_{_data_stamp}.xlsx"
    final_result.to_excel(os.path.join(_RESULTS, _out_name), index=False)
    logger.info("Final result saved -> %s", _out_name)
except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)