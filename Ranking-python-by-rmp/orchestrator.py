import sys, os, importlib.util, logging
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

try:
    from pandas._libs.parsers import STR_NA_VALUES as _PD_NA
    _NA_KEEP_NULL = list(_PD_NA - {"null", "NULL", "Null"})
except Exception:                                   # pragma: no cover - version drift
    _NA_KEEP_NULL = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                     "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
                     "NaN", "None", "n/a", "nan"]

def _load(alias: str, rel_path: str):
    path = os.path.join(os.path.dirname(__file__), rel_path)
    sys.path.insert(0, os.path.dirname(path))   # let the module resolve its own sibling imports
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── 1. Run ranking-common-process ─────────────────────────────────────────────
common_main = _load("common_main", "ranking-common-process/main.py")

macros = common_main.build_default_macros()
store  = common_main.build_io_store(
    tag_parameter_mapping = pd.read_excel(os.path.join(_INPUTS, "tag_parameter_mapping_newname_rev3.xlsx"),
                                           keep_default_na=False, na_values=_NA_KEEP_NULL),
    text_code_mapping     = pd.read_excel(os.path.join(_INPUTS, "text-code-mapping.xlsx")),
    ccp_status            = pd.read_excel(os.path.join(_INPUTS, "ccp-status.xlsx")).assign(
                              Timestamp=lambda df: pd.to_datetime(df["Timestamp"])),
    entity                = pd.read_excel(os.path.join(_INPUTS, "entity.xlsx")),
    parameters            = pd.DataFrame(),
    entity_parameter      = pd.DataFrame(),
    tag                   = pd.DataFrame(),
    furnace_ranking_info  = pd.read_excel(os.path.join(_INPUTS, "furnace-ranking-info.xlsx")),
)
input_df = pd.read_excel(os.path.join(_INPUTS, "wide_format_data_12jan-1am_newname.xlsx"))
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
handoff_path = os.path.join(_RESULTS, "common-result-12jan-26-1am-renamed.xlsx")
common_result.to_excel(handoff_path, index=False)
logger.info(f"COMMON PIPELINIE RESULT SHAPE: {common_result.shape}")
logger.info("----------------COMMON PIPELINE COMPLETED---------------")

# ── 2. Run ranking-case-specific ──────────────────────────────────────────────
try:
    case_main = _load("case_main", "ranking-case-specific/main.py")

    case_main.load_store_data(
        tag_parameter_mapping_csv     = os.path.join(_INPUTS, "tag_parameter_mapping_newname_rev3.xlsx"),
        ropt_extract_macro_values_csv = os.path.join(_INPUTS, "parameters.xlsx"),
        inferred_tags_1_csv           = os.path.join(_INPUTS, "inferred_tags_1.xlsx"),
        inferred_tags_2_csv           = os.path.join(_INPUTS, "inferred_tags_2.xlsx"),
        inferred_tags_3_csv           = os.path.join(_INPUTS, "inferred_tags_3.xlsx"),
        inferred_tags_4_csv           = os.path.join(_INPUTS, "inferred_tags_4.xlsx"),
    )

    final_result = case_main.run_pipeline(input_df=common_result)
    final_result.to_excel(os.path.join(_RESULTS, "12jan-26-1am-case-specific-renamed-result.xlsx"), index=False)
except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)