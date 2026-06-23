import sys, os, importlib.util, logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

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
    tag_parameter_mapping = pd.DataFrame(),
    text_code_mapping     = pd.DataFrame() ,  # supply your real files
    ccp_status            = pd.DataFrame(),
    entity                = pd.DataFrame(),
    parameters            = pd.DataFrame(),
    entity_parameter      = pd.DataFrame(),
    tag                   = pd.DataFrame(),
    furnace_ranking_info  = pd.DataFrame()
)
input_df = pd.read_csv("common-process-wide-format.csv")       # upstream join-data

common_result = common_main.run_process(input_df, macros, store)

# ── Handoff: save to a temp file ──────────────────────────────────────────────
handoff_path = "Results/common_process_output.xlsx"
common_result.to_excel(handoff_path, index=False)

# ── 2. Run ranking-case-specific ──────────────────────────────────────────────
try:
    case_main = _load("case_main", "ranking-case-specific/main.py")

    case_main.load_store_data(
        tag_parameter_mapping_csv     = "Python-Inputs/tag-parameter-mapping (1).xlsx",
        ropt_extract_macro_values_csv = "Python-Inputs/parameters.xlsx",
        inferred_tags_1_csv           = "Python-Inputs/inferred_tags_1.xlsx",
        inferred_tags_2_csv           = "Python-Inputs/inferred_tags_2.xlsx",
        inferred_tags_3_csv           = "Python-Inputs/inferred_tags_3.xlsx",
        inferred_tags_4_csv           = "Python-Inputs/inferred_tags_4.xlsx",
    )

    final_result = case_main.run_pipeline(csv_path=handoff_path)
    final_result.to_excel("Results/case-specific-long-format-result.xlsx", index=False)
except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)