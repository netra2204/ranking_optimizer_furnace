# Furnace Ranking Optimisation – Python Pipeline

Python replica of `ranking_overall_optimization_14th_april.rmp` (RapidMiner).

---

## Project Structure

```
furnace_ranking/
├── main.py                          # Orchestrator – run this
├── config.py                        # Global MACROS dict + constants
│
├── module_01_inputs.py              # INPUTS (2) subprocess
├── module_02_initialization.py      # Initialization (3) subprocess
├── module_03_parameterization.py    # Parameterization subprocess
├── module_04_preprocessing.py       # Preprocess subprocess
├── module_05_past_hour_logic.py     # Branch (6) / past time subprocess
├── module_06_pre_grid.py            # Pre Grid subprocess
├── module_07_grid_main.py           # Grid-Main subprocess (FEED + CONVERSION grid)
├── module_08_post_grid.py           # Post Grid subprocess
├── module_09_generate_output.py     # Generate Output (deviation branch)
└── module_10_output_format_check.py # Output_Format_Check subprocess
```

---

## Module-by-Module Summary

| # | Module | RapidMiner Block | Key Responsibility |
|---|--------|------------------|--------------------|
| 01 | `module_01_inputs` | `INPUTS (2)` | Load raw join-data CSV/DB; derive temp columns; extract end_time; load pipeline macros |
| 02 | `module_02_initialization` | `Initialization (3)` | Join tag + tag_details; dedup; filter ROPT extract-macro-value rows; split inferred-tag stores |
| 03 | `module_03_parameterization` | `Parameterization` | Parse numerics; de-pivot to long; join parameter mapping; pivot back to wide |
| 04 | `module_04_preprocessing` | `Preprocess` | Extract limit macros; compute margin flags; external constraints; rename columns; evaluate inferred tags; decoking furnace; coupling logic; Good/No-Good split |
| 05 | `module_05_past_hour_logic` | `Branch (6) → past time` | Compute prev-hour timestamps; load past 24h output; detect deviation; set `deviation_exists` macro |
| 06 | `module_06_pre_grid` | `Pre Grid` | Compute Ethane_Feed, factor; upper/lower/step limits; balance-feed loops; recycle-ethane macros; extract Row_N_* macros |
| 07 | `module_07_grid_main` | `Grid-Main` | FEED GRID exhaustive search; MAIN CONVERSION GRID; store best feed_delta + conversion_delta per furnace |
| 08 | `module_08_post_grid` | `Post Grid` | Build grid log; extract best biases; merge back; compute New_Feed_flow / New_Overall_conversion; del_ethylene; ranking_opportunity |
| 09 | `module_09_generate_output` | `Generate Output` (deviation branch) | deviation_exists == 1 → fresh output; == 0 → recall prev-hour + join current state |
| 10 | `module_10_output_format_check` | `Output_Format_Check` | Melt wide → long (Timestamp \| sub_model_id \| tag \| value); validate schema |

---

## Data Flow

```
CSV / DB
   │
   ▼
[01 INPUTS] ──► df_main
   │
   ▼
[02 INITIALIZATION] ──► STORE: tag, inferred_tags_1..4, ROPT_extract_macro_value
   │
   ▼
[03 PARAMETERIZATION] ──► df_param (wide, one row per furnace)
   │
   ▼
[04 PRE-PROCESSING] ──► df_preprocessed (margins, conditions, counts)
   │
   ▼
[05 PAST HOUR LOGIC] ──► MACROS["deviation_exists"] = 0 or 1
   │
   ├─ deviation_exists == 1 ──►
   │     [06 PRE-GRID]   ──► Row_N_* macros
   │     [07 GRID MAIN]  ──► Row_N_feed_delta, Grid_Row_N_conversion_delta
   │     [08 POST GRID]  ──► df_post_grid (with biases + del_ethylene)
   │
   └─ deviation_exists == 0 ──► recall prev-hour output
         │
         ▼
[09 GENERATE OUTPUT] ──► df_output (wide, final)
   │
   ▼
[10 OUTPUT FORMAT CHECK] ──► df_long (Timestamp | sub_model_id | tag | value)
```

---

## Global State

All RapidMiner **macros** (`%{macro_name}`) are stored in the `MACROS` dict in `config.py`.

All RapidMiner **remember/recall** stores are in the `STORE` dict in `config.py`.

Both dicts are imported by every module:

```python
from config import MACROS, STORE
```

---

## Running the Pipeline

### CLI

```bash
# Basic run with local CSV
python main.py --csv data/join_data.csv

# With previous-hour CSV for deviation detection
python main.py --csv data/join_data.csv --prev-csv data/prev_output.csv

# Save output to CSV
python main.py --csv data/join_data.csv --output results/output.csv

# Wide format output (per-furnace, not melted)
python main.py --csv data/join_data.csv --wide --output results/wide_output.csv

# Verbose debug logging
python main.py --csv data/join_data.csv --log-level DEBUG
```

### Python API

```python
from main import run_pipeline

# Long format (default)
df_long = run_pipeline(csv_path="data/join_data.csv")

# Wide format
df_wide = run_pipeline(csv_path="data/join_data.csv", return_wide=True)

# Access final MACROS after run
from config import MACROS
print("sum_del_ethylene_final:", MACROS["sum_del_ethylene_final"])
print("ranking_cause_indicator:", MACROS["ranking_cause_indicator"])
```

---

## Input CSV Format

The main input CSV (`join_data.csv`) should be the export of the
`data/join_data_12march_4pm` RapidMiner repository entry. Expected columns:

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | datetime | Measurement timestamp |
| `entity_name` | string | Furnace ID (e.g. F1, F2 … F9) |
| `wet_feed_total_flow` | float | Feed flow (t/h) |
| `overall_conversion` | float | Ethane conversion (%) |
| `overall_ranking` | int | Current ranking |
| `Furnace_condition` | string | Good / Bad / SOR / Semi Good / No Optimization |
| `days_remaining` | float | Days until next decoking |
| `ethylene_production` | float | Current ethylene production (t/h) |
| `Feed_flow` | float | *(alias for wet_feed_total_flow)* |
| `percent_above_threshold` | float | Run-length margin indicator |
| `shc_ratio` | float | Steam-to-hydrocarbon ratio |
| … | … | All sensor/tag columns referenced in margin formulas |

---

## Key Configuration

Edit `config.py` to change:

| Setting | Location | Description |
|---------|----------|-------------|
| `fresh_feed_input` | `INPUTS` | Target fresh feed (t/h) |
| `fresh_feed_change_set` | `INPUTS` | 0 = maintain; -1 = force reduction |
| `Fur_change_recycle_ethane_limit` | `INPUTS` | Max recycle-ethane change |
| `ROPT_furnace_coupling` | `PIPELINE_MACROS` | Enable coupling constraint |
| `ROPT_external_constraint` | `PIPELINE_MACROS` | Enable external constraint re-ranking |
| `ROPT_use_past_time_output` | `PIPELINE_MACROS` | Enable deviation detection |
| `pull_tables_from_db` | `INPUTS` | 0 = CSV; 1 = SQL DB |
| `db_connection_string` | `MACROS` | SQLAlchemy connection string |

---

## Dependencies

```
pandas >= 1.5
numpy >= 1.23
sqlalchemy  (optional – only for DB mode)
pyodbc      (optional – only for SQL Server)
```

Install:
```bash
pip install pandas numpy
pip install sqlalchemy pyodbc   # only if using DB mode
```
