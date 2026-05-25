# united_optimizer — Python replica of `United (2)` (RapidMiner)

This package is a faithful, operator-by-operator port of the RapidMiner
sub-process exported as `united-optimizer-main-block.rmp` (5 513 lines,
511 operators). The .rmp itself is the `United (2)` block inside the
larger parent process `Furnace_optimizer_DB_163_latest_kishan_ver5.rmp`.

## File layout

```
united_optimizer/
├── __init__.py              Public surface
├── orchestrator.py          run_united_optimizer + UnitedOptimizerInputs
├── united_optimizer.py      Fe_inferred (2), Handle Timestamp, Re-formatting,
│                            Generate tags (2), ccp_status_reform (4)
├── bias_and_grid.py         Bias Constants, Overall_Calcs, Inferred calculations,
│                            FEED GRID (2), Conversion / COKE grid, Coilsim apply
├── act_opt.py               ACT=OPT (Branch 9, Subprocess 43, Subprocess 4)
│                            and COUPLED_CCP_USED_CHECK
└── rm_runtime.py            MacroStore, expression evaluator, ExampleSet
                             helpers, Aggregate / Extract Macro / Generate
                             Attributes / filter_examples runtime primitives
```

## Execution order (matches the .rmp connect graph)

```
in 1 ──► Fe_inferred (2) ──► Remove Duplicates (16) ──► Handle Timestamp
        │
        └─ writes Tag_for_Optimizer + macros (TMT_Name, Mixed_Feed_Name, …)

──► Re-formatting
        ├─ Generate Attributes (25)          ─ Pass / Furnace derivation
        ├─ Filtering_Data
        │   ├─ Gen_Pass_Values                Loop Values (6) × Loop (9) 8x
        │   ├─ Filter Examples (30)           keep Pass not missing
        │   ├─ Renaming_Systemwise            Replace (3) regex
        │   ├─ Renaming_Passwise              Replace (9) regex
        │   └─ Subprocess (27)                left-join + Append
        ├─ Renaming as per optimizer (2)     Create ExampleSet → Remember
        ├─ Join (74)                         left-join with Rename_data
        ├─ Generate Attributes (28)          new_name fallback
        ├─ Pivot (10)                        long → wide
        ├─ Rename by Replacing (7)           strip "average(value)_" prefix
        ├─ Numerical to Polynominal (6)
        ├─ Set Role (10)                     Timestamp → id
        ├─ Generate Attributes               COP_Old+1, COT_Old+20
        └─ ccp_status_reform (4)

──► Generate tags (2)                       Tube_Flow_Old, COT_From_Equation_Old,
                                              physics correlation, bias gating
──► Date to Nominal (7)                     Timestamp → string
──► Join (2)                                left-join "constraint"
──► Main_Process
        ├─ Extract Macro (88)               Furnace_Status, ccp_status_curr,
        │                                    total_optimizer_run_check
        ├─ Bias Constants                   bias_max_constant, bias_min_constant
        ├─ Overall_Calcs                    Furnace_Weighted_COT_Old, etc.
        ├─ GRID_AND_BIASING (Branch)        Furnace_Status==1 && ccp_curr!=0 &&
        │                                     total_optimizer_run_check==1
        │   ├─ Inferred calculations
        │   │   ├─ Subprocess (3,100)       8 limiting-pass macros
        │   │   ├─ Subprocess (106)         last_, second_last_, …
        │   │   ├─ Subprocess (96)          pass1_…pass8_
        │   │   ├─ Generate Macro (30)      furnace_in_SOR / EOR
        │   │   ├─ Branch (3)               valid-state guard
        │   │   ├─ Branch (144)             Ranking_Coupled == 0
        │   │   │   ├─ Generate Macro (334) Biasing_in_pass_<ord> decoupled
        │   │   │   ├─ Generate Macro (336) Increase/Decrease flow
        │   │   │   ├─ Generate Macro (5)   Bias_In_Pass_1..8 remap
        │   │   │   └─ Branch (Bias Limits 2)
        │   │   │       Inc<Dec → balanced search (Loop While)
        │   │   └─ Branch (Bias Limits / 4) Ranking_Coupled == 1
        │   │       ├─ Generate Macro (331) up-bias
        │   │       ├─ Generate Macro (335) down-bias
        │   │       ├─ Generate Macro (332) Max_Feed_Bias / Proceed
        │   │       ├─ Generate Macro (2)   Bias_In_Pass_1..8 remap
        │   │       └─ Branch (7) Max_Feed_Bias==Feed_Bias
        │   │           true:  Generate Macro (3) extremes
        │   │           false: Loop While balanced search
        │   ├─ Generate Macro (7) step-sizes per pass (Loop "Step Size")
        │   ├─ FEED GRID (2)                Branch on Proceed==1
        │   │   ├─ Fe_input
        │   │   │   ├─ Generate Macro (55)  run_branch / then_block
        │   │   │   ├─ optimize_parameters_grid "FEED GRID"
        │   │   │   │   pass_1..pass_8 ∈ [min;max;step]
        │   │   │   ├─ Set Macro pass_<n>=NaN init
        │   │   │   ├─ Loop (86) 8x: Generate Macro (280)/(10) quantize
        │   │   │   ├─ Generate Macro (194) Net_Bias / check_feed_min_max
        │   │   │   ├─ Feed_Grid_Character (3) dedup hash
        │   │   │   └─ Branch (155) → Pass_initializer + Objective_function (2)
        │   │   ├─ Loop (3)                runs Fe_input loop_for_bias_count times
        │   │   ├─ Append (2) / Sort (19)
        │   │   ├─ Extract Macro (28) Final_Loop_run_id
        │   │   ├─ Append (3) / Filter Examples (13) pick winner
        │   │   ├─ Generate Attributes (202) Overall_Opt_Branch_Indicator
        │   │   └─ Rename by Replacing "_Coke_Grid$" → "_New"
        │   ├─ Generate Attributes (619)   Overall_Opt_Branch_Indicator = -1.0
        │   └─ Generate Macro (25)         alt indicator when grid skipped
        ├─ ACT=OPT
        │   ├─ Branch (9) then_block == 1
        │   │   true:  pass-through
        │   │   false: Loop Attributes (7) `.*_Old` → `*_New` copies
        │   │            Set Macro (13) Total_Benefit_Per_Day_Coke_Grid=-10000
        │   ├─ Subprocess (43)
        │   │   ├─ Generate Attributes (211) COT-20, days-remaining cap
        │   │   ├─ Aggregate (25) by Furnace
        │   │   ├─ Sort (24, 42)
        │   │   ├─ Mass Cot Calc (4)
        │   │   ├─ Extract Macro (186, Limiting_pass 4, Limiting_pass_new 4)
        │   │   ├─ Branch (10) Generate Macro (100)/(101)
        │   │   ├─ Rename (11), Generate Attributes (239)
        │   │   ├─ Branch (45) min_days_diff_check
        │   │   │   true:  Subprocess (4) final rename + benefit_tags
        │   │   │   false: Generate Macro (104), Loop Attributes (12)
        │   │   └─ Subprocess (4): Multiply, Rename (917), Subprocess (149),
        │   │       Loop Attributes (13/14), benefit_tags (5/6), Merges
        └─ COUPLED_CCP_USED_CHECK
            ├─ Extract Macro (189) sub_model_id
            ├─ Select Attributes (344) exclude sub_model_id
            ├─ Set Role (66) Timestamp → id
            ├─ De-Pivot (6) long format
            ├─ Generate Attributes (622) sub_model_id
            ├─ Branch (23) macro_defined post_optimizer_transformation_utd
            └─ Branch (33) ...=='active' → Generate Attributes (623) coupled_mode

──► Remove Duplicates (37) ─► Parse Numbers (34) ─► Numerical to Real (57) ─► out 1
```

## Plugging in the Coilsim model

The .rmp pulls four pre-trained models from the RapidMiner repository:

```
//ing_manufacturing_furnace_2024_dev/02_Model_Files/163_United_Olf_Furnace_System/
    482_Main_<Y>                      (per-output regression models)
    482_Main_Fur_Coil_Normalized      (z-para normalisation parameters)
```

In Python, supply a subclass of `CoilsimModelProvider`:

```python
class MyCoilsim(CoilsimModelProvider):
    def __init__(self, weights_dir): self.weights_dir = weights_dir
    def available(self, y_name): return os.path.exists(f"{self.weights_dir}/{y_name}.pkl")
    def predict(self, df, y_name): ...   # load + apply normalised model
```

Pass an instance through `UnitedOptimizerInputs(coilsim=…)`.

## Running

```python
from united_optimizer import run_united_optimizer, UnitedOptimizerInputs

result = run_united_optimizer(UnitedOptimizerInputs(
    main_data                = ...,    # long-format pi data
    tag                      = ...,    # tag table
    tag_child                = ...,    # tag_child table
    tag_details              = ...,    # tag_details table
    ccp_status               = ...,    # ccp_status table
    pipeline_parameters_opt  = ...,    # pipeline parameters table
    constraint               = ...,    # constraints table
    coilsim                  = MyCoilsim(...),
    initial_macros           = {
        "sub_model_id": 488,
        "decoke_time": 14,
        "Max_Permissible_TMT": 1105,
        "Use_Optimizer_Opportunity": 1,
        "Use_Uptime_Benefit_In_Opportunity": 1,
        "post_optimizer_transformation_utd": "active",
    },
))
```
