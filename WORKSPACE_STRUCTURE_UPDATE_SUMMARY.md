# Workspace Structure Update Summary

## Changes Made

### 1. Updated Trace Storage Location
- Changed default workspace path from `./workspaces/` to `./configs/`
- Modified `src/core/events/tracer_setup.py` to implement the new directory structure
- Added `results/` subdirectory to keep run results organized

### 2. New Directory Structure
Each config now gets its own directory with organized results:

```
configs/
└── <config_name>/
    ├── config.yaml              # The actual config file
    ├── results/                 # All run results go here
    │   ├── <timestamp>/
    │   │   ├── metadata.json
    │   │   └── traces/
    │   │       ├── <strategy_type>/
    │   │       │   └── *.parquet
    │   │       └── classifiers/
    │   │           └── <classifier_type>/
    │   │               └── *.parquet
    │   ├── <timestamp>/
    │   │   └── ...
    │   └── latest -> <timestamp>/  # Symlink to most recent run
    └── notebooks/               # Optional analysis notebooks
```

### 3. Simplified Trace Structure
- Removed `SYMBOL_TIMEFRAME` subdirectory level from traces
- Signals are now stored directly under strategy type: `traces/bollinger_bands/*.parquet`
- Classifiers stored under: `traces/classifiers/regime/*.parquet`

### 4. Disabled Analytics Database
- Commented out DuckDB analytics workspace creation in `src/core/coordinator/coordinator.py`
- The system no longer creates `analytics.duckdb` files as requested

## Benefits

1. **Clean Organization**: Config files stay in main directory, results are contained in `results/`
2. **No Clutter**: Main config directory only has config.yaml, results/, and notebooks/
3. **Easy Navigation**: Each config has all its runs organized under results/
4. **Latest Symlink**: Quick access to most recent results via `results/latest`
5. **Simpler Trace Structure**: Removed unnecessary symbol_timeframe nesting
6. **Development Friendly**: Clear separation between configs and their results

## Testing

Created test config at `config/test_new_workspace_structure.yaml` to verify the implementation.

When you run signal generation, traces will now be stored in:
```
configs/test_new_workspace_structure/results/<timestamp>/traces/...
configs/test_new_workspace_structure/results/latest -> <timestamp>
```

## Example Structure

```
configs/
└── mean_reversion_research/
    ├── config.yaml
    ├── results/
    │   ├── 2024_12_20_143022/
    │   │   ├── metadata.json
    │   │   └── traces/
    │   │       ├── bollinger_bands/
    │   │       │   ├── period_20_std_2.0.parquet
    │   │       │   └── period_20_std_2.5.parquet
    │   │       └── rsi_threshold/
    │   │           └── period_14_oversold_30_overbought_70.parquet
    │   │
    │   ├── 2024_12_21_091533/
    │   │   ├── metadata.json
    │   │   └── traces/
    │   │       └── ...
    │   │
    │   └── latest -> 2024_12_21_091533/  # Symlink
    │
    └── notebooks/
        ├── analysis.ipynb
        ├── parameter_comparison.ipynb
        └── best_params_deepdive.ipynb
```

## Notes

- The symlink creation is attempted but may fail on some systems (Windows without admin rights)
- The system will still log the workspace location even if symlink creation fails
- WFV (Walk Forward Validation) runs use a slightly different structure for organized study results