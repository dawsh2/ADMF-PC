# Debug Summary: Trace Creation Issues

## Problems Identified

1. **Wrong Data File Loaded**
   - Loading `SPY.csv` (4 bars) instead of `SPY_5m.csv`
   - Symbol extraction issue when using `data: ["SPY_5m"]`

2. **No Strategies Created**
   - "Saved 0 component signal files" indicates no strategies were instantiated
   - Likely due to parameter_space optimization not creating compiled strategies

3. **Container Naming Issue**
   - Shows "SPY_5m_5m_data" because symbol includes timeframe

## Quick Test

Try running with the simple config:
```bash
python main.py --config config/keltner/config_simple.yaml --signal-generation --bars 100
```

This uses a basic strategy without optimization to verify the system works.

## Root Causes

1. **Data Loading**: The data parser is correctly keeping "SPY_5m" as the symbol for file loading, but the data handler might be stripping the timeframe part.

2. **Strategy Creation**: When using `--optimize` with parameter_space, strategies are created as "compiled_strategy_N" which might not be getting traced properly.

3. **Workspace Logging**: The workspace creation log isn't showing because it happens during tracer setup, which might be happening after the main logging.

## Solutions

1. **For Data Loading**: Ensure data handler uses the full symbol name ("SPY_5m") when loading files
2. **For Strategies**: Check why compiled strategies aren't generating signals
3. **For Logging**: Add more debug logging to trace the workflow