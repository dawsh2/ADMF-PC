# CLI Piping Flow Visualization

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1: SIGNAL GENERATION                │
├─────────────────────────────────────────────────────────────────┤
│ Input:  grid_search.yaml                                         │
│ • Parameter grid: 3x3x3 = 27 combinations per strategy         │
│ • Walk-forward: 12 windows                                      │
│ • 2 strategies (momentum, mean_reversion)                       │
│                                                                 │
│ Process:                                                        │
│ • Generate 27 × 12 × 2 = 648 signal sets                      │
│ • Label signals with regime (bull/bear/sideways)               │
│ • Save to parquet files                                        │
│                                                                 │
│ Output (JSON):                                                  │
│ {                                                              │
│   "metadata": {                                                │
│     "phase": "signal_generation",                              │
│     "execution_id": "signal_gen_20250106_120000"              │
│   },                                                           │
│   "artifacts": {                                               │
│     "signals": "workspaces/signal_gen_20250106/signals/",      │
│     "summary": "workspaces/signal_gen_20250106/summary.json"   │
│   },                                                           │
│   "config": {                                                  │
│     "strategies": [...],                                       │
│     "features": [...]                                          │
│   },                                                           │
│   "results": {                                                 │
│     "total_signals": 324000,                                   │
│     "parameter_sets": 54,                                      │
│     "windows": 12                                              │
│   }                                                            │
│ }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                         │ Pipe
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: ENSEMBLE OPTIMIZATION                │
├─────────────────────────────────────────────────────────────────┤
│ Input: Piped from Phase 1                                       │
│ • Inherits signal paths                                        │
│ • Inherits strategy configs                                    │
│ • Inherits feature definitions                                 │
│                                                                 │
│ Process:                                                        │
│ • Load signals from artifacts.signals path                     │
│ • Group by regime and window                                   │
│ • Optimize ensemble weights per regime:                        │
│   - Bull: 70% momentum, 30% mean_reversion                     │
│   - Bear: 30% momentum, 70% mean_reversion                     │
│   - Sideways: 50% momentum, 50% mean_reversion                 │
│                                                                 │
│ Output (JSON):                                                  │
│ {                                                              │
│   "metadata": {                                                │
│     "phase": "signal_replay",                                  │
│     "signal_source": "workspaces/signal_gen_20250106/signals/" │
│   },                                                           │
│   "artifacts": {                                               │
│     "weights": "workspaces/ensemble_20250106/weights.json",    │
│     "signals": "workspaces/signal_gen_20250106/signals/"       │
│   },                                                           │
│   "config": {                                                  │
│     "strategies": [...],                                       │
│     "ensemble_weights": {...}                                  │
│   },                                                           │
│   "results": {                                                 │
│     "optimal_sharpe": 1.85,                                    │
│     "regimes_identified": 3                                    │
│   }                                                            │
│ }                                                              │
└─────────────────────────┬───────────────────────────────────────┘
                         │ Pipe
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: FINAL VALIDATION                     │
├─────────────────────────────────────────────────────────────────┤
│ Input: Piped from Phase 2                                       │
│ • Inherits ensemble weights                                    │
│ • Inherits strategy configs                                    │
│ • Out-of-sample data (2024)                                   │
│                                                                 │
│ Process:                                                        │
│ • Run full backtest with regime-adaptive ensemble             │
│ • Switch weights based on detected regime                      │
│ • Generate complete performance metrics                        │
│                                                                 │
│ Output (Human Readable):                                        │
│ ============================================================   │
│ Phase: backtest                                                │
│ Status: SUCCESS                                                │
│                                                                │
│ Key Metrics:                                                   │
│   total_return: 34.2%                                          │
│   sharpe_ratio: 1.76                                           │
│   max_drawdown: -8.7%                                          │
│   win_rate: 62.3%                                              │
│   regime_switches: 28                                          │
│                                                                │
│ Generated Artifacts:                                           │
│   results: workspaces/backtest_20250106/results.json          │
│   trades: workspaces/backtest_20250106/trades.csv             │
│   report: workspaces/backtest_20250106/report.html            │
│ ============================================================   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Detail

### 1. Configuration Inheritance
```
grid_search.yaml ──┐
                   ▼
              Phase 1 ────► {strategies, features, data}
                               │
                               ▼ (automatic inheritance)
                           Phase 2 ────► {+ signal paths}
                               │
                               ▼ (automatic inheritance)  
                           Phase 3 ────► {+ ensemble weights}
```

### 2. Artifact Chaining
```
Phase 1: Generate Signals
         └─► signals/ ────────┐
                              │
Phase 2: Read signals ◄───────┘
         └─► weights.json ────┐
                              │
Phase 3: Use weights ◄────────┘
         └─► final_report.html
```

### 3. Error Propagation
```
Phase 1 ──► Success ──► Phase 2 ──► Success ──► Phase 3
   │                         │                       │
   └─► Failure               └─► Failure            └─► Failure
        │                         │                       │
        ▼                         ▼                       ▼
    Pipeline Stops           Pipeline Stops         Pipeline Stops
    Exit Code: 1             Exit Code: 1          Exit Code: 1
```

## Key Benefits Visualized

### Reusability
```
                    ┌─► Ensemble Method A ─► Backtest A
                    │
Signal Generation ──┼─► Ensemble Method B ─► Backtest B
(expensive: 1hr)    │
                    └─► Ensemble Method C ─► Backtest C
                         (cheap: 1min each)
```

### Parallelization
```
Config 1 ──► Signal Gen ──┐
Config 2 ──► Signal Gen ──┼──► Merge ──► Optimize ──► Validate
Config 3 ──► Signal Gen ──┘
         (parallel)
```

### Debugging
```
Phase 1 ──► tee phase1.json ──► Phase 2 ──► tee phase2.json ──► Phase 3
              │                               │
              ▼                               ▼
         Inspect/Debug                   Inspect/Debug
```