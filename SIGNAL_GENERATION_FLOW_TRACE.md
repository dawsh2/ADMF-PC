# Signal Generation Execution Flow with Parameter Expansion

## Overview

The signal generation flow in ADMF-PC supports parameter expansion, allowing users to test multiple parameter combinations in a single run. This document traces the complete execution flow from configuration to stored signals.

## 1. Configuration with Parameter Expansion

### Input Configuration (`config/test_parameter_expansion.yaml`)
```yaml
strategies:
  - name: 'ma_crossover'
    type: 'ma_crossover'
    params:
      fast_period: [5, 10, 20]  # 3 values
      slow_period: [20, 30, 50]  # 3 values
      # Creates 9 combinations total

topology: 'signal_generation'
```

## 2. Parameter Expansion in TopologyBuilder

### Location: `src/core/coordinator/topology.py`

#### Step 1: Context Building (`_build_context()`)
- Line 166: `_expand_strategy_parameters()` is called before building context
- This transforms the strategy list from 1 strategy with list parameters to 9 individual strategy configurations

#### Step 2: Parameter Expansion (`_expand_strategy_parameters()`)
- Lines 804-896: Handles the expansion logic
- For each strategy with list parameters:
  1. Identifies which parameters are lists vs scalars
  2. Uses `itertools.product()` to generate all combinations
  3. Validates combinations (e.g., fast_period < slow_period for MA strategies)
  4. Creates individual strategy configs with descriptive names

Example expansion:
```python
# Input
{'name': 'ma_crossover', 'params': {'fast_period': [5, 10], 'slow_period': [20, 30]}}

# Output
[
  {'name': 'ma_crossover_5_20', 'type': 'ma_crossover', 'params': {'fast_period': 5, 'slow_period': 20}},
  {'name': 'ma_crossover_5_30', 'type': 'ma_crossover', 'params': {'fast_period': 5, 'slow_period': 30}},
  {'name': 'ma_crossover_10_20', 'type': 'ma_crossover', 'params': {'fast_period': 10, 'slow_period': 20}},
  {'name': 'ma_crossover_10_30', 'type': 'ma_crossover', 'params': {'fast_period': 10, 'slow_period': 30}}
]
```

## 3. Feature Inference

### Location: `src/core/coordinator/topology.py`

#### Step 3: Feature Inference (`_infer_and_inject_features()`)
- Lines 1165-1246: Automatically determines required features
- For MA crossover strategies, it identifies needed SMA features:
  - `sma_5`, `sma_10`, `sma_20`, `sma_30`, `sma_50`
- Creates feature configurations and injects into context

## 4. Container Creation

### Signal Generation Topology Pattern (`config/patterns/topologies/signal_generation.yaml`)

The topology creates:
1. **Root Container**: Parent container with hierarchical event bus
2. **Data Containers**: One per symbol/timeframe (e.g., `SPY_1m_data`)
3. **Strategy Container**: Single container that receives all strategies

### Strategy Container Creation
- Gets expanded strategies from config (now 9 strategies instead of 1)
- Receives feature configurations
- Creates `StrategyState` component

## 5. Strategy Execution

### Location: `src/strategy/state.py`

#### StrategyState Component
- Lines 206-258: `_load_strategies_from_config()` loads all strategies
- Each strategy gets a unique ID: `{symbol}_{strategy_name}`
  - Example: `SPY_ma_crossover_5_20`

#### Bar Processing Flow
1. **BAR Event Received** (line 273: `on_bar()`)
2. **Feature Calculation** (line 311: `update_bar()`)
   - Updates rolling price windows
   - Computes configured features (SMAs)
3. **Strategy Execution** (line 324: `_execute_strategies()`)
   - Calls each strategy function with features
   - MA crossover strategy generates signal based on SMA positions

#### Signal Publishing
- Line 362: `_publish_signal()` creates SIGNAL event
- Published to parent (root) container for cross-container visibility

## 6. Signal Storage with Sparse Format

### Location: `src/core/events/observers/sparse_portfolio_tracer.py`

#### Sparse Storage Concept
- Only stores signal **changes**, not every signal
- If strategy stays long for 25 bars, only stores 1 event (not 25)

#### Storage Process
1. **Signal Reception** (line 62: `on_event()`)
   - Filters for SIGNAL events from managed strategies
2. **Change Detection** (line 84: `storage.process_signal()`)
   - Compares with previous signal state
   - Only stores if direction changed
3. **Compression Tracking**
   - Tracks total signals vs stored changes
   - Reports compression ratio (e.g., 50:1 for stable strategies)

### Location: `src/core/events/storage/temporal_sparse_storage.py`

#### Storage Format
```json
{
  "metadata": {
    "total_bars": 100,
    "total_changes": 4,
    "compression_ratio": 0.04,
    "strategy_parameters": {
      "ma_crossover_5_20": {
        "type": "ma_crossover",
        "params": {"fast_period": 5, "slow_period": 20}
      }
    }
  },
  "changes": [
    {"idx": 20, "ts": "2024-01-01T09:50:00", "sym": "SPY", "val": 1, "strat": "SPY_ma_crossover_5_20", "px": 470.5},
    {"idx": 45, "ts": "2024-01-01T10:15:00", "sym": "SPY", "val": -1, "strat": "SPY_ma_crossover_5_20", "px": 469.8}
  ]
}
```

## 7. Output Organization

### Workspace Structure
```
workspaces/
└── parameter_expansion_test/
    └── 20241206_143000/
        └── strategy/
            ├── signals_strategy_20241206_143010.json  # All 9 strategies' signals
            └── performance_metrics.json               # Optional performance data
```

## Key Benefits of This Architecture

1. **Efficient Parameter Search**: Test many parameter combinations in one run
2. **Automatic Feature Detection**: No need to manually specify features
3. **Sparse Storage**: 50-100x compression for stable strategies
4. **Clean Organization**: Results organized by workflow/run/container
5. **Strategy Metadata**: Parameters stored with signals for analysis

## Example: 9 Strategy Combinations

For the test configuration with 9 MA crossover variants:
- **Without sparse storage**: 900 signal records (9 strategies × 100 bars)
- **With sparse storage**: ~18-36 records (2-4 changes per strategy)
- **Compression ratio**: 25-50x

The sparse storage makes it practical to run large parameter sweeps and store all results efficiently.