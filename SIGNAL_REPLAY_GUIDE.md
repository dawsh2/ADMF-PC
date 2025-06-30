# Signal Replay System Guide

## Overview

The signal replay system allows you to:
1. Generate trading signals once and save them to disk
2. Replay those signals through portfolio/execution with different risk parameters
3. Quickly test different exit strategies without recomputing entry signals
4. Detect when strategies need recomputation due to new data

## Architecture

### Phase 1: Global Trace Storage ✅
- **StreamingMultiStrategyTracer** writes signal traces to `traces/` directory
- Traces are stored in Parquet format with strategy metadata
- Each strategy gets its own subdirectory: `traces/{strategy_type}/{strategy_name}.parquet`

### Phase 2: Analysis Tools ✅
- **TraceStore** class provides unified access to signal traces
- Notebook templates for analyzing signals and backtests
- Located in `src/analytics/notebooks/`

### Phase 3: Signal Replay ✅
- **SignalReplayHandler** reads traces and publishes SIGNAL events
- Integrates with existing container/topology system
- Accessible via `--signal-replay` flag

### Phase 4: Intelligent Re-running ✅
- **StrategyFreshnessChecker** detects when traces are outdated
- Checks for new data, missing symbols, or parameter changes
- Provides suggestions for updating only what's needed

## Usage Examples

### 1. Generate Signals

First, generate signals for your strategies:

```bash
python main.py --config config/my_strategy.yaml --signal-generation
```

This creates traces in `./traces/` directory.

### 2. Replay with Original Risk Parameters

Replay signals using the same risk parameters:

```bash
python main.py --config config/my_strategy.yaml --signal-replay
```

### 3. Replay with Different Risk Parameters

Create a replay config with risk overrides:

```yaml
# replay_config.yaml
name: my_strategy_replay
symbols: ["SPY"]
initial_capital: 100000

# Same strategy config (to identify traces)
strategy: [
  {
    sma_crossover: {fast_period: 15, slow_period: 50},
    risk: {
      stop_loss: 0.002,
      take_profit: 0.004
    }
  }
]

# Override risk parameters for replay
risk: {
  stop_loss: 0.001,   # Tighter stop loss
  take_profit: 0.006  # Wider take profit
}
```

Run replay:

```bash
python main.py --config replay_config.yaml --signal-replay
```

### 4. Check Strategy Freshness

The system automatically checks if traces are up-to-date during replay:

```
⚠️ Some strategies need updating:
  - sma_crossover_15_50: New data available after 2024-01-15
💡 To update: python main.py --config config/my_strategy.yaml --signal-generation
```

## Configuration Options

### Signal Generation Config

```yaml
# Standard strategy configuration
strategy: [
  {
    strategy_type: {param1: value1, param2: value2},
    risk: {
      stop_loss: 0.002,
      take_profit: 0.004
    }
  }
]

# Optional: specify trace directory
traces_dir: ./my_traces  # defaults to ./traces
```

### Signal Replay Config

```yaml
# Required: same strategy config to identify traces
strategy: [...]

# Optional: override risk parameters
risk: {
  stop_loss: 0.001,
  take_profit: 0.006,
  position_size: 0.1,
  max_positions: 3
}

# Optional: execution parameters
execution: {
  commission: 0.0001,
  slippage: 0.0001
}
```

## Implementation Details

### SignalReplayHandler

Located in `src/data/handlers.py`, this handler:
- Loads signal traces based on strategy configurations
- Publishes SIGNAL events in chronological order
- Supports multi-symbol, multi-strategy replay
- Provides clear error messages when traces are missing

### signal_replay Topology

The `config/patterns/topologies/signal_replay.yaml` topology:
- Creates a data container with SignalReplayHandler
- Sets up portfolio and execution containers
- Uses standard event flow: SIGNAL → ORDER → FILL

### Risk Parameter Flow

1. Original risk params are embedded in signals during generation
2. Portfolio checks for risk overrides in config during replay
3. If overrides exist, they take precedence over embedded params
4. This allows testing different exit strategies without regeneration

## Benefits

1. **Performance**: Signal generation can be expensive (indicators, ML models)
2. **Experimentation**: Quickly test different risk parameters
3. **Reproducibility**: Signals are stored and can be analyzed
4. **Optimization**: Only recompute when data actually changes

## Troubleshooting

### "Signal trace not found" Error

This means signals haven't been generated yet. Run:
```bash
python main.py --config your_config.yaml --signal-generation
```

### Outdated Traces Warning

The system detected new data or parameter changes. Update traces:
```bash
python main.py --config your_config.yaml --signal-generation
```

### Risk Overrides Not Working

Ensure your replay config includes:
1. The exact same strategy configuration (for trace identification)
2. Risk overrides at the top level of the config

## Future Enhancements

1. **Partial Signal Generation**: Only regenerate specific strategies
2. **Signal Filtering**: Replay subset of signals based on criteria
3. **Multi-Config Replay**: Test multiple risk profiles in one run
4. **Signal Analysis UI**: Web interface for exploring traces

## Example Test Script

See `test_signal_replay.py` for a complete example that:
1. Generates signals with one set of risk parameters
2. Runs a normal backtest for comparison
3. Replays signals with different risk parameters
4. Shows how results differ based on exit strategy