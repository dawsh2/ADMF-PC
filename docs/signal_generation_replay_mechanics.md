# Signal Generation and Replay: Precise Mechanics

This document explains the step-by-step mechanics of how grid search signal generation and regime-filtered replay work in ADMF-PC.

## Overview

The signal generation and replay system allows us to:
1. Pre-compute signals for multiple strategy parameter combinations
2. Store signals sparsely with classifier state changes
3. Replay signals efficiently, filtering by regime and strategy
4. Test different portfolio configurations without recomputing signals

## Signal Generation Mechanics

### 1. Grid Search Parameter Expansion

When you define a strategy with parameter grid:

```python
{
    'name': 'momentum',
    'function': momentum_strategy,
    'base_parameters': {'position_size': 0.1},
    'parameter_grid': {
        'fast_period': [5, 10, 20],
        'slow_period': [20, 50, 100],
        'signal_threshold': [0.005, 0.01, 0.02]
    }
}
```

The system generates 3 × 3 × 3 = 27 unique strategy variants:
- `momentum_fast_period_5_slow_period_20_signal_threshold_0.005`
- `momentum_fast_period_5_slow_period_20_signal_threshold_0.01`
- ... (25 more combinations)

### 2. Signal Generation Flow

```
Symbol/Timeframe Containers → Feature Container → Signal Storage
                                     ↓
                           (TimeAlignmentBuffer)
                                     ↓
                           Classifiers → Strategies
```

**Step-by-step process:**

1. **Bar Synchronization**: TimeAlignmentBuffer waits for all required bars
   ```python
   # Example: Strategy needs SPY@1m and QQQ@1m
   # Buffer waits until both arrive for timestamp T
   synchronized_bars = {
       'SPY_1m': Bar(timestamp=T, ...),
       'QQQ_1m': Bar(timestamp=T, ...)
   }
   ```

2. **Feature Calculation**: Features computed on synchronized data
   ```python
   features = {
       'SPY': {'sma_20': 150.5, 'rsi': 65, ...},
       'QQQ': {'sma_20': 380.2, 'rsi': 58, ...},
       'cross_symbol': {'correlation': 0.92, ...}
   }
   ```

3. **Classifier Execution**: Stateless classifiers determine regime
   ```python
   # Classifier called for every bar
   new_regime = trend_classifier(features)  # Returns 'TRENDING'
   
   # Only state CHANGES are stored
   if new_regime != current_regime:
       classifier_index.record_change(bar_idx=1000, 
                                    old='CHOPPY', 
                                    new='TRENDING')
   ```

4. **Strategy Execution**: Each strategy variant processes features
   ```python
   # All 27 momentum variants run
   for strategy_id, params in momentum_variants:
       signal = momentum_strategy(features, 
                                classifier_states={'trend': 'TRENDING'},
                                parameters=params)
       
       if signal.value != 0:  # Only non-zero signals stored
           signal_index.append_signal(bar_idx=1000, 
                                    signal_value=1.0,
                                    classifier_states=states)
   ```

### 3. Sparse Storage Structure

After generation, the storage looks like:

```
./signals/grid_search_2024/
├── signals/
│   ├── momentum_fast_5_slow_20_thresh_0.005.parquet
│   ├── momentum_fast_5_slow_20_thresh_0.005.meta.json
│   ├── momentum_fast_10_slow_50_thresh_0.01.parquet
│   ├── momentum_fast_10_slow_50_thresh_0.01.meta.json
│   └── ... (27 momentum files + 9 pairs files)
└── classifier_changes/
    └── trend_classifier.parquet
```

**Signal Index Format** (momentum_fast_10_slow_50_thresh_0.01.parquet):
```
bar_idx | symbol | value | classifiers                | bar_data
--------|--------|-------|---------------------------|----------
1000    | SPY    | 1.0   | {'trend': 'TRENDING'}     | {...}
1523    | SPY    | -1.0  | {'trend': 'TRENDING'}     | {...}
2105    | SPY    | 0.0   | {'trend': 'TRENDING'}     | {...}
3420    | SPY    | 1.0   | {'trend': 'TRENDING'}     | {...}
```

**Classifier Index Format** (trend_classifier.parquet):
```
bar_idx | old      | new
--------|----------|----------
0       | None     | CHOPPY
1000    | CHOPPY   | TRENDING
2500    | TRENDING | VOLATILE
3200    | VOLATILE | CHOPPY
4100    | CHOPPY   | TRENDING
```

## Signal Replay Mechanics

### 1. Topology Differences

**Signal Generation Topology:**
```
Data → Features → Strategies → Signal Storage
```

**Signal Replay Topology:**
```
Signal Storage → Signal Streamer → Portfolios → Risk → Execution
```

### 2. Regime-Filtered Loading

When replaying with `regime_filter='TRENDING'`:

1. **Load Indices**: SignalStreamer loads all signal files and classifier changes

2. **Prepare Replay Data**: For each bar with signals:
   ```python
   # Reconstruct classifier state at bar 1523
   classifier_state = classifier_index.get_state_at_bar(1523)
   # Returns: 'TRENDING' (from change at bar 1000)
   ```

3. **Apply Boundary-Aware Filtering**:
   ```python
   # BoundaryAwareReplay tracks positions
   if signal.value != 0 and not in_position:
       # Only emit if in target regime
       if current_regime == 'TRENDING':
           emit_signal()
           in_position = True
   elif signal.value == 0 and in_position:
       # ALWAYS emit close signals
       emit_signal()
       in_position = False
   ```

### 3. Sparse Replay Efficiency

With `sparse_replay=True`, only bars with signals are processed:

```
Traditional Replay: Process 97,500 bars (1 year @ 1min)
Sparse Replay: Process ~2,000 bars (only bars with signals)

Efficiency Gain: 48x faster
```

### 4. Portfolio Signal Filtering

Each portfolio subscribes only to its assigned strategies:

```python
# Portfolio configuration
{
    'id': 'conservative',
    'strategy_assignments': ['momentum_fast_10_slow_50_thresh_0.01']
}

# Event bus subscription with filter
event_bus.subscribe(
    EventType.SIGNAL,
    portfolio.receive_event,
    filter_func=lambda e: e.payload['strategy_id'] in strategy_assignments
)
```

## Event Flow Examples

### Example 1: Signal Generation Event Flow

```
Bar Event (T=1000)
├─ Feature Container receives synchronized bars
├─ Calculates features
├─ Calls trend_classifier → 'TRENDING' (changed from 'CHOPPY')
├─ Stores classifier change
├─ Calls all 36 strategy variants
│  ├─ momentum_fast_5_slow_20: No signal (threshold not met)
│  ├─ momentum_fast_10_slow_50: BUY signal → Store
│  ├─ pairs_zscore_2.0: No signal (wrong regime)
│  └─ ... (33 more)
└─ Continue to next bar
```

### Example 2: Regime-Filtered Replay

```
Signal Replay (TRENDING regime only)
├─ Load classifier index → Find TRENDING periods: [1000-2500, 4100-5000]
├─ Load signal indices → Filter to TRENDING bars
├─ For bar 1523 (in TRENDING):
│  ├─ Emit signal from momentum_fast_10_slow_50
│  └─ Skip pairs strategies (wrong regime)
├─ For bar 2600 (in VOLATILE):
│  └─ Skip all signals (wrong regime)
└─ For bar 4200 (in TRENDING):
   └─ Emit signals again
```

### Example 3: Boundary Trade Handling

```
Position opened in TRENDING at bar 4200
├─ Regime changes to CHOPPY at bar 4500
├─ Close signal at bar 4600 (in CHOPPY)
└─ Signal STILL emitted (boundary handling)
   └─ Ensures proper position closure
```

## Storage Efficiency Analysis

For a typical momentum strategy over 1 year of minute data:

**Traditional Storage** (store signal value for every bar):
- 97,500 bars × 4 bytes = 390 KB per strategy
- 27 strategies × 390 KB = 10.5 MB

**Sparse Storage** (store only non-zero signals + changes):
- ~2,000 signals × 20 bytes = 40 KB per strategy
- 27 strategies × 40 KB = 1.08 MB
- 1 classifier × 10 changes × 12 bytes = 120 bytes

**Total: 1.08 MB vs 10.5 MB = 90% reduction**

## Key Benefits

1. **Computational Efficiency**: Calculate features and run strategies once, replay many times
2. **Storage Efficiency**: 90%+ reduction through sparse storage
3. **Replay Speed**: 48x faster with sparse replay (skip empty bars)
4. **Regime Analysis**: Test strategies in specific market conditions
5. **Parameter Optimization**: Test portfolio configurations without recomputing signals
6. **Event Tracing**: Full observability of signal generation and replay

## Implementation Checklist

- [x] Signal storage components (SignalIndex, ClassifierChangeIndex)
- [x] Signal generation component for Feature Container
- [x] Signal streamer component for replay
- [x] Signal generation topology
- [x] Signal replay topology
- [x] Boundary-aware replay for regime filtering
- [x] Grid search parameter expansion
- [ ] Integration with existing container factory
- [ ] Integration with coordinator/sequencer
- [ ] Event tracing observers for metrics collection