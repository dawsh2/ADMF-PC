# Strategy Trace Storage Enhancement

## Overview

This document outlines improvements to our strategy trace storage system to make strategies fully queryable by their parameters while maintaining our efficient sparse storage format.

## Current Limitations

### The Problem with metadata.json

Currently, strategy configurations are stored in a single `metadata.json` file per run:

```
results/
└── run_20241224_123456/
    ├── metadata.json          # All strategy configs buried here
    └── traces/
        └── SPY_5m/
            └── signals/
                └── bollinger_bands/
                    └── SPY_5m_bollinger_bands_0_1234.parquet
```

**Why this fails:**
- No direct mapping from strategy_id to its parquet file
- Difficult to query strategies by parameters across runs
- No performance metrics linked to configurations
- Cannot answer: "Show me all bollinger strategies where period=20"

### The Filename Ambiguity Problem

Current filenames attempt to encode parameters but fail for complex strategies:

```
SPY_5m_bollinger_bands_0_1234.parquet    # What parameters?
SPY_5m_ensemble_ma_crossover_0_5678.parquet    # What's inside?
```

## Proposed Solution: Self-Documenting Traces

### 1. Enhanced Parquet Files

Each trace file becomes self-documenting with embedded metadata:

```
Current Sparse Format (Keep This!)
┌─────┬──────────────┬─────┬─────┬───────────┬──────┐
│ idx │ ts           │ sym │ val │ strat     │ px   │
├─────┼──────────────┼─────┼─────┼───────────┼──────┤
│ 0   │ 2024-01-01.. │ SPY │ 1   │ strategy_0│ 470.5│
│ 45  │ 2024-01-01.. │ SPY │ 0   │ strategy_0│ 471.2│
│ 127 │ 2024-01-01.. │ SPY │ -1  │ strategy_0│ 469.8│
└─────┴──────────────┴─────┴─────┴───────────┴──────┘

Enhanced with Metadata (New!)
┌─────┬──────────────┬─────┬─────┬───────────┬──────┬──────────────┬─────────────────────────┐
│ idx │ ts           │ sym │ val │ strat     │ px   │ strat_hash   │ metadata                │
├─────┼──────────────┼─────┼─────┼───────────┼──────┼──────────────┼─────────────────────────┤
│ 0   │ 2024-01-01.. │ SPY │ 1   │ strategy_0│ 470.5│ a3f4b2c1d5e6 │ {"type": "bollinger..."}│
│ 45  │ 2024-01-01.. │ SPY │ 0   │ strategy_0│ 471.2│ a3f4b2c1d5e6 │ NULL                    │
│ 127 │ 2024-01-01.. │ SPY │ -1  │ strategy_0│ 469.8│ a3f4b2c1d5e6 │ NULL                    │
└─────┴──────────────┴─────┴─────┴───────────┴──────┴──────────────┴─────────────────────────┘
```

**Key points:**
- Metadata stored ONCE on first signal (no duplication)
- Strategy hash enables cross-run identification
- Original sparse format preserved

### 2. PyArrow Table Metadata

Additionally, store configuration in the file's metadata (not visible in data):

```
PyArrow Table Properties:
{
  "strategy_id": "strategy_0",
  "strategy_hash": "a3f4b2c1d5e6",
  "strategy_config": {
    "bollinger_bands": {
      "period": 20,
      "std_dev": 2.0
    },
    "constraints": "intraday"
  }
}
```

This makes files self-documenting even without reading the data.

### 3. Strategy Index File

Create a queryable index of all strategies in a run:

```
strategy_index.parquet
┌────────────┬──────────────┬─────────────────┬────────┬──────────┬─────────────┬───────────────────────────┐
│ strategy_id│ strategy_hash│ strategy_type   │ period │ std_dev  │ constraints │ trace_path                │
├────────────┼──────────────┼─────────────────┼────────┼──────────┼─────────────┼───────────────────────────┤
│ strategy_0 │ a3f4b2c1d5e6 │ bollinger_bands │ 20     │ 2.0      │ intraday    │ traces/SPY_5m/signals/... │
│ strategy_1 │ b7e8f9a0c1d2 │ bollinger_bands │ 25     │ 2.5      │ intraday    │ traces/SPY_5m/signals/... │
│ strategy_2 │ c3d4e5f6a7b8 │ ma_crossover    │ 10     │ 30       │ NULL        │ traces/SPY_5m/signals/... │
└────────────┴──────────────┴─────────────────┴────────┴──────────┴─────────────┴───────────────────────────┘
```

## Why This Approach Wins

### Storage Efficiency

Compare with the "flattened columns" approach that was considered:

```
BAD: Flattened Approach (Duplicates metadata on every row)
┌─────┬─────┬──────────────┬─────────────────┬──────────────┐
│ idx │ val │ meta_period  │ meta_std_dev    │ meta_type    │
├─────┼─────┼──────────────┼─────────────────┼──────────────┤
│ 0   │ 1   │ 20          │ 2.0             │ bollinger... │  ← Duplicated
│ 45  │ 0   │ 20          │ 2.0             │ bollinger... │  ← Duplicated
│ 127 │ -1  │ 20          │ 2.0             │ bollinger... │  ← Duplicated
└─────┴─────┴──────────────┴─────────────────┴──────────────┘

GOOD: Our Approach (Metadata stored once)
┌─────┬─────┬──────────────┬─────────────────────────┐
│ idx │ val │ strat_hash   │ metadata                │
├─────┼─────┼──────────────┼─────────────────────────┤
│ 0   │ 1   │ a3f4b2c1d5e6 │ {"type": "bollinger..."}│  ← Stored once
│ 45  │ 0   │ a3f4b2c1d5e6 │ NULL                    │
│ 127 │ -1  │ a3f4b2c1d5e6 │ NULL                    │
└─────┴─────┴──────────────┴─────────────────────────┘
```

### Query Power

The strategy index enables powerful queries without opening every file:

```sql
-- Find all bollinger strategies with specific parameters
SELECT * FROM strategy_index.parquet
WHERE strategy_type = 'bollinger_bands'
  AND period = 20
  AND std_dev = 2.0

-- Find strategies used across multiple runs
SELECT strategy_hash, COUNT(*) as run_count
FROM 'results/*/strategy_index.parquet'
GROUP BY strategy_hash
HAVING run_count > 1

-- Load traces for top performers
WITH top_strategies AS (
  SELECT trace_path
  FROM strategy_index.parquet
  WHERE sharpe_ratio > 1.5
)
SELECT * FROM top_strategies t
CROSS JOIN read_parquet(t.trace_path)
```

## Handling Composite Strategies

For ensemble/composite strategies, the metadata captures the full hierarchy using the modern nested structure:

```json
{
  "strategy_type": "ensemble",
  "strategy_hash": "d4e5f6a7b8c9",
  "config": {
    "strategy": [
      {
        "sma_crossover": {
          "fast_period": 15,
          "slow_period": 50,
          "weight": 0.4
        }
      },
      {
        "bollinger_bands": {
          "period": 20,
          "std": 2.0,
          "weight": 0.6
        }
      },
      {
        "threshold": "0.5 AND volume > sma(volume, 20) AND market_hours() == 'regular'"
      }
    ]
  }
}

```

For more complex nested compositions:

```json
{
  "strategy_type": "composite",
  "strategy_hash": "e5f6a7b8c9d0",
  "config": {
    "strategy": [
      {
        "weight": 0.6,
        "strategy": [
          {"ma_crossover": {"fast": 10, "slow": 30, "weight": 0.5}},
          {"momentum": {"period": 14, "weight": 0.5}}
        ],
        "threshold": "0.3 AND adx(14) > 25"
      },
      {
        "weight": 0.4,
        "strategy": [
          {"bollinger_bands": {"period": 20, "std": 2.0, "weight": 0.6}},
          {"rsi_extreme": {"period": 14, "oversold": 30, "weight": 0.4}}
        ],
        "threshold": "0.5 AND adx(14) < 20"
      }
    ]
  }
}
```

## Strategy Hashing

Each unique configuration gets a deterministic hash:

```
Configuration:                          Hash:
bollinger(20, 2.0)                 →   a3f4b2c1d5e6
bollinger(20, 2.0)                 →   a3f4b2c1d5e6  (same!)
bollinger(25, 2.0)                 →   b7e8f9a0c1d2  (different)
```

This enables:
- Finding identical strategies across runs
- Deduplicating optimization results
- Building performance histories for specific configurations

## Performance Analytics Integration

The strategy index becomes a powerful analytical tool:

```
Enhanced strategy_index.parquet (after analysis)
┌──────────────┬─────────────────┬──────────┬────────────┬─────────────┬──────────────────┐
│ strategy_hash│ strategy_type   │ sharpe   │ total_pnl  │ max_drawdown│ win_rate         │
├──────────────┼─────────────────┼──────────┼────────────┼─────────────┼──────────────────┤
│ a3f4b2c1d5e6 │ bollinger_bands │ 1.45     │ 12,500     │ -8.2%       │ 0.58             │
│ b7e8f9a0c1d2 │ bollinger_bands │ 0.89     │ 5,200      │ -12.5%      │ 0.52             │
│ c3d4e5f6a7b8 │ ma_crossover    │ 2.10     │ 18,900     │ -5.1%       │ 0.64             │
└──────────────┴─────────────────┴──────────┴────────────┴─────────────┴──────────────────┘
```

## Migration Path

1. **Phase 1**: Add metadata column and strategy hash to new traces
2. **Phase 2**: Generate strategy_index.parquet during finalization
3. **Phase 3**: Add PyArrow table metadata
4. **Phase 4**: Build analytics queries using the new structure
5. **Phase 5**: Migration tool for existing workspaces (optional)

## Summary

This enhancement provides:
- **Self-documenting trace files** with embedded configurations
- **Efficient storage** maintaining our sparse format
- **Powerful queries** through the strategy index
- **Cross-run analysis** via strategy hashing
- **Performance tracking** linked to exact configurations

No more guessing what parameters a strategy used - it's all in the data.