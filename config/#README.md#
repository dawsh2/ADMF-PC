# Progressive Configuration Examples
# From simple to complex trading system configurations

## Directory Structure

Each strategy lives in its own directory under `configs/`:

```
configs/
├── simple_ma_crossover/
│   ├── config.yaml
│   ├── results/
│   │   └── 2024_12_20_143022/
│   │       ├── metadata.json
│   │       └── traces/
│   │           └── sma_crossover/
│   │               └── fast_10_slow_30.parquet
│   └── notebooks/
│       └── analysis.ipynb
│
├── ma_crossover_research/
│   ├── config.yaml
│   ├── results/
│   │   └── 2024_12_21_091533/
│   │       ├── metadata.json
│   │       └── traces/
│   │           └── sma_crossover/
│   │               ├── fast_5_slow_20.parquet
│   │               ├── fast_5_slow_30.parquet
│   │               ├── fast_10_slow_20.parquet
│   │               └── ... (16 more combinations)
│   ├── notebooks/
│   │   └── parameter_analysis.ipynb
│   │
│   └── ma_crossover_filtered/  # Branched research
│       ├── config.yaml
│       ├── results/
│       │   └── 2024_12_22_114523/
│       │       ├── metadata.json
│       │       └── traces/
│       │           └── sma_crossover/
│       │               ├── fast_10_slow_30_volume_filter.parquet
│       │               └── fast_15_slow_50_rsi_filter.parquet
│       └── notebooks/
│           └── filter_analysis.ipynb
│
└── production_system/
    ├── config.yaml
    ├── production.yaml  # Optional production overrides
    └── results/
        └── ...
```

---

# ============================================================================
# LEVEL 1: Hello World - Single Strategy
# ============================================================================
# Location: configs/simple_ma_crossover/config.yaml

```yaml
name: simple_ma_crossover
symbols: ["SPY"]
timeframe: "5m"

strategy: [
  {sma_crossover: {fast_period: 10, slow_period: 30}}
]
```

**Run:**
```bash
python main.py -c configs/simple_ma_crossover/config.yaml
```

**Results in:**
```
configs/simple_ma_crossover/results/2024_12_20_143022/
├── metadata.json  # Contains full config + performance metrics
└── traces/
    └── sma_crossover/
        └── fast_10_slow_30.parquet  # Sparse signal trace
```

---

# ============================================================================
# LEVEL 2: Adding Parameter Research
# ============================================================================
# Location: configs/ma_crossover_research/config.yaml

```yaml
name: ma_crossover_research
symbols: ["SPY"]
timeframe: "5m"

# Exploring different parameters
strategy: [
  {sma_crossover: {
    fast_period: [5, 10, 15, 20],
    slow_period: [20, 30, 50, 100]
  }},
  {bollinger_bands: {period: [11, 12, 13], std_dev: range(0.5, 4.0, 0.5)}}
]
```

**Results in:**
```
configs/ma_crossover_research/results/2024_12_21_091533/
├── metadata.json  # Summary of all 16 combinations tested
└── traces/
    └── sma_crossover/
        ├── fast_5_slow_20.parquet
        ├── fast_5_slow_30.parquet
        ├── fast_5_slow_50.parquet
        └── ... (13 more files)
```

**In notebook:**
```python
# Load results
import json
import pandas as pd
from pathlib import Path

with open('../results/latest/metadata.json') as f:
    metadata = json.load(f)

# See best performing parameters
best = metadata['strategies_tested']['sma_crossover']['best_params']
print(f"Best params: fast={best['fast_period']}, slow={best['slow_period']}")
```

---

# ============================================================================
# LEVEL 3: Feature-Based Thresholds
# ============================================================================
# Location: configs/filtered_ma/config.yaml

```yaml
name: filtered_ma_crossover
symbols: ["SPY"]
timeframe: "5m"

strategy: [
  {
    sma_crossover: {fast_period: 15, slow_period: 50},
    threshold: "sma_crossover AND volume > sma(volume, 20) * 1.2"
  }
]
```

**Advanced with boolean logic:**
```yaml
strategy: [
  {momentum: {period: 14}},
  {rsi_reversal: {period: 14, oversold: 30}},
  {
    threshold: |
      (momentum OR rsi_reversal) AND 
      (volume > sma(volume, 20) * 1.5 OR atr(14) > 0.002) AND
      market_hours() == 'regular'
  }
]
```

**Directional thresholds:**
```yaml
strategy: [
  {
    macd: {fast: 12, slow: 26, signal: 9},
    threshold: {
      long: "macd > 0 AND rsi(14) < 70 AND sma(close, 50) > sma(close, 200)",
      short: "macd < 0 AND rsi(14) > 30 AND sma(close, 50) < sma(close, 200)"
    }
  }
]
```

---

# ============================================================================
# LEVEL 4: Multiple Strategies with Thresholds
# ============================================================================
# ============================================================================
# LEVEL 4: Multiple Strategies
# ============================================================================
# Location: configs/mean_reversion_sweep/config.yaml

```yaml
name: mean_reversion_sweep
symbols: ["SPY"]
timeframe: "5m"

# Test multiple strategy types with conditions
strategy: [
  {
    bollinger_bands: {
      period: [15, 20, 25],
      std: [1.5, 2.0, 2.5]
    },
    threshold: "bollinger_bands AND volume > sma(volume, 20) * 1.2"
  },
  {
    rsi_threshold: {
      period: [10, 14],
      oversold: [25, 30],
      overbought: [70, 75]
    },
    threshold: "rsi_threshold AND atr_percentile(20) < 70"
  }
]
```

**Results structure:**
```
results/2024_12_22_104511/
├── metadata.json
└── traces/
    ├── bollinger_bands/
    │   ├── period_15_std_1.5.parquet
    │   ├── period_15_std_2.0.parquet
    │   └── ... (9 total)
    └── rsi_threshold/
        ├── period_10_oversold_25_overbought_70.parquet
        └── ... (8 total)
```

---

# ============================================================================
# LEVEL 5: Risk Management
# ============================================================================
# Location: configs/dynamic_risk/config.yaml

```yaml
name: filtered_ma_with_risk
symbols: ["SPY"]
timeframe: "5m"

# Global risk settings
risk:
  position_size: 0.1  # 10% per trade
  max_positions: 3

strategy: [
  {
    sma_crossover: {fast_period: 15, slow_period: 50},
    threshold: |
      sma_crossover AND 
      volume > sma(volume, 20) * 1.2 AND
      atr(14) > 0.001
    risk: {
      stop_loss: 0.002,  # 0.2%
      take_profit: 0.004,  # 0.4%
      max_holding_period: 100  # bars
    }
  }
]
```

---

# ============================================================================
# LEVEL 6: Ensemble with Weighted Strategies
# ============================================================================
# Location: configs/ensemble_research/config.yaml

```yaml
name: ensemble_research
symbols: ["SPY"]
timeframe: "5m"

# Weighted ensemble with threshold
strategy: [
  {sma_crossover: {fast_period: 15, slow_period: 50, weight: 0.4}},
  {rsi_threshold: {period: 14, oversold: 30, overbought: 70, weight: 0.3}},
  {bollinger_bands: {period: 20, std: 2.0, weight: 0.3}},
  {threshold: "0.5 AND volume > sma(volume, 20) AND market_hours() == 'regular'"}
]
```

**Testing different thresholds:**
```yaml
strategy: [
  {momentum: {period: 14, weight: 0.6}},
  {mean_reversion: {period: 20, weight: 0.4}},
  {
    threshold: [
      "0.5",  # Pure weighted
      "0.5 AND volume > sma(volume, 20)",  # With volume
      "0.3 AND atr(14) > 0.001",  # Lower threshold with volatility
      "0.7 OR volume > sma(volume, 20) * 3.0"  # High confidence OR spike
    ]
  }
]
```

---

# ============================================================================
# LEVEL 7: Conditional Strategies (Regime-Based)
# ============================================================================
# Location: configs/regime_adaptive/config.yaml

```yaml
name: regime_adaptive_system
symbols: ["SPY"]
timeframe: "5m"

strategy: [
  {
    momentum: {period: 14, weight: 0.5},
    threshold: |
      momentum > 0 AND 
      atr(14) > sma(atr(14), 50) * 1.1 AND
      trend_strength() > 0.5
  },
  {
    bollinger_bands: {period: 20, std: 2.0, weight: 0.5},
    threshold: |
      bollinger_bands != 0 AND
      atr(14) < sma(atr(14), 50) * 0.9 AND
      abs(price - vwap()) / vwap() > 0.002
  },
  {threshold: 0.5}  # Overall threshold
]
```

**Alternative: Explicit regime in threshold:**
```yaml
strategy: [
  {momentum: {period: 14}},
  {bollinger_bands: {period: 20, std: 2.0}},
  {
    threshold: |
      (volatility_regime() == 'high' AND momentum) OR
      (volatility_regime() == 'low' AND bollinger_bands)
  }
]
```

---

# ============================================================================
# LEVEL 8: Research to Production Workflow
# ============================================================================
# Location: configs/bb_production/

## Step 1: Broad Research (config.yaml)
```yaml
name: bollinger_research
symbols: ["SPY", "QQQ"]
timeframe: "5m"

# Cast wide net
strategy:
  - bollinger_bands:
      period: range(15, 30, 5)
      std: range(1.5, 2.5, 0.5)
```

## Step 2: Refine Based on Results
```yaml
# After reviewing results/2024_12_24_*/metadata.json
# Update config.yaml for next iteration

name: bollinger_refined
symbols: ["SPY"]  # QQQ had poor performance
timeframe: "5m"

strategy:
  - bollinger_bands:
      period: [18, 20, 22]  # Best were around 20
      std: [1.8, 2.0, 2.2]  # Best were around 2.0
    filter: "volume > sma(volume, 20) * 1.2"  # Add filter
```

## Step 3: Production Config (production.yaml)
```yaml
# After multiple research iterations
# Lock in exact parameters for production

name: bollinger_production
symbols: ["SPY"]
timeframe: "5m"

strategy:
  bollinger_bands:
    params: {period: 20, std: 2.0}  # No lists, fixed values
    filter: |
      volume > sma(volume, 20) * 1.2 and
      market_hours() == 'regular'
    risk:
      stop_loss: 0.002
      position_size: 0.05
```

---

# ============================================================================
# LEVEL 9: Complex Composite Strategies
# ============================================================================
# Location: configs/advanced_composite/config.yaml

```yaml
name: advanced_composite_system
symbols: ["SPY"]
timeframe: "5m"

strategy: [
  # Trend following group (60% of capital)
  {
    weight: 0.6,
    strategy: [
      {ma_crossover: {fast: 10, slow: 30, weight: 0.5}},
      {momentum: {period: 14, weight: 0.5}}
    ],
    threshold: "0.3 AND adx(14) > 25 AND volume > sma(volume, 20) * 1.2"
  },
  
  # Mean reversion group (40% of capital)
  {
    weight: 0.4,
    strategy: [
      {bollinger_bands: {period: 20, std: 2.0, weight: 0.6}},
      {rsi_extreme: {period: 14, oversold: 30, weight: 0.4}}
    ],
    threshold: "0.5 AND adx(14) < 20 AND abs(price - vwap()) / vwap() > 0.002"
  },
  
  # Overall system threshold
  {threshold: "0.5 AND market_hours() == 'regular'"}
]
```

**Multi-timeframe composite:**
```yaml
strategy: [
  # Hourly trend confirmation
  {
    weight: 0.3,
    timeframe: "1h",
    trend_following: {ma_period: 20},
    threshold: "trend_following AND rsi(14, '1h') > 50"
  },
  
  # 5-minute entries
  {
    weight: 0.7,
    momentum: {period: 14},
    threshold: |
      momentum AND 
      close > vwap() AND
      volume > sma(volume, 20, '5m') * 1.5
  },
  
  # Combined threshold
  {threshold: 0.7}
]
```

---


# ============================================================================
# LEVEL 11: Dynamic Risk Management
# ============================================================================
# Location: configs/dynamic_exits/config.yaml

```yaml
name: dynamic_risk_management
symbols: ["SPY"]
timeframe: "5m"

strategy:
  - momentum:
      period: 14
    filter: {volume_above: {multiplier: 1.2}}
    
    # Static risk parameters
    risk:
      stop_loss: 0.002  # Default 0.2%
      take_profit: 0.004  # Default 0.4%

---
# Advanced: Dynamic exits based on market conditions
strategy:
  - momentum:
      period: 14
    
    risk:
      # Base stops (fallback)
      stop_loss: 0.002
      take_profit: 0.004
      
      # Dynamic exit conditions
      exit_filters: [
        # Wider stops in strong trends
        {
          condition: {trend_strength: {min: 0.7}}
          stop_loss: 0.003  # 0.3%
          take_profit: 0.008  # 0.8%
        },
        
        # Tighter stops in low volatility
        {
          condition: {atr: {below_percentile: 20}}
          stop_loss: 0.001
          take_profit: 0.002
        },
        
        # Trailing stop when profitable
        {
          condition: {position_pnl: {above: 0.003}}
          trailing_stop: {activate: 0.003, distance: 0.002}
        },
        
        # Time-based exit
        {
          condition: {holding_time: {bars: 100}}
          exit: true
        },
        
        # ATR-based stops
        {
          condition: {volatility_regime: "high"}
          stop_loss: "2 * atr(14)"
          take_profit: "4 * atr(14)"
        }
      ]

---
# Research: Testing different exit strategies
strategy:
  - bollinger_bands:
      period: 20
      std: 2.0
    
    # Test multiple risk variants
    risk_variants: [
      {
        name: "fixed"
        stop_loss: 0.002
        take_profit: 0.004
      },
      {
        name: "atr_based"
        stop_loss: "1.5 * atr(14)"
        take_profit: "3 * atr(14)"
      },
      {
        name: "trend_adaptive"
        exit_filters: [
          {
            condition: {trend_strength: {above: 0.5}}
            stop_loss: 0.003
            take_profit: 0.008
          },
          {
            condition: {trend_strength: {below: 0.3}}
            stop_loss: 0.001
            take_profit: 0.002
          }
        ]
      },
      {
        name: "partial_exits"
        exit_filters: [
          # Take half profit at first target
          {
            condition: {position_pnl: {above: 0.002}}
            action: {reduce_position: 0.5}
          },
          # Move stop to breakeven
          {
            condition: {position_pnl: {above: 0.001}}
            action: {move_stop_to: "entry_price"}
          }
        ]
      }
    ]
```

**Results show exit strategy impact:**
```
results/2024_12_23_104532/
└── traces/
    └── bollinger_bands/
        ├── period_20_std_2.0_risk_fixed.parquet
        ├── period_20_std_2.0_risk_atr_based.parquet
        ├── period_20_std_2.0_risk_trend_adaptive.parquet
        └── period_20_std_2.0_risk_partial_exits.parquet
```

---

# ============================================================================
# LEVEL 12: Full Production System
# ============================================================================
# Location: configs/production_system/config.yaml

```yaml
name: production_ready_system
symbols: ["SPY", "QQQ"]
timeframes: ["5m"]

# Execution costs
execution:
  commission: 0.0001
  slippage: 0.0001
  initial_capital: 100000

# Risk limits
risk:
  max_drawdown: 0.10
  position_size: 0.05
  max_positions: 5
  correlation_limit: 0.7

# Production ensemble
strategy: [
  # Trend component (35% weight)
  {
    weight: 0.35,
    strategy: [
      {sma_crossover: {fast_period: 10, slow_period: 30, weight: 0.6}},
      {momentum: {period: 14, weight: 0.4}}
    ],
    threshold: "0.5 AND market_hours() == 'regular' AND volume > sma(volume, 50) * 0.8"
  },
  
  # Mean reversion component (35% weight)
  {
    weight: 0.35,
    bollinger_bands: {period: 20, std: 2.0},
    threshold: "bollinger_bands AND volatility_percentile(50) < 0.7",
    risk: {stop_loss: 0.0015}
  },
  
  # Breakout component (30% weight)
  {
    weight: 0.3,
    breakout_strategy: {lookback: 20, threshold: 0.002},
    threshold: "breakout_strategy AND volatility_regime() == 'high'"
  },
  
  # Overall system requires minimum signal
  {threshold: 0.5}
]
```

---

## Working with Results

### Loading and Analyzing Results

```python
# In notebooks/analysis.ipynb
import json
import pandas as pd
from pathlib import Path

# Load latest run metadata
with open('../results/latest/metadata.json') as f:
    run_info = json.load(f)

print(f"Tested {run_info['summary']['total_combinations_tested']} combinations")
print(f"Best strategy: {run_info['summary']['overall_best_strategy']}")

# Load specific signal trace
best_params = run_info['strategies_tested']['bollinger_bands']['best_params']
filename = f"period_{best_params['period']}_std_{best_params['std']}.parquet"
signals = pd.read_parquet(f'../results/latest/traces/bollinger_bands/{filename}')

# Merge with market data to reconstruct equity curve
market_data = pd.read_parquet('../../../../data/SPY_5m.parquet')
df = market_data.merge(signals, on=['timestamp', 'symbol'], how='left')
df['signal'] = df['signal'].ffill().fillna(0)
```

### Comparing Across Runs

```python
# Compare performance across multiple research iterations
results_dir = Path('../results')
all_runs = []

for run_dir in results_dir.iterdir():
    if run_dir.is_dir() and run_dir.name != 'latest':
        with open(run_dir / 'metadata.json') as f:
            meta = json.load(f)
            all_runs.append({
                'timestamp': meta['timestamp'],
                'config_name': meta['config']['name'],
                'best_sharpe': max(s['best_sharpe'] 
                                 for s in meta['strategies_tested'].values()),
                'total_combinations': meta['summary']['total_combinations_tested']
            })

df_runs = pd.DataFrame(all_runs).sort_values('timestamp')
```

## Key Concepts Summary

### 1. **Composability**
- Strategies can contain strategies: `strategy: [{strategy: [...]}, ...]`
- Filters can be combined: `filter: [{filter1}, {filter2}]`
- Everything uses the same nested structure

### 2. **Parameter Expansion**
- Single value: `period: 20`
- List of values: `period: [10, 20, 30]`
- Range: `period: range(10, 50, 10)`
- Any parameter can have multiple values for optimization

### 3. **Filters as First-Class Citizens**
- Filters can be named and reused
- Filters can be strategies themselves
- Filters compose just like strategies

### 4. **Directory Structure**
- Each strategy gets its own directory
- Results are timestamped and self-contained
- Metadata includes full config for reproducibility

This structure provides:
- **Clean organization** - Each strategy in its own directory
- **Full reproducibility** - Complete config stored with each run
- **Easy comparison** - All results in standardized format
- **Natural workflow** - Edit config.yaml, run, analyze in notebooks


# THE FOLLOWING IS POSSIBLY DEPRECATED:


# Configuration Guide

This directory contains configuration files for the ADMF-PC (Adaptive Decision Making Framework - Protocol Components) system. Configurations define strategies, data sources, execution parameters, and workflow patterns.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration Structure](#configuration-structure)
- [Strategy Syntax](#strategy-syntax)
- [Signal Filtering](#signal-filtering)
- [Feature Configuration](#feature-configuration)
- [Execution Modes](#execution-modes)
- [Examples](#examples)

## Quick Start

Basic configuration example:
```yaml
name: my_strategy
mode: backtest
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-12-31"
initial_capital: 100000

# Simple moving average crossover
strategy:
  sma_crossover:
    params:
      fast_period: 10
      slow_period: 30

# Required features (auto-discovered from strategy)
features:
  - sma
```

Run with:
```bash
python main.py --config config/my_strategy.yaml --signal-generation --dataset test
```

## Configuration Structure

### Top-Level Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `name` | string | Configuration name | Yes |
| `mode` | string | Execution mode: `backtest`, `signal_generation`, `universal` | Yes |
| `symbols` | list | Trading symbols (e.g., ["SPY", "QQQ"]) | Yes |
| `start_date` | string | Start date (YYYY-MM-DD) | Yes |
| `end_date` | string | End date (YYYY-MM-DD) | Yes |
| `initial_capital` | number | Starting capital for backtesting | For backtest |
| `strategy` | object/list | Strategy configuration (see below) | Yes |
| `features` | list | Required features (often auto-discovered) | Optional |
| `classifiers` | list | Market regime classifiers | Optional |
| `dataset` | string | Data split: `full`, `train`, `test` | Optional |

## Strategy Syntax

The framework supports a powerful compositional strategy syntax with four main patterns:

### 1. Atom (Single Strategy)

Simple strategy with parameters:
```yaml
strategy:
  sma_crossover:
    params:
      fast_period: 10
      slow_period: 30
```

With signal filter:
```yaml
strategy:
  sma_crossover:
    params:
      fast_period: 10
      slow_period: 30
    filter: "signal > 0 and price > vwap()"  # Only long signals above VWAP
```

### 2. List Composition (Multiple Strategies)

Combine multiple strategies with weights:
```yaml
strategy: [
  {
    weight: 0.5
    sma_crossover:
      params: {fast_period: 10, slow_period: 30}
  },
  {
    weight: 0.5
    rsi_bands:
      params: {rsi_period: 14, oversold: 30, overbought: 70}
  }
]
```

With combination method:
```yaml
strategy:
  combination: weighted_vote  # or: majority, unanimous
  weight_threshold: 0.6      # Minimum weight for signal
  strategies: [
    {momentum: {weight: 0.4, params: {period: 14}}},
    {rsi_bands: {weight: 0.6, params: {period: 14}}}
  ]
```

### 3. Conditional Strategies

Execute strategies based on market conditions:
```yaml
strategy:
  condition: volatility_percentile(20) > 70
  bollinger_breakout:
    params: {period: 20, std_dev: 2.5}
```

### 4. Complex Compositions

Nested strategies with conditions:
```yaml
strategy: [
  {
    # High volatility regime
    condition: volatility_regime(20) == 'high'
    weight: 0.4
    strategy: [
      {bollinger_breakout: {weight: 0.6, params: {period: 20}}},
      {keltner_breakout: {weight: 0.4, params: {period: 20}}}
    ]
  },
  {
    # Low volatility regime
    condition: volatility_regime(20) == 'low'
    weight: 0.4
    mean_reversion:
      params: {period: 20, threshold: 2.0}
  },
  {
    # Always-on baseline
    weight: 0.2
    vwap_deviation:
      params: {std_multiplier: 2.0}
  }
]
```

## Signal Filtering

Strategies can include filters that evaluate after signal generation:

### Simple Filters
```yaml
strategy:
  sma_crossover:
    params: {fast_period: 10, slow_period: 30}
    filter: "signal > 0"  # Only accept long signals
```

### Complex Filters
```yaml
strategy:
  momentum:
    params: {period: 14}
    filter: "signal != 0 and rsi(14) > 30 and rsi(14) < 70 and volume > sma(20)"
```

### Parameterized Filters (for optimization)
```yaml
strategy:
  macd_crossover:
    params: {fast_period: 12, slow_period: 26}
    filter: "abs(signal) > ${threshold} and volume > ma(${vol_period})"
    filter_params:
      threshold: 0.5      # Can be optimized
      vol_period: 20      # Can be optimized
```

### Filter Expression Reference

Available variables:
- `signal` - The signal value from strategy (-1, 0, 1)
- `price`, `open`, `high`, `low`, `close` - Current bar data
- `volume` - Current volume
- All computed features from FeatureHub

Available functions:
- `ma(period)` - Moving average (SMA or EMA)
- `sma(period)` - Simple moving average
- `ema(period)` - Exponential moving average
- `vwap()` - Volume weighted average price
- `session_vwap()` - Session-based VWAP (resets at market open)
- `rsi(period)` - Relative strength index
- `atr(period)` - Average true range
- `abs()`, `min()`, `max()` - Math functions

## Feature Configuration

### Automatic Discovery
Most features are auto-discovered from strategies:
```yaml
strategy:
  sma_crossover:
    params: {fast_period: 10, slow_period: 30}
# Features sma_10 and sma_30 will be auto-discovered
```

### Explicit Features
Add features not discovered automatically:
```yaml
features:
  - vwap
  - session_vwap
  - atr
  - volume_sma
```

### Feature Parameters
Some features require parameters:
```yaml
features:
  - name: bollinger_bands
    params: {period: 20, std_dev: 2}
  - name: rsi
    params: {period: 14}
```

## Execution Modes

### Signal Generation
Generate and store signals for analysis:
```bash
python main.py --config config/my_strategy.yaml --signal-generation --dataset test
```

### Backtesting
Run full backtest with portfolio simulation:
```bash
python main.py --config config/my_strategy.yaml --backtest
```

### Optimization
Optimize parameters with grid search:
```bash
python main.py --config config/my_strategy.yaml --optimize
```

Add parameter space for optimization:
```yaml
parameter_space:
  strategies:
    - type: sma_crossover
      param_overrides:
        fast_period: [5, 10, 15, 20]
        slow_period: [20, 30, 40, 50]
```

#### Wildcard Discovery

Use wildcards to automatically discover and test all strategies and classifiers:

##### Strategy Discovery
```yaml
parameter_space:
  # Test all indicator strategies
  indicators: "*"
  
  # Test specific indicator categories
  indicators:
    crossover: "*"      # All crossover strategies (sma_crossover, ema_crossover, etc.)
    momentum: "*"       # All momentum strategies (rsi, macd, roc, etc.)
    oscillator: "*"     # All oscillator strategies (cci, stochastic, williams_r, etc.)
    trend: "*"          # All trend strategies (adx, aroon, supertrend, etc.)
    volatility: "*"     # All volatility strategies (bollinger, keltner, donchian, etc.)
    volume: "*"         # All volume strategies (obv, vwap, mfi, chaikin, etc.)
    structure: "*"      # All structure strategies (pivot, trendline, support/resistance, etc.)
```

##### Classifier Discovery
```yaml
parameter_space:
  # Test all classifiers
  classifiers: "*"
  
  # Or test specific classifiers
  classifiers: [
    "trend_classifier",
    "volatility_classifier",
    "momentum_regime_classifier",
    "multi_timeframe_trend_classifier",
    "volatility_momentum_classifier"
  ]
```

##### Mixed Discovery
```yaml
parameter_space:
  # Discover all indicators
  indicators: "*"
  
  # Discover all classifiers
  classifiers: "*"
  
  # Add custom strategies with overrides
  strategies:
    - type: custom_ensemble
      param_overrides:
        weight_threshold: [0.5, 0.6, 0.7]
```

The system will automatically:
- Discover all strategies in `src/strategy/strategies/indicators/`
- Discover all classifiers in `src/strategy/classifiers/`
- Extract their parameter spaces from the `@strategy` and `@classifier` decorators
- Generate parameter combinations based on range specifications (using tuples for ranges)
- Run optimization across all discovered components

##### Granularity Control
Control the number of samples for parameter ranges:
```yaml
optimization:
  granularity: 5  # Default: 5 samples per range
                  # Higher = more parameter combinations
                  # Lower = faster optimization
```

## Examples

### Directory Structure
```
config/
├── README.md                          # This file
├── examples/                          # Example configurations
│   ├── compositional-strategies.yaml  # Strategy composition examples
│   └── signal_filtering_example.yaml  # Filter examples
├── patterns/                          # Reusable workflow patterns
│   ├── topologies/                    # Execution topologies
│   └── workflows/                     # Workflow definitions
└── [your configs].yaml               # Your strategy configurations
```

### Key Examples

1. **Simple Moving Average** - `test_ma_crossover.yaml`
   - Basic SMA crossover strategy

2. **Volatility Regimes** - `volatility_momentum_three_regimes.yaml`
   - Complex multi-regime adaptive strategy

3. **Signal Filtering** - `examples/signal_filtering_example.yaml`
   - Demonstrates various filter patterns

4. **Compositional Strategies** - `examples/compositional-strategies.yaml`
   - Shows all composition patterns

### Creating Your Own

1. Start with a simple example
2. Add your strategy type and parameters
3. Configure data range and symbols
4. Add filters if needed
5. Test with small dataset first

```yaml
# my_first_strategy.yaml
name: my_first_strategy
mode: backtest
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"
initial_capital: 100000

strategy:
  rsi_bands:
    params:
      rsi_period: 14
      oversold: 30
      overbought: 70
    filter: "volume > 50000000"  # Min 50M volume

features:
  - rsi
  - volume
```

## Best Practices

1. **Start Simple** - Test single strategies before compositions
2. **Use Filters** - Filter out low-quality signals
3. **Test Data Split** - Use `--dataset test` for out-of-sample testing
4. **Small Date Ranges** - Test with 1-month ranges first
5. **Version Control** - Keep configs in git for reproducibility
6. **Descriptive Names** - Use clear, descriptive configuration names
7. **Comments** - Document your strategy logic with comments

## Troubleshooting

Common issues:

1. **Missing Features** - Add required features to `features:` section
2. **Filter Syntax** - Keep filters on single line or use proper YAML multiline
3. **Parameter Types** - Ensure parameters match expected types (int vs float)
4. **Date Formats** - Use YYYY-MM-DD format for dates
5. **Symbol Names** - Use standard ticker symbols

For more help, see the main project documentation or run:
```bash
python main.py --help
```
