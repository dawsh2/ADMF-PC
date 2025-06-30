# Strategy Composition Design

## Overview

This document defines the compositional strategy system for ADMF-PC. Instead of special "ensemble" or "regime-switching" strategy types, all strategies are composable using a minimal, uniform syntax inspired by Lisp.

## Core Principles

1. **No special types** - Composition is structural, not type-based
2. **Uniform syntax** - Same pattern at every level of nesting  
3. **Minimal keywords** - Only `strategy`, `condition`, and strategy type names
4. **Everything composes** - Any strategy can contain any other strategy

## Syntax Patterns

### Atomic Strategies

A single strategy is a dict with the strategy type as key:

```yaml
# Simplest form
strategy:
  sma_crossover:
    params: {fast: 10, slow: 30}

# With more parameters
strategy:
  rsi_threshold:
    params: {period: 14, threshold: 70}
    weight: 1.0  # Optional, defaults to 1.0
```

### Composite Strategies

Multiple strategies are expressed as an array:

```yaml
# Simple composition
strategy: [
  {sma_crossover: {weight: 0.5, params: {fast: 10, slow: 30}}},
  {rsi_threshold: {weight: 0.5, params: {threshold: 70}}}
]

# With combination method
strategy:
  combination: weighted_vote  # or 'majority', 'unanimous', 'average'
  weight_threshold: 0.6
  strategies: [
    {sma_crossover: {weight: 0.5}},
    {rsi_threshold: {weight: 0.5}}
  ]
```

### Nested Composition

Strategies compose recursively using the `strategy:` key:

```yaml
strategy: [
  {
    weight: 0.4
    combination: majority
    strategy: [
      {sma_crossover: {weight: 0.6, params: {fast: 10, slow: 30}}},
      {ema_crossover: {weight: 0.4, params: {fast: 12, slow: 26}}}
    ]
  },
  {rsi_threshold: {weight: 0.6, params: {threshold: 70}}}
]
```

### Conditional Strategies

Conditions are just another compositional element:

```yaml
# Simple conditional
strategy:
  condition: volatility_percentile(20) > 70
  if_true:
    rsi_bands:
      params: {oversold: 20, overbought: 80}
  if_false:
    sma_crossover:
      params: {fast: 10, slow: 30}

# Conditions in lists
strategy: [
  {
    condition: trend_strength(50) > 0.7
    weight: 0.5
    momentum:
      params: {period: 14}
  },
  {
    condition: trend_strength(50) < 0.3
    weight: 0.5
    mean_reversion:
      params: {period: 20}
  }
]

# Nested conditions
strategy: [
  {
    condition: market_hours() == 'regular'
    strategy: [
      {
        condition: volatility_percentile(5) > 80
        gap_fade:
          params: {threshold: 0.02}
      },
      {
        condition: volume_ratio(20) > 2.0
        volume_breakout:
          params: {multiplier: 1.5}
      }
    ]
  }
]
```

### Multi-State Regimes

For classifiers that return multiple states (not just binary conditions):

```yaml
# Using 'cases' for clean multi-state handling
strategy:
  regime: market_regime_classifier()
  cases:
    trending_up: 
      momentum:
        params: {period: 14}
    trending_down: 
      mean_reversion:
        params: {period: 20}
    ranging: [  # Can be composite
      {rsi_bands: {weight: 0.5}},
      {bollinger_mean_reversion: {weight: 0.5}}
    ]

# Or using multiple conditions with the same expression
strategy: [
  {
    conditions:
      - {condition: volatility_regime(20) == 'low', weight: 0.3}
      - {condition: volatility_regime(20) == 'medium', weight: 0.5}  
      - {condition: volatility_regime(20) == 'high', weight: 0.7}
    strategy: 
      atr_breakout:
        params: {period: 14}
  }
]

# Combining multiple classifiers
strategy:
  regimes:
    volatility: volatility_regime_classifier()
    trend: trend_strength_classifier()
  cases:
    # Nested state matching
    volatility:
      low:
        trend:
          strong: {momentum: {params: {period: 14}}}
          weak: {mean_reversion: {params: {period: 20}}}
      high:
        trend:
          strong: {breakout: {params: {period: 20}}}
          weak: {volatility_squeeze: {}}
```

## Detection Rules

The topology builder uses these rules to interpret configurations:

1. **Dict with strategy type key** → Atomic strategy
2. **Array** → Composite strategy with equal default weights
3. **Dict with `strategy:` key** → Container with nested strategies
4. **Dict with `condition:` key** → Conditional execution
5. **Dict with `if_true/if_false:` keys** → Conditional branching

## Examples

### Progressive Enhancement

Start simple and add complexity as needed:

```yaml
# 1. Single strategy
strategy:
  sma_crossover:
    params: {fast: 10, slow: 30}

# 2. Add another strategy
strategy: [
  {sma_crossover: {params: {fast: 10, slow: 30}}},
  {rsi_threshold: {params: {threshold: 70}}}
]

# 3. Add weights and combination method
strategy:
  combination: weighted_vote
  weight_threshold: 0.6
  strategies: [
    {sma_crossover: {weight: 0.6}},
    {rsi_threshold: {weight: 0.4}}
  ]

# 4. Add conditions
strategy: [
  {
    condition: volatility_percentile(20) > 50
    weight: 0.6
    sma_crossover:
      params: {fast: 10, slow: 30}
  },
  {
    weight: 0.4
    rsi_threshold:
      params: {threshold: 70}
  }
]

# 5. Add nesting
strategy: [
  {
    condition: trend_strength(50) > 0.5
    weight: 0.6
    strategy: [
      {sma_crossover: {weight: 0.7}},
      {ema_crossover: {weight: 0.3}}
    ]
  },
  {
    weight: 0.4
    rsi_threshold:
      params: {threshold: 70}
  }
]
```

### Market Regime Adaptation

No special regime types needed:

```yaml
strategy: [
  # Trending market strategies
  {
    condition: adx(14) > 25 and trend_strength(50) > 0.5
    weight: 0.5
    strategy: [
      {momentum: {weight: 0.6, params: {period: 14}}},
      {sma_crossover: {weight: 0.4, params: {fast: 10, slow: 30}}}
    ]
  },
  # Ranging market strategies
  {
    condition: adx(14) < 20
    weight: 0.5
    strategy: [
      {rsi_bands: {weight: 0.5, params: {period: 14}}},
      {bollinger_mean_reversion: {weight: 0.5, params: {period: 20}}}
    ]
  },
  # Always-on base strategy
  {
    weight: 0.2
    vwap_deviation:
      params: {threshold: 0.02}
  }
]
```

### Time-Based Strategies

```yaml
strategy: [
  {
    condition: market_hours() == 'pre_market'
    gap_fade:
      params: {min_gap: 0.02}
  },
  {
    condition: market_hours() == 'regular' and time_until_close() > 30
    strategy: [
      {momentum: {weight: 0.5}},
      {trend_follow: {weight: 0.5}}
    ]
  },
  {
    condition: time_until_close() <= 30
    close_positions: {}
  }
]
```

## Implementation Notes

### Condition Evaluation

Conditions are evaluated against a context containing:
- Current bar data
- Feature values  
- Market state
- Portfolio state

The condition expressions use a safe evaluator with access to:
- Indicator functions: `sma(20)`, `rsi(14)`, etc.
- Market functions: `market_hours()`, `time_until_close()`
- State functions: `position_size()`, `unrealized_pnl()`
- Comparison operators: `>`, `<`, `==`, `and`, `or`

### Weight Normalization

When conditions filter active strategies:
1. Evaluate all conditions
2. Include only strategies where `condition` is true or absent
3. Renormalize weights among active strategies
4. Apply combination method

### Execution Order

Strategies execute in depth-first order:
1. Evaluate conditions at current level
2. For active strategies with nested `strategy:`, recurse
3. Execute atomic strategies
4. Combine signals based on method and weights

## Benefits

1. **No special ensemble types** - Natural composition
2. **Unified syntax** - Learn once, use everywhere
3. **Progressive complexity** - Start simple, enhance as needed
4. **Clear semantics** - Structure implies behavior
5. **Infinite flexibility** - Compose anything with anything

## Migration

Existing configurations using `type: ensemble` or `type: regime_switch` should be migrated to this compositional syntax. The old types will be deprecated.