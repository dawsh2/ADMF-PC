# Strategy Module Organization Proposal

## Current Issues

### 1. **Duplicate Implementations**
- `momentum.py` vs `simple_momentum.py` - Nearly identical functionality
- `mean_reversion.py` vs `mean_reversion_simple.py` - One is just a wrapper
- `ma_crossover.py` vs crossover rules in `crossovers.py`

### 2. **Mixed Paradigms**
- Old: Class-based strategies (ArbitrageStrategy, MarketMakingStrategy)
- New: Function-based with @strategy decorator
- Some files have both (trend_following.py)

### 3. **Inconsistent Naming**
- File suffixes: `rsi_strategy.py` vs `momentum.py`
- Prefixes: `simple_momentum.py` vs `momentum.py`
- No clear convention

## Proposed Organization

### Phase 1: Immediate Cleanup

```
strategies/
├── __init__.py
├── README.md                    # Strategy implementation guide
│
├── core/                        # Core trading strategies
│   ├── momentum.py             # All momentum strategies
│   ├── mean_reversion.py       # Mean reversion strategies
│   ├── trend_following.py      # Trend following strategies
│   └── breakout.py             # Breakout strategies
│
├── indicators/                  # Indicator-based strategies
│   ├── moving_averages.py      # MA, EMA, DEMA, TEMA crossovers
│   ├── oscillators.py          # RSI, CCI, Stochastic strategies
│   ├── channels.py             # Bollinger, Keltner, Donchian
│   └── momentum_indicators.py  # MACD, Vortex, ADX strategies
│
├── advanced/                    # Complex multi-component strategies
│   ├── arbitrage.py            # Statistical arbitrage
│   ├── market_making.py        # Market making strategies
│   └── multi_factor.py         # Multi-factor strategies
│
├── rules/                       # Rule-based strategies (16 trading rules)
│   ├── crossovers.py           # Rules 1-9: Crossover strategies
│   ├── thresholds.py           # Rules 10-13: Threshold strategies
│   └── bands.py                # Rules 14-16: Band strategies
│
└── experimental/                # Test and experimental strategies
    └── null.py                 # Null strategy for testing
```

### Phase 2: Content Migration

#### 1. **Consolidate Momentum Strategies**
```python
# momentum.py - Combine all momentum variants
@strategy(name='momentum_sma_rsi')
def momentum_sma_rsi_strategy(...): ...

@strategy(name='momentum_price_change')
def momentum_price_change_strategy(...): ...

@strategy(name='momentum_with_exits')  # From simple_momentum.py
def momentum_with_exits_strategy(...): ...
```

#### 2. **Clean Mean Reversion**
- Delete `mean_reversion.py` (wrapper)
- Rename `mean_reversion_simple.py` → `mean_reversion.py`
- Remove class implementation, keep only @strategy function

#### 3. **Reorganize Crossovers**
- Move content from `ma_crossover.py` to `moving_averages.py`
- Keep rule-based crossovers in `rules/crossovers.py`
- Delete redundant `ma_crossover.py`

#### 4. **Separate Concerns**
- Move MACD from `macd_strategy.py` to `indicators/momentum_indicators.py`
- Move RSI from `rsi_strategy.py` to `indicators/oscillators.py`
- Group by indicator type, not individual files

### Phase 3: Standardization

#### 1. **Naming Convention**
```python
# File names: descriptive, no suffix
momentum.py              # ✓ Good
momentum_strategy.py     # ✗ Redundant suffix

# Function names: {approach}_{variant}_strategy
@strategy(name='momentum_sma_crossover')
def momentum_sma_crossover_strategy(...): ...

@strategy(name='mean_reversion_bollinger')
def mean_reversion_bollinger_strategy(...): ...
```

#### 2. **Remove Class Implementations**
All strategies should be pure functions with @strategy decorator:
```python
# ✓ Good: Pure function
@strategy(name='arbitrage_pairs')
def arbitrage_pairs_strategy(features, bar, params):
    # Implementation
    pass

# ✗ Bad: Class-based
class ArbitrageStrategy:
    def generate_signals(self, ...):
        pass
```

#### 3. **Consistent Structure**
Each strategy file should follow:
```python
"""
Module description explaining the trading approach.
"""

from typing import Dict, Any, Optional
from ...core.components.discovery import strategy

# All strategies as @strategy decorated functions
@strategy(
    name='strategy_name',
    feature_config={...}
)
def strategy_name_strategy(features: Dict[str, Any], 
                          bar: Dict[str, Any], 
                          params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strategy docstring."""
    # Binary signal logic (-1, 0, 1)
    pass
```

### Benefits of Reorganization

1. **Clear Hierarchy**: Core → Indicators → Advanced → Rules
2. **No Duplicates**: Each strategy has one canonical location
3. **Easy Discovery**: Grouped by trading approach
4. **Consistent Patterns**: All use @strategy decorator
5. **Maintainable**: Clear where to add new strategies

### Migration Steps

1. **Create new directory structure**
2. **Move and consolidate files**
3. **Update imports in dependent code**
4. **Remove deprecated class implementations**
5. **Update __init__.py exports**
6. **Run tests to ensure nothing breaks**

### Future Considerations

1. **Strategy Metadata**: Add metadata file describing each strategy's characteristics
2. **Performance Metrics**: Track which strategies are most effective
3. **Strategy Families**: Group related strategies for ensemble trading
4. **Version Control**: Tag strategies with version numbers for reproducibility

This organization would make the strategy module much cleaner and easier to navigate, while maintaining backward compatibility during the transition.