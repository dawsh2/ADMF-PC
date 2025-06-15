# Strategy Development Guide for ADMF-PC

This guide provides comprehensive instructions for developing trading strategies that work seamlessly with the ADMF-PC framework, including proper decorators, feature naming conventions, and discovery mechanisms.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Strategy Architecture](#strategy-architecture)
3. [Creating a New Strategy](#creating-a-new-strategy)
4. [Feature Naming Conventions](#feature-naming-conventions)
5. [Strategy Discovery System](#strategy-discovery-system)
6. [Testing Your Strategy](#testing-your-strategy)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Minimal Working Strategy

```python
from ...core.components.discovery import strategy

@strategy(
    name='simple_sma_crossover',
    feature_config={
        'sma': {
            'params': ['fast_period', 'slow_period'],
            'defaults': {'fast_period': 10, 'slow_period': 20}
        }
    }
)
def simple_sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Simple SMA crossover strategy."""
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    sma_fast = features.get(f'sma_{fast_period}')
    sma_slow = features.get(f'sma_{slow_period}')
    
    if sma_fast is None or sma_slow is None:
        return None
    
    signal_value = 1 if sma_fast > sma_slow else -1
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'simple_sma_crossover',
        'symbol_timeframe': f"{bar.get('symbol', 'UNKNOWN')}_{bar.get('timeframe', '1m')}"
    }
```

## Strategy Architecture

### Stateless Design
All strategies are **stateless pure functions** that:
- Take current features and bar data as input
- Return a signal dictionary or None
- Do not maintain any internal state between calls

### Two-Tier System
1. **FeatureHub (Stateful)**: Maintains rolling windows and computes indicators
2. **Strategies (Stateless)**: Consume pre-computed features and generate signals

### Signal Types
Strategies return sustained signals (sparse storage handles state transitions):
- `1`: Buy/Long signal
- `-1`: Sell/Short signal  
- `0`: Neutral/No position

## Creating a New Strategy

### Step 1: Choose the Right File

Place your strategy in the appropriate category:

```
strategies/
├── indicators/
│   ├── crossovers.py      # MA crossovers, MACD crossovers
│   ├── oscillators.py     # RSI, Stochastic, Williams %R
│   ├── trend.py           # ADX, Aroon, Supertrend
│   ├── volatility.py      # Bollinger, Keltner, ATR
│   └── volume.py          # OBV, MFI, VWAP
├── patterns/              # Chart patterns
├── ml/                    # Machine learning strategies
└── composite/             # Multi-indicator strategies
```

### Step 2: Import Required Modules

```python
from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)
```

### Step 3: Define the Strategy Decorator

The `@strategy` decorator is **MANDATORY** for automatic discovery:

```python
@strategy(
    name='your_strategy_name',  # Unique identifier
    feature_config={
        'feature_type': {
            'params': ['param_names'],  # Maps to strategy parameters
            'defaults': {'param_name': default_value}
        }
    }
)
```

### Step 4: Implement the Strategy Function

```python
def your_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Your strategy description.
    
    Args:
        features: Pre-computed features from FeatureHub
        bar: Current bar with symbol, timestamp, OHLCV
        params: Strategy parameters from configuration
        
    Returns:
        Signal dict or None if not ready
    """
    # Extract parameters
    param1 = params.get('param1', default_value)
    
    # Get features using exact naming convention
    feature1 = features.get(f'feature_type_{param1}')
    
    # Check readiness
    if feature1 is None:
        return None
    
    # Strategy logic
    signal_value = 0  # Your logic here
    
    # Return signal
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'your_strategy_name',
        'symbol_timeframe': f"{bar.get('symbol')}_{bar.get('timeframe')}"
    }
```

## Feature Naming Conventions

### Single-Value Features

Features that return one value:

| Feature Type | Naming Pattern | Example |
|-------------|----------------|---------|
| SMA | `sma_{period}` | `sma_20` |
| EMA | `ema_{period}` | `ema_50` |
| DEMA | `dema_{period}` | `dema_20` |
| TEMA | `tema_{period}` | `tema_20` |
| RSI | `rsi_{period}` | `rsi_14` |
| ATR | `atr_{period}` | `atr_14` |
| Momentum | `momentum_{period}` | `momentum_10` |
| ROC | `roc_{period}` | `roc_12` |
| Volume | `volume` | `volume` |
| Volume SMA | `volume_sma_{period}` | `volume_sma_20` |
| Volume Ratio | `volume_ratio_{period}` | `volume_ratio_20` |
| Williams %R | `williams_r_{period}` | `williams_r_14` |
| CCI | `cci_{period}` | `cci_20` |
| MFI | `mfi_{period}` | `mfi_14` |
| OBV | `obv` | `obv` |
| CMF | `cmf_{period}` | `cmf_20` |
| AD | `ad` | `ad` |
| VWAP | `vwap` | `vwap` |
| Volatility | `volatility_{period}` | `volatility_20` |
| ATR SMA | `atr_sma_{atr_period}_{sma_period}` | `atr_sma_14_20` |
| Volatility SMA | `volatility_sma_{vol_period}_{sma_period}` | `volatility_sma_20_10` |
| High | `high_{period}` | `high_20` |
| Low | `low_{period}` | `low_20` |
| Linear Regression Slope | `lr_slope_{period}` | `lr_slope_20` |
| Linear Regression Intercept | `lr_intercept_{period}` | `lr_intercept_20` |
| Linear Regression R² | `lr_r2_{period}` | `lr_r2_20` |

### Multi-Value Features

Features that return multiple values use suffixes:

| Feature Type | Naming Pattern | Values |
|-------------|----------------|--------|
| Bollinger Bands | `bollinger_{period}_{std}_upper/middle/lower` | `bollinger_20_2.0_upper` |
| MACD | `macd_{fast}_{slow}_{signal}_macd/signal/histogram` | `macd_12_26_9_macd` |
| Stochastic | `stochastic_{k_period}_{d_period}_k/d` | `stochastic_14_3_k` |
| ADX | `adx_{period}` + `adx_{period}_di_plus/di_minus` | `adx_14`, `adx_14_di_plus` |
| Aroon | `aroon_{period}_up/down/oscillator` | `aroon_25_up` |
| Supertrend | `supertrend_{period}_{mult}` + `_direction/upper/lower` | `supertrend_10_3.0` |
| Keltner | `keltner_{period}_{mult}_upper/middle/lower` | `keltner_20_2.0_upper` |
| Donchian | `donchian_{period}_upper/lower` | `donchian_20_upper` |
| PSAR | `psar_{af_start}_{af_max}` | `psar_0.02_0.2` |
| Ichimoku | `ichimoku_{conversion}_{base}_{span}_X` | `ichimoku_9_26_52_tenkan` |
| Vortex | `vortex_{period}_vi_plus/vi_minus` | `vortex_14_vi_plus` |
| Stochastic RSI | `stochastic_rsi_{rsi_period}_{k_period}_{d_period}_k/d` | `stochastic_rsi_14_14_3_k` |
| Ultimate Oscillator | `ultimate_{fast}_{medium}_{slow}` | `ultimate_7_14_28` |
| Pivot Points | `pivot_{period}_pivot/r1/r2/s1/s2` | `pivot_20_pivot` |
| Fibonacci | `fibonacci_{period}_{level}` | `fibonacci_20_0.382` |
| Support/Resistance | `support_resistance_{period}_support/resistance` | `support_resistance_50_support` |
| Swing Points | `swing_{period}_high/low` | `swing_20_high` |

### Special Cases

Some features have unique naming:

```python
# ADX with DI lines
adx_value = features.get('adx_14')
di_plus = features.get('adx_14_di_plus')
di_minus = features.get('adx_14_di_minus')

# Linear regression components
slope = features.get('lr_slope_20')
intercept = features.get('lr_intercept_20')
r_squared = features.get('lr_r2_20')

# Fibonacci retracement levels
fib_0 = features.get('fibonacci_20_0.0')
fib_236 = features.get('fibonacci_20_0.236')
fib_382 = features.get('fibonacci_20_0.382')
```

## Strategy Discovery System

### How Discovery Works

1. **Decorator Registration**: The `@strategy` decorator registers your function
2. **Component Registry**: Functions are stored in a global registry
3. **Topology Builder**: Scans registry and infers feature requirements
4. **Feature Deduplication**: Combines features from all strategies
5. **FeatureHub Configuration**: Configures centralized feature computation

### Import Requirements

For your strategy to be discovered, it must be imported when the module loads:

```python
# In strategies/indicators/__init__.py
from .trend import (
    adx_trend_strength,
    parabolic_sar,
    aroon_crossover,
    supertrend,
    linear_regression_slope
)

# In strategies/__init__.py
from .indicators import *
from .patterns import *
from .composite import *
```

Without proper imports in `__init__.py`, your strategy won't be discovered!

### Feature Configuration Format

```python
feature_config={
    'bollinger': {
        'params': ['bb_period', 'bb_std'],  # Parameter names in your strategy
        'defaults': {'bb_period': 20, 'bb_std': 2.0}
    },
    'rsi': {
        'params': ['rsi_period'],
        'defaults': {'rsi_period': 14}
    }
}
```

### Parameter Mapping

The `params` list maps strategy parameters to feature parameters:

```python
# In decorator
'params': ['bb_period', 'bb_std']

# In strategy function
bb_period = params.get('bb_period', 20)
bb_std = params.get('bb_std', 2.0)

# In feature name
upper_band = features.get(f'bollinger_{bb_period}_{bb_std}_upper')
```

## Required Signal Dictionary Fields

Every strategy must return a dictionary with these required fields:

```python
return {
    'signal_value': signal_value,      # Required: -1, 0, or 1
    'timestamp': bar.get('timestamp'), # Required: Current bar timestamp
    'strategy_id': 'strategy_name',    # Required: Must match decorator name
    'symbol_timeframe': f"{symbol}_{timeframe}", # Required: e.g. "SPY_1m"
    'metadata': {                      # Optional: Additional context
        'price': current_price,
        'indicators': {...}
    }
}
```

Missing any required field will cause signal storage errors!

## Testing Your Strategy

### Unit Test Template

```python
def test_your_strategy():
    """Test your strategy signal generation."""
    from src.strategy.strategies.category.your_file import your_strategy
    
    # Mock features
    features = {
        'sma_10': 100.5,
        'sma_20': 99.5,
        'rsi_14': 45
    }
    
    # Mock bar
    bar = {
        'symbol': 'TEST',
        'close': 101.0,
        'volume': 1000000,
        'timeframe': '1m',
        'timestamp': '2024-01-01T10:00:00'
    }
    
    # Parameters
    params = {
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14
    }
    
    # Test signal generation
    result = your_strategy(features, bar, params)
    
    assert result is not None
    assert result['signal_value'] in [-1, 0, 1]
    assert result['strategy_id'] == 'your_strategy'
```

### Integration Test

```python
def test_strategy_with_feature_hub():
    """Test strategy with real FeatureHub."""
    from src.strategy.components.features.hub import FeatureHub
    from src.strategy.strategies.category.your_file import your_strategy
    
    # Create FeatureHub
    hub = FeatureHub(['TEST'])
    hub.configure_features({
        'sma_10': {'type': 'sma', 'period': 10},
        'sma_20': {'type': 'sma', 'period': 20}
    })
    
    # Feed data
    for i in range(30):
        hub.update_bar('TEST', {
            'open': 100 + i * 0.1,
            'high': 100.5 + i * 0.1,
            'low': 99.5 + i * 0.1,
            'close': 100 + i * 0.1,
            'volume': 1000000
        })
    
    # Get features and generate signal
    features = hub.get_features('TEST')
    result = your_strategy(features, bar, params)
    
    assert result is not None
```

## Common Patterns

### Pattern 1: Multi-Indicator Confirmation

```python
@strategy(
    name='triple_confirmation',
    feature_config={
        'sma': {'params': ['sma_period'], 'defaults': {'sma_period': 50}},
        'rsi': {'params': ['rsi_period'], 'defaults': {'rsi_period': 14}},
        'macd': {'params': ['macd_fast', 'macd_slow', 'macd_signal'], 
                 'defaults': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9}}
    }
)
def triple_confirmation(features, bar, params):
    # Get all indicators
    sma = features.get(f'sma_{params.get("sma_period", 50)}')
    rsi = features.get(f'rsi_{params.get("rsi_period", 14)}')
    macd = features.get(f'macd_{params.get("macd_fast")}_{params.get("macd_slow")}_{params.get("macd_signal")}_macd')
    macd_signal = features.get(f'macd_{params.get("macd_fast")}_{params.get("macd_slow")}_{params.get("macd_signal")}_signal')
    
    if any(f is None for f in [sma, rsi, macd, macd_signal]):
        return None
    
    price = bar.get('close', 0)
    
    # Triple confirmation logic
    bullish = price > sma and rsi < 70 and macd > macd_signal
    bearish = price < sma and rsi > 30 and macd < macd_signal
    
    signal_value = 1 if bullish else (-1 if bearish else 0)
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'triple_confirmation',
        'symbol_timeframe': f"{bar.get('symbol')}_{bar.get('timeframe')}"
    }
```

### Pattern 2: Dynamic Feature Selection

```python
@strategy(
    name='adaptive_ma',
    feature_config={
        'sma': {'params': ['ma_period'], 'defaults': {'ma_period': 20}},
        'ema': {'params': ['ma_period'], 'defaults': {'ma_period': 20}},
        'volatility': {'params': ['vol_period'], 'defaults': {'vol_period': 20}}
    }
)
def adaptive_ma(features, bar, params):
    period = params.get('ma_period', 20)
    vol_period = params.get('vol_period', 20)
    
    # Get features
    sma = features.get(f'sma_{period}')
    ema = features.get(f'ema_{period}')
    volatility = features.get(f'volatility_{vol_period}')
    
    if any(f is None for f in [sma, ema, volatility]):
        return None
    
    # Use EMA in volatile markets, SMA in stable markets
    ma_value = ema if volatility > 0.02 else sma
    price = bar.get('close', 0)
    
    signal_value = 1 if price > ma_value else -1
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'adaptive_ma',
        'symbol_timeframe': f"{bar.get('symbol')}_{bar.get('timeframe')}"
    }
```

## Troubleshooting

### Strategy Not Discovered

**Problem**: Your strategy doesn't appear in the topology
**Solution**: Check that you have:
1. Used the `@strategy` decorator
2. Imported the strategy in `__init__.py`
3. Unique strategy name
4. Valid Python syntax

### Features Always None

**Problem**: All features return None
**Solution**: 
1. Check feature naming matches exactly
2. Ensure sufficient bars have been processed
3. Verify feature configuration in decorator
4. Enable debug logging to see available features

### Wrong Feature Values

**Problem**: Feature values seem incorrect
**Solution**:
1. Verify parameter values in feature names
2. Check if using correct feature type (e.g., `bollinger` vs `bollinger_bands`)
3. Ensure parameters are within valid ranges

### Debugging Tips

```python
# Enable debug logging
import logging
logging.getLogger('src.strategy').setLevel(logging.DEBUG)

# Print available features
logger.info(f"Available features: {list(features.keys())}")

# Print feature values
for name, value in features.items():
    logger.info(f"{name}: {value}")

# Check strategy registration
from src.core.components.discovery import get_component_registry
strategies = get_component_registry().get('strategies', {})
print(f"Registered strategies: {list(strategies.keys())}")
```

## Best Practices

1. **Always Return Signal State**: Return current signal value even if unchanged (sparse storage handles transitions)

2. **Handle None Features Gracefully**: Check all features before using them

3. **Use Descriptive Names**: Strategy names should indicate their logic

4. **Document Parameters**: Include parameter descriptions in docstrings

5. **Keep Logic Simple**: Complex strategies are harder to debug and optimize

6. **Test Edge Cases**: Test with missing data, extreme values, etc.

7. **Log Sparingly**: Use debug level for detailed logs

8. **Follow Naming Conventions**: Consistency helps with maintenance

## Example: Complete Strategy Implementation

Here's a full example following all best practices:

```python
# File: strategies/indicators/volatility.py

from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)

@strategy(
    name='bollinger_squeeze',
    feature_config={
        'bollinger': {
            'params': ['bb_period', 'bb_std'],
            'defaults': {'bb_period': 20, 'bb_std': 2.0}
        },
        'keltner': {
            'params': ['kc_period', 'kc_mult'],
            'defaults': {'kc_period': 20, 'kc_mult': 1.5}
        },
        'momentum': {
            'params': ['mom_period'],
            'defaults': {'mom_period': 12}
        },
        'volume_sma': {
            'params': ['vol_period'],
            'defaults': {'vol_period': 20}
        }
    }
)
def bollinger_squeeze(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Band Squeeze strategy - trades volatility breakouts.
    
    Detects when Bollinger Bands contract inside Keltner Channels (squeeze),
    then trades the breakout direction based on momentum.
    
    Parameters:
        bb_period: Bollinger Bands period (default: 20)
        bb_std: Bollinger Bands standard deviations (default: 2.0)
        kc_period: Keltner Channel period (default: 20)
        kc_mult: Keltner Channel ATR multiplier (default: 1.5)
        mom_period: Momentum period (default: 12)
        vol_period: Volume SMA period (default: 20)
        vol_filter: Require volume > average (default: True)
    """
    # Extract parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    kc_period = params.get('kc_period', 20)
    kc_mult = params.get('kc_mult', 1.5)
    mom_period = params.get('mom_period', 12)
    vol_period = params.get('vol_period', 20)
    vol_filter = params.get('vol_filter', True)
    
    # Get required features
    bb_upper = features.get(f'bollinger_{bb_period}_{bb_std}_upper')
    bb_lower = features.get(f'bollinger_{bb_period}_{bb_std}_lower')
    kc_upper = features.get(f'keltner_{kc_period}_{kc_mult}_upper')
    kc_lower = features.get(f'keltner_{kc_period}_{kc_mult}_lower')
    momentum = features.get(f'momentum_{mom_period}')
    volume_avg = features.get(f'volume_sma_{vol_period}')
    
    # Check if all features are available
    required_features = [bb_upper, bb_lower, kc_upper, kc_lower, momentum]
    if vol_filter:
        required_features.append(volume_avg)
    
    if any(f is None for f in required_features):
        logger.debug(f"Bollinger squeeze not ready for {bar.get('symbol')}")
        return None
    
    # Get current bar data
    current_volume = bar.get('volume', 0)
    
    # Detect squeeze: BB inside KC
    in_squeeze = bb_upper < kc_upper and bb_lower > kc_lower
    
    # Volume filter
    volume_ok = not vol_filter or (volume_avg and current_volume > volume_avg)
    
    # Generate signal based on squeeze release and momentum
    signal_value = 0
    
    if not in_squeeze and volume_ok:  # Squeeze has released
        if momentum > 0:
            signal_value = 1   # Bullish breakout
        elif momentum < 0:
            signal_value = -1  # Bearish breakout
    
    # Always return signal state
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_squeeze',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'in_squeeze': in_squeeze,
            'momentum': momentum,
            'bb_width': bb_upper - bb_lower,
            'kc_width': kc_upper - kc_lower,
            'volume_ratio': current_volume / volume_avg if volume_avg else 1.0
        }
    }
```

## Configuration Example

Add to your YAML configuration:

```yaml
strategies:
  - type: bollinger_squeeze
    name: bb_squeeze_grid
    params:
      bb_period: [15, 20, 25]
      bb_std: [1.5, 2.0, 2.5]
      kc_period: [15, 20, 25]
      kc_mult: [1.0, 1.5, 2.0]
      mom_period: [10, 12, 14]
      vol_filter: [true, false]
      # Total combinations: 3 * 3 * 3 * 3 * 3 * 2 = 486
```

## Performance Considerations

### Feature Computation Performance

With the centralized FeatureHub using incremental computation:
- **Feature updates**: ~0.3ms per bar (667x faster than pandas-based)
- **Strategy execution**: <0.1ms per strategy
- **Total latency**: <5ms for 1000 strategies on a single bar

### Memory Efficiency

- **FeatureHub**: Fixed memory footprint with rolling windows
- **Strategies**: Zero memory overhead (stateless)
- **Sparse Storage**: Only stores signal transitions

## Strategy Categories

### Indicator-Based Strategies
Located in `strategies/indicators/`:
- **crossovers.py**: MA crossovers, MACD crossovers
- **oscillators.py**: RSI, Stochastic, CCI, Williams %R
- **trend.py**: ADX, Aroon, Supertrend, PSAR, Linear Regression
- **volatility.py**: Bollinger Bands, Keltner Channels, ATR
- **volume.py**: OBV, MFI, VWAP, Volume patterns

### Pattern-Based Strategies
Located in `strategies/patterns/`:
- Chart patterns (head & shoulders, triangles, etc.)
- Candlestick patterns
- Market structure patterns

### Composite Strategies
Located in `strategies/composite/`:
- Multi-indicator combinations
- Regime-aware strategies
- Complex signal generation

## Grid Search Configuration

For parameter optimization, strategies support grid search expansion:

```yaml
strategies:
  - type: your_strategy
    name: your_strategy_grid
    params:
      # Single values
      param1: 20
      
      # Lists for grid search
      param2: [10, 20, 30]
      param3: [1.5, 2.0, 2.5]
      
      # Boolean flags
      use_filter: [true, false]
```

## Summary

Key points for successful strategy development:

1. **Always use the @strategy decorator** - Required for discovery
2. **Follow exact feature naming conventions** - Critical for feature lookup
3. **Return sustained signals** - Let sparse storage handle transitions
4. **Handle None features gracefully** - Check before using
5. **Keep strategies stateless** - No internal state between calls
6. **Test thoroughly** - Unit and integration tests
7. **Document clearly** - Help future developers
8. **Use debug logging** - Essential for troubleshooting feature issues
9. **Leverage grid search** - Test multiple parameter combinations
10. **Profile performance** - Ensure strategies execute quickly

With these guidelines, your strategies will integrate seamlessly with the ADMF-PC framework's high-performance architecture.