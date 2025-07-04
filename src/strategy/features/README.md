# Feature Hub and Feature System

This directory contains the stateful feature computation engine (Tier 2) that supports all stateless strategies and classifiers.

## Architecture Overview

The feature system follows a clean protocol + composition architecture:

- **FeatureHub** (`hub.py`): THE canonical stateful engine managing O(1) incremental computation
- **Technical Indicators** (`indicators/`): All technical analysis indicators organized by category
- **Protocols** (`protocols.py`): Feature protocol definitions for type safety
- **Feature Specs** (`specs/`): Feature specification system for deterministic naming

## Feature Hub (`hub.py`)

The centralized feature computation engine that:
- Maintains rolling windows for all symbols
- Provides incremental O(1) updates for streaming data
- Caches computed features for efficiency
- Manages feature dependencies

### Usage Example

```python
from .hub import FeatureHub

# Create hub for multiple symbols
hub = FeatureHub(symbols=['SPY', 'QQQ'], use_incremental=True)

# Configure features
feature_configs = {
    'sma_20': {'type': 'sma', 'period': 20},
    'rsi_14': {'type': 'rsi', 'period': 14},
    'bollinger': {'type': 'bollinger', 'period': 20, 'std_dev': 2.0}
}
hub.configure_features(feature_configs)

# Update with new bar data
bar_data = {
    'open': 100.0, 'high': 101.0, 'low': 99.0,
    'close': 100.5, 'volume': 1000000
}
hub.update_bar('SPY', bar_data)

# Get computed features
features = hub.get_features('SPY')
# Returns: {'sma_20': 100.2, 'rsi_14': 55.5, 'bollinger_20_2.0_upper': 102.1, ...}
```

## Technical Indicators (`indicators/`)

All technical analysis indicators are organized by category in the `indicators/` subdirectory:

```
indicators/
├── trend.py        # SMA, EMA, DEMA, TEMA, WMA, HMA, VWMA (7 features)
├── oscillators.py  # RSI, Stochastic, Williams %R, CCI, etc. (6 features)
├── volatility.py   # ATR, Bollinger Bands, Keltner, Donchian, etc. (7 features)
├── volume.py       # Volume SMA, OBV, VPT, CMF, etc. (8 features)
├── momentum.py     # MACD, ROC, ADX, Aroon, Vortex (6 features)
└── structure.py    # Pivot Points, Support/Resistance, etc. (6 features)
```

**Total: 40 technical indicators** all using O(1) incremental updates.

The `FEATURE_REGISTRY` in `hub.py` automatically imports all indicators:

```python
from .indicators import ALL_INDICATOR_FEATURES
FEATURE_REGISTRY = ALL_INDICATOR_FEATURES.copy()
```

## Feature Naming Convention

Features follow a predictable naming pattern based on their parameters:

### Single-Value Features
- `{feature}_{period}` → `sma_20`, `rsi_14`, `atr_10`

### Multi-Value Features
Features that return multiple values use suffixes:

- **Bollinger Bands**: `bollinger_bands_{period}_{std_dev}_{upper|middle|lower}`
  - Example: `bollinger_bands_20_2.0_upper`
  
- **MACD**: `macd_{fast}_{slow}_{signal}_{macd|signal|histogram}`
  - Example: `macd_12_26_9_signal`
  
- **Stochastic**: `stochastic_{k_period}_{d_period}_{k|d}`
  - Example: `stochastic_14_3_k`

## Protocol + Composition Architecture

All indicators follow the same protocol-based architecture with no inheritance:

```python
# Every indicator implements the Feature protocol
class SMA:
    def __init__(self, period: int, name: str = "sma"):
        self._state = FeatureState(name)  # Composition, not inheritance
        self.period = period
        self._sum = 0.0
        self._buffer = deque(maxlen=period)
    
    @property
    def name(self) -> str:
        return self._state.name
    
    @property 
    def value(self) -> Optional[float]:
        return self._state.value
    
    def update(self, price: float) -> Optional[float]:
        # O(1) incremental update logic
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        self._buffer.append(price)
        self._sum += price
        
        if len(self._buffer) == self.period:
            self._state.set_value(self._sum / self.period)
        
        return self._state.value
```

### Performance Benefits

- **O(1) updates**: Each new data point updates in constant time
- **Memory efficient**: Fixed-size rolling windows, no growing arrays  
- **Stateful computation**: Maintains rolling state for real-time processing
- **Protocol compliance**: All indicators implement the same interface

## Feature Configuration Best Practices

### 1. Use Consistent Parameter Names

```python
# Good - standard parameter names
feature_configs = {
    'sma_fast': {'type': 'sma', 'period': 10},
    'sma_slow': {'type': 'sma', 'period': 50},
    'rsi': {'type': 'rsi', 'period': 14}
}

# Avoid - non-standard names
feature_configs = {
    'my_average': {'type': 'sma', 'window': 10}  # Use 'period' not 'window'
}
```

### 2. Handle Multi-Value Features

```python
# Bollinger Bands returns multiple values
features = hub.get_features('SPY')

# Access individual components
upper_band = features['bollinger_20_2.0_upper']
middle_band = features['bollinger_20_2.0_middle']  
lower_band = features['bollinger_20_2.0_lower']
```

### 3. Check Feature Readiness

```python
# Features may be None during warmup period
if hub.has_sufficient_data('SPY', min_bars=50):
    features = hub.get_features('SPY')
    # Safe to use features
else:
    # Still warming up
    pass
```

## Adding New Features

### 1. Choose Appropriate Category

Add your indicator to the correct file in `indicators/`:
- `trend.py` - Moving averages, trend-following indicators
- `oscillators.py` - Bounded oscillators (RSI, Stochastic, etc.)  
- `volatility.py` - Volatility measures (ATR, Bollinger Bands, etc.)
- `volume.py` - Volume-based indicators (OBV, VPT, etc.)
- `momentum.py` - Momentum indicators (MACD, ROC, etc.)
- `structure.py` - Support/resistance, pivots, patterns

### 2. Implement the Feature Class

```python
# In indicators/trend.py (example)
class MyTrendIndicator:
    """My custom trend indicator with O(1) updates."""
    
    def __init__(self, period: int = 20, name: str = "my_trend"):
        self._state = FeatureState(name)  # Use composition
        self.period = period
        # Initialize your rolling state here
        
    @property
    def name(self) -> str:
        return self._state.name
    
    @property
    def value(self) -> Optional[float]:
        return self._state.value
    
    @property
    def is_ready(self) -> bool:
        return self._state.is_ready
    
    def update(self, price: float, **kwargs) -> Optional[float]:
        # O(1) incremental update logic here
        # self._state.set_value(computed_value)
        return self._state.value
    
    def reset(self) -> None:
        self._state.reset()
        # Reset your rolling state here
```

### 3. Register in Feature Registry

```python
# At the bottom of indicators/trend.py
TREND_FEATURES = {
    # ... existing features
    "my_trend": MyTrendIndicator,
}
```

The feature will automatically be available in FeatureHub through the consolidated registry.

## Feature Dependencies

Some features depend on others. The system handles this automatically:

```python
# Bollinger Bands depends on SMA
# When you request bollinger_bands, it computes SMA internally

# Supertrend depends on ATR
# When you request supertrend, it computes ATR first
```

## Integration with Strategies

Strategies declare required features, and the topology builder infers the specific feature configurations:

```python
@strategy(
    name='momentum_strategy',
    feature_config=['sma', 'rsi']  # Simple declaration
)
def momentum_strategy(features, bar, params):
    # Access inferred features
    fast_sma = features[f'sma_{params["fast_period"]}']
    slow_sma = features[f'sma_{params["slow_period"]}']
    rsi = features[f'rsi_{params["rsi_period"]}']
```

## Memory Management

The FeatureHub uses rolling windows with fixed memory footprint:

```python
# Configurable window size (default: 1000 bars)
self.price_data[symbol] = {
    'open': deque(maxlen=1000),
    'high': deque(maxlen=1000),
    'low': deque(maxlen=1000),
    'close': deque(maxlen=1000),
    'volume': deque(maxlen=1000)
}
```

## Thread Safety

The current implementation is designed for single-threaded use. For multi-threaded environments, wrap access with appropriate locks:

```python
import threading

class ThreadSafeFeatureHub:
    def __init__(self, *args, **kwargs):
        self.hub = FeatureHub(*args, **kwargs)
        self.lock = threading.Lock()
        
    def update_bar(self, symbol, bar):
        with self.lock:
            return self.hub.update_bar(symbol, bar)
```

## Directory Structure

The clean, organized features directory:

```
features/
├── indicators/              # All technical indicators by category
│   ├── __init__.py         # Exports all indicators
│   ├── trend.py            # 7 trend indicators
│   ├── oscillators.py      # 6 oscillator indicators  
│   ├── volatility.py       # 7 volatility indicators
│   ├── volume.py           # 8 volume indicators
│   ├── momentum.py         # 6 momentum indicators
│   └── structure.py        # 6 structure indicators
├── specs/                  # Feature specification system
│   ├── __init__.py        # Exports feature specs
│   ├── feature_spec.py    # Core feature specification
│   ├── composite_features.py # Composite feature support
│   └── feature_spec_improvements.py # Enhanced specs
├── hub.py                  # THE canonical FeatureHub implementation
├── protocols.py            # Feature protocol definitions
├── __init__.py            # Exports FeatureHub and all indicators
└── README.md              # This documentation
```

## Performance Tips

1. **All Features Are Incremental**: No configuration needed - all 40 indicators use O(1) updates
2. **Configure Only Needed Features**: Don't compute unused features
3. **Batch Symbol Updates**: Update all symbols before getting features
4. **Cache Feature Access**: Store feature references in tight loops

```python
# Good - cache feature access
features = hub.get_features('SPY')
sma = features['sma_20']
for i in range(100):
    if close[i] > sma:  # Use cached value
        # ...

# Avoid - repeated lookups
for i in range(100):
    if close[i] > hub.get_features('SPY')['sma_20']:  # Repeated dict access
        # ...
```

## Testing Features

```python
def test_feature_computation():
    """Test feature hub computation."""
    hub = FeatureHub(['TEST'])
    hub.configure_features({
        'sma_10': {'type': 'sma', 'period': 10}
    })
    
    # Feed test data
    for i in range(20):
        hub.update_bar('TEST', {
            'open': 100, 'high': 101, 'low': 99,
            'close': 100 + i * 0.1, 'volume': 1000
        })
    
    # Verify computation
    features = hub.get_features('TEST')
    assert 'sma_10' in features
    assert features['sma_10'] is not None
```

## Common Issues and Solutions

### Feature Not Found
```python
# Problem: Feature name mismatch
sma = features.get('sma')  # Returns None

# Solution: Include parameters in name
sma = features.get('sma_20')  # Correct
```

### Features Still None After Many Bars
```python
# Problem: Insufficient warmup period
# Solution: Check has_sufficient_data()
if hub.has_sufficient_data(symbol, min_bars=50):
    features = hub.get_features(symbol)
```

### Import Errors
```python
# Problem: Import from old structure
from .trend import sma_feature  # OLD - no longer exists

# Solution: Import from indicators
from .indicators.trend import SMA  # NEW - organized structure
```

## No Legacy Code

This features directory contains **only canonical implementations**:
- No `enhanced_`, `improved_`, `advanced_` files
- No pandas-based legacy implementations  
- No duplicate/experimental code
- Single source of truth for each indicator type