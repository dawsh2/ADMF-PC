# Strategy Module

This module provides THE canonical implementations for trading strategies, market classifiers, and strategy components for ADMF-PC.

## Architecture Reference
- **System Architecture**: docs/SYSTEM_ARCHITECTURE_v5.MD#strategy-module  
- **Style Guide**: STYLE.md - Canonical strategy implementations
- **Core Patterns**: docs/new/arch-101.md - Protocol + Composition

## Module Overview

The Strategy module implements trading strategies and market analysis using Protocol + Composition patterns with **no inheritance**. All components are designed as composable, protocol-compliant building blocks that can be mixed and matched through configuration.

## Core Principle: Stateless Strategies + Stateful Feature Engine

This module follows a **Two-Tier Architecture**:

### Tier 1: Stateless Strategy Components
Components that perform pure calculations based on inputs:
- **Trading Strategies**: Generate signals based on current feature values
- **Market Classifiers**: Classify market regimes from current data
- **Rules and Aggregators**: Combine signals using pure logic

### Tier 2: Stateful Feature Engine
The **FeatureHub** component that maintains state for efficient computation:
- **Feature Computation**: Incremental calculation of technical indicators
- **Rolling Windows**: Maintains price/volume history for indicators
- **Caching**: Optimized feature storage and retrieval
- **Real-time Updates**: Streaming data processing for live trading

## Files

### Core Protocol Definitions
- **`protocols.py`** - THE strategy protocol definitions
  - `Strategy`: Core trading strategy interface
  - `FeatureProvider`: Stateful feature computation engine interface
  - `FeatureExtractor`: Stateless feature extraction function interface
  - `Rule`: Trading rule evaluation interface
  - `SignalAggregator`: Signal combination interface
  - `Classifier`: Market regime classification interface
  - `RegimeAdaptive`: Regime-aware strategy interface
  - `Optimizable`: Parameter optimization interface

### Trading Strategies
- **`strategies/momentum.py`** - THE momentum strategy implementation
  - `MomentumStrategy`: Dual moving average crossover with RSI filter
  - Stateless: Consumes features from FeatureHub, no internal state
  - Protocol compliant: Implements Strategy protocol directly

- **`strategies/mean_reversion_simple.py`** - THE mean reversion strategy
  - `MeanReversionStrategy`: Bollinger Bands mean reversion
  - Stateless: Pure decision logic based on current feature values
  - Protocol compliant: No inheritance required

- **`strategies/`** directory contains additional canonical strategies:
  - `trend_following.py`: Donchian channel breakout strategies
  - `arbitrage.py`: Statistical arbitrage implementations
  - `market_making.py`: Bid-ask spread capture strategies

### Feature System
- **`components/indicators.py`** - THE feature computation engine
  - `FeatureHub`: Stateful feature computation engine (Tier 2)
  - Manages rolling windows for all technical indicators
  - Provides incremental updates for streaming data
  - Caches computed features for efficiency

- **`components/features.py`** - THE stateless feature functions
  - `sma_feature`, `ema_feature`, `rsi_feature`: Technical indicators
  - `bollinger_bands_feature`, `macd_feature`: Complex indicators
  - `FEATURE_REGISTRY`: Registry of all available feature functions
  - Pure functions: No state, work on pandas DataFrames

### Market Classifiers
- **`classifiers/hmm_classifier.py`** - THE HMM regime classifier
  - `HMMClassifier`: Hidden Markov Model for regime detection
  - Uses price returns, volume, and volatility for classification
  - Identifies bull, bear, and neutral market states

- **`classifiers/pattern_classifier.py`** - THE pattern-based classifier
  - `PatternClassifier`: Technical pattern recognition
  - Identifies trends, ranges, and reversal patterns

- **`classifiers/simple_classifiers.py`** - Simple regime classifiers
  - `TrendClassifier`: Basic trend detection
  - `VolatilityClassifier`: Volatility regime identification

### Strategy Components
- **`components/signal_aggregation.py`** - THE signal combination engine
  - `SignalCombiner`: Aggregates multiple signals within strategies
  - `WeightedAggregator`: Weighted signal combination
  - Used for intra-strategy signal processing (not multi-strategy)

### Optimization Framework
- **`optimization/`** - THE strategy optimization implementations
  - Complete optimization framework with Protocol + Composition
  - Multi-phase optimization workflows
  - Signal replay for efficient weight optimization
  - Walk-forward validation capabilities
  - See `optimization/README.md` for detailed documentation

## Usage Examples

### Basic Strategy Usage
```python
from src.strategy import MomentumStrategy
from src.strategy.components import FeatureHub, DEFAULT_MOMENTUM_FEATURES

# Create stateful feature engine
feature_hub = FeatureHub(symbols=["SPY", "QQQ"])
feature_hub.configure_features(DEFAULT_MOMENTUM_FEATURES)

# Create stateless strategy
strategy = MomentumStrategy(
    momentum_threshold=0.02,
    rsi_oversold=30,
    rsi_overbought=70
)

# Process market data
for bar in market_data:
    # Update features (stateful)
    feature_hub.update_bar(bar.symbol, {
        'open': bar.open,
        'high': bar.high, 
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    })
    
    # Generate signals (stateless)
    if feature_hub.has_sufficient_data(bar.symbol):
        features = feature_hub.get_all_features()
        strategy_input = {
            'market_data': {bar.symbol: bar},
            'features': features,
            'timestamp': bar.timestamp
        }
        
        signals = strategy.generate_signals(strategy_input)
        for signal in signals:
            print(f"Generated {signal.side} signal for {signal.symbol}")
```

### Mean Reversion Strategy
```python
from src.strategy.strategies import MeanReversionStrategy
from src.strategy.components import DEFAULT_MEAN_REVERSION_FEATURES

# Create feature engine for mean reversion
feature_hub = FeatureHub(symbols=["SPY"])
feature_hub.configure_features(DEFAULT_MEAN_REVERSION_FEATURES)

# Create strategy
strategy = MeanReversionStrategy(
    entry_threshold=2.0,  # 2 standard deviations
    exit_threshold=0.5    # 0.5 standard deviations
)

# Strategy consumes Bollinger Bands and RSI features
# from FeatureHub - no internal state management
```

### Market Classification
```python
from src.strategy.classifiers import HMMClassifier

# Create HMM classifier
classifier = HMMClassifier(
    parameters=HMMParameters(
        n_states=3,
        lookback_period=20,
        confidence_threshold=0.6
    )
)

# Classify market regime
regime = classifier.classify_regime(market_data, timestamp)
print(f"Current regime: {regime}")
```

### Custom Feature Configuration
```python
# Define custom feature set
custom_features = {
    "sma_10": {"feature": "sma", "period": 10},
    "sma_50": {"feature": "sma", "period": 50},
    "rsi_14": {"feature": "rsi", "period": 14},
    "bollinger_20": {"feature": "bollinger", "period": 20, "std_dev": 2.0},
    "atr_14": {"feature": "atr", "period": 14}
}

# Configure FeatureHub
feature_hub = FeatureHub(symbols=["AAPL", "GOOGL"])
feature_hub.configure_features(custom_features)

# Features available to all strategies:
# sma_10, sma_50, rsi_14, bollinger_20_upper, 
# bollinger_20_middle, bollinger_20_lower, atr_14
```

## Protocol Compliance

All strategy components implement standard protocols:

```python
# Strategies implement Strategy protocol
def generate_signals(self, strategy_input: Dict[str, Any]) -> List[Signal]:
    """Generate trading signals from market data and features"""

# Feature providers implement FeatureProvider protocol
def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
    """Update with new bar data for incremental feature calculation"""

def get_features(self, symbol: str) -> Dict[str, Any]:
    """Get current feature values for a symbol"""

# Classifiers implement Classifier protocol
def classify(self, data: Dict[str, Any]) -> str:
    """Classify current market conditions"""
```

## Strategy Discovery and Registration

### IMPORTANT: Strategy Decorator Requirement

All pure function strategies MUST use the `@strategy` decorator for automatic discovery and feature inference:

```python
from ...core.components.discovery import strategy

# RECOMMENDED: Simplified list format
@strategy(
    name='my_strategy',
    feature_config=['sma', 'rsi', 'bollinger_bands']  # Simple list of features
)
def my_strategy_function(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strategy implementation"""
    pass

# LEGACY: Complex dict format (still supported but not recommended)
@strategy(
    name='my_strategy',
    feature_config={
        'sma': {'params': ['sma_period'], 'defaults': {'sma_period': 20}},
        'rsi': {'params': [], 'default': 14}
    }
)
```

### Why the Decorator is Required

1. **Automatic Discovery**: The topology builder uses the component registry to find strategies
2. **Feature Inference**: The decorator metadata enables automatic feature requirement detection
3. **Parameter Expansion**: Grid search and parameter optimization rely on the metadata

Without this decorator, the strategy will NOT be discovered by the topology builder and feature inference will fail!

### Best Practices for Feature Declaration

1. **Use Simple List Format**: Declare features as a simple list when possible
2. **Let Inference Handle Parameters**: The topology builder will infer feature parameters from strategy parameters
3. **Follow Naming Conventions**: Strategy parameters should follow patterns like `{feature}_period`, `fast_{feature}_period`, etc.

## Complete Guide: Adding a New Strategy

### Step 1: Define Your Strategy Function

Create your strategy in the appropriate file under `strategies/` directory:

```python
from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)

@strategy(
    name='bollinger_mean_reversion',  # Unique name for discovery
    feature_config={
        # Define required features with parameter mappings
        'bollinger': {
            'params': ['bb_period', 'bb_std'],  # Maps to strategy params
            'defaults': {'bb_period': 20, 'bb_std': 2.0}
        },
        'rsi': {
            'params': ['rsi_period'],
            'defaults': {'rsi_period': 14}
        },
        'volume_sma': {
            'params': ['volume_period'],
            'defaults': {'volume_period': 20}
        }
    }
)
def bollinger_mean_reversion(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger Bands mean reversion strategy with RSI and volume filters.
    
    Args:
        features: Pre-computed features from FeatureHub
        bar: Current bar data with symbol, timestamp, OHLCV
        params: Strategy parameters from configuration
        
    Returns:
        Signal dict with signal_value (-1, 0, 1) or None if not ready
    """
    # Extract parameters with defaults
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    volume_period = params.get('volume_period', 20)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Get required features - use exact naming convention
    upper_band = features.get(f'bollinger_{bb_period}_{bb_std}_upper')
    lower_band = features.get(f'bollinger_{bb_period}_{bb_std}_lower')
    middle_band = features.get(f'bollinger_{bb_period}_{bb_std}_middle')
    rsi = features.get(f'rsi_{rsi_period}')
    volume_ma = features.get(f'volume_sma_{volume_period}')
    
    # Check if all required features are available
    if any(f is None for f in [upper_band, lower_band, middle_band, rsi, volume_ma]):
        logger.debug(f"Bollinger mean reversion not ready. Features: upper={upper_band}, lower={lower_band}, rsi={rsi}")
        return None
    
    current_price = bar.get('close', 0)
    current_volume = bar.get('volume', 0)
    
    # Strategy logic
    signal_value = 0
    
    # Buy signal: Price below lower band, RSI oversold, volume above average
    if (current_price < lower_band and 
        rsi < rsi_oversold and 
        current_volume > volume_ma):
        signal_value = 1
        
    # Sell signal: Price above upper band, RSI overbought, volume above average
    elif (current_price > upper_band and 
          rsi > rsi_overbought and 
          current_volume > volume_ma):
        signal_value = -1
    
    # Always return signal state (sparse storage handles unchanged signals)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_mean_reversion',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': current_price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'rsi': rsi,
            'volume': current_volume,
            'volume_ma': volume_ma,
            'band_position': (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        }
    }
```

### Step 2: Feature Naming Conventions

Features are named based on their type and parameters. The incremental feature system generates names like:

**Single-value features:**
- `sma_{period}` → e.g., `sma_20`
- `ema_{period}` → e.g., `ema_50`
- `rsi_{period}` → e.g., `rsi_14`
- `atr_{period}` → e.g., `atr_14`

**Multi-value features (with suffixes):**
- `bollinger_{period}_{std_dev}_upper/middle/lower` → e.g., `bollinger_20_2.0_upper`
- `macd_{fast}_{slow}_{signal}_macd/signal/histogram` → e.g., `macd_12_26_9_macd`
- `aroon_{period}_up/down/oscillator` → e.g., `aroon_25_up`
- `supertrend_{period}_{multiplier}_supertrend/trend/upper/lower` → e.g., `supertrend_10_3.0_trend`
- `stochastic_{k_period}_{d_period}_k/d` → e.g., `stochastic_14_3_k`

### Step 3: Configuration File Entry

Add your strategy to the configuration YAML:

```yaml
strategies:
  - type: bollinger_mean_reversion
    name: bb_mean_reversion_grid  # Name for grid expansion
    params:
      # Parameters for grid search
      bb_period: [15, 20, 25]
      bb_std: [1.5, 2.0, 2.5]
      rsi_period: [10, 14, 21]
      rsi_oversold: [25, 30]
      rsi_overbought: [70, 75]
      volume_period: [10, 20]
      # Total combinations: 3 * 3 * 3 * 2 * 2 * 2 = 216
```

### Step 4: Common Pitfalls and Solutions

#### 1. **Feature Not Found**
```python
# WRONG - Feature naming mismatch
upper_band = features.get('bollinger_upper')  # Missing parameters in name

# CORRECT - Include all parameters
upper_band = features.get(f'bollinger_{bb_period}_{bb_std}_upper')
```

#### 2. **Missing Decorator**
```python
# WRONG - No decorator, won't be discovered
def my_strategy(features, bar, params):
    pass

# CORRECT - With decorator
@strategy(name='my_strategy', feature_config={...})
def my_strategy(features, bar, params):
    pass
```

#### 3. **Wrong Parameter Mapping**
```python
# WRONG - Parameter name doesn't match what strategy expects
feature_config={
    'sma': {'params': ['period']}  # Strategy expects 'sma_period'
}

# CORRECT - Parameter names match
feature_config={
    'sma': {'params': ['sma_period']}
}
```

#### 4. **Not Handling None Features**
```python
# WRONG - Will crash if features not ready
signal = 1 if features['rsi_14'] < 30 else 0

# CORRECT - Check for None
rsi = features.get('rsi_14')
if rsi is None:
    return None
signal = 1 if rsi < 30 else 0
```

### Step 5: Testing Your Strategy

Create a test to verify your strategy works:

```python
def test_bollinger_mean_reversion():
    """Test bollinger mean reversion strategy."""
    from src.strategy.strategies.indicators import bollinger_mean_reversion
    
    # Test oversold condition
    features = {
        'bollinger_20_2.0_upper': 105,
        'bollinger_20_2.0_middle': 100,
        'bollinger_20_2.0_lower': 95,
        'rsi_14': 25,
        'volume_sma_20': 1000
    }
    
    bar = {
        'symbol': 'SPY',
        'close': 94,  # Below lower band
        'volume': 1500,  # Above average
        'timeframe': '1m',
        'timestamp': '2024-01-01T10:00:00'
    }
    
    params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volume_period': 20
    }
    
    result = bollinger_mean_reversion(features, bar, params)
    assert result is not None
    assert result['signal_value'] == 1  # Buy signal
```

### Step 6: File Organization

Place strategies in the appropriate category:

```
strategies/
├── indicators/          # Technical indicator-based strategies
│   ├── crossovers.py   # MA crossovers, MACD crossovers, etc.
│   ├── oscillators.py  # RSI, Stochastic, Williams %R strategies
│   ├── trend.py        # ADX, Aroon, Supertrend strategies
│   ├── volatility.py   # Bollinger, Keltner, ATR strategies
│   └── volume.py       # OBV, MFI, VWAP strategies
├── patterns/           # Chart pattern strategies
├── ml/                 # Machine learning strategies
└── __init__.py         # Exports all strategies
```

### Complete Working Example

Here's a complete example that follows all best practices:

```python
# File: strategies/indicators/volatility.py

from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)

@strategy(
    name='keltner_squeeze',
    feature_config={
        'keltner': {
            'params': ['kc_period', 'kc_mult'],
            'defaults': {'kc_period': 20, 'kc_mult': 2.0}
        },
        'bollinger': {
            'params': ['bb_period', 'bb_std'],
            'defaults': {'bb_period': 20, 'bb_std': 2.0}
        },
        'momentum': {
            'params': ['mom_period'],
            'defaults': {'mom_period': 12}
        }
    }
)
def keltner_squeeze(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keltner Channel Squeeze strategy - detects volatility compression.
    
    Signals when Bollinger Bands are inside Keltner Channels (squeeze)
    and trades on momentum direction when squeeze releases.
    """
    # Parameters
    kc_period = params.get('kc_period', 20)
    kc_mult = params.get('kc_mult', 2.0)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    mom_period = params.get('mom_period', 12)
    
    # Get features - note the exact naming format
    kc_upper = features.get(f'keltner_{kc_period}_{kc_mult}_upper')
    kc_lower = features.get(f'keltner_{kc_period}_{kc_mult}_lower')
    bb_upper = features.get(f'bollinger_{bb_period}_{bb_std}_upper')
    bb_lower = features.get(f'bollinger_{bb_period}_{bb_std}_lower')
    momentum = features.get(f'momentum_{mom_period}')
    
    # Readiness check
    if any(f is None for f in [kc_upper, kc_lower, bb_upper, bb_lower, momentum]):
        return None
    
    # Detect squeeze: BB inside KC
    in_squeeze = bb_upper < kc_upper and bb_lower > kc_lower
    
    # Look for squeeze release with momentum
    signal_value = 0
    if not in_squeeze:  # Squeeze has released
        if momentum > 0:
            signal_value = 1   # Bullish breakout
        elif momentum < 0:
            signal_value = -1  # Bearish breakout
    
    # Return signal
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'keltner_squeeze',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'in_squeeze': in_squeeze,
            'momentum': momentum,
            'kc_width': kc_upper - kc_lower,
            'bb_width': bb_upper - bb_lower,
            'price': bar.get('close', 0)
        }
    }
```

### Debugging Tips

1. **Enable debug logging** to see why strategies aren't generating signals:
   ```python
   import logging
   logging.getLogger('src.strategy').setLevel(logging.DEBUG)
   ```

2. **Check feature availability** by printing available features:
   ```python
   logger.info(f"Available features: {list(features.keys())}")
   ```

3. **Verify strategy registration**:
   ```python
   from src.core.components.discovery import get_component_registry
   strategies = get_component_registry().get('strategies', {})
   print(f"Registered strategies: {list(strategies.keys())}")
   ```

4. **Test feature inference**:
   ```python
   # The topology builder will log inferred features
   # Look for: "Inferred X unique features from Y strategies"
   ```

## Configuration Patterns

### YAML Configuration
```yaml
strategy_config:
  momentum_strategy:
    type: "momentum"
    momentum_threshold: 0.02
    rsi_oversold: 30
    rsi_overbought: 70
    
  mean_reversion_strategy:
    type: "mean_reversion"
    entry_threshold: 2.0
    exit_threshold: 0.5

feature_config:
  sma_fast: 
    feature: "sma"
    period: 10
  sma_slow:
    feature: "sma" 
    period: 20
  rsi:
    feature: "rsi"
    period: 14

classifier_config:
  hmm_classifier:
    type: "hmm"
    n_states: 3
    lookback_period: 20
    confidence_threshold: 0.6
```

### Factory Functions
```python
def create_momentum_strategy(config: Dict[str, Any]) -> MomentumStrategy:
    """Factory function for momentum strategy creation"""
    return MomentumStrategy(
        momentum_threshold=config.get('momentum_threshold', 0.02),
        rsi_oversold=config.get('rsi_oversold', 30),
        rsi_overbought=config.get('rsi_overbought', 70)
    )

def create_feature_hub(symbols: List[str], 
                      feature_type: str) -> FeatureHub:
    """Factory function for FeatureHub creation"""
    feature_configs = {
        'momentum': DEFAULT_MOMENTUM_FEATURES,
        'mean_reversion': DEFAULT_MEAN_REVERSION_FEATURES,
        'volatility': DEFAULT_VOLATILITY_FEATURES
    }
    
    hub = FeatureHub(symbols)
    hub.configure_features(feature_configs[feature_type])
    return hub
```

## Testing Strategy

### Unit Testing
```python
def test_momentum_strategy_signal_generation():
    """Test momentum strategy signal logic"""
    strategy = MomentumStrategy(momentum_threshold=0.02)
    
    # Test data with momentum signal
    strategy_input = {
        'market_data': {'SPY': {'close': 100}},
        'features': {
            'SPY': {
                'sma_fast': 102,
                'sma_slow': 100,
                'rsi': 50
            }
        },
        'timestamp': datetime.now()
    }
    
    signals = strategy.generate_signals(strategy_input)
    assert len(signals) == 1
    assert signals[0].side == OrderSide.BUY

def test_feature_hub_incremental_updates():
    """Test FeatureHub incremental computation"""
    hub = FeatureHub(symbols=['TEST'])
    hub.configure_features({'sma_10': {'feature': 'sma', 'period': 10}})
    
    # Add bars incrementally
    for i in range(20):
        hub.update_bar('TEST', {'close': 100 + i})
    
    features = hub.get_features('TEST')
    assert 'sma_10' in features
    assert features['sma_10'] > 100
```

### Integration Testing
```python
def test_strategy_feature_hub_integration():
    """Test full strategy + feature hub integration"""
    # Setup
    hub = FeatureHub(['SPY'])
    hub.configure_features(DEFAULT_MOMENTUM_FEATURES)
    strategy = MomentumStrategy()
    
    # Feed data
    bars = generate_test_bars('SPY', 100)
    for bar in bars:
        hub.update_bar('SPY', bar)
    
    # Generate signals
    if hub.has_sufficient_data('SPY'):
        features = hub.get_all_features()
        strategy_input = {
            'market_data': {'SPY': bars[-1]},
            'features': features,
            'timestamp': datetime.now()
        }
        signals = strategy.generate_signals(strategy_input)
        
        # Verify signal quality
        assert all(isinstance(s.strength, Decimal) for s in signals)
        assert all(0 <= s.strength <= 1 for s in signals)
```

## Performance Considerations

- **FeatureHub Efficiency**: Uses rolling windows with fixed memory footprint
- **Feature Caching**: Computed features cached until next bar update
- **Stateless Strategies**: No memory overhead from strategy state
- **Lazy Computation**: Features computed only when configured
- **Batch Updates**: Multiple symbols processed efficiently

## Architecture Benefits

### Two-Tier Separation
1. **FeatureHub (Tier 2)**: Handles ALL stateful computation
   - Rolling windows for technical indicators
   - Incremental updates for real-time processing
   - Optimized caching and memory management

2. **Strategies (Tier 1)**: Purely stateless decision logic
   - Consume pre-computed features
   - Generate signals based on current state
   - No memory overhead or state management

### Protocol + Composition Advantages
- **No Inheritance**: All components implement protocols directly
- **Flexible Composition**: Mix and match any compatible components  
- **Easy Testing**: Pure functions and clear interfaces
- **Container Isolation**: Each container gets fresh strategy instances
- **Optimization Friendly**: Stateless strategies optimize efficiently

## No "Enhanced" Versions

Do not create `enhanced_momentum_strategy.py`, `improved_feature_hub.py`, etc. Use composition and configuration to add capabilities to the canonical implementations in this module.

## Integration Points

- **Data Module**: Receives market data bars for feature computation
- **Risk Module**: Sends generated trading signals for risk processing
- **Core Events**: Integrates with event bus for signal publishing
- **Coordinator**: Orchestrates strategy lifecycle and configuration

## Future Enhancements

- **ML Feature Extractors**: Machine learning-based feature computation
- **Alternative Data**: News, sentiment, economic indicators
- **Multi-Asset Strategies**: Cross-asset signal generation
- **Dynamic Feature Selection**: Adaptive feature sets based on market conditions

## Strategy-Feature Integration Best Practices

### 1. Feature Configuration Format

**Preferred**: Use the simplified list format for feature declarations
```python
@strategy(
    name='momentum_breakout',
    feature_config=['sma', 'rsi', 'atr']  # Simple and clear
)
```

**Avoid**: Complex dictionary format unless you need special parameter mappings
```python
# Only use this format if you have non-standard parameter names
@strategy(
    feature_config={
        'sma': {'params': ['custom_ma_period'], 'defaults': {'custom_ma_period': 20}}
    }
)
```

### 2. Parameter Naming Conventions

Follow standard naming patterns for automatic feature inference:
- `{feature}_period`: For single-period features (e.g., `sma_period`, `rsi_period`)
- `fast_{feature}_period`, `slow_{feature}_period`: For dual-period features
- `{feature}_{param}`: For other parameters (e.g., `bollinger_std_dev`)

### 3. Feature Access Patterns

Always use the exact feature naming convention when accessing features:
```python
# Correct - includes all parameters in the feature name
rsi_value = features.get(f'rsi_{rsi_period}')
bollinger_upper = features.get(f'bollinger_bands_{period}_{std_dev}_upper')

# Wrong - missing parameters or wrong format
rsi_value = features.get('rsi')  # Missing period
bollinger_upper = features.get('bollinger_upper')  # Missing all parameters
```

### 4. Feature Readiness Checks

Always check if features are available before using them:
```python
# Get all required features
required_features = [
    features.get(f'sma_{fast_period}'),
    features.get(f'sma_{slow_period}'),
    features.get(f'rsi_{rsi_period}')
]

# Check if any are None
if any(f is None for f in required_features):
    return None  # Not ready yet

# Safe to use features
fast_sma, slow_sma, rsi = required_features
```

### 5. Strategy Registration Patterns

Ensure your strategy module is imported for registration:
```python
# In strategies/__init__.py or strategies/indicators/__init__.py
from .crossovers import *  # Imports all @strategy decorated functions
from .oscillators import *
from .trend import *
```

### 6. Testing Feature Integration

Test that your strategy correctly declares and uses features:
```python
def test_strategy_feature_requirements():
    """Test that strategy declares correct features."""
    from src.core.components.discovery import get_component_registry
    
    registry = get_component_registry()
    strategy_info = registry.get_component('my_strategy')
    
    # Verify feature config
    feature_config = strategy_info.metadata.get('feature_config', [])
    assert 'sma' in feature_config
    assert 'rsi' in feature_config
```