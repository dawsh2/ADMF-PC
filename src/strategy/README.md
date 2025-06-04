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