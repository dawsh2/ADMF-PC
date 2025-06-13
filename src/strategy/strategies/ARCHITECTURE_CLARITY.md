# Architecture Clarity: Features vs Indicators vs Classifiers

## Current Architecture (Correct)

```
src/strategy/
├── components/
│   ├── features.py              # Stateful computation engine (Tier 2)
│   └── signal_strength_filter.py
├── classifiers/                 # Market regime classification
│   ├── hmm_classifier.py
│   └── pattern_classifier.py
└── strategies/
    ├── indicators/              # Indicator-based signal strategies (Tier 1)
    └── ensemble/               # Multi-indicator strategies
```

## Role Definitions

### 1. **Features (Tier 2 - Stateful)**
**Location**: `src/strategy/components/features.py`
**Role**: Stateful computation engine that calculates technical indicators
**Responsibility**: 
- Maintains rolling windows and state
- Calculates SMA, EMA, RSI, Bollinger Bands, etc.
- Provides pre-computed indicator values to strategies

**Example**:
```python
@feature(name='rsi', params=['period'])
def rsi_feature(prices: pd.Series, period: int = 14) -> pd.Series:
    # Calculates RSI values from price series
    return rsi_values
```

### 2. **Indicator Strategies (Tier 1 - Stateless)**
**Location**: `src/strategy/strategies/indicators/`
**Role**: Stateless signal generators that consume feature values
**Responsibility**:
- Takes pre-computed features as input
- Applies trading logic to generate binary signals (-1, 0, 1)
- No state management - pure functions

**Example**:
```python
@strategy(name='rsi_threshold')
def rsi_threshold(features: Dict, bar: Dict, params: Dict) -> Optional[Dict]:
    rsi = features.get('rsi_14')  # Consumes pre-computed RSI
    if rsi > 70: return {'signal_value': -1}  # Trading logic
```

### 3. **Classifiers (Separate Domain)**
**Location**: `src/strategy/classifiers/`
**Role**: Market regime detection (not trading signals)
**Responsibility**:
- Classify market conditions (trending, ranging, volatile)
- Emit CLASSIFICATION events (not SIGNAL events)
- Used by adaptive strategies to switch behavior

## Key Distinctions

### Features vs Indicator Strategies
```
Features (FeatureHub):          Indicator Strategies:
─────────────────────           ──────────────────
• Calculates RSI values         • Uses RSI values for signals
• Stateful (rolling windows)    • Stateless (pure functions)
• Shared across all strategies  • Specific trading logic
• Technical computation         • Business logic
• Returns indicator values      • Returns trading signals
```

### Signals vs Classifications
```
Signal Events:                  Classification Events:
──────────────                 ─────────────────────
• Trading decisions             • Market regime detection
• Binary values (-1, 0, 1)      • Regime labels ("trending", "ranging")
• Used by risk module           • Used by adaptive strategies
• Trigger position changes      • Trigger strategy switching
```

## No Redundancy Needed

### ❌ **Don't Create**: `classifiers/indicators/`
Classifiers should NOT have their own indicator strategies because:

1. **Different Purpose**: Classifiers detect market regimes, not generate trades
2. **Different Output**: Classification labels, not binary signals
3. **Different Consumers**: Adaptive strategies, not risk module

### ✅ **Correct Pattern**:
```python
# Classifier uses features directly
@classifier(name='volatility_regime')
def volatility_regime_classifier(features: Dict, params: Dict) -> Dict:
    atr = features.get('atr_14')
    volatility = features.get('volatility_20')
    
    if volatility > 0.03:
        return {'regime': 'high_volatility', 'confidence': 0.8}
    else:
        return {'regime': 'low_volatility', 'confidence': 0.7}
```

## Data Flow

```
Market Data → FeatureHub → [Features] → Strategies → [Signals] → Risk Module
                     ↓
                [Features] → Classifiers → [Classifications] → Adaptive Strategies
```

### Example Flow:
1. **Market bar arrives**: OHLCV data
2. **FeatureHub processes**: Calculates RSI, SMA, Bollinger Bands
3. **Features available**: `{'rsi_14': 75, 'sma_20': 100, 'bollinger_upper': 105}`
4. **Strategies consume**: 
   - `rsi_threshold` uses `rsi_14` → Signal: -1 (sell)
   - `ma_crossover` uses `sma_10`, `sma_20` → Signal: 1 (buy)
5. **Classifiers consume**:
   - `volatility_classifier` uses `atr_14` → Classification: "high_volatility"
6. **Ensemble strategies**:
   - Combine signals: -1 + 1 = 0 (no trade)
   - Consider regime: high volatility → reduce position size

## Resolution: Keep Current Architecture

### ✅ **What We Have is Correct**:
1. **Features**: Stateful indicator computation (shared resource)
2. **Indicator Strategies**: Stateless signal generators (building blocks)
3. **Classifiers**: Regime detection (separate concern)
4. **Ensemble Strategies**: Multi-indicator combination

### ✅ **No Changes Needed**:
- Features and indicator strategies serve different layers
- No redundancy - they have distinct responsibilities
- Clean separation of concerns
- Each component has a single responsibility

## Usage Examples

### Feature Calculation (Stateful Tier):
```python
# In FeatureHub
feature_hub.update_bar('AAPL', {'close': 150, 'high': 151, 'low': 149})
features = feature_hub.get_features('AAPL')
# {'rsi_14': 72.5, 'sma_20': 148.5, 'bollinger_upper': 152.1}
```

### Strategy Signal Generation (Stateless Tier):
```python
# Indicator strategy uses pre-computed features
signal = rsi_threshold_strategy(features, bar, {'threshold': 70})
# {'signal_value': -1, 'reason': 'RSI overbought'}
```

### Classification (Separate Domain):
```python
# Classifier uses same features for different purpose
regime = volatility_classifier(features, {})
# {'regime': 'high_volatility', 'confidence': 0.8}
```

This architecture provides clean separation while avoiding duplication.