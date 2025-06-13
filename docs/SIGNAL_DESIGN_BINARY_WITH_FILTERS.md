# Signal Design: Binary Signals with Separate Strength Layer

## Core Design Principles

### 1. Binary Signal Values (-1, 0, 1)
- **1**: Long signal (sustained while conditions met)
- **-1**: Short signal (sustained while conditions met)
- **0**: No signal/neutral

### 2. Signal Persistence
Strategies emit sustained signals, not events:
- MA crossover: Signal stays 1 while fast > slow, -1 while fast < slow
- RSI: Signal stays 1 while oversold, -1 while overbought
- Bollinger: Signal stays 1 while below lower band, -1 while above upper

### 3. No Signal Types
- Remove ENTRY/EXIT types - the signal value itself indicates position
- Remove REBALANCE - handled by risk module
- CLASSIFICATION is a separate event type, not a signal

## Signal Strength as Separate Layer

### Option A: Signal Filter Component

```python
class SignalStrengthFilter:
    """
    Separate component that enhances binary signals with strength scores.
    Maintains sparse storage while providing rich information to risk module.
    """
    
    def calculate_strength(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """
        Calculate signal strength based on multiple factors.
        Binary signal remains unchanged in storage.
        """
        if signal['signal_value'] == 0:
            return 0.0
            
        strength_factors = []
        
        # Factor 1: Indicator extremity
        if 'rsi' in features:
            rsi = features['rsi']
            if signal['signal_value'] == 1:  # Long
                strength_factors.append(max(0, (30 - rsi) / 30))
            else:  # Short
                strength_factors.append(max(0, (rsi - 70) / 30))
        
        # Factor 2: Trend alignment
        if 'sma_50' in features and 'sma_200' in features:
            trend_aligned = (features['sma_50'] > features['sma_200']) == (signal['signal_value'] == 1)
            strength_factors.append(1.0 if trend_aligned else 0.5)
        
        # Factor 3: Volume confirmation
        if 'volume_ratio' in features:
            strength_factors.append(min(1.0, features['volume_ratio'] / 2))
        
        # Combine factors
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.5
```

### Option B: Risk Module Integration

```python
class RiskModule:
    """
    Risk module calculates position sizes based on binary signals
    and additional market context.
    """
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio_state: Dict[str, Any],
                              market_context: Dict[str, Any]) -> float:
        """
        Determine position size from binary signal.
        """
        if signal['signal_value'] == 0:
            return 0.0
            
        base_size = self.base_position_size
        
        # Adjust based on portfolio risk
        risk_multiplier = self.get_risk_multiplier(portfolio_state)
        
        # Adjust based on market volatility
        vol_multiplier = self.get_volatility_adjustment(market_context)
        
        # Adjust based on signal metadata (if strategy provides hints)
        confidence_multiplier = self.get_confidence_multiplier(signal.get('metadata', {}))
        
        return base_size * risk_multiplier * vol_multiplier * confidence_multiplier
```

### Option C: Signal Enrichment Pipeline

```python
class SignalEnrichmentPipeline:
    """
    Pipeline that enriches binary signals without modifying stored values.
    """
    
    def __init__(self):
        self.enrichers = [
            IndicatorStrengthEnricher(),
            MarketRegimeEnricher(),
            VolatilityEnricher(),
            CorrelationEnricher()
        ]
    
    def enrich_signal(self, signal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add strength and context to binary signal without changing core value.
        """
        enriched = signal.copy()
        enriched['enrichment'] = {}
        
        for enricher in self.enrichers:
            enrichment = enricher.enrich(signal, context)
            enriched['enrichment'].update(enrichment)
        
        # Calculate composite strength
        enriched['composite_strength'] = self._calculate_composite_strength(enriched['enrichment'])
        
        return enriched
```

## Benefits of Separate Strength Layer

### 1. **Maintains Sparse Storage**
- Signals remain binary (-1, 0, 1)
- Storage only when signal changes
- ~95% storage savings maintained

### 2. **Maximum Flexibility**
- Strength calculation can be changed without regenerating signals
- Different strength algorithms for different market conditions
- A/B testing of strength calculations

### 3. **Clean Separation of Concerns**
- Strategies: Detect conditions (binary decision)
- Strength Layer: Assess conviction (graduated assessment)
- Risk Module: Size positions (portfolio-aware decision)

### 4. **Better for Ensembles**
- Binary signals easy to combine (majority vote, etc.)
- Strength calculated on ensemble output
- No complex strength aggregation logic

## Implementation Examples

### Binary MA Crossover
```python
@strategy(
    name='ma_crossover_binary',
    feature_config={
        'sma': {
            'params': ['fast_period', 'slow_period'],
            'defaults': {'fast_period': 10, 'slow_period': 20}
        }
    }
)
def ma_crossover_binary(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple MA crossover with sustained binary signals.
    """
    fast_ma = features.get(f'sma_{params.get("fast_period", 10)}')
    slow_ma = features.get(f'sma_{params.get("slow_period", 20)}')
    
    if fast_ma is None or slow_ma is None:
        return None
    
    # Determine current signal state
    if fast_ma > slow_ma:
        signal_value = 1
    elif fast_ma < slow_ma:
        signal_value = -1
    else:
        signal_value = 0
    
    # Check if signal changed
    prev_signal = features.get('prev_signal_ma', 0)
    if signal_value != prev_signal:
        return {
            'symbol_timeframe': f"{bar['symbol']}_{bar.get('timeframe', '5m')}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'ma_crossover_binary',
            'metadata': {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'separation_pct': abs(fast_ma - slow_ma) / slow_ma * 100  # For strength layer
            }
        }
    
    return None
```

### Signal Strength Filter Usage
```python
# In the coordinator or risk module
signal_filter = SignalStrengthFilter()

# When processing signals
for event in signal_events:
    signal = event.payload
    
    # Calculate strength without modifying stored signal
    strength = signal_filter.calculate_strength(signal, current_features)
    
    # Risk module uses both signal and strength
    position_size = risk_module.calculate_position(
        signal_value=signal['signal_value'],
        signal_strength=strength,
        portfolio_state=portfolio_state
    )
```

## Storage Comparison

### With Binary Signals + Strength Layer
```json
// Stored signals (sparse)
{"t": 1705329600, "st": "AAPL_5m", "v": 1, "id": "ma"}
{"t": 1705330800, "st": "AAPL_5m", "v": -1, "id": "ma"}
{"t": 1705331400, "st": "AAPL_5m", "v": 0, "id": "ma"}

// Strength calculated on-demand, not stored
```

### With Embedded Strength (Not Recommended)
```json
// Would need to store every bar
{"t": 1705329600, "st": "AAPL_5m", "v": 1, "s": 0.73, "id": "ma"}
{"t": 1705329660, "st": "AAPL_5m", "v": 1, "s": 0.71, "id": "ma"}
{"t": 1705329720, "st": "AAPL_5m", "v": 1, "s": 0.69, "id": "ma"}
// ... every bar ...
```

## Recommendation

Use **Option A: Signal Filter Component** because:

1. **Modularity**: Clean separation between signal detection and strength assessment
2. **Flexibility**: Can swap strength algorithms without touching strategies
3. **Performance**: Strength only calculated when needed (not on every bar)
4. **Testing**: Can unit test signal detection and strength calculation separately
5. **Evolution**: Can start simple and add sophisticated strength models later

The binary signal approach with separate strength layer provides the best balance of simplicity, efficiency, and flexibility.