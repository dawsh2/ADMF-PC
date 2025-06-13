# Trading Rules Implementation Plan

## Proposed File Organization

Based on the 16 trading rules and your existing architecture, I propose organizing the strategies into these files:

### 1. **crossovers.py** - Moving Average and Crossover Rules
- Rule 1: Simple MA Crossover (MA vs MA)
- Rule 2: EMA vs MA Crossover
- Rule 3: EMA vs EMA Crossover
- Rule 4: DEMA vs MA Crossover
- Rule 5: DEMA vs DEMA Crossover
- Rule 6: TEMA vs MA Crossover
- Rule 7: Stochastic Crossover
- Rule 8: Vortex Indicator Crossover
- Rule 9: Ichimoku Cloud Crossover

### 2. **oscillators.py** - Oscillator-based Rules
- Rule 10: RSI Threshold (single threshold)
- Rule 11: CCI Threshold (single threshold)
- Rule 12: RSI Band Strategy (overbought/oversold bands)
- Rule 13: CCI Band Strategy (high/low bands)

### 3. **channels.py** - Channel and Band-based Rules
- Rule 14: Keltner Channel Strategy
- Rule 15: Donchian Channel Strategy
- Rule 16: Bollinger Band Strategy

## Implementation Pattern

Each rule will follow the event-driven pattern:

```python
@strategy(
    name='rule1_ma_crossover',
    feature_config={
        'sma': {
            'params': ['fast_period', 'slow_period'],
            'defaults': {'fast_period': 10, 'slow_period': 20}
        }
    }
)
def rule1_ma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rule 1: Simple Moving Average Crossover"""
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Get features
    fast_ma = features.get(f'sma_{fast_period}')
    slow_ma = features.get(f'sma_{slow_period}')
    price = bar.get('close', 0)
    
    # Check conditions
    if fast_ma is None or slow_ma is None:
        return None
    
    # Generate signal
    if fast_ma > slow_ma:
        return {
            'symbol': bar.get('symbol', 'UNKNOWN'),
            'direction': 'long',
            'signal_type': 'entry',
            'strength': min(1.0, (fast_ma - slow_ma) / slow_ma),
            'price': price,
            'reason': f'MA{fast_period} > MA{slow_period}',
            'indicators': {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'price': price
            }
        }
    elif fast_ma < slow_ma:
        return {
            'symbol': bar.get('symbol', 'UNKNOWN'),
            'direction': 'short',
            'signal_type': 'entry',
            'strength': min(1.0, (slow_ma - fast_ma) / fast_ma),
            'price': price,
            'reason': f'MA{fast_period} < MA{slow_period}',
            'indicators': {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'price': price
            }
        }
    
    return None
```

## Required Features

The following features need to be available in the FeatureHub:

### Basic Indicators
- SMA (Simple Moving Average) ✓
- EMA (Exponential Moving Average) ✓
- DEMA (Double Exponential Moving Average) - needs implementation
- TEMA (Triple Exponential Moving Average) - needs implementation
- RSI (Relative Strength Index) ✓
- CCI (Commodity Channel Index) - needs implementation
- Stochastic Oscillator - needs implementation
- Vortex Indicator - needs implementation
- Ichimoku Cloud - needs implementation
- Keltner Channel - needs implementation
- Donchian Channel - needs implementation
- Bollinger Bands ✓

## Grid Search Compatibility

Each rule will be optimizable through the existing grid search framework:

```yaml
# Example configuration for grid search
strategies:
  - name: rule1_ma_crossover
    parameters:
      fast_period: [5, 10, 15, 20]
      slow_period: [20, 30, 40, 50]
  
  - name: rule10_rsi_threshold
    parameters:
      rsi_period: [14, 21, 28]
      threshold: [20, 30, 40, 50, 60, 70, 80]
```

## Integration with Existing Architecture

1. All rules will be **stateless** pure functions
2. They'll use the `@strategy` decorator for automatic discovery
3. Features will be computed by the FeatureHub (stateful tier)
4. Rules will consume pre-computed features (stateless tier)
5. Each rule can be used standalone or combined in ensemble strategies