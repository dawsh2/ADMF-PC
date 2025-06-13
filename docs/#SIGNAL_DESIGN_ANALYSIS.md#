# Signal Design Analysis: Binary vs Graduated Signals

## Current Design Issues

### 1. Symbol vs Symbol_Timeframe
You're right - we should use `symbol_timeframe` instead of just `symbol`. This is crucial for:
- Multi-timeframe strategies
- Proper signal aggregation
- Avoiding conflicts between different timeframes

**Recommendation**: Update all strategies to use `symbol_timeframe` format (e.g., "AAPL_5m", "AAPL_1h")

### 2. Direction Values
Current implementation uses string values ('long', 'short', 'flat'), but the sparse storage expects numeric values (-1, 0, 1).

**Current Mapping**:
- `long` → 1
- `short` → -1  
- `flat` → 0

### 3. Signal Types
From the codebase:
- `ENTRY` - Opening a position
- `EXIT` - Closing a position
- `REBALANCE` - Adjusting position size
- `CLASSIFICATION` - Market regime classification

## Binary vs Graduated Signals: Analysis

### Option 1: Binary/Ternary Signals Only (-1, 0, 1)

**Advantages**:
1. **Sparse Storage Efficiency**: Signals only stored when they change
2. **Minimal Storage**: ~16 bytes per signal change
3. **Fast Retrieval**: Simple integer comparisons
4. **Clear Decisions**: No ambiguity in signal interpretation

**Disadvantages**:
1. **Loss of Information**: Can't express confidence/conviction
2. **Poor Position Sizing**: All trades are equal size
3. **No Gradual Scaling**: Can't scale in/out of positions

**Example Storage with Binary Signals**:
```json
{"t": 1705329600, "s": "AAPL_5m", "v": 1, "id": "rule1"}
{"t": 1705329900, "s": "AAPL_5m", "v": 0, "id": "rule1"}  
{"t": 1705330200, "s": "AAPL_5m", "v": -1, "id": "rule1"}
```
Storage: ~50 bytes per signal change

### Option 2: Graduated Signals with Strength (0.0 - 1.0)

**Advantages**:
1. **Rich Information**: Express confidence and conviction
2. **Dynamic Position Sizing**: Size trades based on signal strength
3. **Risk Management**: Reduce position size in uncertain conditions
4. **Ensemble Benefits**: Better signal aggregation from multiple strategies

**Disadvantages**:
1. **Storage Explosion**: Must store every bar (strength changes constantly)
2. **No Sparse Storage**: Loses 90%+ storage efficiency
3. **Complex Aggregation**: How to combine different strengths?

**Example Storage with Graduated Signals**:
```json
{"t": 1705329600, "s": "AAPL_5m", "v": 1, "str": 0.73, "id": "rule1"}
{"t": 1705329660, "s": "AAPL_5m", "v": 1, "str": 0.71, "id": "rule1"}
{"t": 1705329720, "s": "AAPL_5m", "v": 1, "str": 0.68, "id": "rule1"}
```
Storage: ~70 bytes per bar (not per change!)

## Hybrid Solution: Discrete Strength Levels

### Proposed Design: Quantized Strength Levels

Instead of continuous strength (0.0-1.0), use discrete levels that change infrequently:

**Signal Values**:
- Strong Long: 3
- Medium Long: 2
- Weak Long: 1
- Flat/Exit: 0
- Weak Short: -1
- Medium Short: -2
- Strong Short: -3

**Benefits**:
1. **Maintains Sparsity**: Strength only changes at meaningful thresholds
2. **Position Sizing**: Maps to 3 position sizes (33%, 66%, 100%)
3. **Clear Thresholds**: Reduces noise from minor strength fluctuations
4. **Efficient Storage**: Still uses single integer value

### Implementation Examples

#### Rule 10: RSI Threshold with Discrete Strength
```python
def rule10_rsi_threshold_discrete(features, bar, params):
    rsi = features.get('rsi')
    
    if rsi < 20:
        signal_value = 3  # Strong long
    elif rsi < 30:
        signal_value = 2  # Medium long
    elif rsi < 40:
        signal_value = 1  # Weak long
    elif rsi > 80:
        signal_value = -3  # Strong short
    elif rsi > 70:
        signal_value = -2  # Medium short
    elif rsi > 60:
        signal_value = -1  # Weak short
    else:
        signal_value = 0  # Flat
    
    # Only emit signal if value changed
    if signal_value != features.get('prev_signal_value', 0):
        return {
            'symbol_timeframe': f"{bar['symbol']}_{bar['timeframe']}",
            'signal_value': signal_value,
            'signal_type': 'ENTRY' if signal_value != 0 else 'EXIT',
            'reason': f'RSI={rsi:.1f}'
        }
```

#### Rule 1: MA Crossover with Discrete Strength
```python
def rule1_ma_crossover_discrete(features, bar, params):
    fast_ma = features.get('sma_10')
    slow_ma = features.get('sma_20')
    
    # Calculate separation percentage
    separation = abs(fast_ma - slow_ma) / slow_ma * 100
    
    if fast_ma > slow_ma:
        # Long signal with strength based on separation
        if separation > 2.0:
            signal_value = 3  # Strong trend
        elif separation > 1.0:
            signal_value = 2  # Medium trend
        else:
            signal_value = 1  # Weak trend
    elif fast_ma < slow_ma:
        # Short signal with strength based on separation
        if separation > 2.0:
            signal_value = -3  # Strong trend
        elif separation > 1.0:
            signal_value = -2  # Medium trend
        else:
            signal_value = -1  # Weak trend
    else:
        signal_value = 0
```

### Storage Comparison

**Scenario**: 1 day of 5-minute bars (78 bars) with 5 signal changes

1. **Binary Signals**: 5 records × 50 bytes = 250 bytes
2. **Continuous Strength**: 78 records × 70 bytes = 5,460 bytes
3. **Discrete Levels**: 5 records × 50 bytes = 250 bytes

**Storage Efficiency**: Discrete levels maintain same efficiency as binary!

## Recommendation

### Use Discrete Strength Levels (-3 to 3)

**Implementation Plan**:

1. **Update Signal Structure**:
```python
@dataclass
class Signal:
    symbol_timeframe: str  # e.g., "AAPL_5m"
    signal_value: int      # -3 to 3
    timestamp: datetime
    strategy_id: str
    signal_type: SignalType
    metadata: Optional[Dict[str, Any]] = None
```

2. **Position Sizing in Risk Module**:
```python
def calculate_position_size(signal_value: int, base_size: float) -> float:
    strength_map = {
        3: 1.0,    # 100% of base size
        2: 0.66,   # 66% of base size
        1: 0.33,   # 33% of base size
        0: 0.0,    # No position
        -1: 0.33,  # 33% short
        -2: 0.66,  # 66% short
        -3: 1.0    # 100% short
    }
    return base_size * strength_map.get(abs(signal_value), 0)
```

3. **Sparse Storage Format**:
```json
{"t": 1705329600, "st": "AAPL_5m", "v": 3, "id": "r1"}
{"t": 1705330200, "st": "AAPL_5m", "v": 1, "id": "r1"}
{"t": 1705330800, "st": "AAPL_5m", "v": -2, "id": "r1"}
```

### Benefits of This Approach

1. **Maintains Sparse Storage**: Only stores changes, not every bar
2. **Expresses Conviction**: 3 levels of strength in each direction
3. **Clear Position Sizing**: Maps directly to risk management
4. **Backward Compatible**: Can treat as binary by checking v != 0
5. **Efficient Aggregation**: Integer math for combining signals

### Example Rules Benefiting from Graduated Signals

1. **RSI Extremes**: Stronger signals at 10/90 than at 30/70
2. **Bollinger Band Distance**: Signal strength based on penetration depth
3. **MA Separation**: Stronger trends have larger MA gaps
4. **Volume Confirmation**: Higher volume = stronger signal
5. **Multi-Timeframe Agreement**: More timeframes = stronger signal

This design provides the best of both worlds: rich signal information while maintaining storage efficiency.