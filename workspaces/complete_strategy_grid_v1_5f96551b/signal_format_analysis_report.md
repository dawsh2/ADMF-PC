# Signal Value Format Analysis Report

## Executive Summary

**No leading zero issues found** - Signal values are stored correctly and trade sequencing logic should work as expected.

## Signal Value Format Analysis

### 1. Strategy Signals (Numerical)
- **Data Type**: BIGINT (integer)
- **Values**: Clean integers (-1, 1) with no leading zeros
- **No "0" (neutral) signals** - strategies use direct transitions between -1 and 1

**Examples from MACD Crossover:**
```
Signal Value: -1 (3,270 occurrences)
Signal Value: 1  (3,270 occurrences)
```

**Examples from EMA Crossover:**
```
Signal Value: -1 (1,860 occurrences) 
Signal Value: 1  (1,859 occurrences)
```

### 2. Classifier Signals (Categorical)
- **Data Type**: VARCHAR (string)
- **Values**: Descriptive text categories

**Example from Volatility Momentum Classifier:**
```
- "high_vol_bearish" (1 occurrence)
- "low_vol_bearish"  (4,682 occurrences)
- "low_vol_bullish"  (4,837 occurrences)
- "neutral"          (4,748 occurrences)
```

## Trade Sequencing Impact

### Current Signal Transitions
Since strategies use **only -1 and 1** (no "0"), the trade patterns are:

```sql
CASE 
    WHEN prev_signal = -1 AND signal_value = 1 THEN 'SHORT_TO_LONG'
    WHEN prev_signal = 1 AND signal_value = -1 THEN 'LONG_TO_SHORT'
    ELSE 'NO_CHANGE'
END
```

### Sample Transition Data
```
2024-03-26 14:10:00  -1    1     LONG_TO_SHORT
2024-03-26 14:17:00   1   -1     SHORT_TO_LONG  
2024-03-26 14:22:00  -1    1     LONG_TO_SHORT
2024-03-26 14:24:00   1   -1     SHORT_TO_LONG
```

## Recommendations

### 1. Trade Construction Logic
**Use integer comparison directly** - no need for string handling:

```sql
-- Correct approach for these signals
CASE 
    WHEN LAG(val) OVER (ORDER BY ts) = -1 AND val = 1 THEN 'ENTRY_LONG'
    WHEN LAG(val) OVER (ORDER BY ts) = 1 AND val = -1 THEN 'ENTRY_SHORT'
    -- Exit logic: reverse the position on opposite signal
    WHEN LAG(val) OVER (ORDER BY ts) = 1 AND val = -1 THEN 'EXIT_LONG'  
    WHEN LAG(val) OVER (ORDER BY ts) = -1 AND val = 1 THEN 'EXIT_SHORT'
END
```

### 2. No Leading Zero Concerns
- ✅ Values stored as clean integers (-1, 1)
- ✅ No string formatting issues
- ✅ Direct integer comparison is reliable
- ✅ No "01" vs "1" problems

### 3. Classifier Handling
For classifiers with text values, different logic needed:
```sql
-- For classifier signals (text-based)
CASE 
    WHEN prev_val != 'neutral' AND val = 'neutral' THEN 'EXIT'
    WHEN prev_val = 'neutral' AND val IN ('low_vol_bullish', 'high_vol_bullish') THEN 'ENTRY_LONG'
    WHEN prev_val = 'neutral' AND val IN ('low_vol_bearish', 'high_vol_bearish') THEN 'ENTRY_SHORT'
END
```

## Conclusion

**No signal format issues detected.** The integer-based strategy signals are clean and should work correctly with existing trade sequencing logic. The main consideration is that these strategies don't use neutral (0) signals - they transition directly between long (1) and short (-1) positions.