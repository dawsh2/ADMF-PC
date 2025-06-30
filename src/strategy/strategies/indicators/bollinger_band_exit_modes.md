# Bollinger Band Exit Modes Explained

## Current Implementation

The strategy currently supports three exit modes:

### 1. Middle Exit (exit_mode='middle')
- **Long Entry**: Price <= lower_band + buffer
- **Long Exit**: Price >= middle_band
- **Short Entry**: Price >= upper_band - buffer  
- **Short Exit**: Price <= middle_band

This is a "mean reversion to middle" approach.

### 2. Opposite Exit (exit_mode='opposite')
- **Long Entry**: Price <= lower_band + buffer
- **Long Exit**: Price >= upper_band - buffer
- **Short Entry**: Price >= upper_band - buffer
- **Short Exit**: Price <= lower_band + buffer

This captures the full band width as profit.

### 3. Percent Exit (exit_mode='percent')
- **Long Entry**: Price <= lower_band + buffer
- **Long Exit**: Price >= lower_band + (band_width * exit_percent)
- **Short Entry**: Price >= upper_band - buffer
- **Short Exit**: Price <= upper_band - (band_width * exit_percent)

This allows partial profit targets.

## Missing: Classic "Re-enter Band" Exit

Traditional Bollinger Band strategy often exits when price comes back inside the bands:

```python
# Classic Bollinger exit logic (not currently implemented)
if in_long_position:
    if price > lower_band + small_buffer:
        exit_long()  # Price came back inside band
        
if in_short_position:
    if price < upper_band - small_buffer:
        exit_short()  # Price came back inside band
```

## Proposed Addition

Add a fourth exit mode: `exit_mode='band_reentry'`

```python
elif exit_mode == 'band_reentry':
    # Exit when price re-enters the band
    exit_long = lower_band + (band_width * 0.05)  # 5% inside band
    exit_short = upper_band - (band_width * 0.05)  # 5% inside band
```

This would give you:
- Entry: Price breaks outside band
- Exit: Price comes back inside band
- More reactive to price reversals
- Shorter holding periods
- Better for choppy markets