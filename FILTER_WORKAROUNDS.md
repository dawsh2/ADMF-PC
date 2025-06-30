# Filter Workarounds

## For atr_sma_50 Error

### Option 1: Manual Feature Definition
Add this to your config file:
```yaml
feature_configs:
  atr_14:
    type: atr
    period: 14
  # Define atr_sma_50 manually since auto-discovery doesn't support composite features yet
  atr_sma_50:
    type: sma
    period: 50
    # Note: You may need to compute this as SMA of ATR values in your implementation
```

### Option 2: Use Different Filter Logic
Instead of:
```yaml
filter: "atr(14) > atr_sma(50) * 1.2"
```

Use a simpler volatility check:
```yaml
filter: "atr(14) > ${atr_threshold}"
filter_params:
  atr_threshold: 0.5  # Adjust based on your needs
```

## For bar_of_day Error

### Option 1: Remove Time-Based Filters (Quick Fix)
If you don't need time filtering, remove bar_of_day from filters:
```yaml
# Instead of:
filter: "signal == 0 or (volume > volume_sma_20 * 1.2 and bar_of_day > 10)"

# Use:
filter: "signal == 0 or volume > volume_sma_20 * 1.2"
```

### Option 2: Ensure Timestamp in Data
Make sure your data loader includes timestamps. Check your data files or data loader configuration.

### Option 3: Use a Different Time Filter
Instead of bar_of_day, you could use other available features:
```yaml
# If you have hour-of-day or other time features available
filter: "signal == 0 or hour >= 10"  # If hour is available
```

## Immediate Workaround to Continue Testing

Add this minimal feature config to your YAML file:
```yaml
# Add this section to your config
feature_configs:
  # Volume SMA should work automatically now
  # But add ATR features manually
  atr_14:
    type: atr
    period: 14
  # Skip atr_sma_50 for now if it's causing issues
```

And simplify your filters to avoid problematic features:
```yaml
# Simplified filter without atr_sma or bar_of_day
filter: "signal == 0 or (volume > volume_sma_20 * 1.2 and atr(14) > 0.5)"
```

This will let you continue testing while we work on proper fixes for composite features.