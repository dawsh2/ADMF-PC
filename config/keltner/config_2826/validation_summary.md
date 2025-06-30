# 2826 Strategy Validation Summary

## Initial Run Result
- **Expected**: 2826 signals
- **Got**: 1535 signals
- **Issue**: Wrong parameter combination

## Parameter Discovery
Strategy 3 (which produces 2826 signals) uses:
- **Period**: 30
- **Multiplier**: 1.0  
- **Filter**: volatility_above with threshold 1.1

## Updated Config
The config has been updated with the correct parameters:
```yaml
period: [30]
multiplier: [1.0]
filter: {volatility_above: {threshold: 1.1}}
```

## Next Steps
1. Re-run with the updated config
2. Should now get exactly 2826 signals
3. Performance should match:
   - 0.68 bps/trade gross
   - 73.7% win rate
   - 5.7 trades/day
   - 9.7% annual gross

## Note
The 1535-signal strategy you got initially is also in our analysis (ranked #12):
- 0.19 bps/trade gross
- 66.7% win rate
- 3.1 trades/day
- Not as good as the 2826 strategy