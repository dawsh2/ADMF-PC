# Top Strategies with 2-3 Trades Per Day

## Summary of Findings

Only **ONE strategy group** meets both criteria (2-3 trades/day + positive returns):

### 1500-Signal Group (Directional Filter)
- **Performance**: 0.41 bps/trade
- **Frequency**: 2.98 trades/day
- **Annual Return**: 3.1%
- **Win Rate**: 63.2%
- **Filter Type**: Directional filter (long/short specific)
- **Key Feature**: Strong long bias (6:1 ratio over shorts)

## Other Groups in 2-3 Trade Range

### 1202-Signal Group (Volume Filter)
- **Performance**: 0.22 bps/trade
- **Frequency**: 2.38 trades/day
- **Annual Return**: 1.3%
- **Win Rate**: 69.6%
- **Issue**: Returns too low to be practical

### 1535-Signal Group (Light Volatility)
- **Performance**: 0.19 bps/trade
- **Frequency**: 3.05 trades/day
- **Annual Return**: 1.4%
- **Win Rate**: 66.7%
- **Issue**: Marginal profitability

## Reality Check

The harsh truth is that forcing a 2-3 trades/day constraint severely limits performance:
- Only 1 viable strategy group found
- Best return is only 0.41 bps/trade
- This translates to just 3.1% annual return before costs

## Recommendations

1. **Relax the frequency constraint**: Allow 4-6 trades/day to access better strategies:
   - 2826 signals: 0.68 bps/trade at 5.7 trades/day (9.7% annual)
   - 2305 signals: 0.42 bps/trade at 4.7 trades/day (4.9% annual)

2. **If you must have 2-3 trades/day**:
   - Use the 1500-signal directional filter strategy
   - Implement as long-only given the 6:1 bias
   - Add 10-20 bps stop loss
   - Expect modest 2-3% annual returns after costs

3. **Consider alternative approaches**:
   - Different timeframes (15-min, 30-min bars)
   - Different indicators (not Keltner-based)
   - Ensemble of multiple strategies

The data strongly suggests that Keltner Bands on 5-minute bars don't have sufficient edge to support both high returns AND your specific frequency requirement.