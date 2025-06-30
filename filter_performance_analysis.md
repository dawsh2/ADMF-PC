# Filter Performance Analysis: Why Filters Aren't Improving Returns

## Key Findings

### 1. The Performance Paradox
- **Baseline (no filter)**: 0.41 bps/trade with 6.7 trades/day
- **Filtered strategies (>50% reduction)**: 0.33 bps/trade with 2.1 trades/day
- **Filters are actually REDUCING performance by -0.08 bps**

### 2. Best Performing Strategy Meeting Requirements
- **Signal count**: 2,826 (only 18.8% reduction)
- **Filter type**: Volatility filter (minimal filtering)
- **Performance**: 0.68 bps/trade with 5.7 trades/day
- **Annual return**: 9.7%

### 3. Why Heavy Filtering Fails

#### Problem 1: Removing Good Trades
- Heavy filters (>90% reduction) drop to 0.5 trades/day
- The few remaining trades aren't significantly better quality
- We're filtering out profitable opportunities along with losers

#### Problem 2: Wrong Filter Logic
Looking at the correlation (0.159) between filter reduction and returns, there's almost no relationship. This suggests:
- Filters aren't targeting the right market conditions
- Current filters may be arbitrary rather than edge-based

#### Problem 3: Overfitting to Specific Conditions
The "Master Regime" filter (98.6% reduction) shows 4.09 bps/trade but only 23 trades total:
- This is likely overfitted to very specific market conditions
- Not practical for real trading (0.1 trades/day)

### 4. What's Actually Working

The best performing strategies that meet your requirements have:
- **Minimal filtering** (< 20% signal reduction)
- **High frequency** (5-7 trades/day)
- **Consistent small edge** (0.45-0.68 bps/trade)

### 5. Root Cause Analysis

The fundamental issue is that **Keltner Bands at 5-minute frequency may not have enough edge** to support heavy filtering:
- Base strategy only produces 0.41 bps/trade
- With execution costs, this is barely profitable
- Heavy filtering can't create edge that isn't there

## Recommendations

### 1. Accept Current Performance
The volatility-filtered strategy (2,826 signals) is your best option:
- 0.68 bps/trade
- 5.7 trades/day
- 9.7% annual return

### 2. Alternative Approaches
Instead of more filtering, consider:
- **Different timeframes**: Test 15-minute or 1-hour Keltner
- **Different indicators**: The base Keltner strategy may be too weak
- **Position sizing**: Use Kelly criterion or volatility-based sizing
- **Ensemble approach**: Combine multiple weak signals

### 3. Execution Optimization
With only 0.68 bps/trade edge:
- Minimize slippage through limit orders
- Negotiate lower commission rates
- Consider maker-taker rebates

### 4. Risk Management
- Implement stop losses (test 0.1-0.3% stops)
- Use position limits to avoid overexposure
- Monitor real-time performance closely

## Conclusion

The filters aren't improving performance because the base Keltner strategy has minimal edge to begin with. You're better off with light filtering that maintains trade frequency rather than heavy filtering that reduces already-thin margins.