# Ensemble Strategy Analysis: Last 22,000 Bars (~56 Trading Days)

## Executive Summary

The analysis covers the period from bar index 80,236 to 102,235, during which SPY declined 6.65% with 16% annualized volatility and a maximum drawdown of -10.79%.

### Key Performance Metrics

| Metric | Default Ensemble | Custom Ensemble | Market (SPY) |
|--------|-----------------|-----------------|--------------|
| **Gross Return** | +7.11% | +0.77% | -6.65% |
| **Net Return (0.01% costs)** | -32.71% | -53.70% | -6.65% |
| **Sharpe Ratio** | 1.99 | 0.29 | N/A |
| **Max Drawdown** | -5.29% | -10.31% | -10.79% |
| **Win Rate** | 46.8% | 40.3% | N/A |
| **Total Trades** | 2,324 | 3,888 | N/A |
| **Avg Trade Duration** | 9.5 bars | 5.7 bars | N/A |
| **Outperformance (gross)** | +13.75% | +7.42% | 0% |

## Critical Insights

### 1. Transaction Cost Impact
- **Default Ensemble**: 39.82% cost impact turns +7.11% gross into -32.71% net
- **Custom Ensemble**: 54.47% cost impact turns +0.77% gross into -53.70% net
- **Breakeven costs**: Default needs <0.0015% per trade, Custom needs <0.0001% per trade

### 2. Trading Frequency Analysis
- Default ensemble trades every ~9.5 bars (2,324 trades over 22k bars)
- Custom ensemble trades every ~5.7 bars (3,888 trades over 22k bars)
- High frequency is the primary driver of negative net returns

### 3. Regime Distribution
- **Neutral**: 92.7% of time (20,399 bars)
- **Low Vol Bear**: 4.5% of time (989 bars)
- **Low Vol Bull**: 2.8% of time (612 bars)
- **High Vol regimes**: Not observed in this period

### 4. Performance by Regime

#### Default Ensemble
- **Neutral**: 1,990 trades, +7.42% return, 48.0% win rate
- **Low Vol Bear**: 283 trades, -0.58% return, 41.0% win rate
- **Low Vol Bull**: 51 trades, +0.29% return, 33.3% win rate

#### Custom Ensemble
- **Neutral**: 3,245 trades, +0.59% return, 42.3% win rate
- **Low Vol Bear**: 566 trades, -1.38% return, 29.5% win rate
- **Low Vol Bull**: 77 trades, +1.58% return, 35.1% win rate

### 5. Quarterly Performance Consistency

#### Default Ensemble
- Q1: -2.21% (581 trades)
- Q2: +4.68% (581 trades)
- Q3: +3.14% (581 trades)
- Q4: +1.44% (581 trades)

#### Custom Ensemble
- Q1: -0.20% (972 trades)
- Q2: +2.37% (972 trades)
- Q3: -0.66% (972 trades)
- Q4: -0.70% (972 trades)

## Market Context Analysis

### Regime Transitions
- Total regime changes: 3,199 (very frequent switching)
- Most common transition: neutral ↔ low_vol_bearish (1,974 transitions)
- Average regime duration: 5-10 bars (very short-lived)

### Signal Clustering
- Both ensembles show high signal density (~1 signal per bar in active periods)
- Signals slightly decrease after regime changes (-5.3% for default, -2.7% for custom)
- No significant clustering around regime transitions

## Comparison to 12k Bar Analysis

The 22k bar analysis provides:
1. **Better statistical significance** with more trades
2. **More diverse market conditions** (though still mostly neutral regime)
3. **Clearer picture of transaction cost impact** over longer period
4. **Evidence of performance consistency** across quarters

## Recommendations

### 1. Reduce Trading Frequency
- Target average trade duration of 50+ bars to reduce cost impact
- Consider filtering signals to only high-confidence opportunities

### 2. Optimize for Regime-Specific Performance
- The neutral regime dominates (92.7%) but strategies may not be optimal for it
- Consider developing neutral-specific strategies with lower turnover

### 3. Cost Structure Alternatives
- Current strategies require <0.1 basis points per trade to be profitable
- Consider maker-taker rebates or institutional execution

### 4. Strategy Selection
- Default ensemble shows better risk-adjusted returns (Sharpe 1.99 vs 0.29)
- Focus on improving default ensemble's cost efficiency

### 5. Regime Classification
- Very frequent regime changes (every 5-10 bars) may be noise
- Consider smoothing regime classification to reduce whipsaws

## Conclusion

Both ensemble strategies demonstrate the ability to generate positive gross returns in a declining market, with the default ensemble significantly outperforming. However, transaction costs completely erode profitability due to high trading frequency. The key to success lies in dramatically reducing turnover while maintaining signal quality.