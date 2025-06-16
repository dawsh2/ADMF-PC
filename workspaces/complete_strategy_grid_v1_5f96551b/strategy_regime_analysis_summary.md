# Strategy Performance by Regime Analysis Summary

## Overview
Analyzed 1,235 strategies across 10 months of training data (March 2024 - January 2025) using the top Volatility Momentum classifier (SPY_volatility_momentum_grid_05_65_40).

## Key Findings

### 1. Regime Distribution
- **Low Vol Bullish**: 46.3% of time (36,984 minutes)
- **Low Vol Bearish**: 44.3% of time (35,317 minutes)
- **Neutral**: 9.4% of time (7,499 minutes)
- **High Vol Bearish**: <0.1% of time (4 minutes)

### 2. Strategy Performance by Regime

#### Sample Analysis Results (4 strategies analyzed in detail):

**MACD Crossover (12_26_9)**:
- Low Vol Bullish: 1,388 trades, -66.2% net return (after costs)
- Low Vol Bearish: 1,319 trades, -66.8% net return
- Neutral: 204 trades, -11.0% net return
- High Vol Bearish: 1 trade, +0.4% net return

**EMA Crossover (7_35)**:
- Low Vol Bullish: 563 trades, -31.7% net return
- Low Vol Bearish: 580 trades, -25.9% net return  
- Neutral: 65 trades, -3.6% net return
- High Vol Bearish: 1 trade, +0.8% net return

**RSI Threshold (11_40)**:
- Low Vol Bearish: 3,308 trades, -161.2% net return
- Low Vol Bullish: 122 trades, -7.4% net return
- Neutral: 156 trades, -6.3% net return

### 3. Regime Performance Summary

Average performance across analyzed strategies:
- **High Vol Bearish**: +0.52% net return (only 3 trades total)
- **Neutral**: -6.98% net return (425 trades)
- **Low Vol Bullish**: -35.08% net return (2,073 trades)
- **Low Vol Bearish**: -84.60% net return (5,207 trades)

### 4. Critical Observations

1. **Transaction Costs Dominate**: All strategies show severe negative returns after 0.05% round-trip costs
2. **High Frequency Trading Issues**: Strategies average 12-25 minutes per trade, making transaction costs prohibitive
3. **Regime-Dependent Performance**: Strategies perform differently in each regime, but all are unprofitable after costs
4. **Sparse High Volatility Data**: Only 4 minutes of high volatility bearish regime in 10 months

## Recommendations

1. **Fix Sharpe Ratio Calculations**: Current calculations use per-trade Sharpe, need wall-clock time normalization
2. **Filter for Lower Frequency**: Only consider strategies with average trade duration > 60 minutes
3. **Reduce Transaction Costs**: Model more realistic costs (e.g., 0.01% for liquid ETFs)
4. **Expand Regime Analysis**: Need classifiers with better high volatility regime coverage
5. **Ensemble Construction**: Despite poor individual performance, regime-specific ensembles may still add value

## Next Steps

1. Re-run analysis with corrected Sharpe ratio calculations
2. Apply quality filters (min trade duration, max trades per day)
3. Analyze all 1,235 strategies in batches
4. Build regime-specific ensemble allocation matrices
5. Perform walk-forward validation on selected ensembles

## Technical Notes

- Data timezone alignment: Applied -4 hour correction to match EST trading hours
- Sparse signal storage: Used forward-fill for regime timeline construction
- Processing approach: Individual strategy analysis to avoid memory constraints
- Export location: strategy_regime_results.csv contains detailed results