# Market Regime Analysis Summary

## Dataset Overview
- **Workspace**: `workspaces/complete_strategy_grid_v1_fc4cc700`
- **Total bars**: 81,787
- **Total signals**: 100,388,223
- **Total classifications**: 98,616
- **Strategy files**: 1,229 strategies
- **Classifier files**: 81 classifiers

## Classifier Balance Analysis

### Key Findings
1. **All classifiers show some degree of imbalance** - none met the strict criteria for "excellent" balance
2. **Volatility momentum classifiers are the most balanced** among all available options
3. **Many classifiers have extreme imbalance** with some states occurring <1% of the time

### Most Balanced Classifiers Selected
The following `volatility_momentum_grid` classifiers showed the best balance (lowest balance scores):

1. `SPY_volatility_momentum_grid_12_75_30` - Balance Score: 33.33
2. `SPY_volatility_momentum_grid_16_75_30` - Balance Score: 33.33  
3. `SPY_volatility_momentum_grid_20_75_30` - Balance Score: 33.33
4. `SPY_volatility_momentum_grid_16_80_20` - Balance Score: 33.33
5. `SPY_volatility_momentum_grid_20_75_25` - Balance Score: 33.33
6. `SPY_volatility_momentum_grid_12_80_20` - Balance Score: 33.33

### State Distribution for Primary Classifier
Using `SPY_volatility_momentum_grid_12_75_30`:
- **Neutral**: 50.0% (384 bars)
- **Low Vol Bearish**: 31.1% (239 bars)  
- **Low Vol Bullish**: 18.9% (145 bars)

## Strategy Performance by Regime

### Analysis Results (10 MACD strategies tested)

#### Neutral Regime (50.0% of time)
- **Average Performance**: -14.73% return
- **Average Win Rate**: 31.88%
- **Average Trades**: 6,926 per strategy
- **Best Strategy**: `SPY_macd_crossover_grid_5_20_9` (-1.14% return)

#### Low Vol Bearish Regime (31.1% of time)  
- **Average Performance**: -4.74% return
- **Average Win Rate**: 39.28%
- **Average Trades**: 69 per strategy
- **Best Strategy**: `SPY_macd_crossover_grid_12_20_7` (+10.88% return)

#### Low Vol Bullish Regime (18.9% of time)
- **Average Performance**: -10.26% return  
- **Average Win Rate**: 41.47%
- **Average Trades**: 36 per strategy
- **Best Strategy**: `SPY_macd_crossover_grid_12_26_11` (-1.47% return)

## Key Insights

### 1. Regime-Specific Performance Patterns
- **Bearish low volatility periods show the best absolute performance** (+10.88% best case)
- **Bullish low volatility periods generally underperform** but with higher win rates
- **Neutral periods dominate trading activity** (>6,900 trades vs <100 in other regimes)

### 2. Win Rate Observations
- **Higher win rates in low volatility regimes** (39-41%) vs neutral (32%)
- **Regime-aware strategies could benefit from dynamic allocation**

### 3. Trade Frequency Insights
- **Neutral regime generates 99%+ of all trades** due to its dominance (50% of time)
- **Low volatility regimes have much lower signal frequency**
- **May indicate need for regime-specific strategy parameters**

## Recommendations

### 1. Focus on Balanced Classifiers
Use the selected `volatility_momentum_grid` classifiers for regime analysis as they provide the most balanced state distributions available.

### 2. Regime-Aware Strategy Development
- Develop strategies optimized for **low vol bearish** periods (best performance)
- Consider different parameters for **neutral** vs **low volatility** regimes
- Account for the **trade frequency imbalance** across regimes

### 3. Portfolio Construction
- **Overweight strategies that perform well in bearish low-vol** periods
- **Consider regime-switching models** given the performance differences
- **Account for regime persistence** when position sizing

### 4. Further Analysis Priorities
1. **Expand strategy universe** beyond MACD strategies
2. **Test more balanced classifiers** from the selected list
3. **Analyze regime transition patterns** and persistence
4. **Develop regime-optimized strategy parameters**

## Files Generated
- `classifier_recommendations.json` - Classifier balance analysis
- `regime_performance_analysis_SPY_volatility_momentum_grid_12_75_30.json` - Detailed results
- `analyze_classifier_distributions.py` - Classifier analysis script
- `analyze_strategy_performance.py` - Performance analysis script

## Usage for Future Analysis
The analysis framework is now in place to:
1. Evaluate any subset of the 1,229 available strategies
2. Test performance across different regime classifiers
3. Optimize strategy parameters for specific market regimes
4. Build regime-aware portfolio allocation models