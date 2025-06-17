# Ensemble Selection Guide

## Core Principles for Optimal Ensemble Selection

### 1. Performance Metrics (40% weight)
- **Sharpe Ratio**: Primary metric (minimum 0.5, prefer > 1.0)
- **Consistency Score**: Sharpe × √(win_rate) - balances returns with reliability
- **Regime Stability**: How consistently the strategy performs in this specific regime

### 2. Diversification (30% weight)
- **Strategy Type Diversity**: Maximum 2 strategies per type
- **Signal Correlation**: Avoid strategies with similar entry/exit patterns
- **Market Behavior**: Mix trend-following, mean-reversion, and momentum strategies

### 3. Risk Management (20% weight)
- **Volatility Control**: Prefer lower volatility strategies for stability
- **Downside Protection**: Consider Sortino ratio and maximum drawdown
- **Win Rate**: Minimum 40% win rate for psychological comfort

### 4. Practical Constraints (10% weight)
- **Trading Frequency**: Balance signal generation to avoid overtrading
- **Implementation Complexity**: Simpler strategies are more robust
- **Market Impact**: Consider position sizing and liquidity

## Selection Process

### Step 1: Initial Filtering
```sql
-- Minimum requirements
WHERE sharpe_ratio >= 0.5
  AND trading_days >= 20
  AND win_rate >= 0.40
```

### Step 2: Scoring Formula
```
Selection Score = 
    0.30 × (Sharpe_Ratio / 5.0) +
    0.20 × (Consistency_Score / 3.0) +
    0.20 × (Regime_Stability / 10.0) +
    0.15 × Win_Rate +
    0.15 × (1 - Volatility / 0.10)
```

### Step 3: Diversification Rules
1. Maximum 2 strategies per strategy type
2. At least 3 different strategy types in ensemble
3. Mix of holding periods (short/medium term)

### Step 4: Regime-Specific Adjustments

#### Low Volatility Bullish
- Emphasize trend-following (DEMA, MACD crossovers)
- Include momentum strategies
- Allow higher risk for higher returns

#### Low Volatility Bearish  
- Focus on defensive strategies
- Emphasize short signals and hedging
- Include volatility-based strategies

#### Neutral
- Balance long/short capabilities
- Emphasize mean-reversion strategies
- Include market-neutral approaches

## Recommended Ensemble Sizes

- **Conservative**: 5-7 strategies
- **Balanced**: 8-12 strategies  
- **Aggressive**: 13-15 strategies

## Post-Selection Validation

1. **Correlation Analysis**: Ensure strategies are not too correlated
2. **Stress Testing**: Test ensemble during regime transitions
3. **Transaction Cost Analysis**: Ensure profitability after costs
4. **Rebalancing Frequency**: Monthly or quarterly review

## Example Selection Query

```sql
WITH scored_strategies AS (
    SELECT 
        strategy_name,
        strategy_type,
        current_regime,
        sharpe_ratio,
        -- Composite score
        (0.30 * (sharpe_ratio / 5.0) + 
         0.20 * (sharpe_ratio * SQRT(win_rate)) +
         0.15 * win_rate +
         0.15 * (1 - daily_volatility / 0.10)
        ) as selection_score,
        ROW_NUMBER() OVER (
            PARTITION BY current_regime, strategy_type 
            ORDER BY sharpe_ratio DESC
        ) as type_rank
    FROM strategy_results
    WHERE sharpe_ratio >= 0.5
)
SELECT *
FROM scored_strategies
WHERE type_rank <= 2  -- Max 2 per type
ORDER BY current_regime, selection_score DESC
LIMIT 10;  -- Top 10 per regime
```

## Next Steps After Selection

1. **Weight Optimization**: Use mean-variance optimization or risk parity
2. **Walk-Forward Testing**: Validate ensemble out-of-sample
3. **Position Sizing**: Implement Kelly criterion or fixed fractional
4. **Risk Limits**: Set maximum drawdown and position limits
5. **Monitoring**: Track regime changes and strategy degradation