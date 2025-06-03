# Configuring Strategies

This guide shows you how to configure different types of trading strategies in ADMF-PC. All strategies are configured through YAML - no programming required!

## 🎯 Basic Strategy Configuration

### Simple Momentum Strategy

```yaml
strategies:
  - type: "momentum"
    name: "MA_Crossover"
    params:
      fast_period: 10
      slow_period: 20
      signal_threshold: 0.01  # 1% price movement threshold
    allocation: 1.0  # 100% of portfolio
```

**How it works**:
- Generates BUY signal when fast MA crosses above slow MA
- Generates SELL signal when fast MA crosses below slow MA
- `signal_threshold` filters out weak signals

### Mean Reversion Strategy

```yaml
strategies:
  - type: "mean_reversion"
    name: "RSI_MeanRev"
    params:
      period: 14
      oversold_threshold: 30
      overbought_threshold: 70
      lookback_period: 20
    allocation: 1.0
```

**How it works**:
- Buys when RSI drops below oversold threshold
- Sells when RSI rises above overbought threshold
- Uses lookback period to establish mean levels

### Breakout Strategy

```yaml
strategies:
  - type: "breakout"
    name: "Donchian_Breakout"
    params:
      period: 20
      entry_threshold: 0.5  # 50% of channel width
      exit_threshold: 0.2   # 20% retracement
    allocation: 1.0
```

## 🔧 Strategy Parameters

### Common Parameters

All strategies support these common parameters:

```yaml
strategies:
  - type: "momentum"
    name: "Custom_Name"           # Optional: Strategy identifier
    enabled: true                 # Optional: Enable/disable strategy
    allocation: 0.5               # Portfolio allocation (0.0 to 1.0)
    
    # Strategy-specific parameters
    params:
      fast_period: 10
      slow_period: 20
      
    # Risk management overrides
    risk_overrides:
      max_position_size: 0.1      # Max 10% position size
      stop_loss_pct: 0.02         # 2% stop loss
      
    # Data requirements
    data_requirements:
      min_history_days: 100       # Minimum data needed
      required_timeframes: ["1m"] # Required timeframes
```

### Parameter Ranges and Validation

```yaml
# Parameters are automatically validated
strategies:
  - type: "momentum"
    params:
      fast_period: 10        # Must be positive integer
      slow_period: 20        # Must be > fast_period
      signal_threshold: 0.01 # Must be 0.0 to 1.0
      
      # Invalid examples that will cause errors:
      # fast_period: -5      ❌ Negative values not allowed
      # slow_period: 5       ❌ Must be > fast_period
      # signal_threshold: 2.0 ❌ Must be ≤ 1.0
```

## 📊 Built-in Strategy Types

### 1. Momentum Strategies

#### Simple Moving Average Crossover
```yaml
strategies:
  - type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
      signal_threshold: 0.01
```

#### Exponential Moving Average
```yaml
strategies:
  - type: "ema_momentum"
    params:
      fast_period: 12
      slow_period: 26
      smoothing_factor: 0.15
```

#### MACD Strategy
```yaml
strategies:
  - type: "macd"
    params:
      fast_period: 12
      slow_period: 26
      signal_period: 9
      histogram_threshold: 0.1
```

### 2. Mean Reversion Strategies

#### RSI Mean Reversion
```yaml
strategies:
  - type: "rsi_mean_reversion"
    params:
      period: 14
      oversold: 30
      overbought: 70
      exit_neutral: 50
```

#### Bollinger Bands
```yaml
strategies:
  - type: "bollinger_bands"
    params:
      period: 20
      std_dev: 2.0
      squeeze_threshold: 0.1
```

#### Statistical Arbitrage
```yaml
strategies:
  - type: "stat_arb"
    params:
      lookback_period: 60
      entry_z_score: 2.0
      exit_z_score: 0.5
      correlation_threshold: 0.8
```

### 3. Breakout Strategies

#### Donchian Channel Breakout
```yaml
strategies:
  - type: "donchian_breakout"
    params:
      entry_period: 20
      exit_period: 10
      atr_multiplier: 1.5
```

#### Volume Breakout
```yaml
strategies:
  - type: "volume_breakout"
    params:
      volume_period: 20
      volume_threshold: 2.0  # 2x average volume
      price_threshold: 0.02  # 2% price move
```

### 4. Pairs Trading

```yaml
strategies:
  - type: "pairs_trading"
    params:
      pair_symbols: ["SPY", "QQQ"]
      lookback_period: 60
      entry_z_score: 2.0
      exit_z_score: 0.5
      hedge_ratio_method: "ols"  # or "tls", "kalman"
```

## 🤖 Machine Learning Integration

### Scikit-Learn Models

```yaml
strategies:
  - type: "sklearn_model"
    name: "RandomForest_Strategy"
    params:
      model_path: "models/rf_strategy.pkl"
      feature_columns: ["rsi", "macd", "bb_position"]
      prediction_threshold: 0.6
      retrain_frequency: "monthly"
    allocation: 0.3
```

**Model Requirements**:
- Model must be saved as pickle file
- Must have `predict()` or `predict_proba()` method
- Should return predictions in range [-1, 1] or [0, 1]

### TensorFlow/Keras Models

```yaml
strategies:
  - type: "tensorflow_model"
    name: "LSTM_Strategy"
    params:
      model_path: "models/lstm_model.h5"
      sequence_length: 60
      features: ["close", "volume", "rsi", "macd"]
      prediction_threshold: 0.55
      preprocessing: "standard_scaler"
    allocation: 0.4
```

### Custom Model Integration

```yaml
strategies:
  - type: "custom_model"
    name: "Proprietary_Model"
    params:
      model_class: "models.proprietary.PropModel"
      model_config: "config/prop_model.json"
      update_frequency: "daily"
    allocation: 0.3
```

## 🔀 Multi-Strategy Portfolios

### Equal Weight Portfolio

```yaml
strategies:
  - type: "momentum"
    name: "Fast_Momentum"
    params:
      fast_period: 5
      slow_period: 15
    allocation: 0.33
    
  - type: "mean_reversion"
    name: "RSI_MeanRev"
    params:
      period: 14
      oversold: 25
      overbought: 75
    allocation: 0.33
    
  - type: "breakout"
    name: "Channel_Breakout"
    params:
      period: 20
    allocation: 0.34
```

### Risk-Weighted Allocation

```yaml
# Allocate based on historical volatility
strategies:
  - type: "momentum"
    allocation: 0.5  # Lower volatility = higher allocation
    risk_target: 0.10
    
  - type: "breakout"
    allocation: 0.3  # Higher volatility = lower allocation
    risk_target: 0.15
    
  - type: "mean_reversion"
    allocation: 0.2
    risk_target: 0.12

# Alternative: Dynamic allocation based on recent performance
portfolio_optimization:
  method: "risk_parity"
  rebalance_frequency: "monthly"
  lookback_period: 252  # 1 year
```

### Conditional Strategy Allocation

```yaml
# Different strategies for different market conditions
regime_strategies:
  classifier: "hmm"  # Hidden Markov Model for regime detection
  
  regimes:
    trending:
      strategies:
        - type: "momentum"
          allocation: 0.7
        - type: "breakout"
          allocation: 0.3
          
    sideways:
      strategies:
        - type: "mean_reversion"
          allocation: 0.6
        - type: "pairs_trading"
          allocation: 0.4
          
    volatile:
      strategies:
        - type: "defensive"
          allocation: 1.0
```

## 🎛️ Advanced Configuration

### Strategy Ensembles

```yaml
strategies:
  - type: "ensemble"
    name: "Signal_Voting_Ensemble"
    
    members:
      - type: "momentum"
        weight: 0.4
        params:
          fast_period: 10
          slow_period: 20
          
      - type: "rsi_mean_reversion"
        weight: 0.3
        params:
          period: 14
          
      - type: "sklearn_model"
        weight: 0.3
        params:
          model_path: "models/ml_signals.pkl"
          
    ensemble_method: "weighted_average"  # or "majority_vote", "confidence_weighted"
    min_agreement: 0.6  # Require 60% agreement for signal
```

### Time-Based Strategy Switching

```yaml
strategies:
  - type: "time_conditional"
    name: "Market_Hours_Strategy"
    
    conditions:
      market_open:  # 9:30 AM - 11:30 AM
        time_range: ["09:30", "11:30"]
        strategy:
          type: "momentum"
          params:
            fast_period: 5
            slow_period: 15
            
      mid_day:  # 11:30 AM - 2:30 PM
        time_range: ["11:30", "14:30"]
        strategy:
          type: "mean_reversion"
          params:
            period: 10
            
      market_close:  # 2:30 PM - 4:00 PM
        time_range: ["14:30", "16:00"]
        strategy:
          type: "breakout"
          params:
            period: 20
```

### Dynamic Parameter Adjustment

```yaml
strategies:
  - type: "adaptive_momentum"
    name: "Volatility_Adjusted_Momentum"
    
    base_params:
      fast_period: 10
      slow_period: 20
      
    adaptations:
      - condition: "volatility > 0.3"
        adjustments:
          fast_period: 15  # Slower in high volatility
          slow_period: 30
          
      - condition: "volume > 2.0 * avg_volume"
        adjustments:
          signal_threshold: 0.005  # Lower threshold on high volume
```

## 🔧 Custom Strategy Functions

### Python Function Integration

```yaml
strategies:
  - type: "custom_function"
    name: "My_Custom_Strategy"
    function: "strategies.custom.my_strategy_function"
    params:
      parameter1: 10
      parameter2: 0.5
    allocation: 0.5
```

**Function Requirements**:
```python
# File: strategies/custom.py
def my_strategy_function(market_data, parameter1=10, parameter2=0.5):
    """
    Custom strategy function
    
    Args:
        market_data: MarketData object with OHLCV data
        parameter1: Custom parameter 1
        parameter2: Custom parameter 2
        
    Returns:
        TradingSignal object with action and strength
    """
    from src.core.events.types import TradingSignal
    
    # Your custom logic here
    if some_condition:
        return TradingSignal(
            symbol=market_data.symbol,
            action="BUY",
            strength=0.8,
            strategy_id="my_custom_strategy"
        )
    else:
        return TradingSignal(
            symbol=market_data.symbol,
            action="HOLD",
            strength=0.0,
            strategy_id="my_custom_strategy"
        )
```

### External API Integration

```yaml
strategies:
  - type: "external_api"
    name: "Third_Party_Signals"
    params:
      api_endpoint: "https://api.tradingsignals.com/v1/signals"
      api_key: "${TRADING_API_KEY}"  # Environment variable
      update_frequency: 60  # seconds
      signal_mapping:
        buy: "BUY"
        sell: "SELL"
        hold: "HOLD"
    allocation: 0.2
```

## 📊 Strategy Performance Monitoring

### Real-Time Metrics

```yaml
strategies:
  - type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
      
    monitoring:
      track_metrics: true
      metrics:
        - "signal_frequency"
        - "win_rate"
        - "average_holding_period"
        - "sharpe_ratio"
      alert_thresholds:
        min_win_rate: 0.45
        max_drawdown: 0.1
```

### Strategy Attribution

```yaml
# Track performance by strategy in multi-strategy portfolio
portfolio:
  strategy_attribution: true
  attribution_frequency: "daily"
  
reporting:
  include_attribution: true
  attribution_charts: true
```

## 🛠️ Testing and Validation

### Strategy Backtesting

```yaml
# Test individual strategy before adding to portfolio
workflow:
  type: "strategy_validation"
  
strategy_under_test:
  type: "momentum"
  params:
    fast_period: 10
    slow_period: 20
    
validation_tests:
  - type: "historical_backtest"
    data_range: ["2020-01-01", "2023-12-31"]
    
  - type: "walk_forward"
    train_period: 252
    test_period: 63
    
  - type: "monte_carlo"
    iterations: 1000
    
  - type: "stress_test"
    scenarios: ["covid_crash", "tech_selloff"]
```

### A/B Testing

```yaml
# Compare strategy variants
strategies:
  strategy_a:
    type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
    allocation: 0.5
    
  strategy_b:
    type: "momentum"
    params:
      fast_period: 15
      slow_period: 30
    allocation: 0.5
    
ab_test:
  duration_days: 90
  success_metric: "sharpe_ratio"
  significance_level: 0.05
```

## 🤔 Common Questions

**Q: How do I know which strategy parameters to use?**
A: Start with default parameters, then use [optimization](optimization.md) to find better values for your specific data and objectives.

**Q: Can I combine multiple strategy types?**
A: Yes! Multi-strategy portfolios are a key ADMF-PC feature. See the multi-strategy examples above.

**Q: How do I integrate my own ML model?**
A: Save your model as a pickle file and use the `sklearn_model` or `tensorflow_model` strategy types, or create a custom function.

**Q: What if my strategy needs custom indicators?**
A: Use custom functions or see [Custom Components](../08-advanced-topics/custom-components.md) for creating reusable indicators.

## 🎯 Best Practices

1. **Start Simple**: Begin with single strategies before building portfolios
2. **Test Thoroughly**: Use walk-forward validation before production
3. **Monitor Performance**: Set up alerts for strategy degradation
4. **Diversify**: Combine different strategy types and timeframes
5. **Document Everything**: Keep notes on parameter choices and performance

## 📝 Next Steps

- **Optimize Parameters**: [Optimization Guide](optimization.md)
- **Add Risk Management**: [Risk Management Guide](risk-management.md)
- **Build Workflows**: [Multi-Phase Workflows](multi-phase-workflows.md)
- **Deploy to Production**: [Live Trading Guide](live-trading.md)

---

Continue to [Risk Management](risk-management.md) to protect your strategies →