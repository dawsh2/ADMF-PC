# Risk Management

Risk management is crucial for protecting capital and ensuring consistent performance. ADMF-PC provides comprehensive risk controls that can be configured without programming.

## 🛡️ Basic Risk Controls

### Simple Risk Configuration

```yaml
risk_management:
  type: "basic"
  params:
    position_size_pct: 0.02      # 2% of portfolio per position
    max_positions: 5             # Maximum 5 concurrent positions  
    max_exposure_pct: 0.10       # Maximum 10% total exposure
    stop_loss_pct: 0.02          # 2% stop loss
    take_profit_pct: 0.06        # 6% take profit (3:1 ratio)
```

**How it works**:
- Each position limited to 2% of portfolio value
- Maximum of 5 positions can be open simultaneously
- Total exposure cannot exceed 10% of portfolio
- Automatic stop-loss at 2% loss per position
- Automatic take-profit at 6% gain per position

### Position Sizing Methods

#### Fixed Percentage
```yaml
risk_management:
  type: "fixed_percentage"
  params:
    position_size_pct: 0.025    # Always 2.5% per position
```

#### Volatility-Based Sizing
```yaml
risk_management:
  type: "volatility_based"
  params:
    target_volatility: 0.15     # Target 15% portfolio volatility
    lookback_days: 20           # Use 20-day volatility estimate
    max_position_size: 0.05     # Cap at 5% per position
    min_position_size: 0.01     # Minimum 1% per position
```

#### Kelly Criterion
```yaml
risk_management:
  type: "kelly_criterion"
  params:
    lookback_trades: 100        # Use last 100 trades for calculation
    kelly_fraction: 0.25        # Use 25% of full Kelly (more conservative)
    max_position_size: 0.10     # Cap at 10% regardless of Kelly
```

#### Risk Parity
```yaml
risk_management:
  type: "risk_parity"
  params:
    target_risk_per_position: 0.02  # Each position contributes 2% risk
    risk_measure: "volatility"       # or "var", "cvar"
    rebalance_frequency: "weekly"
```

## 📊 Portfolio-Level Risk Controls

### Maximum Drawdown Protection

```yaml
risk_management:
  portfolio_limits:
    max_drawdown_pct: 0.15          # Stop trading at 15% drawdown
    drawdown_action: "reduce_size"   # or "stop_trading", "hedge"
    recovery_threshold: 0.10         # Resume normal size at 10% drawdown
    
    # Alternative: Dynamic sizing based on drawdown
    drawdown_scaling:
      enabled: true
      scaling_factor: 0.5           # Halve position sizes in drawdown
      drawdown_threshold: 0.05      # Start scaling at 5% drawdown
```

### Exposure Limits

```yaml
risk_management:
  exposure_limits:
    max_gross_exposure: 1.0         # 100% gross exposure (long + short)
    max_net_exposure: 0.8           # 80% net exposure (long - short)
    max_sector_exposure: 0.3        # 30% per sector
    max_single_position: 0.05       # 5% per individual position
    
    # Currency exposure (for multi-currency portfolios)
    max_currency_exposure: 0.5      # 50% per currency
    base_currency: "USD"
```

### Correlation Controls

```yaml
risk_management:
  correlation_limits:
    max_portfolio_correlation: 0.7  # Maximum correlation between positions
    correlation_window: 60          # Use 60-day correlation
    correlation_action: "reject"    # or "reduce_size", "delay_entry"
    
    # Sector/industry diversification
    max_positions_per_sector: 2
    min_sectors: 3
```

## ⚡ Dynamic Risk Management

### Volatility-Adaptive Risk

```yaml
risk_management:
  type: "volatility_adaptive"
  params:
    base_position_size: 0.02
    volatility_lookback: 20
    
    # Adjust position size based on market volatility
    volatility_adjustments:
      - volatility_threshold: 0.15   # Low volatility
        size_multiplier: 1.5          # Increase size 50%
      - volatility_threshold: 0.25   # Medium volatility  
        size_multiplier: 1.0          # Normal size
      - volatility_threshold: 0.40   # High volatility
        size_multiplier: 0.5          # Reduce size 50%
```

### Regime-Aware Risk Management

```yaml
risk_management:
  type: "regime_aware"
  regime_classifier: "hmm"  # Use HMM for regime detection
  
  regime_configs:
    bull_market:
      position_size_pct: 0.03
      max_exposure_pct: 0.15
      stop_loss_pct: 0.03
      
    bear_market:
      position_size_pct: 0.015
      max_exposure_pct: 0.08
      stop_loss_pct: 0.015
      
    sideways_market:
      position_size_pct: 0.025
      max_exposure_pct: 0.12
      stop_loss_pct: 0.02
```

### Time-Based Risk Adjustments

```yaml
risk_management:
  time_based_adjustments:
    # Reduce risk before major events
    earnings_season:
      weeks_before_earnings: 1
      size_reduction: 0.5
      
    # Reduce risk on Fridays (weekend risk)
    end_of_week:
      day: "friday"
      time_after: "15:00"
      size_reduction: 0.3
      
    # Increase risk during high-volume periods
    market_open:
      time_range: ["09:30", "10:30"]
      size_multiplier: 1.2
```

## 🎯 Stop-Loss and Take-Profit

### Basic Stop-Loss Configuration

```yaml
risk_management:
  stop_loss:
    type: "percentage"
    value: 0.02                    # 2% stop loss
    
  take_profit:
    type: "percentage" 
    value: 0.06                    # 6% take profit
```

### Advanced Stop-Loss Methods

#### Trailing Stop-Loss
```yaml
risk_management:
  stop_loss:
    type: "trailing"
    initial_stop_pct: 0.02         # Initial 2% stop
    trail_amount_pct: 0.01         # Trail by 1%
    trail_frequency: "daily"       # Update daily
```

#### ATR-Based Stops
```yaml
risk_management:
  stop_loss:
    type: "atr_based"
    atr_period: 14                 # 14-day ATR
    atr_multiplier: 2.0            # 2x ATR stop distance
    min_stop_pct: 0.01             # Minimum 1% stop
    max_stop_pct: 0.05             # Maximum 5% stop
```

#### Technical Level Stops
```yaml
risk_management:
  stop_loss:
    type: "technical_levels"
    methods: ["support_resistance", "moving_averages"]
    lookback_periods: [20, 50]
    buffer_pct: 0.005              # 0.5% buffer below level
```

### Profit Taking Strategies

#### Scaled Profit Taking
```yaml
risk_management:
  take_profit:
    type: "scaled"
    levels:
      - profit_pct: 0.03           # Take 25% profit at 3% gain
        position_reduction: 0.25
      - profit_pct: 0.06           # Take 50% profit at 6% gain  
        position_reduction: 0.50
      - profit_pct: 0.10           # Take remaining at 10% gain
        position_reduction: 1.00
```

#### Time-Based Profit Taking
```yaml
risk_management:
  take_profit:
    type: "time_based"
    max_holding_period: 30         # Force exit after 30 days
    profit_decay:
      enabled: true
      initial_target: 0.08         # 8% initial target
      decay_rate: 0.1              # Reduce target 10% per week
      min_target: 0.03             # Minimum 3% target
```

## 💹 Advanced Risk Models

### Value at Risk (VaR)

```yaml
risk_management:
  type: "var_based"
  params:
    var_method: "historical"       # or "parametric", "monte_carlo"
    confidence_level: 0.95         # 95% confidence
    holding_period: 1              # 1-day VaR
    lookback_days: 252             # 1 year of data
    
    # Portfolio VaR limits
    max_portfolio_var: 0.02        # Max 2% daily VaR
    max_position_var: 0.005        # Max 0.5% VaR per position
    
    # Stress testing
    stress_scenarios:
      - name: "market_crash"
        percentile: 0.01           # 1% worst case
      - name: "sector_rotation"
        correlation_shock: 0.5     # 50% correlation increase
```

### Conditional Value at Risk (CVaR)

```yaml
risk_management:
  type: "cvar_based"
  params:
    confidence_level: 0.95
    max_portfolio_cvar: 0.03       # Max 3% expected shortfall
    optimization_frequency: "weekly"
    
    # Dynamic CVaR adjustment
    cvar_scaling:
      low_volatility: 1.2          # Allow higher CVaR in low vol
      high_volatility: 0.8         # Reduce CVaR in high vol
      volatility_threshold: 0.20
```

### Factor Risk Models

```yaml
risk_management:
  type: "factor_based"
  factor_model: "fama_french_5"    # or "custom", "pca"
  
  factor_limits:
    market_beta: [-0.5, 1.5]       # Beta exposure limits
    size_factor: [-0.3, 0.3]       # Size factor limits  
    value_factor: [-0.2, 0.2]      # Value factor limits
    momentum_factor: [-0.2, 0.2]   # Momentum factor limits
    
  # Risk decomposition
  risk_attribution: true
  rebalance_frequency: "monthly"
```

## 🔄 Multi-Strategy Risk Management

### Strategy-Level Risk Allocation

```yaml
# Different risk limits per strategy
strategies:
  - name: "momentum_strategy"
    type: "momentum"
    allocation: 0.4
    risk_limits:
      max_position_size: 0.03
      max_exposure: 0.12
      stop_loss_pct: 0.02
      
  - name: "mean_reversion_strategy"  
    type: "mean_reversion"
    allocation: 0.3
    risk_limits:
      max_position_size: 0.02
      max_exposure: 0.08
      stop_loss_pct: 0.015
      
  - name: "ml_strategy"
    type: "sklearn_model"
    allocation: 0.3
    risk_limits:
      max_position_size: 0.025
      max_exposure: 0.10
      stop_loss_pct: 0.025
```

### Cross-Strategy Risk Controls

```yaml
risk_management:
  cross_strategy_limits:
    max_overlapping_positions: 2   # Max 2 strategies in same symbol
    correlation_threshold: 0.8     # Limit correlated strategies
    
    # Risk budgeting across strategies
    risk_budget_allocation:
      momentum_strategy: 0.4       # 40% of risk budget
      mean_reversion_strategy: 0.3 # 30% of risk budget
      ml_strategy: 0.3             # 30% of risk budget
      
    # Strategy interaction rules
    interaction_rules:
      - if: "momentum_strategy.signal == BUY"
        then: "reduce_mean_reversion_size(0.5)"
      - if: "portfolio.correlation > 0.8"
        then: "halt_new_positions"
```

## 📈 Risk Monitoring and Alerts

### Real-Time Risk Monitoring

```yaml
risk_monitoring:
  enabled: true
  update_frequency: "1m"          # Update every minute
  
  # Risk metrics to track
  metrics:
    - "portfolio_var"
    - "gross_exposure"
    - "net_exposure" 
    - "max_drawdown"
    - "position_concentration"
    - "correlation"
    
  # Alert thresholds
  alerts:
    portfolio_var_threshold: 0.025
    exposure_threshold: 0.9
    drawdown_threshold: 0.12
    correlation_threshold: 0.8
    
  # Alert actions
  alert_actions:
    email: "risk@trading.com"
    webhook: "https://alerts.trading.com/risk"
    auto_reduce_size: true
```

### Risk Reporting

```yaml
risk_reporting:
  frequency: "daily"
  include_metrics:
    - "var_breakdown"
    - "exposure_analysis" 
    - "correlation_matrix"
    - "drawdown_analysis"
    - "risk_attribution"
    
  # Automated reports
  automated_reports:
    - type: "daily_risk_summary"
      time: "17:00"
      recipients: ["portfolio_manager@trading.com"]
      
    - type: "weekly_risk_review"
      day: "friday"
      time: "16:00"
      include_recommendations: true
```

## 🛠️ Custom Risk Models

### Custom Risk Functions

```yaml
risk_management:
  type: "custom_function"
  function: "risk_models.custom.advanced_risk_model"
  params:
    lookback_days: 60
    risk_target: 0.15
    custom_param: 1.5
```

**Function Requirements**:
```python
# File: risk_models/custom.py
def advanced_risk_model(signal, portfolio, lookback_days=60, risk_target=0.15, custom_param=1.5):
    """
    Custom risk model function
    
    Args:
        signal: TradingSignal object
        portfolio: Current portfolio state
        lookback_days: Days to look back for risk calculation
        risk_target: Target portfolio risk level
        custom_param: Custom parameter
        
    Returns:
        Dict with risk decision: {'approved': bool, 'position_size': float, 'reason': str}
    """
    # Your custom risk logic here
    current_risk = calculate_portfolio_risk(portfolio, lookback_days)
    position_risk = estimate_position_risk(signal, portfolio)
    
    if current_risk + position_risk > risk_target:
        return {
            'approved': False, 
            'position_size': 0.0,
            'reason': 'Would exceed risk target'
        }
    
    # Calculate appropriate position size
    optimal_size = optimize_position_size(signal, portfolio, risk_target)
    
    return {
        'approved': True,
        'position_size': optimal_size,
        'reason': f'Risk-optimized size: {optimal_size:.1%}'
    }
```

## 🎯 Risk Management Best Practices

### 1. **Layered Risk Controls**
```yaml
# Multiple layers of protection
risk_management:
  # Position level
  position_limits:
    max_position_size: 0.05
    stop_loss_pct: 0.02
    
  # Portfolio level  
  portfolio_limits:
    max_exposure: 0.8
    max_drawdown: 0.15
    
  # System level
  system_limits:
    daily_loss_limit: 0.03
    monthly_loss_limit: 0.10
```

### 2. **Risk-Adjusted Position Sizing**
```yaml
# Size positions based on risk, not returns
risk_management:
  type: "risk_adjusted"
  target_risk_per_trade: 0.01      # Risk 1% per trade
  risk_measure: "volatility"        # Use volatility to measure risk
  max_position_size: 0.10           # Cap at 10% regardless
```

### 3. **Dynamic Risk Adjustment**
```yaml
# Adjust risk based on market conditions
risk_management:
  dynamic_adjustment:
    volatility_scaling: true
    correlation_adjustment: true
    drawdown_scaling: true
    regime_awareness: true
```

### 4. **Comprehensive Monitoring**
```yaml
# Monitor all aspects of risk
monitoring:
  real_time_metrics: true
  automated_alerts: true
  daily_reporting: true
  stress_testing: "weekly"
```

## 🤔 Common Questions

**Q: Should I use percentage-based or volatility-based position sizing?**
A: Volatility-based is generally better as it accounts for varying risk levels across assets and time periods.

**Q: How tight should my stop-losses be?**
A: Tight enough to limit losses, loose enough to avoid premature exits. ATR-based stops often work well.

**Q: What's a good maximum drawdown limit?**
A: 10-20% for most strategies. More aggressive strategies might allow 25%, conservative ones 5-10%.

**Q: How often should I rebalance risk limits?**
A: Daily monitoring with weekly/monthly limit adjustments based on changing market conditions.

## 🎯 Risk Management Checklist

- [ ] Set appropriate position sizing method
- [ ] Configure stop-loss and take-profit levels
- [ ] Set portfolio-level exposure limits
- [ ] Configure maximum drawdown protection
- [ ] Set up correlation controls
- [ ] Enable real-time risk monitoring
- [ ] Configure automated alerts
- [ ] Test risk controls with historical data
- [ ] Document risk management rationale
- [ ] Review and adjust limits regularly

## 📝 Next Steps

- **Backtest with Risk Controls**: [Backtesting Guide](backtesting.md)
- **Optimize Risk-Adjusted Returns**: [Optimization Guide](optimization.md)
- **Monitor Risk in Production**: [Live Trading Guide](live-trading.md)
- **Advanced Risk Models**: [Custom Components](../08-advanced-topics/custom-components.md)

---

Continue to [Backtesting](backtesting.md) to test your risk-managed strategies →