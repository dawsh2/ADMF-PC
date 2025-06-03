# Backtesting

Backtesting validates trading strategies using historical data. ADMF-PC provides realistic backtesting with transaction costs, slippage, and comprehensive performance analysis.

## 🚀 Quick Start Backtesting

### Basic Backtest Configuration

```yaml
# config/my_backtest.yaml
workflow:
  type: "backtest"
  name: "My Strategy Backtest"
  
data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
      
risk_management:
  position_size_pct: 0.02
  max_exposure_pct: 0.10
  stop_loss_pct: 0.02
  
execution:
  slippage_bps: 10              # 10 basis points slippage
  commission_per_share: 0.01    # $0.01 per share
  
portfolio:
  initial_capital: 100000
  
reporting:
  output_path: "reports/my_backtest.html"
  include_charts: true
```

### Run the Backtest

```bash
# Run the backtest
python main.py config/my_backtest.yaml

# Run with options
python main.py config/my_backtest.yaml --verbose --bars 1000
```

## 📊 Data Configuration

### CSV Data Source

```yaml
data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
    
    # Optional: Data preprocessing
    preprocessing:
      fill_missing: "forward_fill"  # or "interpolate", "drop"
      remove_outliers: true
      outlier_threshold: 3.0        # 3 standard deviations
      
  # Time range
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
  # Optional: Limit data for faster testing
  max_bars: 10000                   # Use only first 10k bars
```

### Multiple Symbols

```yaml
data:
  symbols: ["SPY", "QQQ", "IWM"]    # Multiple ETFs
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
  # Symbol-specific settings
  symbol_config:
    SPY:
      weight: 0.5                   # 50% allocation
    QQQ:
      weight: 0.3                   # 30% allocation  
    IWM:
      weight: 0.2                   # 20% allocation
```

### Database Data Source

```yaml
data:
  source:
    type: "database"
    connection_string: "postgresql://user:pass@localhost/market_data"
    table: "daily_bars"
    symbol_column: "symbol"
    timestamp_column: "date"
    
  query_params:
    additional_filters: "volume > 1000000"  # Minimum volume filter
```

### Live Data for Paper Trading

```yaml
data:
  source:
    type: "live"
    provider: "alpaca"              # or "interactive_brokers", "polygon"
    api_key: "${ALPACA_API_KEY}"
    secret_key: "${ALPACA_SECRET_KEY}"
    paper_trading: true
```

## ⚙️ Execution Configuration

### Realistic Trading Costs

```yaml
execution:
  # Slippage modeling
  slippage:
    type: "linear"                  # or "sqrt", "impact"
    slippage_bps: 10               # 10 basis points base slippage
    volume_impact: 0.1             # Additional impact based on volume
    
  # Commission structure
  commission:
    type: "per_share"              # or "per_trade", "percentage"
    per_share: 0.005               # $0.005 per share
    minimum: 1.0                   # $1 minimum commission
    
  # Market impact modeling
  market_impact:
    enabled: true
    impact_model: "almgren_chriss"  # Academic market impact model
    participation_rate: 0.1         # Trade 10% of volume
```

### Order Types and Timing

```yaml
execution:
  # Default order type
  default_order_type: "market"     # or "limit", "market_on_close"
  
  # Order timing
  order_timing:
    signal_to_order_delay: "1m"    # 1 minute delay from signal to order
    order_to_fill_delay: "30s"     # 30 second delay from order to fill
    
  # Limit order settings (if using limit orders)
  limit_orders:
    price_offset_bps: 5            # 5bp offset from current price
    timeout_minutes: 30            # Cancel after 30 minutes
    
  # Fill modeling
  fill_probability:
    market_orders: 1.0             # Market orders always fill
    limit_orders: 0.95             # 95% fill rate for limit orders
```

### Advanced Execution Models

```yaml
execution:
  type: "advanced"
  
  # Time-weighted average price (TWAP)
  twap:
    enabled: true
    duration_minutes: 30           # Execute over 30 minutes
    slice_interval: "2m"           # 2-minute slices
    
  # Volume-weighted average price (VWAP)
  vwap:
    enabled: true
    participation_rate: 0.15       # 15% of volume
    urgency: "medium"              # low, medium, high
    
  # Implementation shortfall
  implementation_shortfall:
    risk_aversion: 0.01            # Balance speed vs cost
    market_impact_model: "linear"
```

## 📈 Portfolio Configuration

### Basic Portfolio Settings

```yaml
portfolio:
  initial_capital: 100000          # Starting capital
  currency: "USD"
  
  # Cash management
  cash_management:
    min_cash_pct: 0.05            # Keep 5% cash minimum
    reinvest_dividends: true       # Reinvest dividend payments
    interest_rate: 0.02            # 2% annual interest on cash
```

### Multi-Currency Portfolios

```yaml
portfolio:
  initial_capital: 100000
  base_currency: "USD"
  
  # Currency exposure management
  currency_hedging:
    enabled: true
    hedge_ratio: 0.8              # Hedge 80% of FX exposure
    rebalance_frequency: "weekly"
    
  # Currency-specific allocations
  currency_limits:
    EUR: 0.3                      # Max 30% EUR exposure
    GBP: 0.2                      # Max 20% GBP exposure
    JPY: 0.1                      # Max 10% JPY exposure
```

### Leverage and Margin

```yaml
portfolio:
  leverage:
    enabled: true
    max_leverage: 2.0             # 2:1 maximum leverage
    margin_rate: 0.05             # 5% annual margin rate
    maintenance_margin: 0.25      # 25% maintenance margin
    
  # Margin call handling
  margin_call_action: "reduce_positions"  # or "add_cash", "liquidate"
  margin_buffer: 0.1              # 10% buffer above maintenance
```

## 📊 Performance Analysis

### Basic Performance Metrics

```yaml
reporting:
  metrics:
    returns:
      - "total_return"
      - "annualized_return"
      - "monthly_returns"
      - "daily_returns"
      
    risk:
      - "volatility"
      - "sharpe_ratio"
      - "sortino_ratio"
      - "max_drawdown"
      - "calmar_ratio"
      
    trading:
      - "total_trades"
      - "win_rate"
      - "avg_win"
      - "avg_loss"
      - "profit_factor"
      
    portfolio:
      - "turnover"
      - "avg_position_size"
      - "time_in_market"
```

### Advanced Analytics

```yaml
reporting:
  advanced_analytics:
    # Factor attribution
    factor_attribution:
      enabled: true
      factor_model: "fama_french_3"
      
    # Risk decomposition
    risk_decomposition:
      enabled: true
      decomposition_method: "marginal_var"
      
    # Transaction cost analysis
    transaction_cost_analysis:
      enabled: true
      include_slippage: true
      include_commissions: true
      include_market_impact: true
      
    # Regime analysis
    regime_analysis:
      enabled: true
      regime_method: "hidden_markov"
      n_regimes: 3
```

### Custom Performance Metrics

```yaml
reporting:
  custom_metrics:
    - name: "underwater_time"
      function: "analytics.custom.calculate_underwater_time"
      
    - name: "tail_ratio"
      function: "analytics.custom.calculate_tail_ratio"
      params:
        percentile: 0.95
        
    - name: "kelly_criterion"
      function: "analytics.custom.calculate_kelly"
```

## 📋 Comprehensive Backtest Example

### Multi-Strategy Portfolio Backtest

```yaml
# config/comprehensive_backtest.yaml
workflow:
  type: "backtest"
  name: "Multi-Strategy Portfolio Backtest"
  description: "Comprehensive backtest with multiple strategies and risk controls"
  
data:
  symbols: ["SPY", "QQQ", "IWM", "TLT", "GLD"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  timeframe: "1h"
  
strategies:
  # Momentum strategy for trending markets
  - type: "momentum"
    name: "Trend_Following"
    params:
      fast_period: 12
      slow_period: 26
      signal_threshold: 0.02
    allocation: 0.4
    symbols: ["SPY", "QQQ", "IWM"]
    
  # Mean reversion for range-bound markets  
  - type: "mean_reversion"
    name: "Mean_Reversion"
    params:
      period: 20
      std_dev: 2.0
      entry_threshold: 0.95
    allocation: 0.3
    symbols: ["SPY", "QQQ"]
    
  # Defensive strategy for volatility
  - type: "defensive"
    name: "Flight_to_Quality"
    params:
      volatility_threshold: 0.25
      safe_haven_symbols: ["TLT", "GLD"]
    allocation: 0.3
    symbols: ["TLT", "GLD"]

risk_management:
  # Position-level risk
  position_limits:
    max_position_size: 0.05        # 5% max per position
    stop_loss_pct: 0.025           # 2.5% stop loss
    take_profit_pct: 0.075         # 7.5% take profit
    
  # Portfolio-level risk
  portfolio_limits:
    max_gross_exposure: 0.95       # 95% max gross exposure
    max_drawdown_pct: 0.15         # 15% max drawdown
    max_correlation: 0.8           # Max 80% correlation between positions
    
  # Dynamic risk adjustment
  volatility_scaling:
    enabled: true
    base_volatility: 0.16          # 16% base volatility assumption
    scaling_factor: 0.5            # Halve size when vol doubles

execution:
  # Realistic execution costs
  slippage_bps: 8                  # 8bp slippage
  commission_per_share: 0.005      # $0.005 per share
  
  # Advanced execution
  execution_algorithm: "twap"
  twap_duration_minutes: 15
  
  # Order timing
  signal_delay_minutes: 2          # 2-minute implementation delay

portfolio:
  initial_capital: 500000          # $500k starting capital
  cash_interest_rate: 0.03         # 3% on cash
  dividend_reinvestment: true

reporting:
  output_formats: ["html", "pdf", "json"]
  output_path: "reports/comprehensive_backtest"
  
  # Detailed reporting
  include_charts: true
  include_trade_log: true
  include_daily_metrics: true
  
  # Benchmark comparison
  benchmarks:
    - symbol: "SPY"
      name: "S&P 500"
    - symbol: "60_40"
      name: "60/40 Portfolio"
      composition:
        SPY: 0.6
        TLT: 0.4
```

## 🔍 Analyzing Results

### Performance Report Structure

```
reports/comprehensive_backtest.html
├── Executive Summary
│   ├── Key Metrics
│   ├── Performance vs Benchmarks
│   └── Risk-Adjusted Returns
├── Strategy Performance
│   ├── Individual Strategy Returns
│   ├── Strategy Attribution
│   └── Strategy Correlation
├── Risk Analysis
│   ├── Drawdown Analysis
│   ├── Volatility Analysis
│   └── Risk Decomposition
├── Trading Analysis
│   ├── Trade Statistics
│   ├── Win/Loss Analysis
│   └── Transaction Costs
├── Charts and Visualizations
│   ├── Equity Curve
│   ├── Drawdown Chart
│   ├── Monthly Returns Heatmap
│   └── Rolling Metrics
└── Detailed Trade Log
```

### Key Metrics to Focus On

**Return Metrics**:
- Total Return vs Benchmark
- Risk-Adjusted Returns (Sharpe, Sortino)
- Consistency (monthly win rate)

**Risk Metrics**:
- Maximum Drawdown
- Volatility vs Benchmark
- Downside Deviation

**Trading Metrics**:
- Win Rate and Profit Factor
- Average Holding Period
- Transaction Costs Impact

### Red Flags to Watch For

```yaml
# Warning signs in backtest results
red_flags:
  performance:
    - sharpe_ratio < 0.5           # Poor risk-adjusted returns
    - max_drawdown > 0.25          # Excessive drawdowns
    - win_rate < 0.35              # Too many losing trades
    
  trading:
    - avg_holding_period < "1h"    # Excessive trading
    - transaction_costs > 0.02     # High transaction costs
    - trades_per_month > 100       # Overtrading
    
  data:
    - total_trades < 30            # Insufficient trades for significance
    - time_in_market < 0.3         # Too much time in cash
    - lookback_bias: true          # Using future data
```

## 🛠️ Debugging Poor Performance

### Common Performance Issues

#### Issue: Low Sharpe Ratio
```yaml
# Potential solutions
debugging:
  low_sharpe_ratio:
    check:
      - signal_threshold: "Is threshold too low, generating weak signals?"
      - transaction_costs: "Are costs eating into returns?"
      - risk_management: "Are stops too tight?"
    solutions:
      - increase signal threshold
      - reduce trading frequency
      - optimize position sizing
```

#### Issue: High Drawdowns
```yaml
debugging:
  high_drawdowns:
    check:
      - position_sizing: "Are positions too large?"
      - diversification: "Are strategies too correlated?"
      - stop_losses: "Are stops too loose or missing?"
    solutions:
      - reduce position sizes
      - add uncorrelated strategies
      - tighten risk controls
```

#### Issue: Excessive Trading
```yaml
debugging:
  excessive_trading:
    check:
      - signal_noise: "Are signals too noisy?"
      - parameter_sensitivity: "Are parameters over-optimized?"
      - market_regime: "Strategy not suited for regime?"
    solutions:
      - add signal filters
      - use regime detection
      - increase holding periods
```

### Diagnostic Configuration

```yaml
# Enable detailed debugging
debugging:
  enabled: true
  
  # Signal analysis
  signal_analysis:
    track_signal_strength: true
    analyze_false_signals: true
    measure_signal_persistence: true
    
  # Execution analysis
  execution_analysis:
    track_slippage: true
    measure_fill_rates: true
    analyze_timing_impact: true
    
  # Portfolio analysis
  portfolio_analysis:
    track_turnover: true
    measure_capacity: true
    analyze_position_correlation: true
```

## 🎯 Backtest Validation

### Out-of-Sample Testing

```yaml
# Reserve data for final validation
data:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  
  # Data splits
  splits:
    train: ["2020-01-01", "2022-12-31"]  # 3 years for optimization
    test: ["2023-01-01", "2023-12-31"]   # 1 year for validation
    
validation:
  out_of_sample:
    enabled: true
    test_split: "test"
    require_positive_results: true       # Fail if OOS results poor
```

### Walk-Forward Analysis

```yaml
# Rolling validation
validation:
  walk_forward:
    enabled: true
    train_period_months: 12            # 12 months training
    test_period_months: 3              # 3 months testing
    step_months: 1                     # Move 1 month at a time
    min_trades_per_period: 10          # Require minimum trades
```

## 🤔 Common Questions

**Q: How much historical data do I need?**
A: Minimum 2-3 years, ideally 5+ years including different market conditions. Ensure at least 100 trades for statistical significance.

**Q: Should I include transaction costs?**
A: Always! Backtests without realistic costs are misleading. Include slippage, commissions, and market impact.

**Q: What's a good Sharpe ratio?**
A: Above 1.0 is good, above 1.5 is excellent, above 2.0 is exceptional (and rare).

**Q: How do I know if my backtest is realistic?**
A: Compare with benchmark, check trade frequency is reasonable, ensure transaction costs are realistic, validate with out-of-sample data.

## 📝 Backtest Checklist

- [ ] Include realistic transaction costs
- [ ] Use sufficient historical data
- [ ] Validate with out-of-sample testing
- [ ] Check for look-ahead bias
- [ ] Compare against relevant benchmarks
- [ ] Ensure adequate number of trades
- [ ] Document assumptions and limitations
- [ ] Stress test with different market conditions

## 📈 Next Steps

- **Optimize Parameters**: [Optimization Guide](optimization.md)
- **Validate Robustness**: [Walk-Forward Analysis](walk-forward-analysis.md)
- **Build Complex Workflows**: [Multi-Phase Workflows](multi-phase-workflows.md)
- **Deploy to Production**: [Live Trading Guide](live-trading.md)

---

Continue to [Optimization](optimization.md) to improve your strategy parameters →