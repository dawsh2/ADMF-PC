# Live Trading

Live trading deploys your strategies to production with real money. ADMF-PC ensures the same configuration that worked in backtesting works identically in live trading.

## 🎯 Production Readiness

### Prerequisites for Live Trading

Before deploying to live trading, ensure:

**Strategy Validation**:
- [ ] Positive out-of-sample performance
- [ ] Walk-forward validation completed
- [ ] Risk controls tested and validated
- [ ] Transaction costs realistically modeled
- [ ] Sufficient trade frequency for statistical significance

**Infrastructure Readiness**:
- [ ] Broker API credentials configured
- [ ] Real-time data feeds operational
- [ ] Risk monitoring systems active
- [ ] Logging and alerting configured
- [ ] Backup and recovery procedures tested

**Risk Management**:
- [ ] Position size limits appropriate for live capital
- [ ] Maximum drawdown limits conservative
- [ ] Emergency stop procedures defined
- [ ] Risk monitoring thresholds set

## 🔧 Broker Integration

### Supported Brokers

```yaml
# config/live_trading.yaml
data:
  source:
    type: "live"
    provider: "alpaca"              # or "interactive_brokers", "td_ameritrade"
    
    # API credentials (use environment variables)
    api_credentials:
      api_key: "${ALPACA_API_KEY}"
      secret_key: "${ALPACA_SECRET_KEY}"
      base_url: "${ALPACA_BASE_URL}"  # Paper: paper-api.alpaca.markets
      
    # Data configuration
    real_time_data: true
    market_data_subscription: "premium"  # or "basic"
```

### Interactive Brokers Configuration

```yaml
data:
  source:
    type: "live"
    provider: "interactive_brokers"
    
    connection:
      host: "127.0.0.1"
      port: 7497                    # TWS paper trading port
      client_id: 1
      
    api_credentials:
      username: "${IB_USERNAME}"
      password: "${IB_PASSWORD}"
      
    # IB-specific settings
    ib_settings:
      enable_outside_rth: false     # Regular trading hours only
      enable_paper_trading: true    # Start with paper trading
      request_market_data: true
      place_orders: true
```

### Alpaca Configuration

```yaml
data:
  source:
    type: "live"
    provider: "alpaca"
    
    api_credentials:
      api_key: "${ALPACA_API_KEY}"
      secret_key: "${ALPACA_SECRET_KEY}"
      
    # Environment selection
    environment: "paper"            # or "live"
    base_url: "https://paper-api.alpaca.markets"  # Paper trading
    
    # Alpaca-specific settings
    alpaca_settings:
      enable_crypto: false
      enable_options: false
      data_feed: "iex"              # or "sip"
```

## 📊 Live Trading Configuration

### Basic Live Trading Setup

```yaml
# config/live_momentum_strategy.yaml
workflow:
  type: "live_trading"
  name: "Live Momentum Strategy"
  
# Real-time data configuration
data:
  source:
    type: "live"
    provider: "alpaca"
    environment: "paper"            # Start with paper trading
    
  symbols: ["SPY", "QQQ"]
  timeframe: "1m"                   # 1-minute bars
  
  # Market hours
  market_hours:
    start_time: "09:30"
    end_time: "16:00"
    timezone: "US/Eastern"
    trading_calendar: "NYSE"

# Strategy configuration (same as backtesting)
strategies:
  - type: "momentum"
    name: "Live_Momentum"
    params:
      fast_period: 12               # From optimization
      slow_period: 28               # From optimization
      signal_threshold: 0.025       # From optimization
    allocation: 1.0

# Risk management (conservative for live trading)
risk_management:
  position_size_pct: 0.02           # 2% per position
  max_positions: 3                  # Conservative limit
  max_exposure_pct: 0.06            # 6% total exposure
  stop_loss_pct: 0.015              # Tight stop loss
  daily_loss_limit_pct: 0.03       # 3% daily loss limit

# Execution configuration
execution:
  order_type: "market"              # Start with market orders
  execution_delay_seconds: 30       # 30-second delay for review
  position_management:
    enable_partial_fills: true
    timeout_seconds: 300            # 5-minute order timeout
    
# Portfolio configuration
portfolio:
  initial_capital: 10000            # Start small
  currency: "USD"
  cash_management:
    min_cash_pct: 0.1               # Keep 10% cash
    
# Monitoring and alerts
monitoring:
  enabled: true
  alert_channels:
    email: "trader@example.com"
    webhook: "https://hooks.slack.com/services/..."
    
  alert_conditions:
    position_opened: true
    position_closed: true
    stop_loss_triggered: true
    daily_loss_limit_reached: true
    system_error: true

# Logging
logging:
  level: "INFO"
  log_trades: true
  log_signals: true
  log_portfolio_updates: true
  output_path: "logs/live_trading/"
```

## 🚨 Risk Management for Live Trading

### Enhanced Risk Controls

```yaml
risk_management:
  # Position-level controls
  position_controls:
    max_position_size_usd: 500      # Absolute dollar limit
    position_size_pct: 0.02         # Percentage limit
    max_leverage: 1.0               # No leverage initially
    
  # Portfolio-level controls
  portfolio_controls:
    max_drawdown_pct: 0.10          # 10% maximum drawdown
    daily_loss_limit_pct: 0.03      # 3% daily loss limit
    monthly_loss_limit_pct: 0.08    # 8% monthly loss limit
    
  # Time-based controls
  time_controls:
    no_trading_near_close: true
    close_positions_minutes: 30     # Close 30 min before market close
    max_holding_period_days: 10     # Force exit after 10 days
    
  # Volatility controls
  volatility_controls:
    max_symbol_volatility: 0.30     # Skip high volatility stocks
    portfolio_volatility_limit: 0.15
    volatility_adjustment: true     # Reduce size in high vol
    
  # Emergency controls
  emergency_controls:
    kill_switch: true               # Manual emergency stop
    auto_liquidation_drawdown: 0.15 # Auto-liquidate at 15% drawdown
    broker_connection_timeout: 60   # Stop if disconnected > 1 min
```

### Real-Time Risk Monitoring

```yaml
monitoring:
  risk_monitoring:
    enabled: true
    check_frequency_seconds: 30     # Check every 30 seconds
    
    # Metrics to monitor
    monitored_metrics:
      - "current_drawdown"
      - "daily_pnl"
      - "position_concentration"
      - "gross_exposure"
      - "cash_level"
      - "unrealized_pnl"
      
    # Alert thresholds
    alert_thresholds:
      drawdown_warning: 0.05        # 5% drawdown warning
      drawdown_critical: 0.08       # 8% drawdown critical
      daily_loss_warning: 0.02     # 2% daily loss warning
      exposure_warning: 0.08       # 8% exposure warning
      
    # Automated responses
    automated_responses:
      reduce_size_on_drawdown: true
      halt_trading_on_critical: true
      email_on_warning: true
      sms_on_critical: true
```

## 📈 Staged Deployment Process

### Phase 1: Paper Trading

```yaml
# Start with paper trading to validate everything works
deployment:
  phase: "paper_trading"
  duration_days: 30
  
data:
  source:
    environment: "paper"            # Paper trading environment
    
portfolio:
  initial_capital: 100000          # Realistic paper capital
  
validation:
  paper_trading_validation:
    min_trades: 20                  # Require minimum trades
    track_slippage_estimates: true  # Validate execution assumptions
    compare_to_backtest: true       # Compare with backtest results
    
  success_criteria:
    min_sharpe_ratio: 0.8           # Lower than backtest due to costs
    max_drawdown: 0.12              # Slightly higher tolerance
    correlation_with_backtest: 0.7  # 70% correlation minimum
```

### Phase 2: Small Live Capital

```yaml
deployment:
  phase: "small_live_capital"
  duration_days: 60
  
data:
  source:
    environment: "live"             # Real money
    
portfolio:
  initial_capital: 5000            # Start very small
  
risk_management:
  position_size_pct: 0.01          # Even more conservative
  max_positions: 2                 # Very limited positions
  
monitoring:
  enhanced_monitoring: true
  real_time_alerts: true
  daily_reports: true
```

### Phase 3: Gradual Scale-Up

```yaml
deployment:
  phase: "gradual_scale_up"
  
  # Scale up schedule based on performance
  scaling_schedule:
    weeks_1_4:
      capital: 5000
      position_size_pct: 0.01
      max_positions: 2
      
    weeks_5_8:
      capital: 10000               # 2x scale up
      position_size_pct: 0.015
      max_positions: 3
      
    weeks_9_16:
      capital: 25000               # 2.5x scale up
      position_size_pct: 0.02
      max_positions: 4
      
    weeks_17+:
      capital: 50000               # Final target
      position_size_pct: 0.02
      max_positions: 5
      
  # Scaling criteria
  scaling_criteria:
    min_sharpe_ratio: 1.0
    max_drawdown: 0.10
    min_win_rate: 0.45
    stable_performance_weeks: 4   # 4 weeks stable before scaling
```

## 🔄 Operational Procedures

### Daily Operations

```yaml
daily_operations:
  # Pre-market procedures
  pre_market:
    time: "08:30"
    procedures:
      - "check_system_health"
      - "validate_data_feeds"
      - "review_overnight_news"
      - "check_position_limits"
      - "validate_risk_controls"
      
  # Market open procedures
  market_open:
    time: "09:30"
    procedures:
      - "enable_trading"
      - "confirm_data_flowing"
      - "validate_first_signals"
      
  # Intraday monitoring
  intraday:
    frequency: "hourly"
    procedures:
      - "check_position_status"
      - "monitor_pnl"
      - "validate_risk_metrics"
      - "check_system_performance"
      
  # Market close procedures
  market_close:
    time: "16:00"
    procedures:
      - "disable_new_positions"
      - "review_daily_performance"
      - "generate_daily_report"
      - "backup_trading_logs"
      
  # Post-market procedures
  post_market:
    time: "17:00"
    procedures:
      - "reconcile_positions"
      - "update_performance_tracking"
      - "prepare_next_day_config"
```

### Weekly Procedures

```yaml
weekly_operations:
  # Weekly review
  weekly_review:
    day: "sunday"
    time: "10:00"
    procedures:
      - "analyze_weekly_performance"
      - "review_parameter_stability"
      - "check_strategy_health"
      - "update_risk_limits"
      - "review_transaction_costs"
      
  # Parameter reoptimization (if needed)
  parameter_review:
    frequency: "monthly"
    trigger_conditions:
      - "performance_degradation > 0.3"
      - "parameter_drift > 0.5"
      - "new_regime_detected"
      
    reoptimization_process:
      - "collect_recent_data"
      - "run_walk_forward_update"
      - "validate_new_parameters"
      - "gradual_parameter_transition"
```

## 📊 Performance Monitoring

### Real-Time Performance Tracking

```yaml
performance_monitoring:
  real_time_metrics:
    update_frequency: "1m"
    
    # Core metrics
    core_metrics:
      - "current_pnl"
      - "unrealized_pnl"
      - "realized_pnl"
      - "current_drawdown"
      - "gross_exposure"
      - "net_exposure"
      - "cash_balance"
      
    # Risk metrics
    risk_metrics:
      - "portfolio_var"
      - "position_concentration"
      - "correlation_risk"
      - "leverage_ratio"
      
    # Trading metrics
    trading_metrics:
      - "trades_today"
      - "win_rate_today"
      - "avg_trade_pnl"
      - "transaction_costs"
      
  # Performance alerts
  performance_alerts:
    pnl_alerts:
      daily_loss_threshold: -0.02   # -2% daily loss
      weekly_loss_threshold: -0.05  # -5% weekly loss
      
    position_alerts:
      large_position_threshold: 0.05 # 5% position size
      concentrated_exposure: 0.10    # 10% in one symbol
      
    trading_alerts:
      excessive_trading: 20          # >20 trades per day
      poor_execution: 0.005          # >0.5% slippage
```

### Performance Reporting

```yaml
reporting:
  # Daily reports
  daily_reports:
    enabled: true
    delivery_time: "17:30"
    recipients: ["trader@example.com", "risk@example.com"]
    
    content:
      - "daily_pnl_summary"
      - "position_summary"
      - "risk_metrics"
      - "trade_log"
      - "system_health"
      
  # Weekly reports
  weekly_reports:
    enabled: true
    delivery_day: "monday"
    delivery_time: "08:00"
    
    content:
      - "weekly_performance_analysis"
      - "strategy_attribution"
      - "risk_analysis"
      - "transaction_cost_analysis"
      - "parameter_stability"
      
  # Monthly reports
  monthly_reports:
    enabled: true
    delivery_day: "first_business_day"
    
    content:
      - "monthly_performance_review"
      - "strategy_health_assessment"
      - "risk_review"
      - "optimization_recommendations"
```

## 🚨 Emergency Procedures

### Emergency Stop Procedures

```yaml
emergency_procedures:
  # Manual kill switch
  kill_switch:
    trigger_methods:
      - "web_interface_button"
      - "email_command"
      - "phone_call_verification"
      - "sms_command"
      
    actions:
      - "halt_all_new_orders"
      - "cancel_pending_orders"
      - "close_all_positions"      # Optional
      - "send_emergency_alerts"
      - "log_emergency_action"
      
  # Automatic emergency stops
  automatic_stops:
    triggers:
      - drawdown_limit: 0.15       # 15% drawdown
      - daily_loss_limit: 0.05     # 5% daily loss
      - system_error: true         # Critical system error
      - broker_disconnection: 300  # 5 minutes disconnected
      
    actions:
      - "halt_new_orders"
      - "reduce_position_sizes"
      - "send_critical_alerts"
      - "attempt_system_recovery"
      
  # Recovery procedures
  recovery_procedures:
    system_restart:
      - "validate_system_health"
      - "check_position_integrity"
      - "verify_data_feeds"
      - "test_order_placement"
      - "gradual_trading_resumption"
```

## 🔧 System Configuration

### Hardware Requirements

```yaml
system_requirements:
  # Minimum specifications
  minimum:
    cpu_cores: 4
    memory_gb: 16
    storage_gb: 100
    network: "broadband"
    
  # Recommended specifications
  recommended:
    cpu_cores: 8
    memory_gb: 32
    storage_gb: 500
    network: "dedicated_line"
    backup_internet: true
    ups_power: true
    
  # High-frequency specifications
  high_frequency:
    cpu_cores: 16
    memory_gb: 64
    storage_type: "nvme_ssd"
    network_latency: "<1ms"
    co_location: "preferred"
```

### Software Configuration

```yaml
software_setup:
  # Operating system
  operating_system:
    type: "linux"                  # Preferred for stability
    distribution: "ubuntu_lts"
    hardening: true
    
  # Python environment
  python_environment:
    version: "3.11+"
    virtual_environment: true
    dependency_pinning: true
    
  # Database
  database:
    type: "postgresql"             # For trade logging
    backup_frequency: "daily"
    
  # Monitoring
  monitoring:
    system_monitoring: "prometheus"
    log_aggregation: "elk_stack"
    uptime_monitoring: "external"
```

## 🤔 Common Questions

**Q: How do I know when my strategy is ready for live trading?**
A: Positive out-of-sample results, successful walk-forward validation, realistic transaction costs included, and stable parameters.

**Q: Should I start with paper trading?**
A: Always! Paper trading validates your infrastructure and helps identify issues without risk.

**Q: How much capital should I start with?**
A: Start very small (1-5% of intended capital) and scale gradually based on performance.

**Q: What if my live results don't match backtesting?**
A: Common causes: unrealistic transaction costs in backtest, execution delays, or data differences. Start with paper trading to debug.

## 📝 Live Trading Checklist

- [ ] Strategy validated with out-of-sample testing
- [ ] Risk controls tested and conservative
- [ ] Broker API credentials configured
- [ ] Real-time data feeds operational
- [ ] Monitoring and alerting systems active
- [ ] Emergency procedures documented
- [ ] Started with paper trading
- [ ] Small initial capital allocation
- [ ] Daily operational procedures defined

## 📈 Next Steps

- **Advanced Patterns**: [Patterns Documentation](../06-patterns/README.md)
- **Custom Components**: [Advanced Topics](../08-advanced-topics/README.md)
- **System Architecture**: [Architecture Guide](../05-architecture/README.md)

---

🎉 **Congratulations!** You've completed the User Guide. You now have the knowledge to build, optimize, validate, and deploy sophisticated trading strategies using ADMF-PC's zero-code approach.