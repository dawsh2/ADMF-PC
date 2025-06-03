# Component Catalog

Complete catalog of all available ADMF-PC components with detailed specifications, parameters, and usage examples.

## 📊 Strategy Components

### Momentum Strategies

#### Simple Momentum (Moving Average Crossover)
```yaml
type: "momentum"
```

**Parameters**:
- `fast_period` (int, 1-100, default: 10): Fast moving average period
- `slow_period` (int, 2-500, default: 20): Slow moving average period  
- `signal_threshold` (float, 0.0-1.0, default: 0.01): Minimum signal strength threshold
- `ma_type` (str, ["sma", "ema", "wma"], default: "sma"): Moving average type

**Signals Generated**: BUY when fast MA > slow MA, SELL when fast MA < slow MA

**Example**:
```yaml
strategies:
  - type: "momentum"
    params:
      fast_period: 12
      slow_period: 26
      signal_threshold: 0.02
      ma_type: "ema"
```

#### MACD Strategy
```yaml
type: "macd"
```

**Parameters**:
- `fast_period` (int, 1-50, default: 12): Fast EMA period
- `slow_period` (int, 2-100, default: 26): Slow EMA period
- `signal_period` (int, 1-50, default: 9): Signal line EMA period
- `histogram_threshold` (float, 0.0-1.0, default: 0.0): Histogram threshold for signals

**Example**:
```yaml
strategies:
  - type: "macd"
    params:
      fast_period: 12
      slow_period: 26
      signal_period: 9
      histogram_threshold: 0.1
```

#### Exponential Moving Average Momentum
```yaml
type: "ema_momentum"
```

**Parameters**:
- `fast_period` (int, 1-100, default: 12): Fast EMA period
- `slow_period` (int, 2-500, default: 26): Slow EMA period
- `smoothing_factor` (float, 0.01-0.5, default: 0.1): Additional smoothing
- `divergence_threshold` (float, 0.0-0.1, default: 0.01): Minimum divergence for signal

### Mean Reversion Strategies

#### RSI Mean Reversion
```yaml
type: "rsi_mean_reversion"
```

**Parameters**:
- `period` (int, 2-100, default: 14): RSI calculation period
- `oversold` (float, 10.0-40.0, default: 30): Oversold threshold for BUY
- `overbought` (float, 60.0-90.0, default: 70): Overbought threshold for SELL
- `exit_neutral` (float, 40.0-60.0, default: 50): Neutral zone for exits

**Example**:
```yaml
strategies:
  - type: "rsi_mean_reversion"
    params:
      period: 14
      oversold: 25
      overbought: 75
      exit_neutral: 50
```

#### Bollinger Bands
```yaml
type: "bollinger_bands"
```

**Parameters**:
- `period` (int, 5-100, default: 20): Moving average period
- `std_dev` (float, 0.5-5.0, default: 2.0): Standard deviation multiplier
- `squeeze_threshold` (float, 0.0-1.0, default: 0.1): Minimum band width for trading

#### Statistical Arbitrage
```yaml
type: "stat_arb"
```

**Parameters**:
- `lookback_period` (int, 20-252, default: 60): Historical lookback for statistics
- `entry_z_score` (float, 1.0-5.0, default: 2.0): Z-score threshold for entry
- `exit_z_score` (float, 0.0-2.0, default: 0.5): Z-score threshold for exit
- `correlation_threshold` (float, 0.5-1.0, default: 0.8): Minimum correlation for trading

### Breakout Strategies

#### Donchian Channel Breakout
```yaml
type: "donchian_breakout"
```

**Parameters**:
- `entry_period` (int, 5-100, default: 20): Period for breakout calculation
- `exit_period` (int, 2-50, default: 10): Period for exit calculation
- `atr_multiplier` (float, 0.5-5.0, default: 1.5): ATR multiplier for stop loss
- `volume_confirmation` (bool, default: false): Require volume confirmation

**Example**:
```yaml
strategies:
  - type: "donchian_breakout"
    params:
      entry_period: 20
      exit_period: 10
      atr_multiplier: 2.0
      volume_confirmation: true
```

#### Volume Breakout
```yaml
type: "volume_breakout"
```

**Parameters**:
- `volume_period` (int, 5-100, default: 20): Period for volume average
- `volume_threshold` (float, 1.0-10.0, default: 2.0): Volume multiple for breakout
- `price_threshold` (float, 0.001-0.1, default: 0.02): Minimum price move percentage

### Advanced Strategies

#### Pairs Trading
```yaml
type: "pairs_trading"
```

**Parameters**:
- `pair_symbols` (list[str], required): List of exactly 2 symbols to trade
- `lookback_period` (int, 30-252, default: 60): Period for cointegration test
- `entry_z_score` (float, 1.0-5.0, default: 2.0): Z-score for position entry
- `exit_z_score` (float, 0.0-2.0, default: 0.5): Z-score for position exit
- `hedge_ratio_method` (str, ["ols", "tls", "kalman"], default: "ols"): Hedge ratio calculation

**Example**:
```yaml
strategies:
  - type: "pairs_trading"
    params:
      pair_symbols: ["SPY", "QQQ"]
      lookback_period: 60
      entry_z_score: 2.0
      exit_z_score: 0.5
      hedge_ratio_method: "kalman"
```

#### Market Making
```yaml
type: "market_making"
```

**Parameters**:
- `spread_target` (float, 0.001-0.1, default: 0.01): Target bid-ask spread
- `inventory_limit` (int, 100-10000, default: 1000): Maximum inventory position
- `quote_size` (int, 1-1000, default: 100): Size of quotes
- `adverse_selection_protection` (bool, default: true): Enable adverse selection protection

#### Machine Learning Strategy
```yaml
type: "sklearn_model"
```

**Parameters**:
- `model_path` (str, required): Path to saved scikit-learn model
- `feature_columns` (list[str], required): List of feature column names
- `prediction_threshold` (float, 0.0-1.0, default: 0.6): Threshold for signal generation
- `retrain_frequency` (str, ["daily", "weekly", "monthly"], default: "monthly"): Model retraining frequency
- `preprocessing` (str, ["none", "standard_scaler", "min_max_scaler"], default: "standard_scaler"): Feature preprocessing

**Example**:
```yaml
strategies:
  - type: "sklearn_model"
    params:
      model_path: "models/random_forest.pkl"
      feature_columns: ["rsi", "macd", "bb_position", "volume_ratio"]
      prediction_threshold: 0.65
      retrain_frequency: "weekly"
      preprocessing: "standard_scaler"
```

## 🛡️ Risk Management Components

### Position Sizing

#### Fixed Percentage Sizing
```yaml
type: "fixed_percentage"
```

**Parameters**:
- `position_size_pct` (float, 0.001-1.0, default: 0.02): Fixed percentage per position
- `max_positions` (int, 1-100, default: 10): Maximum concurrent positions
- `currency_adjustment` (bool, default: false): Adjust for currency differences

#### Volatility-Based Sizing
```yaml
type: "volatility_based"
```

**Parameters**:
- `target_volatility` (float, 0.05-1.0, default: 0.15): Target portfolio volatility
- `lookback_days` (int, 10-252, default: 20): Days for volatility calculation
- `min_position_size` (float, 0.001-0.1, default: 0.01): Minimum position size
- `max_position_size` (float, 0.01-1.0, default: 0.1): Maximum position size
- `volatility_floor` (float, 0.01-0.5, default: 0.05): Minimum volatility assumption

**Example**:
```yaml
risk_management:
  type: "volatility_based"
  params:
    target_volatility: 0.12
    lookback_days: 30
    min_position_size: 0.01
    max_position_size: 0.08
```

#### Kelly Criterion Sizing
```yaml
type: "kelly_criterion"
```

**Parameters**:
- `lookback_trades` (int, 30-1000, default: 100): Number of trades for Kelly calculation
- `kelly_fraction` (float, 0.1-1.0, default: 0.25): Fraction of full Kelly to use
- `max_position_size` (float, 0.01-1.0, default: 0.1): Hard cap on position size
- `min_win_rate` (float, 0.3-0.7, default: 0.4): Minimum win rate for Kelly calculation

### Risk Limits

#### Portfolio Risk Limits
```yaml
type: "portfolio_limits"
```

**Parameters**:
- `max_gross_exposure` (float, 0.1-5.0, default: 1.0): Maximum gross exposure
- `max_net_exposure` (float, 0.1-2.0, default: 0.8): Maximum net exposure
- `max_drawdown_pct` (float, 0.05-0.5, default: 0.15): Maximum drawdown before action
- `max_correlation` (float, 0.3-1.0, default: 0.8): Maximum correlation between positions
- `sector_limits` (dict, optional): Per-sector exposure limits

**Example**:
```yaml
risk_management:
  type: "portfolio_limits"
  params:
    max_gross_exposure: 0.95
    max_net_exposure: 0.8
    max_drawdown_pct: 0.12
    max_correlation: 0.7
    sector_limits:
      technology: 0.3
      financials: 0.2
      healthcare: 0.2
```

#### Value at Risk (VaR)
```yaml
type: "var_based"
```

**Parameters**:
- `var_method` (str, ["historical", "parametric", "monte_carlo"], default: "historical"): VaR calculation method
- `confidence_level` (float, 0.9-0.99, default: 0.95): Confidence level for VaR
- `holding_period` (int, 1-30, default: 1): Holding period in days
- `lookback_days` (int, 30-1000, default: 252): Historical data period
- `max_portfolio_var` (float, 0.01-0.2, default: 0.02): Maximum daily portfolio VaR

## 📊 Data Components

### Data Sources

#### CSV Data Source
```yaml
type: "csv"
```

**Parameters**:
- `path` (str, required): Path to CSV file
- `symbol_column` (str, default: "symbol"): Column name for symbol
- `timestamp_column` (str, default: "timestamp"): Column name for timestamp
- `price_columns` (dict, default: standard OHLCV): Mapping of price column names
- `date_format` (str, default: "%Y-%m-%d %H:%M:%S"): Timestamp format
- `timezone` (str, default: "UTC"): Timezone for timestamps

**Example**:
```yaml
data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
    symbol_column: "ticker"
    timestamp_column: "datetime"
    date_format: "%Y-%m-%d %H:%M:%S"
    timezone: "US/Eastern"
```

#### Database Source
```yaml
type: "database"
```

**Parameters**:
- `connection_string` (str, required): Database connection string
- `table` (str, required): Table name
- `symbol_column` (str, default: "symbol"): Symbol column name
- `timestamp_column` (str, default: "timestamp"): Timestamp column name
- `query_template` (str, optional): Custom query template
- `connection_pool_size` (int, 1-100, default: 5): Connection pool size

#### Live Data Source
```yaml
type: "live"
```

**Parameters**:
- `provider` (str, ["alpaca", "interactive_brokers", "polygon"], required): Data provider
- `api_credentials` (dict, required): API credentials
- `subscription_level` (str, ["basic", "premium"], default: "basic"): Data subscription level
- `buffer_size` (int, 100-10000, default: 1000): Internal buffer size
- `reconnect_attempts` (int, 1-10, default: 3): Auto-reconnection attempts

### Data Processors

#### Data Cleaning
```yaml
type: "data_cleaner"
```

**Parameters**:
- `fill_missing` (str, ["forward_fill", "backward_fill", "interpolate", "drop"], default: "forward_fill"): Missing data handling
- `remove_outliers` (bool, default: false): Enable outlier removal
- `outlier_threshold` (float, 1.0-5.0, default: 3.0): Standard deviations for outlier detection
- `min_volume_threshold` (int, 0-1000000, default: 0): Minimum volume filter

#### Feature Engineering
```yaml
type: "feature_engineer"
```

**Parameters**:
- `technical_indicators` (list[str], default: []): List of technical indicators to compute
- `rolling_windows` (list[int], default: [5, 10, 20]): Rolling window periods
- `lag_features` (list[int], default: [1, 2, 5]): Lag periods for features
- `normalization` (str, ["none", "z_score", "min_max"], default: "none"): Feature normalization

## ⚙️ Execution Components

### Order Execution

#### Market Order Executor
```yaml
type: "market_order"
```

**Parameters**:
- `slippage_bps` (float, 0-100, default: 10): Expected slippage in basis points
- `execution_delay_seconds` (int, 0-300, default: 0): Delay before order execution
- `partial_fill_handling` (str, ["accept", "reject", "wait"], default: "accept"): Partial fill behavior
- `market_impact_model` (str, ["linear", "sqrt", "none"], default: "linear"): Market impact calculation

#### Limit Order Executor
```yaml
type: "limit_order"
```

**Parameters**:
- `price_offset_bps` (float, -50-50, default: 5): Price offset from current price
- `timeout_seconds` (int, 30-3600, default: 300): Order timeout
- `price_improvement` (bool, default: true): Allow price improvement
- `hidden_quantity_pct` (float, 0.0-1.0, default: 0.0): Hidden quantity percentage

#### Smart Order Router
```yaml
type: "smart_router"
```

**Parameters**:
- `venues` (list[str], required): List of execution venues
- `routing_algorithm` (str, ["price", "liquidity", "speed"], default: "price"): Routing priority
- `venue_weights` (dict, optional): Custom venue weights
- `dark_pool_preference` (float, 0.0-1.0, default: 0.3): Dark pool usage preference

### Brokers

#### Alpaca Broker
```yaml
type: "alpaca"
```

**Parameters**:
- `api_key` (str, required): Alpaca API key
- `secret_key` (str, required): Alpaca secret key
- `base_url` (str, required): Alpaca API base URL
- `data_feed` (str, ["iex", "sip"], default: "iex"): Market data feed
- `paper_trading` (bool, default: true): Enable paper trading mode

**Example**:
```yaml
execution:
  broker:
    type: "alpaca"
    params:
      api_key: "${ALPACA_API_KEY}"
      secret_key: "${ALPACA_SECRET_KEY}"
      base_url: "https://paper-api.alpaca.markets"
      data_feed: "iex"
      paper_trading: true
```

#### Interactive Brokers
```yaml
type: "interactive_brokers"
```

**Parameters**:
- `host` (str, default: "127.0.0.1"): TWS/Gateway host
- `port` (int, default: 7497): TWS/Gateway port
- `client_id` (int, 1-32, default: 1): Client ID
- `account` (str, optional): Account number
- `currency` (str, default: "USD"): Base currency

## 📈 Analysis Components

### Performance Analyzers

#### Basic Performance Analyzer
```yaml
type: "basic_performance"
```

**Parameters**:
- `benchmark_symbol` (str, default: "SPY"): Benchmark for comparison
- `risk_free_rate` (float, 0.0-0.1, default: 0.02): Risk-free rate for calculations
- `confidence_levels` (list[float], default: [0.95, 0.99]): Confidence levels for VaR

#### Drawdown Analyzer
```yaml
type: "drawdown_analysis"
```

**Parameters**:
- `recovery_threshold` (float, 0.8-1.0, default: 0.95): Recovery percentage threshold
- `underwater_threshold_days` (int, 5-100, default: 30): Long underwater period threshold
- `peak_to_trough_analysis` (bool, default: true): Enable peak-to-trough analysis

#### Factor Attribution Analyzer
```yaml
type: "factor_attribution"
```

**Parameters**:
- `factor_model` (str, ["fama_french_3", "fama_french_5", "custom"], default: "fama_french_3"): Factor model
- `factor_data_source` (str, optional): External factor data source
- `rolling_window_days` (int, 30-252, default: 63): Rolling attribution window
- `attribution_frequency` (str, ["daily", "weekly", "monthly"], default: "monthly"): Attribution frequency

### Risk Analyzers

#### VaR Analyzer
```yaml
type: "var_analysis"
```

**Parameters**:
- `var_methods` (list[str], default: ["historical", "parametric"]): VaR calculation methods
- `confidence_levels` (list[float], default: [0.95, 0.99, 0.999]): Confidence levels
- `holding_periods` (list[int], default: [1, 5, 10]): Holding periods in days
- `bootstrap_iterations` (int, 100-10000, default: 1000): Bootstrap iterations

#### Stress Testing Analyzer
```yaml
type: "stress_testing"
```

**Parameters**:
- `stress_scenarios` (list[dict], required): List of stress scenarios
- `scenario_probability` (float, 0.01-0.1, default: 0.05): Scenario probability assumption
- `recovery_analysis` (bool, default: true): Include recovery time analysis
- `correlation_breakdown` (bool, default: true): Analyze correlation breakdown

## 🔧 Infrastructure Components

### Container Types

#### Full Backtest Container
```yaml
type: "full_backtest"
```

**Capabilities**:
- Complete strategy execution
- All risk management features
- Full order simulation
- Comprehensive logging

**Resource Usage**:
- Memory: 50-200MB per container
- CPU: 0.1-1.0 cores per container

#### Signal Replay Container
```yaml
type: "signal_replay"
```

**Capabilities**:
- 10-100x faster than full backtest
- Risk management re-application
- Portfolio tracking
- Signal-based optimization

**Resource Usage**:
- Memory: 20-50MB per container
- CPU: 0.05-0.2 cores per container

#### Signal Generation Container
```yaml
type: "signal_generation"
```

**Capabilities**:
- Pure signal capture
- Strategy analysis
- Feature engineering
- Signal compression

**Resource Usage**:
- Memory: 30-100MB per container
- CPU: 0.1-0.5 cores per container

### Event Adapters

#### Pipeline Adapter
```yaml
type: "pipeline"
```

**Parameters**:
- `containers` (list[str], required): Ordered list of containers
- `buffer_size` (int, 100-10000, default: 1000): Event buffer size
- `timeout_seconds` (int, 1-300, default: 30): Event timeout
- `error_handling` (str, ["propagate", "skip", "retry"], default: "propagate"): Error handling strategy

#### Broadcast Adapter
```yaml
type: "broadcast"
```

**Parameters**:
- `source` (str, required): Source container name
- `targets` (list[str], required): Target container names
- `event_filters` (dict, optional): Per-target event filters
- `delivery_guarantee` (str, ["at_most_once", "at_least_once"], default: "at_most_once"): Delivery guarantee

#### Hierarchical Adapter
```yaml
type: "hierarchical"
```

**Parameters**:
- `parent` (str, required): Parent container name
- `children` (list[str], required): Child container names
- `context_propagation` (bool, default: true): Enable context propagation
- `aggregation_method` (str, ["sum", "average", "weighted"], default: "weighted"): Result aggregation

## 🤔 Common Questions

**Q: How do I know which component type to use?**
A: Check the component description and parameters. Start with basic types and move to advanced as needed.

**Q: Can I mix different component types?**
A: Yes! ADMF-PC's Protocol + Composition architecture allows mixing any compatible components.

**Q: Where do I find the exact parameter ranges?**
A: Each parameter listing includes the valid range. Values outside these ranges will cause validation errors.

**Q: How do I create custom components?**
A: See [Advanced Topics](../08-advanced-topics/custom-components.md) for custom component development.

## 📝 Usage Notes

### Parameter Validation

All parameters are validated at configuration time:
- Type checking (int, float, str, bool, list, dict)
- Range validation (min/max values)
- Option validation (valid choices)
- Required field checking

### Default Behavior

When parameters are omitted:
- Default values are automatically used
- Defaults are chosen for typical use cases
- Override defaults for specialized needs

### Performance Considerations

Component choice affects performance:
- **Signal Replay**: 10-100x faster for optimization
- **Volatility-Based Sizing**: More computation than fixed sizing
- **Live Data Sources**: Network latency considerations
- **Complex Analyzers**: Higher memory usage

---

Continue to [Configuration Schema](configuration-schema.md) for complete YAML reference →