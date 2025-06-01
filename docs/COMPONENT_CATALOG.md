# ADMF-PC Component Catalog

This document provides a comprehensive reference of all available components in the ADMF-PC system. These components implement standardized protocols, enabling seamless integration regardless of their internal complexity.

## Trading Strategies

| Component | Configuration Type | Purpose | Key Parameters |
|-----------|-------------------|---------|----------------|
| `momentum` | Built-in | Trend-following using moving average crossovers | `fast_period`, `slow_period`, `signal_threshold` |
| `mean_reversion` | Built-in | Counter-trend trading on price deviations | `lookback_period`, `std_threshold`, `hold_period` |
| `breakout` | Built-in | Momentum trading on price breakouts | `breakout_period`, `volume_threshold`, `atr_multiplier` |
| `pairs_trading` | Built-in | Statistical arbitrage between correlated assets | `symbol_1`, `symbol_2`, `entry_threshold`, `exit_threshold` |
| `ensemble` | Built-in | Combines multiple strategies with weighted voting | `strategies`, `weights`, `combination_method` |
| `regime_adaptive` | Built-in | Switches parameters based on market regime | `regime_parameters`, `switching_mode` |

## Technical Indicators

| Component | Type | Purpose | Common Parameters |
|-----------|------|---------|------------------|
| `sma` | Moving Average | Simple moving average calculation | `period` |
| `ema` | Moving Average | Exponential moving average with decay | `period`, `alpha` |
| `rsi` | Momentum | Relative strength index oscillator | `period`, `overbought`, `oversold` |
| `macd` | Momentum | Moving average convergence divergence | `fast_period`, `slow_period`, `signal_period` |
| `atr` | Volatility | Average true range for volatility measurement | `period` |
| `bollinger_bands` | Volatility | Price channels with standard deviation bands | `period`, `std_dev_multiplier` |
| `custom_indicator` | Function | User-defined calculation function | `function`, `parameters` |

## Market Regime Classifiers

| Component | Type | Purpose | Configuration |
|-----------|------|---------|---------------|
| `hmm_classifier` | Statistical | Hidden Markov Model for regime detection | `n_states`, `features`, `regime_labels` |
| `pattern_classifier` | Technical | Pattern recognition for market states | `patterns`, `lookback_period`, `confirmation_bars` |
| `volatility_regime` | Statistical | Volatility-based regime classification | `short_window`, `long_window`, `thresholds` |
| `trend_regime` | Technical | Trend-based market state detection | `trend_window`, `trend_threshold`, `noise_filter` |
| `ml_classifier` | Machine Learning | Custom ML model for regime detection | `model_class`, `features`, `training_config` |

## Risk Management Components

| Component | Type | Purpose | Parameters |
|-----------|------|---------|------------|
| `fixed_fraction` | Position Sizing | Fixed percentage of capital per trade | `position_size_pct`, `max_positions` |
| `volatility_based` | Position Sizing | Size based on asset volatility | `risk_per_trade`, `lookback_period` |
| `kelly_criterion` | Position Sizing | Optimal position sizing using Kelly formula | `win_rate`, `avg_win`, `avg_loss` |
| `var_risk` | Risk Limit | Value-at-Risk based position limits | `confidence_level`, `lookback_period` |
| `drawdown_limit` | Risk Limit | Maximum drawdown circuit breaker | `max_drawdown_pct`, `stop_trading` |
| `exposure_limit` | Risk Limit | Maximum portfolio exposure controls | `max_gross_exposure`, `max_net_exposure` |

## Data Sources and Handlers

| Component | Type | Purpose | Configuration |
|-----------|------|---------|---------------|
| `csv_data` | Historical | Load data from CSV files | `file_path`, `date_format`, `symbols` |
| `database_data` | Historical | Load data from database connections | `connection_string`, `query`, `table` |
| `live_data` | Real-time | Stream live market data | `provider`, `api_credentials`, `symbols` |
| `alternative_data` | External | Integration with alternative data sources | `provider`, `data_type`, `update_frequency` |

## Execution and Portfolio Components

| Component | Type | Purpose | Configuration |
|-----------|------|---------|---------------|
| `simulated_execution` | Backtest | Simulated order execution for backtesting | `slippage`, `commission`, `market_impact` |
| `live_execution` | Trading | Real broker integration for live trading | `broker`, `account_id`, `order_types` |
| `portfolio_tracker` | Portfolio | Track positions, performance, and attribution | `initial_capital`, `benchmark`, `metrics` |
| `performance_analyzer` | Analysis | Calculate trading performance metrics | `metrics`, `benchmark`, `reporting_frequency` |

## Advanced Components

| Component | Type | Purpose | Use Cases |
|-----------|------|---------|-----------|
| `signal_generator` | Analysis | Pure signal generation without execution | Signal quality research, MAE/MFE analysis |
| `signal_replayer` | Optimization | Replay pre-generated signals for ensemble optimization | Fast ensemble weight optimization |
| `walk_forward_validator` | Validation | Rolling window out-of-sample testing | Robustness testing, parameter stability |
| `monte_carlo_simulator` | Analysis | Statistical simulation of trading outcomes | Risk analysis, scenario testing |

## Integration Components

| Component | Type | Purpose | Examples |
|-----------|------|---------|----------|
| `sklearn_model` | ML Integration | Scikit-learn model wrapper | `RandomForestClassifier`, `SVM`, `LogisticRegression` |
| `tensorflow_model` | ML Integration | TensorFlow model integration | Neural networks, deep learning models |
| `zipline_strategy` | External | Import strategies from Zipline | Existing Zipline algorithm migration |
| `custom_function` | Function | Wrap any Python function as component | Custom calculations, external libraries |

## Component Enhancement Capabilities

Any component can be enhanced with additional capabilities without modifying its core implementation:

| Capability | Purpose | Configuration |
|------------|---------|---------------|
| `logging` | Structured logging with correlation tracking | `log_level`, `trace_methods`, `correlation_context` |
| `monitoring` | Performance metrics and health checks | `track_performance`, `health_checks`, `alert_thresholds` |
| `error_handling` | Robust error boundaries and retry logic | `retry_policy`, `fallback_strategy`, `error_boundaries` |
| `optimization` | Automatic parameter optimization support | `parameter_space`, `constraints`, `validation_rules` |
| `validation` | State and configuration validation | `validation_rules`, `lifecycle_checks`, `config_schema` |

## Usage Examples

```yaml
# Mix any component types seamlessly
strategies:
  - type: "momentum"           # Built-in strategy
    fast_period: 10
    slow_period: 30
    
  - type: "sklearn_model"      # ML model integration
    model: "RandomForestClassifier"
    features: ["rsi", "macd", "volume_ratio"]
    
  - type: "custom_function"    # Custom function
    function: "my_custom_strategy"
    parameters: {lookback: 20}
    
  - type: "zipline_strategy"   # External library
    algorithm: "MeanReversion"
    import_path: "zipline.examples"
```

This component library demonstrates the system's composability—any component implementing the appropriate protocols can be mixed with any other component, enabling unlimited strategy development flexibility.
