# Indicator Strategy Testing

This directory contains test configurations for each indicator strategy in `src/strategy/strategies/indicators/`.

## Testing Approach

Each test configuration:
1. Uses a single strategy with fixed parameters (no optimization)
2. Runs on limited data (--bars 100 or one month)
3. Generates signals only (no backtesting)
4. Verifies signal logic matches feature values

## Running Tests

```bash
# Test a single indicator
python main.py --config config/indicators/test_rsi_bands.yaml --signal-generation --bars 100

# Test with verbose output to see feature values
python main.py --config config/indicators/test_rsi_bands.yaml --signal-generation --bars 100 --verbose-signals

# Test with specific date range
python main.py --config config/indicators/test_sma_crossover.yaml --signal-generation --dataset test
```

## Verification Checklist

For each strategy test:
- [ ] Signal generation matches documented logic
- [ ] Feature values are correctly computed
- [ ] Signal storage includes accurate price data
- [ ] No errors or warnings during execution
- [ ] Signals change appropriately with market conditions

## Test Files

### Crossover Strategies
- `test_sma_crossover.yaml` - Simple moving average crossover
- `test_ema_crossover.yaml` - Exponential moving average crossover
- `test_macd_crossover.yaml` - MACD line vs signal line crossover
- `test_stochastic_crossover.yaml` - Stochastic %K vs %D crossover

### Oscillator Strategies
- `test_rsi_bands.yaml` - RSI overbought/oversold bands
- `test_rsi_threshold.yaml` - RSI single threshold
- `test_cci_bands.yaml` - CCI overbought/oversold bands
- `test_williams_r.yaml` - Williams %R oscillator

### Volatility Strategies
- `test_bollinger_breakout.yaml` - Bollinger band breakout
- `test_keltner_breakout.yaml` - Keltner channel breakout
- `test_donchian_breakout.yaml` - Donchian channel breakout

### Volume Strategies
- `test_vwap_deviation.yaml` - VWAP with standard deviation bands
- `test_obv_trend.yaml` - On-balance volume trend
- `test_mfi_bands.yaml` - Money flow index bands

### Structure Strategies
- `test_pivot_points.yaml` - Pivot point support/resistance
- `test_trendline_breaks.yaml` - Trendline breakout detection
- `test_support_resistance_breakout.yaml` - S/R level breakouts

## Signal Logic Reference

See each test file for the expected signal generation logic and verification steps.