# Indicator Strategy Test Summary

## Test Structure

Each test configuration:
- Uses single strategy with fixed parameters
- Tests on 1 month of SPY data
- Can be run with `--bars 100` for quick testing
- Verifies signal generation logic

## Running Tests

```bash
# Run all tests
./config/indicators/run_tests.sh

# Run single test
python main.py --config config/indicators/crossover/test_sma_crossover.yaml --signal-generation --bars 100

# Run with verbose signals to see feature values
python main.py --config config/indicators/oscillator/test_rsi_bands.yaml --signal-generation --bars 100 --verbose-signals
```

## Test Configurations by Category

### Crossover Strategies (Trend Following)
- `crossover/test_sma_crossover.yaml` - Simple MA crossover
- `crossover/test_ema_crossover.yaml` - Exponential MA crossover  
- `crossover/test_ema_sma_crossover.yaml` - EMA vs SMA crossover
- `crossover/test_dema_sma_crossover.yaml` - DEMA vs SMA crossover
- `crossover/test_dema_crossover.yaml` - Double EMA fast/slow crossover
- `crossover/test_tema_sma_crossover.yaml` - TEMA vs SMA crossover
- `crossover/test_macd_crossover.yaml` - MACD vs signal line
- `crossover/test_stochastic_crossover.yaml` - Stochastic %K vs %D
- `crossover/test_vortex_crossover.yaml` - Vortex VI+ vs VI-
- `crossover/test_ichimoku_cloud_position.yaml` - Price vs Ichimoku cloud

### Oscillator Strategies
- `oscillator/test_rsi_bands.yaml` - RSI overbought/oversold (mean reversion)
- `oscillator/test_rsi_threshold.yaml` - RSI directional (momentum)
- `oscillator/test_williams_r.yaml` - Williams %R extremes
- `oscillator/test_cci_threshold.yaml` - CCI zero-line crossover (momentum)
- `oscillator/test_cci_bands.yaml` - CCI extreme bands (mean reversion)
- `oscillator/test_stochastic_rsi.yaml` - Stochastic RSI extremes (mean reversion)
- `oscillator/test_roc_threshold.yaml` - ROC momentum threshold
- `oscillator/test_ultimate_oscillator.yaml` - Ultimate Oscillator extremes (mean reversion)

### Volatility Strategies
- `volatility/test_bollinger_bands.yaml` - Bollinger mean reversion
- `volatility/test_bollinger_breakout.yaml` - Bollinger breakout (trend)
- `volatility/test_keltner_bands.yaml` - Keltner mean reversion
- `volatility/test_keltner_bands_filtered.yaml` - Keltner with volatility filter
- `volatility/test_keltner_breakout.yaml` - Keltner breakout (trend)
- `volatility/test_donchian_bands.yaml` - Donchian mean reversion
- `volatility/test_donchian_breakout.yaml` - Donchian breakout (trend)

### Volume Strategies
- `volume/test_vwap_deviation.yaml` - VWAP mean reversion
- `volume/test_obv_trend.yaml` - OBV vs moving average (trend)
- `volume/test_mfi_bands.yaml` - MFI overbought/oversold (mean reversion)
- `volume/test_chaikin_money_flow.yaml` - CMF buying/selling pressure
- `volume/test_accumulation_distribution.yaml` - A/D line vs EMA (trend)

### Structure Strategies
- `structure/test_pivot_points.yaml` - Pivot point breakouts (1m timeframe)
- `structure/test_pivot_bounces.yaml` - Pivot point bounces (mean reversion)
- `structure/test_swing_pivot_breakout.yaml` - Swing-based channel breakouts
- `structure/test_swing_pivot_bounce.yaml` - Dynamic S/R bounces
- `structure/test_trendline_breaks.yaml` - Trendline breakout strategy
- `structure/test_trendline_bounces.yaml` - Trendline bounce strategy
- `structure/test_trendline_bounces_filtered.yaml` - Trendline bounces with bounce count filter
- `structure/test_trendline_acceleration.yaml` - Trendline breaks filtered by slope/branches
- `structure/test_diagonal_channel_reversion.yaml` - Diagonal channel mean reversion (configurable)
- `structure/test_diagonal_channel_breakout.yaml` - Diagonal channel breakout with sloped exits

### Trend Strategies
- `trend/test_parabolic_sar.yaml` - Parabolic SAR trend following
- `trend/test_aroon_crossover.yaml` - Aroon Up/Down crossover
- `trend/test_supertrend.yaml` - Supertrend with ATR bands
- `trend/test_linear_regression_slope.yaml` - Linear regression trend direction

### Momentum Strategies
- `momentum/test_macd_crossover.yaml` - MACD histogram crossover
- `momentum/test_momentum_breakout.yaml` - Rate of change momentum breakout
- `momentum/test_roc_trend.yaml` - ROC trend acceleration
- `momentum/test_adx_trend_strength.yaml` - ADX with directional indicators
- `momentum/test_aroon_oscillator.yaml` - Aroon oscillator trend strength
- `momentum/test_vortex_trend.yaml` - Vortex indicator trend reversals
- `momentum/test_elder_ray.yaml` - Elder Ray bull/bear power analysis

### Divergence Strategies (Reversal)
- `divergence/test_rsi_divergence.yaml` - RSI vs price divergence at swings
- `divergence/test_macd_histogram_divergence.yaml` - MACD histogram divergence
- `divergence/test_stochastic_divergence.yaml` - Stochastic divergence in extreme zones
- `divergence/test_momentum_divergence.yaml` - Momentum (ROC) divergence
- `divergence/test_obv_price_divergence.yaml` - OBV accumulation/distribution divergence

## Signal Logic Summary

### Mean Reversion Strategies
These expect price to revert to average:
- `bollinger_bands`: Sell at upper band, buy at lower band
- `keltner_bands`: Sell at upper channel, buy at lower channel
- `donchian_bands`: Sell at period high, buy at period low
- `vwap_deviation`: Sell above VWAP+band, buy below VWAP-band
- `rsi_bands`: Buy when RSI<30, sell when RSI>70

### Trend Following Strategies
These trade in direction of breakout:
- `bollinger_breakout`: Buy above upper band, sell below lower band
- `keltner_breakout`: Buy above upper channel, sell below lower channel
- `donchian_breakout`: Buy at new highs, sell at new lows
- `sma_crossover`: Buy when fast>slow, sell when fast<slow
- `pivot_points`: Buy above R1, sell below S1

### Compositional Examples

```yaml
# Combining strategies
strategy: [
  {bollinger_bands: {weight: 0.5}},
  {rsi_bands: {weight: 0.5}}
]

# Adding filters
strategy:
  keltner_bands:
    filter: "volume > sma(20) and atr(14) / price < 0.02"

# Using bounce counts (NEW)
strategy:
  trendline_bounces:
    filter: "trendlines_support_bounces >= 3 or trendlines_resistance_bounces >= 3"

# Conditional strategies
strategy:
  condition: volatility_regime(20) == 'high'
  bollinger_breakout: {}
```

### Enhanced Features

**Trendline Advanced Metrics**: The trendlines feature now provides:
- `trendlines_support_bounces`: Count of successful bounces from support lines
- `trendlines_resistance_bounces`: Count of successful bounces from resistance lines
- `trendlines_support_slope`: Slope of nearest support trendline
- `trendlines_resistance_slope`: Slope of nearest resistance trendline
- `trendlines_support_angle`: Angle in degrees (positive = upward)
- `trendlines_resistance_angle`: Angle in degrees (negative = downward)
- `trendlines_uptrend_branches`: Count of steeper uptrends branching off
- `trendlines_downtrend_branches`: Count of steeper downtrends branching off

**Diagonal Channel Tracking**: New diagonal_channel feature provides:
- `diagonal_channel_upper_bounces`: Bounces from upper channel
- `diagonal_channel_lower_bounces`: Bounces from lower channel
- `diagonal_channel_position_in_channel`: 0-1 scale position
- `diagonal_channel_channel_angle`: Channel angle in degrees
- `diagonal_channel_channel_strength`: Based on touches and validity

These enable sophisticated filtering like:
```yaml
# Trade steep uptrends with acceleration
filter: "trendlines_support_angle > 20 and trendlines_uptrend_branches > 0"

# Trade strong channels with proven bounces
filter: "diagonal_channel_lower_bounces >= 3 and diagonal_channel_channel_strength > 0.8"
```