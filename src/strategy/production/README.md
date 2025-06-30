# Production Strategies

This directory contains strategies that have been validated and are ready for paper/live trading.

## bollinger_rsi_simple_signals

**Status**: Ready for paper trading
**Validated**: 2025-06-20

### Strategy Overview
- **Type**: Mean reversion
- **Entry Logic**: Price at Bollinger Band extremes with RSI non-confirmation
  - Long: Price < lower band AND RSI > 40 (shows strength despite price weakness)
  - Short: Price > upper band AND RSI < 60 (shows weakness despite price strength)
- **Exit Logic**: Price returns to middle 20% of Bollinger Bands (40-60% position)
- **Signal Type**: Generates -1, 0, 1 signals (short, flat, long)

### Performance Metrics (Test Dataset)
- **Total trades**: 196 over ~262 trading days
- **Trades per day**: 0.75
- **Win rate**: 70.41%
- **Average return per trade**: 0.0145% (before costs)
- **Average holding period**: 10.7 bars (~11 minutes)
- **Time in market**: ~50% (other 50% is flat)

### Key Characteristics
- Completely stateless - no position tracking
- Quick mean reversion trades
- Clear entry/exit zones based on current conditions only
- No complex state or confirmation requirements

### Risk Considerations
- High frequency of trades means transaction costs are critical
- Performance assumes good fills at signal prices
- Should implement position sizing based on volatility
- Consider filtering during major news events

### Configuration
See `config/production/bollinger_rsi_simple_signals.yaml`

Default parameters:
- Bollinger Bands: 20 period, 2.0 std dev
- RSI: 14 period
- RSI threshold: 10 (RSI must be 10+ points from extremes)

### Important Note
**DO NOT MODIFY** this strategy file. Any improvements should be made as a new strategy version.