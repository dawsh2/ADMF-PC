# Tick-Level Streaming with ADMF-PC

## Overview

The ADMF-PC system now supports tick-level (trade) data streaming through Alpaca's WebSocket API!

## Usage

To stream tick data, use the special timeframe `tick` in your data configuration:

```yaml
# config/bollinger/test_tick.yaml
name: bollinger_tick
data: BTCUSD_tick  # Use _tick suffix for tick data

strategy: [
  {
    bollinger_bands: {
      period: 10,
      std_dev: 1.5
    },
    constraints: "intraday"
  }
]
```

Then run with the --alpaca flag:
```bash
python main.py --config config/bollinger/test_tick.yaml --alpaca
```

## How It Works

1. **Timeframe Detection**: When the system sees `_tick` in the data string, it:
   - Sets timeframe to "tick"
   - Subscribes to trades instead of bars
   - Processes trade messages (type "t") instead of bar messages (type "b")

2. **Trade to Bar Conversion**: Each tick/trade is converted to a bar format where:
   - open = high = low = close = trade price
   - volume = trade size
   - timestamp = trade timestamp

3. **Strategy Processing**: Strategies process ticks as if they were bars:
   - Bollinger Bands on 10 ticks = bands calculated on last 10 trades
   - Moving averages on ticks = average of last N trade prices

## Available Tick Symbols

### Crypto (24/7)
- BTCUSD_tick
- ETHUSD_tick
- LTCUSD_tick
- BCHUSD_tick

### Stocks (Market Hours)
- SPY_tick
- AAPL_tick
- TSLA_tick
- Any stock symbol with _tick suffix

## Key Differences: Bars vs Ticks

| Feature | Bar Data | Tick Data |
|---------|----------|-----------|
| Frequency | Fixed intervals (1m, 5m, etc) | Every trade |
| Data Rate | ~1 per minute | 10-1000+ per minute |
| OHLC | Real OHLC values | All prices = trade price |
| Volume | Bar volume | Individual trade size |
| Use Case | Technical analysis | High-frequency trading |

## Example Output

When streaming ticks, you'll see:
```
Successfully subscribed to trades: ['BTCUSD']
Received tick: BTCUSD 2025-06-26 15:35:22.123456 Price:65432.10 Size:0.15
Received tick: BTCUSD 2025-06-26 15:35:22.456789 Price:65431.50 Size:0.08
Received tick: BTCUSD 2025-06-26 15:35:22.789012 Price:65433.25 Size:0.25
```

## Performance Considerations

1. **Data Volume**: Tick data is MUCH higher volume than bars
   - SPY can have 1000+ ticks per minute during active trading
   - Crypto typically has fewer ticks but still significant

2. **Processing Speed**: Strategies must be fast enough to process each tick
   - Avoid complex calculations on every tick
   - Consider sampling or aggregating ticks

3. **Connection Limits**: Alpaca limits concurrent WebSocket connections
   - Close other connections before starting tick streaming
   - Error: "connection limit exceeded" means too many connections

## Strategy Adaptation

Traditional strategies need adaptation for tick data:

### Moving Average on Ticks
```python
# Instead of 20-period SMA on 1-minute bars (20 minutes of data)
# Use 1200-tick SMA (approximately 20 minutes at 60 ticks/minute)
```

### Bollinger Bands on Ticks
```python
# Instead of 20-period bands on bars
# Use 100-tick bands for similar time coverage
```

## Advanced Usage

### Mixed Timeframes
You could run multiple data handlers:
- One streaming 5m bars for trend detection
- One streaming ticks for precise entry/exit

### Tick Aggregation
Future enhancement: aggregate ticks into custom bars:
- Volume bars (every 1000 shares)
- Dollar bars (every $100k traded)
- Tick bars (every 100 ticks)

## Troubleshooting

### "Connection limit exceeded"
- Close any other Alpaca WebSocket connections
- Check for zombie processes: `ps aux | grep python`
- Wait 1-2 minutes and retry

### No ticks received
- Ensure market is open (or use crypto for 24/7)
- Check symbol is valid and active
- Verify API permissions include real-time data

### High CPU usage
- Tick data is intensive - monitor system resources
- Consider filtering or sampling ticks
- Optimize strategy calculations

## Summary

Tick streaming enables high-frequency trading strategies in ADMF-PC:
- Use `SYMBOL_tick` format in data configuration
- System automatically subscribes to trades instead of bars
- Each tick is processed through the same event pipeline
- Strategies see ticks as "bars" with identical OHLC prices

This opens up possibilities for:
- Market microstructure analysis
- High-frequency trading strategies
- Precise entry/exit timing
- Order flow analysis