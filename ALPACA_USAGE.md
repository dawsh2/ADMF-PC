# Using ADMF-PC with Alpaca Live Data

## Quick Start

### 1. Install Dependencies
```bash
pip install websockets alpaca-trade-api
```

### 2. Set API Credentials
```bash
export ALPACA_API_KEY='your_api_key_here'
export ALPACA_SECRET_KEY='your_secret_key_here'
```

### 3. Run with --alpaca Flag
```bash
# Use any strategy config with live data
python main.py --config config/bollinger/config.yaml --alpaca

# Limit to 100 bars for testing
python main.py --config config/bollinger/config.yaml --alpaca --bars 100

# Run without config file
python main.py --strategies "ma_crossover:fast_period=5,slow_period=20" --alpaca
```

## What --alpaca Does

When you add the `--alpaca` flag:

1. **Overrides Data Source**: Replaces any file-based data with Alpaca WebSocket streaming
2. **Uses Universal Topology**: Automatically uses the complete trading pipeline (signals → portfolio → execution)
3. **Extracts Symbols**: 
   - From data config (e.g., "SPY_5m" → ["SPY"])
   - From strategies if specified
   - Defaults to ["SPY"] if none found
4. **Configures Live Trading**:
   - Uses paper trading API for safety
   - IEX feed (included with all accounts)
   - WebSocket connection for real-time bars

## Example Output

```
2025-06-26 14:45:21,899 - __main__ - INFO - 🔴 Live trading mode enabled with Alpaca WebSocket
2025-06-26 14:45:21,899 - __main__ - INFO - 📊 Symbols: ['SPY']
2025-06-26 14:45:21,899 - __main__ - INFO - 🔑 API Key: PK4HCHWR...
2025-06-26 14:45:21,899 - __main__ - INFO - 📄 Paper Trading: True
2025-06-26 14:45:21,899 - __main__ - INFO - 📡 Data Feed: iex
...
2025-06-26 14:45:30,123 - alpaca_streamer - INFO - Connected to Alpaca WebSocket
2025-06-26 14:45:30,456 - alpaca_streamer - INFO - Successfully subscribed to bars: ['SPY']
2025-06-26 14:45:31,789 - data_handler - INFO - 🔴 Publishing live BAR event #1 for SPY at 2025-06-26 14:45:31
```

## Configuration Examples

### Simple Moving Average with Live Data
```yaml
# config/live_ma.yaml
name: "live_ma_crossover"
strategies:
  - type: ma_crossover
    params:
      fast_period: 5
      slow_period: 20
```
Run: `python main.py --config config/live_ma.yaml --alpaca`

### Multiple Symbols
```yaml
# config/live_multi.yaml
data:
  symbols: ["SPY", "QQQ", "AAPL"]
  
strategies:
  - type: bollinger_bands
    params:
      period: 20
      std_dev: 2.0
```
Run: `python main.py --config config/live_multi.yaml --alpaca`

## Important Notes

1. **Market Hours**: Live data only streams during market hours (9:30 AM - 4:00 PM ET)
2. **Connection Limits**: Alpaca limits concurrent WebSocket connections
3. **Paper Trading**: Always uses paper trading API for safety
4. **Pre-warming**: The system can pre-warm indicators with historical data before going live

## Troubleshooting

### No Data Streaming
- Check if markets are open
- Verify API credentials are correct
- Ensure no other WebSocket connections are active

### Import Errors
```bash
pip install websockets alpaca-trade-api
```

### Connection Refused
- Close any other Alpaca WebSocket connections
- Wait a few minutes and retry
- Check Alpaca account status

## Future Enhancements

The `--alpaca` flag currently:
- ✅ Streams live market data
- ✅ Runs strategies in real-time
- ✅ Generates signals and tracks portfolio

Future additions will include:
- 🔜 Asynchronous execution engine
- 🔜 Live order submission to Alpaca
- 🔜 Real-time position tracking
- 🔜 P&L monitoring