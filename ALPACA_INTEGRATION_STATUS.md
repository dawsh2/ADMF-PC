# Alpaca Integration Status

## ✅ What Works

The `--alpaca` flag is now fully functional! When you run:
```bash
python main.py --config config/bollinger/test.yaml --alpaca
```

It successfully:
1. **Connects to Alpaca WebSocket API** with your paid account credentials
2. **Authenticates** using API keys from environment variables
3. **Subscribes to real-time bar data** (tested with BTCUSD)
4. **Streams live market data** through the ADMF-PC event system
5. **Uses universal topology** automatically (BAR → SIGNAL → ORDER → FILL)

## 📊 Key Differences from live_data_examples/

### Current Implementation (src/data/)
- ✅ Integrated into main ADMF-PC data handler system
- ✅ Works with --alpaca flag seamlessly
- ✅ Properly extracts symbols from config (handles "BTCUSD_5m" → "BTCUSD")
- ✅ Sets up universal topology automatically
- ❌ Missing indicator pre-warming functionality

### Example Implementation (live_data_examples/)
- ✅ Has indicator pre-warming with historical data
- ✅ Better error handling for connection limits
- ❌ Not integrated with main.py
- ❌ Requires manual setup

## 🔧 Configuration

### Symbol Formats
- **Stocks**: Use ticker symbol (e.g., "SPY", "AAPL")
- **Crypto**: Use full pair (e.g., "BTCUSD", "ETHUSD")

### Example Config (config/bollinger/test.yaml)
```yaml
name: bollinger
data: BTCUSD_5m  # Changed from SPY_5m for 24/7 crypto markets

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

## 🚀 Usage Examples

### Basic Usage
```bash
# Stream live crypto data with Bollinger Bands
python main.py --config config/bollinger/test.yaml --alpaca

# Limit to 100 bars for testing
python main.py --config config/bollinger/test.yaml --alpaca --bars 100

# Use verbose logging
python main.py --config config/bollinger/test.yaml --alpaca --verbose
```

### What Happens
1. Parses symbol from config (e.g., "BTCUSD_5m" → "BTCUSD")
2. Overrides data source to use Alpaca WebSocket
3. Creates LiveDataHandler with AlpacaWebSocketStreamer
4. Connects to wss://stream.data.alpaca.markets/v2/iex
5. Authenticates with your API credentials
6. Subscribes to real-time bars for the symbol
7. Streams data through ADMF-PC event system
8. Strategies process live data in real-time

## 🔜 Future Enhancements

1. **Add Indicator Pre-warming**
   - Port the pre-warming code from live_data_examples/
   - Fetch historical data before going live
   - Ensure indicators are ready

2. **Asynchronous Execution Engine**
   - Replace synchronous broker with async Alpaca broker
   - Enable real order submission

3. **Live Position Tracking**
   - Track actual positions from Alpaca account
   - Sync portfolio state with broker

4. **Error Recovery**
   - Handle disconnections gracefully
   - Automatic reconnection logic
   - Connection limit handling

## 📝 Notes

- The system uses paper trading API for safety
- IEX feed is used (included with all accounts)
- Crypto markets are 24/7, great for testing anytime
- Stock data only available during market hours
- Alpaca limits concurrent WebSocket connections