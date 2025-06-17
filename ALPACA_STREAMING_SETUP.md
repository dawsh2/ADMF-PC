# Alpaca Live Streaming Setup

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streaming.txt
```

### 2. Get Alpaca Paper Trading Credentials
1. Go to [Alpaca Paper Trading](https://app.alpaca.markets/paper/dashboard/overview)
2. Sign up for a free paper trading account
3. Get your API Key and Secret Key from the dashboard

### 3. Set Environment Variables
```bash
export ALPACA_API_KEY="your_paper_api_key_here"
export ALPACA_SECRET_KEY="your_paper_secret_key_here"
```

### 4. Test Live Streaming
```bash
python test_alpaca_streaming.py
```

You should see live SPY bars streaming to your console!

## 📊 Expected Output

```
🔴 LIVE ALPACA BARS - SPY
================================================================================
        Time | #   | Sym  | Timestamp | OHLC                               | Volume
--------------------------------------------------------------------------------
[0:00:03.45] #001 | SPY  | 14:30:00 | O: 456.78 H: 456.89 L: 456.65 C: 456.72 | V:   125,432
[0:00:03.52] #002 | SPY  | 14:31:00 | O: 456.72 H: 456.85 L: 456.58 C: 456.80 | V:   98,765
```

## 🛠️ Usage in Code

### Basic Streaming
```python
import asyncio
from src.data.streamers.alpaca_streamer import create_alpaca_streamer, AlpacaBarPrinter

async def stream_live_data():
    # Create streamer
    streamer = create_alpaca_streamer(
        api_key="your_key",
        secret_key="your_secret", 
        symbols=["SPY", "QQQ"],
        paper_trading=True
    )
    
    # Start streaming
    await streamer.start()
    
    # Process bars
    async for timestamp, bars in streamer.stream_bars():
        for symbol, bar in bars.items():
            print(f"{symbol}: {bar.close}")
    
    await streamer.stop()

asyncio.run(stream_live_data())
```

### Integration with ADMF-PC Architecture
```python
from src.data.streamers import AlpacaWebSocketStreamer, AlpacaConfig

# Create with config
config = AlpacaConfig(
    api_key="your_key",
    secret_key="your_secret",
    symbols=["SPY"],
    paper_trading=True,
    timeframe="1Min",
    feed="iex"
)

streamer = AlpacaWebSocketStreamer(config)

# Use with your strategies
async def run_live_strategy():
    await streamer.start()
    
    async for timestamp, bars in streamer.stream_bars():
        # Feed bars to your ensemble strategy
        for symbol, bar in bars.items():
            # Process with ensemble strategy
            signal = your_ensemble_strategy.process_bar(bar.to_bar())
            
            if signal:
                print(f"Strategy signal: {signal}")

asyncio.run(run_live_strategy())
```

## 🔧 Configuration Options

- **symbols**: List of symbols to stream (e.g., ["SPY", "QQQ", "AAPL"])
- **paper_trading**: True for paper trading, False for live trading
- **timeframe**: "1Min", "5Min", "15Min", "1Hour", "1Day"
- **feed**: "iex" (free) or "sip" (premium)

## 📝 Market Hours

Alpaca streaming works during:
- **Regular Hours**: 9:30 AM - 4:00 PM ET
- **Extended Hours**: Available with SIP feed

The streamer will connect anytime but only receive data during market hours.

## 🐛 Troubleshooting

### Connection Issues
- Check your API credentials
- Ensure paper trading keys (not live trading keys)
- Verify internet connection

### No Data
- Markets might be closed
- Try during regular trading hours (9:30 AM - 4:00 PM ET)
- SPY is the most liquid, try it first

### Authentication Errors
- Double-check API key and secret
- Make sure you're using paper trading credentials
- Regenerate keys if needed

## 🔗 Next Steps

Once streaming works, you can:
1. **Connect to ensemble strategy** - feed live bars to your strategy
2. **Add live execution** - connect signals to Alpaca broker
3. **Implement risk management** - position sizing and limits
4. **Add monitoring** - track performance in real-time

The foundation is now ready for full live paper trading!