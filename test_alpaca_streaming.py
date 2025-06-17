#!/usr/bin/env python3
"""
Test script for Alpaca WebSocket streaming.

Run this to see live bars printed to console.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.streamers.alpaca_streamer import demo_alpaca_streaming, create_alpaca_streamer, AlpacaBarPrinter


async def test_streaming():
    """Test Alpaca WebSocket streaming with credentials from environment."""
    
    # Get credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Missing Alpaca credentials!")
        print("\nPlease set environment variables:")
        print("  export ALPACA_API_KEY=your_paper_key_here")
        print("  export ALPACA_SECRET_KEY=your_paper_secret_here")
        print("\nGet paper trading credentials from: https://app.alpaca.markets/paper/dashboard/overview")
        return
    
    # Test symbols - SPY is most liquid
    symbols = ["SPY"]
    
    print(f"🚀 Starting Alpaca WebSocket test...")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"📝 Paper Trading: True")
    print(f"\n💡 Press Ctrl+C to stop streaming\n")
    
    # Create streamer
    streamer = create_alpaca_streamer(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        paper_trading=True,
        timeframe="1Min",
        feed="iex"  # IEX is free
    )
    
    # Create printer
    printer = AlpacaBarPrinter(symbols)
    
    try:
        print("🔌 Connecting to Alpaca WebSocket...")
        
        # Start streaming
        if await streamer.start():
            print("✅ Connected successfully!")
            print("📡 Waiting for live bars...\n")
            
            # Run printer
            await printer.run_with_streamer(streamer)
        else:
            print("❌ Failed to connect to Alpaca WebSocket")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping stream...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await streamer.stop()
        print("✅ Stream stopped cleanly")


async def test_connection_only():
    """Test just the connection without streaming."""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Missing credentials - set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    
    from data.streamers.alpaca_streamer import AlpacaWebSocketStreamer, AlpacaConfig
    
    config = AlpacaConfig(
        api_key=api_key,
        secret_key=secret_key,
        symbols=["SPY"],
        paper_trading=True
    )
    
    streamer = AlpacaWebSocketStreamer(config)
    
    print("🧪 Testing connection...")
    
    try:
        if await streamer.start():
            print("✅ Connection successful!")
            print("⏱️  Waiting 5 seconds...")
            await asyncio.sleep(5)
        else:
            print("❌ Connection failed")
    finally:
        await streamer.stop()
        print("✅ Test complete")


if __name__ == "__main__":
    print("🔴 Alpaca WebSocket Streaming Test")
    print("=" * 50)
    
    # Check if credentials are available
    if len(sys.argv) > 1 and sys.argv[1] == "--test-connection":
        asyncio.run(test_connection_only())
    else:
        asyncio.run(test_streaming())