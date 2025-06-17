#!/usr/bin/env python3
"""
Test live trades (more frequent than bars).
"""

import asyncio
import websockets
import json
import os
from datetime import datetime


async def stream_trades():
    """Stream SPY trades - these come more frequently than bars."""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    print("🔴 LIVE SPY TRADES STREAM")
    print("=" * 40)
    print(f"🕐 {datetime.now()}")
    
    try:
        websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
        print("✅ Connected")
        
        # Auth
        await websocket.send(json.dumps({
            "action": "auth",
            "key": api_key, 
            "secret": secret_key
        }))
        
        # Subscribe to trades (more frequent than bars)
        await websocket.send(json.dumps({
            "action": "subscribe",
            "trades": ["SPY"]  # Trades instead of bars
        }))
        
        print("📡 Subscribed to SPY trades")
        print("🔴 LIVE - Waiting for trades...")
        print("-" * 40)
        
        async for message in websocket:
            try:
                data = json.loads(message)
                messages = data if isinstance(data, list) else [data]
                
                for msg in messages:
                    msg_type = msg.get("T", "?")
                    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    if msg_type == "success":
                        print(f"[{now}] ✅ {msg.get('msg', '')}")
                    elif msg_type == "subscription":
                        print(f"[{now}] 📡 Subscribed: {msg}")
                    elif msg_type == "t":  # Trade data
                        symbol = msg.get("S", "?")
                        price = msg.get("p", 0)
                        size = msg.get("s", 0)
                        trade_time = msg.get("t", "?")
                        
                        print(f"[{now}] 💰 TRADE: {symbol} ${price:.2f} x {size:,} @ {trade_time}")
                    elif msg_type == "b":  # Bar data (if any)
                        symbol = msg.get("S", "?")
                        close_price = msg.get("c", 0)
                        volume = msg.get("v", 0)
                        bar_time = msg.get("t", "?")
                        
                        print(f"[{now}] 📊 BAR: {symbol} ${close_price:.2f} V:{volume:,} @ {bar_time}")
                    else:
                        print(f"[{now}] 📩 {msg}")
                        
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")
            except Exception as e:
                print(f"Processing error: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'websocket' in locals():
            await websocket.close()


if __name__ == "__main__":
    print("💡 Trades come much more frequently than 1-minute bars")
    print("💡 You should see live SPY trades if market is active")
    print("💡 Press Ctrl+C to stop\n")
    
    asyncio.run(stream_trades())