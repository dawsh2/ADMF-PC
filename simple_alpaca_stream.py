#!/usr/bin/env python3
"""
Minimal Alpaca WebSocket streaming test.
"""

import asyncio
import websockets
import json
import os
from datetime import datetime


async def stream_alpaca_bars():
    """Stream SPY bars from Alpaca WebSocket."""
    
    # Credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not secret_key:
        print("❌ Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    print("🚀 Alpaca WebSocket Live Bar Streaming")
    print("=" * 50)
    print(f"🔑 API Key: {api_key[:8]}...")
    
    try:
        # Connect
        websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
        print("✅ Connected to Alpaca WebSocket")
        
        # Auth
        await websocket.send(json.dumps({
            "action": "auth",
            "key": api_key,
            "secret": secret_key
        }))
        print("🔐 Sent authentication...")
        
        # Subscribe (send immediately after auth)
        await websocket.send(json.dumps({
            "action": "subscribe", 
            "bars": ["SPY"]
        }))
        print("📡 Sent subscription for SPY bars...")
        
        print("\n📊 Listening for messages...")
        print("-" * 60)
        
        bar_count = 0
        start_time = datetime.now()
        
        # Listen for all messages
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Handle arrays or single messages
                messages = data if isinstance(data, list) else [data]
                
                for msg in messages:
                    msg_type = msg.get("T", "unknown")
                    
                    if msg_type == "success":
                        print(f"✅ Success: {msg.get('msg', 'No message')}")
                    elif msg_type == "subscription":
                        subscriptions = msg.get('bars', [])
                        print(f"✅ Subscribed to bars: {subscriptions}")
                        print("🔴 LIVE - Waiting for SPY bars...")
                    elif msg_type == "b":  # Bar data
                        bar_count += 1
                        elapsed = datetime.now() - start_time
                        
                        symbol = msg.get("S", "?")
                        timestamp = msg.get("t", "?")
                        open_price = msg.get("o", 0)
                        high_price = msg.get("h", 0)
                        low_price = msg.get("l", 0)
                        close_price = msg.get("c", 0)
                        volume = msg.get("v", 0)
                        
                        print(f"[{elapsed}] #{bar_count:03d} | "
                              f"{symbol:4s} | {timestamp[11:19] if len(str(timestamp)) > 10 else timestamp} | "
                              f"O:{open_price:7.2f} H:{high_price:7.2f} "
                              f"L:{low_price:7.2f} C:{close_price:7.2f} | "
                              f"V:{volume:8,.0f}")
                    else:
                        print(f"📩 Message: {msg}")
                        
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON error: {e}")
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped by user")
        if 'bar_count' in locals():
            print(f"📊 Total bars received: {bar_count}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'websocket' in locals():
            await websocket.close()
        print("✅ Disconnected")


if __name__ == "__main__":
    print("💡 This will stream live SPY bars during market hours")
    print("💡 Press Ctrl+C to stop")
    print("💡 If no bars appear, markets might be closed\n")
    
    asyncio.run(stream_alpaca_bars())