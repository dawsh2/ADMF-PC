#!/usr/bin/env python3
"""
Live bar streaming with debug info.
"""

import asyncio
import websockets
import json
import os
from datetime import datetime


async def debug_stream():
    """Stream with detailed debugging."""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    print("🚀 Live Market Hours - Alpaca Bar Streaming")
    print("=" * 50)
    print(f"🔑 API Key: {api_key}")  # Show full key for verification
    print(f"🕐 Current time: {datetime.now()}")
    
    try:
        websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
        print("✅ Connected")
        
        # Auth
        await websocket.send(json.dumps({
            "action": "auth", 
            "key": api_key,
            "secret": secret_key
        }))
        
        # Subscribe
        await websocket.send(json.dumps({
            "action": "subscribe",
            "bars": ["SPY"]
        }))
        
        print("📡 Sent auth + subscription, listening for messages...")
        print("🔴 LIVE STREAMING - Press Ctrl+C to stop")
        print("-" * 60)
        
        message_count = 0
        
        # Set a timeout for each message
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                message_count += 1
                
                try:
                    data = json.loads(message)
                    messages = data if isinstance(data, list) else [data]
                    
                    for msg in messages:
                        msg_type = msg.get("T", "?")
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        if msg_type == "success":
                            print(f"[{timestamp}] ✅ SUCCESS: {msg.get('msg', '')}")
                        elif msg_type == "subscription":
                            print(f"[{timestamp}] 📡 SUBSCRIBED: {msg.get('bars', [])}")
                        elif msg_type == "b":  # Bar data!
                            symbol = msg.get("S", "?")
                            bar_time = msg.get("t", "?")
                            open_p = msg.get("o", 0)
                            high_p = msg.get("h", 0) 
                            low_p = msg.get("l", 0)
                            close_p = msg.get("c", 0)
                            volume = msg.get("v", 0)
                            
                            print(f"[{timestamp}] 📊 BAR: {symbol} {bar_time[11:19] if len(str(bar_time)) > 10 else bar_time}")
                            print(f"    O:{open_p:7.2f} H:{high_p:7.2f} L:{low_p:7.2f} C:{close_p:7.2f} V:{volume:8,.0f}")
                        else:
                            print(f"[{timestamp}] 📩 MSG #{message_count}: {msg}")
                            
                except json.JSONDecodeError:
                    print(f"[{timestamp}] ⚠️ Invalid JSON: {message}")
                    
            except asyncio.TimeoutError:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏰ No message for 10 seconds...")
                print("   This is normal - bars come every 1 minute during active trading")
                continue
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Connection closed")
                break
                
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped by user after {message_count} messages")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'websocket' in locals():
            await websocket.close()
        print("✅ Disconnected")


if __name__ == "__main__":
    asyncio.run(debug_stream())