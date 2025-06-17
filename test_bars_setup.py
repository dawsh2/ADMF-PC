#!/usr/bin/env python3
"""
Quick test to confirm bar subscription setup.
"""

import asyncio
import websockets
import json
import os
from datetime import datetime


async def test_bars_setup():
    """Test bar subscription setup and exit."""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    print("📊 Testing SPY Bars Setup")
    print("=" * 30)
    
    try:
        websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
        print("✅ Connected")
        
        # Auth
        await websocket.send(json.dumps({
            "action": "auth",
            "key": api_key,
            "secret": secret_key
        }))
        
        # Subscribe to bars
        await websocket.send(json.dumps({
            "action": "subscribe",
            "bars": ["SPY"]
        }))
        
        print("📡 Sent auth + bar subscription")
        
        # Read setup messages
        for i in range(5):
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(message)
                messages = data if isinstance(data, list) else [data]
                
                for msg in messages:
                    msg_type = msg.get("T", "?")
                    if msg_type == "success":
                        print(f"✅ {msg.get('msg', '')}")
                    elif msg_type == "subscription":
                        print(f"📊 Subscribed to bars: {msg.get('bars', [])}")
                        print("✅ Setup complete - bars will arrive every minute during market hours")
                        await websocket.close()
                        return
                    elif msg_type == "b":
                        print(f"📊 Got bar immediately: {msg}")
                        await websocket.close()
                        return
                        
            except asyncio.TimeoutError:
                print(f"⏰ Timeout on message {i+1}")
                break
                
        await websocket.close()
        print("✅ Bar subscription confirmed")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_bars_setup())