#!/usr/bin/env python3
"""
Quick Alpaca connection test - exits after setup.
"""

import asyncio
import websockets
import json
import os


async def quick_test():
    """Quick test of Alpaca WebSocket setup."""
    
    api_key = os.getenv("ALPACA_API_KEY") 
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not secret_key:
        print("❌ Missing credentials")
        return
    
    print(f"🔑 Testing with API key: {api_key[:8]}...")
    
    try:
        # Connect
        websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
        print("✅ WebSocket connected")
        
        # Auth
        await websocket.send(json.dumps({
            "action": "auth",
            "key": api_key,
            "secret": secret_key
        }))
        print("🔐 Auth sent")
        
        # Subscribe
        await websocket.send(json.dumps({
            "action": "subscribe",
            "bars": ["SPY"]
        }))
        print("📡 Subscription sent")
        
        # Read a few messages
        for i in range(5):
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"📩 Message {i+1}: {data}")
            except asyncio.TimeoutError:
                print(f"⏰ Timeout waiting for message {i+1}")
                break
            except Exception as e:
                print(f"❌ Error reading message {i+1}: {e}")
                break
        
        await websocket.close()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(quick_test())