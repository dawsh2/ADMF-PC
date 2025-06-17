#!/usr/bin/env python3
"""
Simple Alpaca WebSocket test without complex imports.
"""

import asyncio
import websockets
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


async def test_alpaca_connection():
    """Test basic Alpaca WebSocket connection."""
    
    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not secret_key:
        print("❌ Missing Alpaca credentials!")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (or ALPACA_API_SECRET) environment variables")
        return
    
    # Alpaca WebSocket URL for IEX data
    ws_url = "wss://stream.data.alpaca.markets/v2/iex"
    
    print("🚀 Testing Alpaca WebSocket Connection")
    print("=" * 50)
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🌐 URL: {ws_url}")
    print(f"📊 Symbol: SPY")
    print(f"📝 Paper Trading: True")
    
    try:
        print("\n🔌 Connecting to Alpaca WebSocket...")
        
        # Connect to WebSocket
        websocket = await websockets.connect(ws_url)
        print("✅ WebSocket connection established")
        
        # Authenticate
        auth_payload = {
            "action": "auth",
            "key": api_key,
            "secret": secret_key
        }
        
        print("🔐 Authenticating...")
        await websocket.send(json.dumps(auth_payload))
        
        # Wait for auth responses (Alpaca sends multiple messages)
        auth_success = False
        for _ in range(3):  # Try up to 3 messages
            response = await websocket.recv()
            auth_data = json.loads(response)
            print(f"🔐 Auth response: {auth_data}")
            
            # Handle response as list or single object
            if isinstance(auth_data, list):
                auth_data = auth_data[0] if auth_data else {}
            
            if auth_data.get("T") == "success" and auth_data.get("msg") == "authenticated":
                auth_success = True
                break
        
        if auth_success:
            print("✅ Authentication successful!")
            
            # Subscribe to SPY bars
            subscribe_payload = {
                "action": "subscribe",
                "bars": ["SPY"]
            }
            
            print("📡 Subscribing to SPY bars...")
            await websocket.send(json.dumps(subscribe_payload))
            
            # Wait for subscription response
            response = await websocket.recv()
            sub_data = json.loads(response)
            print(f"📡 Subscription response: {sub_data}")
            
            # Handle response as list or single object
            if isinstance(sub_data, list):
                sub_data = sub_data[0] if sub_data else {}
            
            if sub_data.get("T") == "subscription":
                print("✅ Successfully subscribed to SPY bars!")
                print("\n📊 Waiting for live bars... (Press Ctrl+C to stop)")
                print("-" * 60)
                
                bar_count = 0
                start_time = datetime.now()
                
                # Listen for bars
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Handle array of messages or single message
                        messages = data if isinstance(data, list) else [data]
                        
                        for msg in messages:
                            if msg.get("T") == "b":  # Bar data
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
                                      f"{symbol:4s} | {timestamp[11:19]} | "
                                      f"O:{open_price:7.2f} H:{high_price:7.2f} "
                                      f"L:{low_price:7.2f} C:{close_price:7.2f} | "
                                      f"V:{volume:8,.0f}")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON decode error: {e}")
                    except Exception as e:
                        print(f"⚠️ Message processing error: {e}")
                        
            else:
                print(f"❌ Subscription failed: {sub_data}")
                
        else:
            print(f"❌ Authentication failed: {auth_data}")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopped by user")
        if 'bar_count' in locals():
            print(f"📊 Received {bar_count} bars")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        if 'websocket' in locals():
            await websocket.close()
        print("✅ Connection closed")


if __name__ == "__main__":
    asyncio.run(test_alpaca_connection())