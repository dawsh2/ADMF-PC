#!/usr/bin/env python3
"""
Simple Alpaca WebSocket connection diagnostics tool.
Helps debug connection limit issues.
"""

import asyncio
import websockets
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_alpaca_connection():
    """Test basic Alpaca WebSocket connection."""
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
    
    if not api_key or not secret_key:
        print("❌ Missing Alpaca credentials")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    print(f"🔑 Testing connection with API key: {api_key[:8]}...")
    
    # Test different endpoints
    endpoints = [
        ("IEX (Live)", "wss://stream.data.alpaca.markets/v2/iex"),
        ("SIP (Live)", "wss://stream.data.alpaca.markets/v2/sip"),
    ]
    
    for name, url in endpoints:
        print(f"\n📡 Testing {name}: {url}")
        
        try:
            # Connect
            async with websockets.connect(url) as websocket:
                print(f"✅ Connected to {name}")
                
                # Send auth
                auth_payload = {
                    "action": "auth",
                    "key": api_key,
                    "secret": secret_key
                }
                
                await websocket.send(json.dumps(auth_payload))
                print(f"📤 Sent authentication")
                
                # Wait for auth response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    print(f"📥 Auth response: {data}")
                    
                    # Handle list response
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    
                    if data.get("T") == "success":
                        if data.get("msg") == "connected":
                            print("🔄 Got connected message, waiting for auth...")
                            # Wait for actual auth response
                            auth_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            auth_data = json.loads(auth_response)
                            print(f"📥 Final auth response: {auth_data}")
                            
                            if isinstance(auth_data, list):
                                auth_data = auth_data[0] if auth_data else {}
                            
                            if auth_data.get("T") == "success" and auth_data.get("msg") == "authenticated":
                                print(f"✅ {name} authentication successful!")
                            else:
                                print(f"❌ {name} authentication failed: {auth_data}")
                        elif data.get("msg") == "authenticated":
                            print(f"✅ {name} authentication successful!")
                        else:
                            print(f"❌ {name} authentication failed: {data}")
                    elif data.get("T") == "error":
                        error_code = data.get("code")
                        error_msg = data.get("msg")
                        print(f"❌ {name} error {error_code}: {error_msg}")
                        
                        if error_code == 406:
                            print("💡 Connection limit exceeded - possible causes:")
                            print("   1. Another WebSocket connection is active")
                            print("   2. Previous connections weren't properly closed")
                            print("   3. Account limits reached")
                            print("   4. Try waiting 1-2 minutes and retry")
                    else:
                        print(f"❓ {name} unexpected response: {data}")
                        
                except asyncio.TimeoutError:
                    print(f"⏰ {name} authentication timeout")
                    
        except Exception as e:
            print(f"❌ {name} connection failed: {e}")
    
    print(f"\n🕐 Test completed at {datetime.now()}")
    print("\n💡 If you see 'connection limit exceeded':")
    print("   1. Close any other applications using Alpaca WebSocket")
    print("   2. Wait 2-3 minutes for connections to time out")
    print("   3. Try again")

def main():
    """Main entry point."""
    print("🔧 Alpaca WebSocket Connection Diagnostics")
    print("=" * 50)
    
    try:
        asyncio.run(test_alpaca_connection())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()