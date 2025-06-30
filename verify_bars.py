#!/usr/bin/env python3

import asyncio
import websockets
import json
import os
from datetime import datetime

api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_API_SECRET')

url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"

async def test():
    async with websockets.connect(url) as ws:
        # Auth flow
        msg = await ws.recv()
        print(f"Connected: {msg}")
        
        await ws.send(json.dumps({"action": "auth", "key": api_key, "secret": secret_key}))
        auth = await ws.recv()
        print(f"Auth: {auth}")
        
        # Subscribe exactly as docs show
        await ws.send(json.dumps({"action": "subscribe", "bars": ["BTC/USD"]}))
        sub = await ws.recv()
        print(f"Subscribe: {sub}")
        
        print(f"\nTime: {datetime.now().strftime('%H:%M:%S')}")
        print("Waiting for bar...")
        
        # Wait for message
        msg = await ws.recv()
        print(f"\nReceived: {msg}")

asyncio.run(test())