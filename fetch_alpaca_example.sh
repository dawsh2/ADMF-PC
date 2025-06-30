#!/bin/bash
# Example script to fetch Alpaca data

# Your API credentials are already set in environment
echo "API Key is set: $ALPACA_API_KEY"
echo "Secret is set: [HIDDEN]"

# Option 1: Fetch maximum available tick data (multiple years)
echo "Fetching maximum tick data..."
python3 src/data/fetch_max_ticks.py --symbol SPY

# Option 2: Fetch specific date range (lighter download)
# python3 src/data/fetch_alpaca_ticks.py --symbol SPY --days 30

# Option 3: Fetch and convert to 5-minute bars
# python3 src/data/fetch_alpaca_ticks.py --symbol SPY --days 30 --timeframe 5m

echo "Done!"