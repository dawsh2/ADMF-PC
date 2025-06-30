#!/usr/bin/env python3
"""
Fetch tick data from Alpaca for SPY
Requires Alpaca API keys set as environment variables:
- APCA_API_KEY_ID
- APCA_API_SECRET_KEY
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.rest import REST, TimeFrame
import argparse
from pathlib import Path
import time

def setup_alpaca_client():
    """Initialize Alpaca REST client"""
    # Try multiple environment variable names
    api_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
    api_secret = os.environ.get('ALPACA_API_SECRET') or os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("❌ Error: Alpaca API keys not found in environment variables")
        print("Please set one of: ALPACA_API_KEY or APCA_API_KEY_ID")
        print("Please set one of: ALPACA_API_SECRET, ALPACA_SECRET_KEY, or APCA_API_SECRET_KEY")
        print(f"   Found API Key: {'Yes' if api_key else 'No'}")
        print(f"   Found Secret: {'Yes' if api_secret else 'No'}")
        sys.exit(1)
    
    # Use paper trading endpoint by default
    base_url = 'https://paper-api.alpaca.markets'
    
    return REST(api_key, api_secret, base_url, api_version='v2')

def fetch_tick_data(api, symbol, start_date, end_date, save_dir, max_trades=None):
    """
    Fetch tick (trade) data from Alpaca
    
    Args:
        api: Alpaca REST client
        symbol: Stock symbol (e.g., 'SPY')
        start_date: Start date (datetime)
        end_date: End date (datetime)
        save_dir: Directory to save data
        max_trades: Maximum number of trades to fetch (None for all)
    """
    
    print(f"📊 Fetching tick data for {symbol}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    if max_trades:
        print(f"Max trades limit: {max_trades:,}")
    
    # Alpaca limits: max 10,000 trades per request
    # Need to fetch in chunks for tick data
    
    all_trades = []
    current_start = start_date
    total_fetched = 0
    
    # Convert to Eastern timezone for market hours
    eastern = pytz.timezone('US/Eastern')
    
    while current_start < end_date:
        # Check if we've hit max trades limit
        if max_trades and total_fetched >= max_trades:
            print(f"\n✅ Reached maximum trades limit: {max_trades:,}")
            break
            
        # Fetch one day at a time for tick data
        current_end = min(current_start + timedelta(days=1), end_date)
        
        print(f"\nFetching {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}...")
        
        try:
            # Calculate remaining trades if we have a limit
            remaining_limit = None
            if max_trades:
                remaining_limit = max_trades - total_fetched
            
            # Get trades (tick data)
            trades_iter = api.get_trades(
                symbol,
                start=current_start.isoformat(),
                end=current_end.isoformat(),
                limit=10000,
                asof=None,
                feed=None,  # Use default feed
                page_limit=1000  # Limit pages to avoid massive downloads
            )
            
            # Convert to list
            trades = []
            count = 0
            for trade in trades_iter:
                # Check if we should stop
                if max_trades and total_fetched + count >= max_trades:
                    print(f"\n  Stopping at trade limit...")
                    break
                    
                trades.append({
                    'timestamp': trade.timestamp,
                    'price': trade.price,
                    'size': trade.size,
                    'exchange': trade.exchange,
                    'conditions': trade.conditions,
                    'tape': trade.tape
                })
                count += 1
                
                # Show progress
                if count % 10000 == 0:
                    print(f"  Processed {count:,} trades... Total: {total_fetched + count:,}", end='\r')
            
            print(f"  Retrieved {count:,} trades for {current_start.strftime('%Y-%m-%d')} (Total: {total_fetched + count:,})")
            all_trades.extend(trades)
            total_fetched += count
            
        except Exception as e:
            print(f"⚠️  Error fetching data for {current_start.strftime('%Y-%m-%d')}: {e}")
        
        # Move to next day
        current_start = current_end
        
        # Rate limit: 200 requests per minute
        time.sleep(0.3)  # Be conservative
    
    if not all_trades:
        print("❌ No trade data retrieved")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    
    # Ensure timestamp is timezone-aware
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Convert to Eastern time
    df['timestamp'] = df['timestamp'].dt.tz_convert(eastern)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Add derived fields
    df['dollar_volume'] = df['price'] * df['size']
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    
    print(f"\n✅ Total trades fetched: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Average trades per day: {len(df) / df['date'].nunique():,.0f}")
    
    # Save to CSV
    filename = f"{symbol}_ticks_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    filepath = Path(save_dir) / filename
    
    # Save with index=False to avoid saving row numbers
    df.to_csv(filepath, index=False)
    print(f"\n💾 Saved to: {filepath}")
    
    # Also save a parquet version for faster loading
    parquet_path = filepath.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"💾 Saved parquet to: {parquet_path}")
    
    return df

def fetch_quotes_data(api, symbol, start_date, end_date, save_dir):
    """
    Fetch quote (bid/ask) data from Alpaca
    
    Args:
        api: Alpaca REST client
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        save_dir: Directory to save data
    """
    
    print(f"\n📊 Fetching quote data for {symbol}")
    
    all_quotes = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(hours=1), end_date)  # Smaller chunks for quotes
        
        print(f"\nFetching quotes {current_start.strftime('%Y-%m-%d %H:%M')}...", end='')
        
        try:
            quotes_iter = api.get_quotes(
                symbol,
                start=current_start.isoformat(),
                end=current_end.isoformat(),
                limit=10000
            )
            
            quotes = []
            count = 0
            for quote in quotes_iter:
                quotes.append({
                    'timestamp': quote.timestamp,
                    'bid_price': quote.bid_price,
                    'bid_size': quote.bid_size,
                    'ask_price': quote.ask_price,
                    'ask_size': quote.ask_size,
                    'bid_exchange': quote.bid_exchange,
                    'ask_exchange': quote.ask_exchange,
                    'conditions': quote.conditions
                })
                count += 1
            
            print(f" ({count} quotes)")
            all_quotes.extend(quotes)
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
        
        current_start = current_end
        time.sleep(0.3)
    
    if all_quotes:
        df = pd.DataFrame(all_quotes)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['spread'] = df['ask_price'] - df['bid_price']
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        
        filename = f"{symbol}_quotes_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = Path(save_dir) / filename
        df.to_csv(filepath, index=False)
        print(f"💾 Saved quotes to: {filepath}")
        
        return df
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Fetch tick data from Alpaca')
    parser.add_argument('--symbol', default='SPY', help='Stock symbol (default: SPY)')
    parser.add_argument('--days', type=int, default=1, help='Number of days to fetch (default: 1)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--save-dir', default='data/ticks', help='Directory to save data')
    parser.add_argument('--quotes', action='store_true', help='Also fetch quote data')
    parser.add_argument('--max-trades', type=int, help='Maximum number of trades to fetch')
    parser.add_argument('--max-days', action='store_true', help='Fetch maximum allowed historical data')
    
    args = parser.parse_args()
    
    # Setup dates
    if args.max_days:
        # Fetch maximum historical data (Alpaca allows ~5-6 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        print("🚀 Fetching MAXIMUM historical tick data (5 years)...")
    elif args.start_date and args.end_date:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
    else:
        # Default: fetch last N trading days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    
    # Make timezone aware (Eastern time)
    eastern = pytz.timezone('US/Eastern')
    if start_date.tzinfo is None:
        start_date = eastern.localize(start_date.replace(hour=9, minute=30))
    if end_date.tzinfo is None:
        end_date = eastern.localize(end_date.replace(hour=16, minute=0))
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Alpaca client
    api = setup_alpaca_client()
    
    # Fetch tick data
    tick_df = fetch_tick_data(api, args.symbol, start_date, end_date, save_dir, max_trades=args.max_trades)
    
    # Optionally fetch quote data
    if args.quotes:
        quote_df = fetch_quotes_data(api, args.symbol, start_date, end_date, save_dir)
    
    # Display summary statistics
    if tick_df is not None and len(tick_df) > 0:
        print("\n📊 Summary Statistics:")
        print(f"Total trades: {len(tick_df):,}")
        print(f"Unique dates: {tick_df['date'].nunique()}")
        print(f"Price range: ${tick_df['price'].min():.2f} - ${tick_df['price'].max():.2f}")
        print(f"Total volume: {tick_df['size'].sum():,}")
        print(f"Total dollar volume: ${tick_df['dollar_volume'].sum():,.0f}")
        
        # Show sample of data
        print("\nSample of tick data:")
        print(tick_df.head(10))
        
        # Market hours analysis
        market_hours = tick_df[
            (tick_df['timestamp'].dt.hour >= 9) & 
            (tick_df['timestamp'].dt.hour < 16)
        ]
        
        if len(market_hours) > 0:
            print(f"\nMarket hours trades: {len(market_hours):,} ({len(market_hours)/len(tick_df)*100:.1f}%)")
            
            # Busiest times
            tick_df['hour'] = tick_df['timestamp'].dt.hour
            hourly_counts = tick_df.groupby('hour').size()
            print("\nTrades by hour:")
            print(hourly_counts.sort_values(ascending=False).head())

if __name__ == "__main__":
    main()