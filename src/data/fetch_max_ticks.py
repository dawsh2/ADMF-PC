#!/usr/bin/env python3
"""
Fetch MAXIMUM tick data from Alpaca - optimized for large downloads
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api.rest import REST
import argparse
from pathlib import Path
import time
import concurrent.futures
from typing import List, Tuple

def setup_alpaca_client():
    """Initialize Alpaca REST client"""
    # Try multiple environment variable names
    api_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
    api_secret = os.environ.get('ALPACA_API_SECRET') or os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("❌ Error: Alpaca API keys not found")
        print("   Looking for: ALPACA_API_KEY or APCA_API_KEY_ID")
        print("   Looking for: ALPACA_API_SECRET, ALPACA_SECRET_KEY, or APCA_API_SECRET_KEY")
        print(f"   Found API Key: {'Yes' if api_key else 'No'}")
        print(f"   Found Secret: {'Yes' if api_secret else 'No'}")
        sys.exit(1)
    
    return REST(api_key, api_secret, 'https://paper-api.alpaca.markets', api_version='v2')

def fetch_day_ticks(args: Tuple[REST, str, datetime, Path]) -> pd.DataFrame:
    """Fetch ticks for a single day - for parallel processing"""
    api, symbol, date, save_dir = args
    
    start = date.replace(hour=4, minute=0)  # Pre-market
    end = date.replace(hour=20, minute=0)   # After-market
    
    try:
        trades = []
        trades_iter = api.get_trades(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=10000
        )
        
        for trade in trades_iter:
            # Handle different trade object structures
            if hasattr(trade, '_raw'):
                # New API structure
                raw = trade._raw
                trades.append({
                    'timestamp': raw.get('t') or raw.get('timestamp'),
                    'price': raw.get('p') or raw.get('price'),
                    'size': raw.get('s') or raw.get('size'),
                    'exchange': raw.get('x', 'N/A'),
                    'conditions': raw.get('c', []),
                    'tape': raw.get('z', 'N/A')
                })
            else:
                # Try direct attributes
                trades.append({
                    'timestamp': getattr(trade, 't', getattr(trade, 'timestamp', None)),
                    'price': getattr(trade, 'p', getattr(trade, 'price', None)),
                    'size': getattr(trade, 's', getattr(trade, 'size', None)),
                    'exchange': getattr(trade, 'x', getattr(trade, 'exchange', 'N/A')),
                    'conditions': getattr(trade, 'c', getattr(trade, 'conditions', [])),
                    'tape': getattr(trade, 'z', getattr(trade, 'tape', 'N/A'))
                })
        
        if trades:
            df = pd.DataFrame(trades)
            # Save daily file
            filename = f"{symbol}_ticks_{date.strftime('%Y%m%d')}.parquet"
            df.to_parquet(save_dir / 'daily' / filename)
            print(f"✓ {date.strftime('%Y-%m-%d')}: {len(trades):,} trades")
            return df
        else:
            print(f"✗ {date.strftime('%Y-%m-%d')}: No trades")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"✗ {date.strftime('%Y-%m-%d')}: Error - {e}")
        return pd.DataFrame()

def get_max_tick_data(symbol='SPY', parallel=True):
    """
    Fetch maximum available tick data from Alpaca
    
    Alpaca Data Limits:
    - Free: IEX data only (limited)
    - Paid: Full SIP feed
    - Historical: ~5-6 years available
    - Rate limit: 200 requests/minute
    """
    
    print(f"🚀 Fetching MAXIMUM tick data for {symbol}")
    print("=" * 60)
    
    api = setup_alpaca_client()
    
    # Determine account type
    account = api.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    
    # Setup dates - Alpaca has tick data from ~2016
    eastern = pytz.timezone('US/Eastern')
    end_date = datetime.now(eastern)
    
    # Try different historical depths
    depths = [
        (6, "6 years"),
        (5, "5 years"),
        (4, "4 years"),
        (3, "3 years"),
        (2, "2 years"),
        (1, "1 year"),
        (0.5, "6 months"),
        (0.25, "3 months"),
        (0.083, "1 month"),
    ]
    
    # Find maximum available depth
    print("\nTesting data availability...")
    max_depth_years = 1
    
    for years, label in depths:
        test_date = end_date - timedelta(days=365*years)
        try:
            # Test with a small request
            test_trades = list(api.get_trades(
                symbol,
                start=test_date.isoformat(),
                end=(test_date + timedelta(hours=1)).isoformat(),
                limit=10
            ))
            if test_trades:
                print(f"✓ Data available for {label} back")
                max_depth_years = years
            else:
                print(f"✗ No data for {label} back")
                break
        except Exception as e:
            print(f"✗ Error testing {label}: {e}")
            break
    
    # Set start date based on available data
    start_date = end_date - timedelta(days=365*max_depth_years)
    print(f"\n📅 Fetching from {start_date.date()} to {end_date.date()}")
    print(f"Total period: ~{max_depth_years} years")
    
    # Create save directory structure
    save_dir = Path('data/ticks/max_download')
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'daily').mkdir(exist_ok=True)
    
    # Generate list of trading days
    trading_days = []
    current = start_date
    
    while current <= end_date:
        # Only weekdays
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            trading_days.append(current)
        current += timedelta(days=1)
    
    print(f"📊 Estimated trading days: {len(trading_days)}")
    
    # Fetch data
    all_data = []
    
    if parallel:
        print("\n🔄 Fetching in parallel (4 threads)...")
        # Parallel fetching - be careful with rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Prepare arguments
            args_list = [(api, symbol, day, save_dir) for day in trading_days]
            
            # Submit in batches to respect rate limits
            batch_size = 40  # ~10 seconds per batch at 4 threads
            
            for i in range(0, len(args_list), batch_size):
                batch = args_list[i:i+batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{len(args_list)//batch_size + 1}")
                
                results = list(executor.map(fetch_day_ticks, batch))
                all_data.extend([r for r in results if not r.empty])
                
                # Rate limit pause between batches
                if i + batch_size < len(args_list):
                    print("Pausing for rate limit...")
                    time.sleep(10)
    else:
        print("\n🔄 Fetching sequentially...")
        for day in trading_days:
            df = fetch_day_ticks((api, symbol, day, save_dir))
            if not df.empty:
                all_data.append(df)
            time.sleep(0.3)  # Rate limit
    
    # Combine all data
    if all_data:
        print(f"\n📦 Combining {len(all_data)} daily files...")
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df = full_df.sort_values('timestamp')
        
        # Save combined file
        print("💾 Saving combined dataset...")
        output_file = save_dir / f"{symbol}_ticks_COMPLETE_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        full_df.to_parquet(output_file)
        
        # Also save metadata
        metadata = {
            'symbol': symbol,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_trades': len(full_df),
            'trading_days': len(all_data),
            'file_size_mb': output_file.stat().st_size / (1024*1024),
            'columns': list(full_df.columns),
            'price_range': [full_df['price'].min(), full_df['price'].max()],
            'total_volume': full_df['size'].sum(),
            'download_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(save_dir / f'{symbol}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Summary statistics
        print("\n" + "="*60)
        print("📊 DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Total trades: {len(full_df):,}")
        print(f"Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
        print(f"Price range: ${full_df['price'].min():.2f} - ${full_df['price'].max():.2f}")
        print(f"Total volume: {full_df['size'].sum():,}")
        print(f"File size: {metadata['file_size_mb']:.1f} MB")
        print(f"\n📁 Saved to: {output_file}")
        
        # Sample data by year
        full_df['year'] = full_df['timestamp'].dt.year
        yearly_stats = full_df.groupby('year').agg({
            'price': ['count', 'mean', 'std'],
            'size': 'sum'
        })
        
        print("\n📅 Yearly Statistics:")
        print(yearly_stats)
        
        return full_df
    else:
        print("\n❌ No data retrieved!")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fetch MAXIMUM tick data from Alpaca')
    parser.add_argument('--symbol', default='SPY', help='Stock symbol (default: SPY)')
    parser.add_argument('--sequential', action='store_true', help='Use sequential fetching instead of parallel')
    
    args = parser.parse_args()
    
    # Start the download
    df = get_max_tick_data(args.symbol, parallel=not args.sequential)
    
    if df is not None:
        print("\n💡 Tips:")
        print("- Daily parquet files are saved in data/ticks/max_download/daily/")
        print("- Use the combined parquet file for analysis")
        print("- Consider downsampling for initial exploration:")
        print("  df = pd.read_parquet('...').sample(n=1000000)")
        print("\n⚠️  Note: Free Alpaca accounts only get IEX data (partial market)")
        print("     Paid accounts get full SIP feed with all exchanges")

if __name__ == "__main__":
    main()