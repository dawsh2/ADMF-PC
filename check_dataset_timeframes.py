#!/usr/bin/env python3
"""
Check time ranges and bar frequencies in all datasets under ./data
to ensure EOD closing logic is compatible.
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_dataset(file_path):
    """Analyze a single dataset for time information."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")
    
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Check if we have a timestamp column
        timestamp_col = None
        for col in ['timestamp', 'Timestamp', 'date', 'Date', 'datetime', 'Datetime']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if not timestamp_col:
            print(f"❌ No timestamp column found. Columns: {list(df.columns)}")
            return
        
        # Parse timestamps
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        print(f"✓ Timestamp column: {timestamp_col}")
        print(f"✓ Total rows: {len(df):,}")
        print(f"✓ Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
        
        # Calculate time differences to determine bar frequency
        if len(df) > 1:
            time_diffs = df[timestamp_col].diff().dropna()
            
            # Get the most common time difference (mode)
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                bar_frequency = mode_diff.iloc[0]
                minutes = bar_frequency.total_seconds() / 60
                
                print(f"✓ Bar frequency: {minutes:.0f} minutes")
                
                # Calculate bars per day based on frequency
                if minutes == 1:
                    bars_per_day = 390  # 6.5 hours * 60 minutes
                elif minutes == 5:
                    bars_per_day = 78   # 6.5 hours * 60 / 5
                elif minutes == 15:
                    bars_per_day = 26   # 6.5 hours * 60 / 15
                elif minutes == 30:
                    bars_per_day = 13   # 6.5 hours * 60 / 30
                elif minutes == 60:
                    bars_per_day = 6.5  # 6.5 hours
                elif minutes == 1440:  # Daily bars
                    bars_per_day = 1
                else:
                    bars_per_day = 390 / minutes
                
                print(f"✓ Bars per trading day: {bars_per_day}")
                
                # Check EOD filter compatibility
                if minutes <= 60:  # Intraday data
                    # For 5-minute bars: 72 bars = 360 min = 3:30 PM, 78 bars = 390 min = 3:50 PM
                    # Need to adjust for different timeframes
                    cutoff_bar = int(360 / minutes)  # 3:30 PM
                    close_bar = int(390 / minutes)   # 3:50 PM
                    
                    print(f"\n📊 EOD Filter Calculations:")
                    print(f"  - Entry cutoff (3:30 PM): bar_of_day < {cutoff_bar}")
                    print(f"  - Force exit (3:50 PM): bar_of_day >= {close_bar}")
                else:
                    print(f"\n📊 Daily or longer timeframe - EOD filters not applicable")
        
        # Check trading hours coverage
        if timestamp_col and len(df) > 0:
            # Get unique times
            df['time'] = df[timestamp_col].dt.time
            unique_times = df['time'].unique()
            
            if len(unique_times) > 1:  # Intraday data
                earliest_time = min(unique_times)
                latest_time = max(unique_times)
                
                print(f"\n⏰ Trading Hours Coverage:")
                print(f"  - Earliest bar: {earliest_time}")
                print(f"  - Latest bar: {latest_time}")
                
                # Check if we have pre/post market data
                market_open = pd.to_datetime('09:30:00').time()
                market_close = pd.to_datetime('16:00:00').time()
                
                has_premarket = earliest_time < market_open
                has_afterhours = latest_time > market_close
                
                if has_premarket:
                    print(f"  - ⚠️  Contains pre-market data (before 9:30 AM)")
                if has_afterhours:
                    print(f"  - ⚠️  Contains after-hours data (after 4:00 PM)")
                
                # Check specific EOD times
                df['hour'] = df[timestamp_col].dt.hour
                df['minute'] = df[timestamp_col].dt.minute
                
                # Count bars near EOD
                bars_330pm = len(df[(df['hour'] == 15) & (df['minute'] == 30)])
                bars_350pm = len(df[(df['hour'] == 15) & (df['minute'] == 50)])
                bars_400pm = len(df[(df['hour'] == 16) & (df['minute'] == 0)])
                
                print(f"\n📍 Bars at Key Times:")
                print(f"  - 3:30 PM bars: {bars_330pm}")
                print(f"  - 3:50 PM bars: {bars_350pm}")
                print(f"  - 4:00 PM bars: {bars_400pm}")
                
                # Sample some actual timestamps near EOD
                eod_bars = df[df['hour'] >= 15].head(10)
                if len(eod_bars) > 0:
                    print(f"\n🔍 Sample EOD timestamps:")
                    for _, row in eod_bars.iterrows():
                        print(f"  - {row[timestamp_col]}")
                
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Check all CSV files in the data directory."""
    data_dir = Path('./data')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # Analyze each file
    for csv_file in sorted(csv_files):
        analyze_dataset(csv_file)
    
    # Summary for EOD implementation
    print(f"\n{'='*80}")
    print("EOD CLOSING IMPLEMENTATION NOTES:")
    print(f"{'='*80}")
    print("\nThe current EOD implementation uses bar_of_day calculations:")
    print("- bar_of_day < 72: Allow entries until 3:30 PM (for 5-min bars)")
    print("- bar_of_day >= 78: Force exit at 3:50 PM (for 5-min bars)")
    print("\nFor different timeframes, the calculation needs adjustment:")
    print("- 1-min bars: cutoff=360, exit=390")
    print("- 5-min bars: cutoff=72, exit=78")
    print("- 15-min bars: cutoff=24, exit=26")
    print("- 30-min bars: cutoff=12, exit=13")
    print("- 60-min bars: cutoff=6, exit=6.5")
    print("\nThe implementation should detect the timeframe and adjust accordingly.")

if __name__ == "__main__":
    main()