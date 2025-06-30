#!/usr/bin/env python3
"""
Verify that trading signals go to zero before market close each day.
This ensures strategies are truly intraday and not holding overnight.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import time

def check_intraday_signals(parquet_file):
    """Check if signals go to zero before day end"""
    # Load sparse signals
    signals = pd.read_parquet(parquet_file)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    # Add date and time components
    signals['date'] = signals['ts'].dt.date
    signals['time'] = signals['ts'].dt.time
    
    # Group by date and check last signal
    daily_summary = []
    
    for date, day_signals in signals.groupby('date'):
        # Sort by timestamp
        day_signals = day_signals.sort_values('ts')
        
        # Get last signal of the day
        last_signal = day_signals.iloc[-1]
        last_time = last_signal['time']
        last_value = last_signal['val']
        
        # Check if last signal is zero
        is_zero = (last_value == 0)
        
        # Market closes at 16:00 ET, but we might see signals until 16:15
        is_near_close = last_time >= time(15, 45)  # 3:45 PM or later
        
        daily_summary.append({
            'date': date,
            'last_time': last_time,
            'last_signal': last_value,
            'is_zero': is_zero,
            'is_near_close': is_near_close
        })
    
    summary_df = pd.DataFrame(daily_summary)
    
    # Calculate statistics
    total_days = len(summary_df)
    zero_days = summary_df['is_zero'].sum()
    non_zero_days = total_days - zero_days
    
    # Show violations (non-zero at day end)
    violations = summary_df[~summary_df['is_zero']]
    
    return {
        'total_days': total_days,
        'zero_days': zero_days,
        'non_zero_days': non_zero_days,
        'compliance_rate': zero_days / total_days if total_days > 0 else 0,
        'violations': violations,
        'summary_df': summary_df
    }

def main():
    # Check a sample of strategy files
    results_dir = Path('config/bollinger/results/20250624_150142/traces/SPY_5m_1m/signals/bollinger_bands')
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return
    
    # Get sample of parquet files
    parquet_files = list(results_dir.glob('*.parquet'))[:10]  # Check first 10
    
    print(f"🔍 Checking {len(parquet_files)} strategy files for intraday compliance...\n")
    
    all_compliant = True
    
    for i, pfile in enumerate(parquet_files):
        print(f"\n{'='*60}")
        print(f"Strategy {i+1}: {pfile.name}")
        
        try:
            result = check_intraday_signals(pfile)
            
            print(f"Total trading days: {result['total_days']}")
            print(f"Days ending at zero: {result['zero_days']}")
            print(f"Days with positions held: {result['non_zero_days']}")
            print(f"Compliance rate: {result['compliance_rate']:.1%}")
            
            if result['non_zero_days'] > 0:
                all_compliant = False
                print(f"\n⚠️  VIOLATIONS FOUND - Positions held overnight:")
                violations = result['violations']
                for _, row in violations.head(5).iterrows():
                    print(f"   {row['date']}: Last signal at {row['last_time']} = {row['last_signal']}")
                
                if len(violations) > 5:
                    print(f"   ... and {len(violations) - 5} more violations")
        
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    print(f"\n{'='*60}")
    if all_compliant:
        print("✅ All checked strategies are intraday compliant!")
    else:
        print("❌ Some strategies are holding positions overnight!")
        
    # Deep dive into one file
    print(f"\n{'='*60}")
    print("DEEP DIVE - Checking signal patterns for first strategy:")
    
    if parquet_files:
        signals = pd.read_parquet(parquet_files[0])
        signals['ts'] = pd.to_datetime(signals['ts'])
        signals['date'] = signals['ts'].dt.date
        signals['time'] = signals['ts'].dt.time
        
        # Show last few signals for each day
        print("\nLast 3 signals per day (first 5 days):")
        for i, (date, day_signals) in enumerate(signals.groupby('date')):
            if i >= 5:
                break
            day_signals = day_signals.sort_values('ts')
            print(f"\n{date}:")
            for _, row in day_signals.tail(3).iterrows():
                print(f"  {row['time']}: {row['val']}")

if __name__ == '__main__':
    main()