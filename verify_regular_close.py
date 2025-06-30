#!/usr/bin/env python3
"""
Verify that trading signals go to zero by regular market close (20:00 UTC / 4:00 PM ET).
"""

import pandas as pd
from pathlib import Path
from datetime import time

def check_regular_close_compliance(parquet_file):
    """Check if signals go to zero by 20:00 UTC (4:00 PM ET)"""
    # Load sparse signals
    signals = pd.read_parquet(parquet_file)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    # Add date and time components
    signals['date'] = signals['ts'].dt.date
    signals['time'] = signals['ts'].dt.time
    signals['hour'] = signals['ts'].dt.hour
    signals['minute'] = signals['ts'].dt.minute
    
    # Group by date and check signals after regular close
    daily_summary = []
    
    for date, day_signals in signals.groupby('date'):
        # Sort by timestamp
        day_signals = day_signals.sort_values('ts')
        
        # Find last signal before or at 20:00 UTC (4:00 PM ET)
        regular_hours = day_signals[day_signals['ts'].dt.time <= time(20, 0)]
        extended_hours = day_signals[day_signals['ts'].dt.time > time(20, 0)]
        
        if len(regular_hours) > 0:
            last_regular_signal = regular_hours.iloc[-1]
            last_regular_value = last_regular_signal['val']
            last_regular_time = last_regular_signal['time']
        else:
            last_regular_value = 0
            last_regular_time = None
        
        # Check if there are extended hours trades
        has_extended = len(extended_hours) > 0
        if has_extended:
            first_extended = extended_hours.iloc[0]
            extended_signal = first_extended['val']
            extended_time = first_extended['time']
        else:
            extended_signal = None
            extended_time = None
        
        daily_summary.append({
            'date': date,
            'last_regular_time': last_regular_time,
            'last_regular_signal': last_regular_value,
            'closed_by_regular': last_regular_value == 0,
            'has_extended_trading': has_extended,
            'extended_signal': extended_signal,
            'extended_time': extended_time
        })
    
    summary_df = pd.DataFrame(daily_summary)
    
    # Calculate statistics
    total_days = len(summary_df)
    closed_days = summary_df['closed_by_regular'].sum()
    extended_days = summary_df['has_extended_trading'].sum()
    
    return {
        'total_days': total_days,
        'closed_by_regular': closed_days,
        'not_closed_by_regular': total_days - closed_days,
        'days_with_extended': extended_days,
        'compliance_rate': closed_days / total_days if total_days > 0 else 0,
        'violations': summary_df[~summary_df['closed_by_regular']],
        'summary_df': summary_df
    }

def main():
    # Check a sample of strategy files
    results_dir = Path('config/bollinger/results/20250624_150142/traces/SPY_5m_1m/signals/bollinger_bands')
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return
    
    # Get sample of parquet files
    parquet_files = list(results_dir.glob('*.parquet'))[:20]  # Check first 20
    
    print(f"🔍 Checking {len(parquet_files)} strategies for regular close compliance...")
    print(f"📍 Regular market close: 20:00 UTC (4:00 PM ET)\n")
    
    compliant_strategies = []
    non_compliant_strategies = []
    
    for i, pfile in enumerate(parquet_files):
        try:
            result = check_regular_close_compliance(pfile)
            
            compliance_pct = result['compliance_rate'] * 100
            strategy_name = pfile.stem
            
            if compliance_pct == 100:
                compliant_strategies.append(strategy_name)
            else:
                non_compliant_strategies.append({
                    'name': strategy_name,
                    'compliance': compliance_pct,
                    'violations': result['not_closed_by_regular'],
                    'extended_days': result['days_with_extended']
                })
            
            if compliance_pct < 100:
                print(f"\n{'='*60}")
                print(f"Strategy: {strategy_name}")
                print(f"Compliance: {compliance_pct:.1f}%")
                print(f"Days not closed by 4PM ET: {result['not_closed_by_regular']}/{result['total_days']}")
                print(f"Days with extended trading: {result['days_with_extended']}")
                
                # Show some violations
                violations = result['violations'].head(5)
                if len(violations) > 0:
                    print("\nExample violations (position held past 4PM ET):")
                    for _, v in violations.iterrows():
                        print(f"  {v['date']}: Signal={v['last_regular_signal']} at {v['last_regular_time']}")
                        if v['has_extended_trading']:
                            print(f"    → Extended trading at {v['extended_time']}: {v['extended_signal']}")
        
        except Exception as e:
            print(f"❌ Error processing {pfile.name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"✅ Fully compliant (always close by 4PM ET): {len(compliant_strategies)} strategies")
    print(f"❌ Non-compliant (trade past 4PM ET): {len(non_compliant_strategies)} strategies")
    
    if non_compliant_strategies:
        print("\n📈 Non-compliant strategies ranked by severity:")
        sorted_violations = sorted(non_compliant_strategies, key=lambda x: x['compliance'])
        for s in sorted_violations[:10]:
            print(f"  {s['name']}: {s['compliance']:.1f}% compliant ({s['violations']} violations)")

if __name__ == '__main__':
    main()