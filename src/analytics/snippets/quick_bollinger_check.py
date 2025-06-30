# Quick check of Bollinger signals
import pandas as pd
from pathlib import Path

# Force the correct directory
signal_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_185742/traces/signals/bollinger_bands')
print(f"Checking directory: {signal_dir}")
print(f"Directory exists: {signal_dir.exists()}")

if signal_dir.exists():
    files = list(signal_dir.glob('*.parquet'))
    print(f"\nFound {len(files)} parquet files")
    
    # Check first 5 files
    for f in files[:5]:
        df = pd.read_parquet(f)
        non_zero = (df['val'] != 0).sum()
        print(f"\n{f.name}:")
        print(f"  Total rows: {len(df)}")
        print(f"  Non-zero signals: {non_zero}")
        print(f"  Signal distribution: {dict(df['val'].value_counts())}")
        
    # Analyze one file in detail
    print("\n" + "="*60)
    print("Detailed analysis of first file with signals:")
    print("="*60)
    
    for f in files:
        df = pd.read_parquet(f)
        if (df['val'] != 0).sum() > 100:  # Find file with good signal count
            print(f"\nAnalyzing {f.name}")
            
            # Convert timestamps
            df['ts'] = pd.to_datetime(df['ts'])
            if hasattr(df['ts'].dtype, 'tz'):
                df['ts'] = df['ts'].dt.tz_localize(None)
            
            # Sort by time
            df = df.sort_values('ts')
            
            # Extract trades
            trades = []
            current_pos = 0
            entry = None
            
            for _, row in df.iterrows():
                if current_pos == 0 and row['val'] != 0:
                    # Entry
                    current_pos = row['val']
                    entry = {'time': row['ts'], 'price': row['px'], 'direction': row['val']}
                elif current_pos != 0 and row['val'] != current_pos:
                    # Exit
                    if entry:
                        exit_price = row['px']
                        if entry['direction'] > 0:
                            ret = (exit_price - entry['price']) / entry['price']
                        else:
                            ret = (entry['price'] - exit_price) / entry['price']
                        
                        trades.append({
                            'entry_time': entry['time'],
                            'exit_time': row['ts'],
                            'entry_price': entry['price'],
                            'exit_price': exit_price,
                            'return': ret,
                            'duration_min': (row['ts'] - entry['time']).total_seconds() / 60
                        })
                    
                    # Update position
                    current_pos = row['val']
                    if row['val'] != 0:
                        entry = {'time': row['ts'], 'price': row['px'], 'direction': row['val']}
                    else:
                        entry = None
            
            if trades:
                trades_df = pd.DataFrame(trades)
                print(f"\nExtracted {len(trades_df)} trades")
                print(f"Total return: {(1 + trades_df['return']).prod() - 1:.2%}")
                print(f"Win rate: {(trades_df['return'] > 0).mean():.1%}")
                print(f"Avg return per trade: {trades_df['return'].mean():.3%}")
                print(f"Avg duration: {trades_df['duration_min'].mean():.1f} minutes")
                
                # Show first 5 trades
                print("\nFirst 5 trades:")
                for i, trade in trades_df.head().iterrows():
                    print(f"  {i+1}. Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} = {trade['return']:.3%}")
            
            break  # Just analyze one file
            
else:
    print(f"\n❌ Directory not found: {signal_dir}")