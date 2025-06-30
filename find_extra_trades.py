#!/usr/bin/env python3
"""Find where the extra trades are coming from."""

import pandas as pd
import json

# Load your positions
positions = pd.read_parquet('config/bollinger/results/latest/traces/portfolio/positions_close/positions_close.parquet')

# Extract exit types and analyze patterns
exit_data = []
for _, row in positions.iterrows():
    meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    exit_data.append({
        'ts': pd.to_datetime(row['ts']),
        'exit_type': meta.get('exit_type', 'unknown'),
        'exit_reason': meta.get('exit_reason', ''),
        'entry_price': float(meta.get('entry_price', 0)),
        'exit_price': float(meta.get('exit_price', 0))
    })

df = pd.DataFrame(exit_data)
df = df.sort_values('ts')

print("=== Trade Analysis ===")
print(f"Total trades: {len(df)}")
print(f"\nExit type breakdown:")
for exit_type, count in df['exit_type'].value_counts().items():
    print(f"  {exit_type}: {count}")

# Look for re-entries after risk exits
print("\n=== Re-entry Analysis ===")

# Find all risk-based exits (stop_loss, take_profit)
risk_exits = df[df['exit_type'].isin(['stop_loss', 'take_profit'])]
print(f"Risk-based exits: {len(risk_exits)}")

# For each risk exit, check if there's a quick re-entry
re_entries = []
for idx, risk_exit in risk_exits.iterrows():
    # Find next trade
    next_trades = df[df['ts'] > risk_exit['ts']].head(1)
    if len(next_trades) > 0:
        next_trade = next_trades.iloc[0]
        time_diff = (next_trade['ts'] - risk_exit['ts']).total_seconds() / 60
        
        if time_diff < 60:  # Within 1 hour
            re_entries.append({
                'exit_time': risk_exit['ts'],
                'exit_type': risk_exit['exit_type'],
                'next_entry_time': next_trade['ts'],
                'minutes_to_re_entry': time_diff
            })

print(f"\nRe-entries within 1 hour of risk exit: {len(re_entries)}")
if re_entries:
    re_df = pd.DataFrame(re_entries)
    print(f"Average time to re-entry: {re_df['minutes_to_re_entry'].mean():.1f} minutes")
    
    # Show first few examples
    print("\nExamples:")
    for i, re in enumerate(re_df.head(5).itertuples()):
        print(f"  {i+1}. {re.exit_type} at {re.exit_time.strftime('%Y-%m-%d %H:%M')}, "
              f"re-entry {re.minutes_to_re_entry:.1f} min later")

# Check if we're getting signal reversals
print("\n=== Signal Reversal Analysis ===")
reversals = df[df['exit_reason'].str.contains('reversal', case=False, na=False)]
print(f"Reversal exits: {len(reversals)}")

# The notebook had 416 trades. We have 453. That's 37 extra.
# Let's see if it's from re-entries after stops/targets
print(f"\n=== Summary ===")
print(f"Expected trades: 416")
print(f"Actual trades: {len(df)}")
print(f"Extra trades: {len(df) - 416}")
print(f"\nPossible sources of extra trades:")
print(f"  - Risk exits (stop/target): {len(risk_exits)}")
print(f"  - Quick re-entries: {len(re_entries)}")
print(f"  - Reversals: {len(reversals)}")

# If risk exits + signal exits > 416, we might be re-entering after risk exits
total_without_reentry = len(df[df['exit_type'] == 'signal'])
print(f"\nSignal-only exits: {total_without_reentry}")
print(f"If no re-entries after risk exits, we'd have: {total_without_reentry} trades")