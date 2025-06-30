"""Debug why profitable trades are exiting at stop loss price."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== ANALYZING PROFITABLE TRADES EXITING AT STOP LOSS PRICE ===\n")

# Track suspicious exits
suspicious_exits = []

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type'):
            exit_type = nested.get('exit_type')
            exit_side = metadata.get('side', '').lower()
            exit_price = float(metadata.get('price', 0))
            
            # Find the previous entry
            for j in range(idx-1, -1, -1):
                prev_row = fills_df.iloc[j]
                prev_meta = prev_row['metadata']
                if isinstance(prev_meta, dict):
                    prev_nested = prev_meta.get('metadata', {})
                    if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                        entry_price = float(prev_meta.get('price', 0))
                        entry_side = prev_meta.get('side', '').lower()
                        
                        if entry_price > 0:
                            # Determine if this was a short position
                            is_short = (entry_side == 'sell' and exit_side == 'buy')
                            
                            # Calculate actual P&L
                            if is_short:
                                pnl_pct = (entry_price - exit_price) / entry_price * 100
                            else:
                                pnl_pct = (exit_price - entry_price) / entry_price * 100
                            
                            # Calculate what the stop loss price should be
                            if is_short:
                                expected_stop_price = entry_price * 1.00075  # Short stop is higher
                                expected_take_price = entry_price * 0.9985   # Short take is lower
                            else:
                                expected_stop_price = entry_price * 0.99925  # Long stop is lower
                                expected_take_price = entry_price * 1.0015   # Long take is higher
                            
                            # Check if this is suspicious:
                            # 1. Trade is profitable (pnl > 0)
                            # 2. Exit price matches stop loss price
                            # 3. Exit type might be wrong
                            
                            is_profitable = pnl_pct > 0
                            exit_matches_stop = abs(exit_price - expected_stop_price) < 0.01
                            exit_matches_take = abs(exit_price - expected_take_price) < 0.01
                            
                            # Also check if exit price is exactly 0.00075
                            is_weird_price = abs(exit_price - 0.00075) < 0.00001
                            
                            if is_profitable and (exit_matches_stop or is_weird_price):
                                suspicious_exits.append({
                                    'order_id': metadata.get('order_id', ''),
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl_pct': pnl_pct,
                                    'is_short': is_short,
                                    'exit_type': exit_type,
                                    'exit_reason': nested.get('exit_reason', ''),
                                    'expected_stop': expected_stop_price,
                                    'expected_take': expected_take_price,
                                    'exit_matches_stop': exit_matches_stop,
                                    'exit_matches_take': exit_matches_take,
                                    'is_weird_price': is_weird_price
                                })
                            
                            # Also check if exit price is literally 0.00075
                            if exit_price < 1:  # Abnormally low price
                                print(f"\n🚨 CRITICAL: Exit price is ${exit_price:.5f} for {metadata.get('order_id', '')}")
                                print(f"   Entry: ${entry_price:.4f}, Exit: ${exit_price:.5f}")
                                print(f"   This looks like a bug - using stop loss percentage as price!")
                            
                            break

print(f"\nFound {len(suspicious_exits)} suspicious exits (profitable but at stop price)\n")

# Analyze the suspicious exits
if suspicious_exits:
    print("Examples of profitable trades exiting at stop loss price:")
    for i, sus in enumerate(suspicious_exits[:10]):
        print(f"\n{i+1}. {sus['order_id']}")
        print(f"   Position: {'SHORT' if sus['is_short'] else 'LONG'}")
        print(f"   Entry: ${sus['entry_price']:.4f}")
        print(f"   Exit:  ${sus['exit_price']:.4f}")
        print(f"   P&L:   {sus['pnl_pct']:.4f}% (PROFITABLE)")
        print(f"   Exit type: {sus['exit_type']}")
        print(f"   Exit reason: {sus['exit_reason']}")
        print(f"   Expected stop: ${sus['expected_stop']:.4f}")
        print(f"   Expected take: ${sus['expected_take']:.4f}")
        if sus['is_weird_price']:
            print(f"   🚨 EXIT PRICE IS 0.00075!")
        elif sus['exit_matches_stop']:
            print(f"   ⚠️  Exit at stop price despite profit!")
        if sus['exit_matches_take']:
            print(f"   ✅ Actually matches take profit price")

# Check for exits at literal 0.00075
literal_stop_exits = [ex for ex in suspicious_exits if ex['is_weird_price']]
if literal_stop_exits:
    print(f"\n\n🚨 CRITICAL ISSUE: {len(literal_stop_exits)} trades exited at price $0.00075")
    print("This suggests the stop loss PERCENTAGE is being used as the PRICE!")