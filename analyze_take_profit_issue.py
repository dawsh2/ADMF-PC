"""Analyze why take profits are exiting at 0.10% instead of 0.15%."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== TAKE PROFIT EXIT ANALYSIS ===\n")

# Analyze all take profit exits
take_profit_returns = []

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type') == 'take_profit':
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
                            
                            # Calculate return based on position type
                            if is_short:
                                return_pct = (entry_price - exit_price) / entry_price * 100
                            else:
                                return_pct = (exit_price - entry_price) / entry_price * 100
                            
                            take_profit_returns.append({
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'return_pct': return_pct,
                                'is_short': is_short,
                                'exit_reason': nested.get('exit_reason', ''),
                                'order_id': metadata.get('order_id', '')
                            })
                            break

print(f"Total take profit exits: {len(take_profit_returns)}\n")

if take_profit_returns:
    # Count exits at different levels
    at_010 = sum(1 for t in take_profit_returns if 0.095 <= t['return_pct'] <= 0.105)
    at_015 = sum(1 for t in take_profit_returns if 0.145 <= t['return_pct'] <= 0.155)
    
    print(f"Exits at ~0.10%: {at_010} ({at_010/len(take_profit_returns)*100:.1f}%)")
    print(f"Exits at ~0.15%: {at_015} ({at_015/len(take_profit_returns)*100:.1f}%)")
    
    # Show distribution
    print(f"\nReturn distribution:")
    for i in range(0, 20):
        lower = i * 0.01
        upper = (i + 1) * 0.01
        count = sum(1 for t in take_profit_returns if lower <= t['return_pct'] < upper)
        if count > 0:
            print(f"  {lower:.2f}% - {upper:.2f}%: {count} trades")
    
    # Show examples
    print(f"\nFirst 10 take profit exits:")
    for i, tp in enumerate(take_profit_returns[:10]):
        print(f"\n{i+1}. {'SHORT' if tp['is_short'] else 'LONG'}")
        print(f"   Entry: ${tp['entry_price']:.4f}")
        print(f"   Exit:  ${tp['exit_price']:.4f}")
        print(f"   Return: {tp['return_pct']:.4f}%")
        print(f"   Exit reason: {tp['exit_reason']}")

# Check the configuration
print("\n=== CHECKING CONFIGURATION ===")

# Load metadata to see configured values
metadata_path = results_path / "metadata.json"
if metadata_path.exists():
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find strategy config
    for component_name, component_config in metadata.get('components', {}).items():
        if 'bollinger' in component_name.lower() or 'strategy_0' in component_name:
            params = component_config.get('parameters', {})
            risk = params.get('_risk', {})
            print(f"\nStrategy: {component_name}")
            print(f"Stop loss: {risk.get('stop_loss')} ({risk.get('stop_loss', 0) * 100:.3f}%)")
            print(f"Take profit: {risk.get('take_profit')} ({risk.get('take_profit', 0) * 100:.3f}%)")

# Check signal files for actual parameters
signal_path = results_path / "traces/signals/bollinger_bands"
if signal_path.exists():
    signal_files = list(signal_path.glob("*.parquet"))
    if signal_files:
        print(f"\n=== CHECKING SIGNAL FILES ===")
        # Read first signal file
        signal_df = pd.read_parquet(signal_files[0])
        if 'metadata' in signal_df.columns and len(signal_df) > 0:
            # Check first row's metadata
            first_meta = signal_df.iloc[0]['metadata']
            if isinstance(first_meta, dict):
                params = first_meta.get('parameters', {})
                risk = params.get('_risk', {})
                if not risk:
                    risk = first_meta.get('risk', {})
                print(f"From signal file:")
                print(f"Stop loss: {risk.get('stop_loss')} ({risk.get('stop_loss', 0) * 100:.3f}%)")
                print(f"Take profit: {risk.get('take_profit')} ({risk.get('take_profit', 0) * 100:.3f}%)")