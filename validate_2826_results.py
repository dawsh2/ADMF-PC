#!/usr/bin/env python3
"""
Validate the results from running config_2826 to ensure it matches expected performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob

def validate_2826_results():
    print("=== VALIDATING 2826 STRATEGY RESULTS ===\n")
    
    # Find the results directory
    base_path = Path("/Users/daws/ADMF-PC/config/keltner/config_2826")
    
    # Look for result directories
    result_dirs = sorted(glob.glob(str(base_path / "results" / "*")))
    
    if not result_dirs:
        print("ERROR: No results found in config/keltner/config_2826/results/")
        return
    
    latest_result = result_dirs[-1]
    print(f"Found results in: {latest_result}")
    
    # Load metadata
    metadata_path = Path(latest_result) / "metadata.json"
    
    if not metadata_path.exists():
        print("ERROR: metadata.json not found")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check signal counts
    print("\n1. SIGNAL COUNT VALIDATION:")
    print("-" * 50)
    
    for comp_name, comp_data in metadata['components'].items():
        if comp_name.startswith('SPY_5m_compiled_strategy_'):
            signal_count = comp_data.get('signal_changes', 0)
            print(f"Strategy: {comp_name}")
            print(f"Signal changes: {signal_count}")
            
            if signal_count == 2826:
                print("✓ CORRECT: Matches expected 2826 signals")
            else:
                print(f"✗ ERROR: Expected 2826, got {signal_count}")
    
    # Load and analyze the actual signals
    traces_dir = Path(latest_result) / "traces" / "keltner_bands"
    signal_files = list(traces_dir.glob("*.parquet"))
    
    if signal_files:
        print("\n2. PERFORMANCE VALIDATION:")
        print("-" * 50)
        
        # Analyze first signal file
        signals_df = pd.read_parquet(signal_files[0])
        
        # Calculate trades
        trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            
            if signal != 0:
                if current_position is not None:
                    # Close existing position
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price']) * 10000
                    else:
                        ret = -np.log(price / current_position['entry_price']) * 10000
                    
                    trades.append({
                        'return_bps': ret,
                        'direction': current_position['direction']
                    })
                
                # Open new position
                current_position = {
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                # Exit signal
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price']) * 10000
                else:
                    ret = -np.log(price / current_position['entry_price']) * 10000
                
                trades.append({
                    'return_bps': ret,
                    'direction': current_position['direction']
                })
                current_position = None
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate metrics (GROSS)
            avg_return_gross = trades_df['return_bps'].mean()
            
            # Apply costs for NET
            trades_df['return_net'] = trades_df['return_bps'] - 0.5
            avg_return_net = trades_df['return_net'].mean()
            
            # Other metrics
            total_trades = len(trades_df)
            trades_per_day = total_trades / 252
            win_rate = (trades_df['return_net'] > 0).mean() * 100
            
            # Directional breakdown
            long_trades = trades_df[trades_df['direction'] == 'long']
            short_trades = trades_df[trades_df['direction'] == 'short']
            
            long_return_net = long_trades['return_net'].mean() if len(long_trades) > 0 else 0
            short_return_net = short_trades['return_net'].mean() if len(short_trades) > 0 else 0
            
            # Annual returns
            annual_gross = avg_return_gross * total_trades / 100
            annual_net = avg_return_net * total_trades / 100
            
            print(f"Total trades: {total_trades}")
            print(f"Trades per day: {trades_per_day:.1f}")
            print(f"\nRETURNS:")
            print(f"Gross return: {avg_return_gross:.2f} bps/trade")
            print(f"Net return: {avg_return_net:.2f} bps/trade (after 0.5 bps costs)")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"\nDIRECTIONAL:")
            print(f"Long trades: {len(long_trades)} ({long_return_net:.2f} bps net)")
            print(f"Short trades: {len(short_trades)} ({short_return_net:.2f} bps net)")
            print(f"\nANNUAL:")
            print(f"Gross annual: {annual_gross:.1f}%")
            print(f"Net annual: {annual_net:.1f}%")
            
            # Validation checks
            print("\n3. VALIDATION AGAINST EXPECTED:")
            print("-" * 50)
            
            expected = {
                'signal_count': 2826,
                'trades': 1429,
                'trades_per_day': 5.7,
                'gross_return': 0.68,
                'net_return': 0.18,
                'win_rate': 71.2,
                'annual_gross': 9.7,
                'annual_net': 2.6
            }
            
            print(f"{'Metric':<20} {'Expected':<10} {'Actual':<10} {'Match':<10}")
            print("-" * 50)
            
            checks = [
                ('Signal count', expected['signal_count'], signal_count, abs(signal_count - expected['signal_count']) < 10),
                ('Total trades', expected['trades'], total_trades, abs(total_trades - expected['trades']) < 50),
                ('Trades/day', expected['trades_per_day'], trades_per_day, abs(trades_per_day - expected['trades_per_day']) < 0.5),
                ('Gross return', expected['gross_return'], avg_return_gross, abs(avg_return_gross - expected['gross_return']) < 0.1),
                ('Net return', expected['net_return'], avg_return_net, abs(avg_return_net - expected['net_return']) < 0.1),
                ('Win rate', expected['win_rate'], win_rate, abs(win_rate - expected['win_rate']) < 5),
                ('Annual gross', expected['annual_gross'], annual_gross, abs(annual_gross - expected['annual_gross']) < 1),
                ('Annual net', expected['annual_net'], annual_net, abs(annual_net - expected['annual_net']) < 0.5)
            ]
            
            all_match = True
            for metric, exp, act, match in checks:
                status = "✓" if match else "✗"
                all_match = all_match and match
                print(f"{metric:<20} {exp:<10.1f} {act:<10.1f} {status:<10}")
            
            print("\n4. OVERALL VALIDATION:")
            print("-" * 50)
            if all_match:
                print("✓ SUCCESS: Results match expected performance!")
                print("  This confirms we've correctly reproduced the 2826 strategy")
            else:
                print("✗ WARNING: Some metrics don't match expected values")
                print("  This could be due to:")
                print("  - Different data period")
                print("  - Implementation differences")
                print("  - Config variations")
    
    # Check config used
    print("\n5. CONFIG VERIFICATION:")
    print("-" * 50)
    
    # Try to load the config that was used
    config_path = Path(latest_result).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
            if 'volatility_above' in config_content and '1.1' in config_content:
                print("✓ Config uses volatility filter with threshold ~1.1")
            else:
                print("✗ Config doesn't match expected volatility filter")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    validate_2826_results()