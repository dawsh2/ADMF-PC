#!/usr/bin/env python3
"""Compare signal traces between notebook analysis and latest system run"""

import pandas as pd
import numpy as np

# Load the signal traces
print("Loading signal traces...")

# Latest system run
latest = pd.read_parquet('config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

# Test run from notebook (June 25)
test_run = pd.read_parquet('config/bollinger/results/20250625_173629/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print(f"\nLatest run (with stops/targets in config):")
print(f"  Rows: {len(latest)}")
print(f"  Non-zero signals: {(latest['val'] != 0).sum()}")
print(f"  Strategy hash: {latest['strategy_hash'].iloc[0] if 'strategy_hash' in latest.columns else 'N/A'}")
print(f"  Date range: {latest['ts'].min()} to {latest['ts'].max()}")

print(f"\nTest run (June 25 - used in notebook):")
print(f"  Rows: {len(test_run)}")
print(f"  Non-zero signals: {(test_run['val'] != 0).sum()}")
print(f"  Strategy hash: {test_run['strategy_hash'].iloc[0] if 'strategy_hash' in test_run.columns else 'N/A'}")
print(f"  Date range: {test_run['ts'].min()} to {test_run['ts'].max()}")

# Compare signals
print("\n" + "="*60)
print("SIGNAL COMPARISON")
print("="*60)

if len(latest) == len(test_run):
    # Compare signal values
    signal_matches = (latest['val'].values == test_run['val'].values).all()
    price_matches = np.allclose(latest['px'].values, test_run['px'].values, rtol=1e-9)
    
    print(f"Signal values match: {signal_matches}")
    print(f"Prices match: {price_matches}")
    
    if not signal_matches:
        # Find differences
        diffs = latest[latest['val'] != test_run['val']]
        print(f"\nFound {len(diffs)} signal differences:")
        print(diffs[['ts', 'val', 'px']].head(10))
    
    # Compare strategy parameters
    print("\n" + "="*60)
    print("STRATEGY PARAMETERS")
    print("="*60)
    
    # Check metadata
    if 'metadata' in latest.columns and latest['metadata'].iloc[0] is not None:
        latest_meta = latest['metadata'].iloc[0]
        if isinstance(latest_meta, dict):
            print("\nLatest run parameters:")
            params = latest_meta.get('parameters', {})
            print(f"  Period: {params.get('period', 'N/A')}")
            print(f"  Std Dev: {params.get('std_dev', 'N/A')}")
            
            risk = latest_meta.get('risk', {})
            print(f"\nRisk parameters in latest run:")
            print(f"  Stop loss: {risk.get('stop_loss', 'N/A')}")
            print(f"  Take profit: {risk.get('take_profit', 'N/A')}")
            print(f"  Trailing stop: {risk.get('trailing_stop', 'N/A')}")
    
    # Check fills and portfolio data
    print("\n" + "="*60)
    print("EXECUTION DATA CHECK")
    print("="*60)
    
    # Check if execution traces exist
    import os
    fills_path = 'config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet'
    if os.path.exists(fills_path):
        fills = pd.read_parquet(fills_path)
        print(f"\nFills found: {len(fills)}")
        if len(fills) > 0:
            print("\nFills columns:", fills.columns.tolist())
            # Extract metadata fields
            if 'metadata' in fills.columns:
                import json
                first_fill = json.loads(fills.iloc[0]['metadata'])
                print("\nMetadata fields:", list(first_fill.keys()))
                print("\nFirst fill details:")
                print(f"  Fill ID: {first_fill.get('fill_id', 'N/A')}")
                print(f"  Side: {first_fill.get('side', 'N/A')}")
                print(f"  Price: {first_fill.get('fill_price', 'N/A')}")
                print(f"  Quantity: {first_fill.get('quantity', 'N/A')}")
                print(f"  Order reason: {first_fill.get('order_reason', 'N/A')}")
    else:
        print("\n❌ No fills data found!")
    
    positions_path = 'config/bollinger/results/latest/traces/portfolio/positions_close/positions_close.parquet'
    if os.path.exists(positions_path):
        positions = pd.read_parquet(positions_path)
        print(f"\nPositions found: {len(positions)}")
        if len(positions) > 0:
            print("\nPositions columns:", positions.columns.tolist())
            # Try to extract position data from metadata
            if 'metadata' in positions.columns:
                import json
                first_pos = json.loads(positions.iloc[0]['metadata'])
                print("\nPosition metadata fields:", list(first_pos.keys()))
                print("\nFirst position details:")
                print(f"  Symbol: {first_pos.get('symbol', 'N/A')}")
                print(f"  Entry price: {first_pos.get('entry_price', 'N/A')}")
                print(f"  Exit price: {first_pos.get('exit_price', 'N/A')}")
                print(f"  Realized P&L: {first_pos.get('realized_pnl', 'N/A')}")
    else:
        print("\n❌ No positions data found!")
        
else:
    print(f"❌ Different number of rows: {len(latest)} vs {len(test_run)}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print("""
The issue might be:

1. Risk parameters are in the config but not being applied during execution
2. The execution engine is not implementing stops/targets correctly
3. The portfolio state is not tracking P&L properly
4. The notebook analysis uses different logic than the system

Check:
- Are stops/targets being triggered in the execution engine?
- Is the portfolio correctly calculating realized P&L?
- Are fills being generated at the right prices?
""")