# Save the extracted trades to CSV
# Run this right after the extract_trades() call in your notebook

import pandas as pd

# Check if 'trades' exists from the extract_trades call
if 'trades' in globals():
    print(f"✅ Found 'trades' variable with {len(trades)} trades")
    
    # Save to CSV
    csv_filename = 'analysis_trades_5edc4365.csv'
    trades.to_csv(csv_filename, index=False)
    print(f"✅ Saved {len(trades)} trades to {csv_filename}")
    
    # Show what was saved
    print(f"\nFirst 5 trades:")
    print(trades.head())
    
    print(f"\nColumns: {trades.columns.tolist()}")
    print(f"\nExit types in saved data:")
    if 'exit_type' in trades.columns:
        print(trades['exit_type'].value_counts())
else:
    print("❌ 'trades' variable not found")
    print("\nMake sure you run this after calling extract_trades() in your notebook")
    print("The code should look like:")
    print("  trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)")
    print("  %run this_script.py")