# Save trades from analysis notebook to CSV for comparison
# Run this in your signal analysis notebook after the trades have been extracted

import pandas as pd
from pathlib import Path

# Assuming 'trades_df' contains your extracted trades from the analysis
# If your trades DataFrame has a different name, replace 'trades_df' below

if 'trades_df' in globals():
    # Save to CSV
    csv_filename = 'analysis_trades_5edc4365.csv'
    trades_df.to_csv(csv_filename, index=False)
    print(f"✅ Saved {len(trades_df)} trades to {csv_filename}")
    
    # Show what was saved
    print(f"\nFirst 5 trades saved:")
    print(trades_df.head())
    
    print(f"\nColumns saved: {trades_df.columns.tolist()}")
    print(f"\nFile saved to: {Path(csv_filename).absolute()}")
else:
    print("❌ 'trades_df' not found in namespace")
    print("\nIf your trades DataFrame has a different name, please update this script.")
    print("Common names: trades, all_trades, trade_list, results_df")
    
    # Show available DataFrames
    print("\nAvailable DataFrames in namespace:")
    for var_name, var_value in globals().items():
        if isinstance(var_value, pd.DataFrame) and not var_name.startswith('_'):
            print(f"  - {var_name}: {len(var_value)} rows, columns: {var_value.columns.tolist()[:5]}...")