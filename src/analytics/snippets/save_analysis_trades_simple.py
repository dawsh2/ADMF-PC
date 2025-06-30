# Save trades from analysis notebook to CSV for comparison
# Run this in your signal analysis notebook after the trades have been extracted

import pandas as pd
from pathlib import Path

# Check common trade DataFrame names
possible_names = ['trades_df', 'trades', 'all_trades', 'trade_list', 'results_df', 'trade_results', 'extracted_trades']

found_df = None
found_name = None

for name in possible_names:
    if name in globals() and isinstance(globals()[name], pd.DataFrame):
        found_df = globals()[name]
        found_name = name
        print(f"✅ Found trades DataFrame: '{found_name}' with {len(found_df)} rows")
        break

if found_df is not None:
    # Save to CSV
    csv_filename = 'analysis_trades_5edc4365.csv'
    found_df.to_csv(csv_filename, index=False)
    print(f"✅ Saved {len(found_df)} trades to {csv_filename}")
    
    # Show what was saved
    print(f"\nFirst 5 trades saved:")
    print(found_df.head())
    
    print(f"\nColumns saved: {found_df.columns.tolist()}")
    print(f"\nFile saved to: {Path(csv_filename).absolute()}")
else:
    print("❌ No trades DataFrame found with common names")
    print("\nTo save your trades, run one of these commands:")
    print("  1. If your trades are in a list called 'trades':")
    print("     trades_df = pd.DataFrame(trades)")
    print("     %run /path/to/this/script.py")
    print("")
    print("  2. If your trades have a different name, replace 'your_trades_variable' below:")
    print("     trades_df = your_trades_variable")
    print("     %run /path/to/this/script.py")
    print("")
    print("  3. Or save directly:")
    print("     your_trades_variable.to_csv('analysis_trades_5edc4365.csv', index=False)")