# Find what variables contain trade data in your notebook
import pandas as pd

print("Looking for variables that might contain trade data...\n")

# Get all variables safely
all_vars = list(globals().keys())

# Look for DataFrames
print("DataFrames found:")
for var_name in all_vars:
    try:
        var_value = globals()[var_name]
        if isinstance(var_value, pd.DataFrame) and not var_name.startswith('_'):
            # Check if it looks like trade data
            cols = var_value.columns.tolist()
            if any(col in cols for col in ['entry_time', 'exit_time', 'return', 'entry_price', 'exit_price', 'dir', 'direction']):
                print(f"  📊 {var_name}: {len(var_value)} rows")
                print(f"     Columns: {cols[:8]}...")
                print("")
    except:
        pass

# Look for lists that might be trades
print("\nLists found:")
for var_name in all_vars:
    try:
        var_value = globals()[var_name]
        if isinstance(var_value, list) and len(var_value) > 0 and not var_name.startswith('_'):
            # Check if first item looks like a trade
            first_item = var_value[0]
            if isinstance(first_item, dict):
                keys = list(first_item.keys())
                if any(key in keys for key in ['entry_time', 'exit_time', 'return', 'entry_price', 'num']):
                    print(f"  📋 {var_name}: {len(var_value)} items")
                    print(f"     Keys: {keys[:8]}...")
                    print("")
    except:
        pass

print("\nTo save your trades:")
print("1. If you found a DataFrame above, run:")
print("   your_dataframe_name.to_csv('analysis_trades_5edc4365.csv', index=False)")
print("")
print("2. If you found a list above, run:")
print("   trades_df = pd.DataFrame(your_list_name)")
print("   trades_df.to_csv('analysis_trades_5edc4365.csv', index=False)")