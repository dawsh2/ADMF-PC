#!/usr/bin/env python3
"""
Find where trades_df is actually being created
"""

def analyze_trades_df_source(trades_df):
    """Analyze the structure to determine source"""
    
    print("=== Analyzing trades_df Structure ===\n")
    
    print("Columns in trades_df:")
    print(trades_df.columns.tolist())
    print()
    
    print("Data types:")
    print(trades_df.dtypes)
    print()
    
    print("This DataFrame structure suggests it might be created by:")
    
    # Check for specific column patterns
    if 'entry_bar' in trades_df.columns and 'exit_bar' in trades_df.columns:
        print("✓ Code that tracks bar indices")
    
    if 'realized_pnl' in trades_df.columns:
        print("✓ Code that calculates P&L (though it's always 0)")
        
    if 'return_bucket' in trades_df.columns:
        print("✓ Code that buckets returns for analysis")
        
    if 'bars_held' in trades_df.columns:
        print("✓ Code that tracks trade duration")
    
    print("\nThe absence of 'entry_signal' or 'direction' columns suggests")
    print("this is NOT from trace_analysis.py's get_trades() method")
    
    print("\nLikely sources:")
    print("1. A custom analysis script")
    print("2. A notebook that processes traces directly")
    print("3. An older version of the analysis code")
    
    # Show how to find the source
    print("\nTo find the exact source, search for code that creates these columns:")
    print("- 'return_bucket'")
    print("- 'return_per_bar'")
    print("- 'bars_held' AND 'realized_pnl' together")

# Run with your data
analyze_trades_df_source(trades_df)