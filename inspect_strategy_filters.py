#!/usr/bin/env python3
"""
Inspect the actual filter expressions in compiled strategies.
"""

import json
from pathlib import Path
import pandas as pd

# Check if there's a parameter export file
params_file = Path("config/keltner/results/latest/parameters.json")
if params_file.exists():
    with open(params_file) as f:
        parameters = json.load(f)
    
    print("📋 Strategy Filter Configurations:\n")
    
    # Group by filter type
    no_filter = []
    with_filter = []
    
    for strategy_id, params in parameters.items():
        if 'filter' in params:
            with_filter.append((strategy_id, params))
        else:
            no_filter.append((strategy_id, params))
    
    print(f"✅ Strategies without filters: {len(no_filter)}")
    print(f"🔧 Strategies with filters: {len(with_filter)}")
    
    # Show some examples
    print("\n📊 Filter Examples:\n")
    
    # Show different filter types
    filter_types = {}
    for strategy_id, params in with_filter[:50]:  # Check first 50
        filter_expr = params.get('filter', 'No filter')
        
        # Categorize by filter content
        if 'rsi' in filter_expr and 'volume' in filter_expr:
            category = "Combined RSI+Volume"
        elif 'rsi' in filter_expr and 'signal > 0' in filter_expr:
            category = "Directional RSI"
        elif 'rsi' in filter_expr:
            category = "RSI Filter"
        elif 'volume' in filter_expr:
            category = "Volume Filter"
        elif 'atr' in filter_expr:
            category = "Volatility Filter"
        elif 'vwap' in filter_expr:
            category = "VWAP Filter"
        elif 'bar_of_day' in filter_expr:
            category = "Time Filter"
        else:
            category = "Other"
        
        if category not in filter_types:
            filter_types[category] = []
        filter_types[category].append((strategy_id, filter_expr))
    
    # Display examples from each category
    for category, examples in filter_types.items():
        print(f"\n{category} ({len(examples)} strategies):")
        # Show first example
        if examples:
            strategy_id, filter_expr = examples[0]
            # Truncate long filters
            if len(filter_expr) > 100:
                filter_expr = filter_expr[:97] + "..."
            print(f"  Example: {filter_expr}")

else:
    print("❌ No parameters.json file found. Checking metadata...")
    
    # Try metadata file
    metadata_file = Path("config/keltner/results/latest/metadata.json")
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        print("\n📊 Metadata Summary:")
        print(f"Config: {metadata.get('config_name', 'Unknown')}")
        print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"Total strategies: {metadata.get('total_strategies', 'Unknown')}")
        
        # Check if strategies info is in metadata
        if 'strategies' in metadata:
            print("\n🔍 Strategy Details:")
            strategies = metadata['strategies']
            if isinstance(strategies, list):
                for i, strategy in enumerate(strategies[:5]):
                    print(f"\nStrategy {i}:")
                    print(f"  Type: {strategy.get('type', 'Unknown')}")
                    if 'filter' in strategy:
                        print(f"  Filter: {strategy['filter'][:80]}...")
    else:
        print("❌ No metadata.json file found either.")
        
    print("\n💡 To see filter details, you may need to:")
    print("   1. Check the compiled strategy objects")
    print("   2. Look at the actual signal differences in the parquet files")
    print("   3. Run with --export-parameters flag if available")