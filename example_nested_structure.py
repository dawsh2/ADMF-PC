#!/usr/bin/env python3
"""
Example showing the nested symbol_timeframe + strategy organization
"""

import sys
sys.path.append('src/core/events/storage')
from temporal_sparse_storage import TemporalSparseStorage
from pathlib import Path
import shutil

def demo_nested_structure():
    """Demo the nested symbol_timeframe/signals+classifiers/strategy_type structure."""
    print("🏗️  Demonstrating Nested Symbol_Timeframe + Strategy Organization")
    
    # Clean up previous test
    test_workspace = Path("./demo_nested")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    
    traces_dir = test_workspace / "traces"
    
    # Test cases with different symbols and timeframes
    test_configs = [
        # SPY 1m strategies
        ("strategy", "rsi", "SPY_rsi_grid_7_20_70", "SPY", "1m"),
        ("strategy", "momentum", "SPY_momentum_grid_10_25_65", "SPY", "1m"),
        ("strategy", "ma_crossover", "SPY_ma_crossover_grid_5_20", "SPY", "1m"),
        
        # SPY 1m classifiers
        ("classifier", "trend", "SPY_trend_grid_001_20_100", "SPY", "1m"),
        ("classifier", "volatility", "SPY_volatility_grid_20_05_30", "SPY", "1m"),
        
        # QQQ 1m strategies
        ("strategy", "rsi", "QQQ_rsi_grid_14_30_80", "QQQ", "1m"),
        ("strategy", "momentum", "QQQ_momentum_grid_20_35_70", "QQQ", "1m"),
        
        # SPY 5m strategies (different timeframe)
        ("strategy", "rsi", "SPY_rsi_grid_21_25_75", "SPY", "5m"),
        ("strategy", "breakout", "SPY_breakout_grid_50_2.0", "SPY", "5m"),
    ]
    
    print("📁 Creating nested organization...")
    
    for component_type, strategy_type, component_id, symbol, timeframe in test_configs:
        # Create the nested directory structure
        symbol_timeframe_dir = traces_dir / f"{symbol}_{timeframe}"
        
        if component_type == 'strategy':
            base_dir = symbol_timeframe_dir / "signals" / strategy_type
        else:
            base_dir = symbol_timeframe_dir / "classifiers" / strategy_type
        
        storage = TemporalSparseStorage(
            base_dir=str(base_dir),
            run_id=component_id,
            timeframe=timeframe,
            source_file_path=f"./data/{symbol}_{timeframe}.csv",
            data_source_type="csv"
        )
        
        # Add a test signal/classification
        signal_value = "long" if component_type == 'strategy' else "trending"
        storage.process_signal(symbol, signal_value, component_id, "2023-01-01T09:30:00", 400.0, 1)
        
        # Save
        filepath = storage.save(tag=component_id)
        relative_path = Path(filepath).relative_to(test_workspace) if filepath else "failed"
        print(f"✅ {component_type}: {relative_path}")
    
    # Show final structure
    print(f"\n📊 Final Nested Workspace Structure:")
    print(f"workspaces/config_id/")
    print(f"├── analytics.duckdb")
    print(f"├── metadata.json")
    print(f"└── traces/")
    
    # Walk through and display the directory tree
    def print_tree(directory, prefix="", is_last=True):
        if directory.is_file():
            return
            
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{directory.name}/")
        
        if directory == test_workspace:
            children = [child for child in directory.iterdir() if child.is_dir()]
        else:
            children = sorted([child for child in directory.iterdir()])
        
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            extension = "    " if is_last else "│   "
            
            if child.is_dir():
                print_tree(child, prefix + extension, is_last_child)
            else:
                child_connector = "└── " if is_last_child else "├── "
                print(f"{prefix}{extension}{child_connector}{child.name}")
    
    for child in sorted(traces_dir.iterdir()):
        if child.is_dir():
            print_tree(child, "    ", child == sorted(traces_dir.iterdir())[-1])
    
    print(f"\n🎯 Benefits of Nested Organization:")
    print(f"  ✅ Symbol/timeframe separation at top level")
    print(f"  ✅ Clear signals/ vs classifiers/ separation")
    print(f"  ✅ Strategy type grouping within each category")
    print(f"  ✅ Easy to find all SPY_1m strategies vs SPY_5m strategies")
    print(f"  ✅ Source metadata preserved in each parquet file")
    print(f"  ✅ Multi-symbol, multi-timeframe support")
    
    # Test reading metadata
    print(f"\n📖 Testing Source Metadata in Nested Structure:")
    import pandas as pd
    
    sample_file = traces_dir / "SPY_1m" / "signals" / "rsi" / "SPY_rsi_grid_7_20_70.parquet"
    if sample_file.exists():
        data = pd.read_parquet(sample_file)
        print(f"  File: {sample_file.relative_to(test_workspace)}")
        print(f"  Timeframe: {data['tf'].iloc[0]}")
        print(f"  Source file: {data['src_file'].iloc[0]}")
        print(f"  Symbol: {data['sym'].iloc[0]}")
    
    # Show example queries this structure enables
    print(f"\n💡 Example Analytics Queries This Structure Enables:")
    print(f"  📈 'Show all SPY 1m RSI strategy variants'")
    print(f"     → traces/SPY_1m/signals/rsi/*.parquet")
    print(f"  📊 'Compare RSI performance across timeframes'")
    print(f"     → traces/SPY_1m/signals/rsi/ vs traces/SPY_5m/signals/rsi/")
    print(f"  🔍 'Analyze all QQQ strategies'")
    print(f"     → traces/QQQ_1m/signals/*/*.parquet")
    print(f"  🎯 'Get all trend classifiers for SPY 1m'")
    print(f"     → traces/SPY_1m/classifiers/trend/*.parquet")
    
    # Cleanup
    shutil.rmtree(test_workspace)
    print(f"\n✅ Demo completed and cleaned up!")

if __name__ == "__main__":
    demo_nested_structure()