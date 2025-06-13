#!/usr/bin/env python3
"""
Example showing the restored strategy-based organization with source metadata
"""

import sys
sys.path.append('src/core/events/storage')
from temporal_sparse_storage import TemporalSparseStorage
from pathlib import Path
import shutil

def demo_restored_structure():
    """Demo the restored strategy-based workspace structure."""
    print("🏗️  Demonstrating Restored Strategy-Based Organization")
    
    # Clean up previous test
    test_workspace = Path("./demo_workspace")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    
    # Create the structure: traces/signals/strategy_type/ and traces/classifiers/classifier_type/
    signals_dir = test_workspace / "traces" / "signals"
    classifiers_dir = test_workspace / "traces" / "classifiers"
    
    # Strategy types with source metadata
    strategy_configs = [
        ("rsi", "SPY_rsi_grid_7_20_70", "SPY", "1m"),
        ("rsi", "QQQ_rsi_grid_14_30_80", "QQQ", "1m"),
        ("momentum", "SPY_momentum_grid_10_25_65", "SPY", "1m"),
        ("ma_crossover", "SPY_ma_crossover_grid_5_20", "SPY", "1m"),
        ("mean_reversion", "SPY_mean_reversion_grid_20_2.0", "SPY", "1m"),
        ("breakout", "SPY_breakout_grid_50_1.5", "SPY", "1m"),
    ]
    
    classifier_configs = [
        ("trend", "SPY_trend_grid_001_20_100", "SPY", "1m"),
        ("volatility", "SPY_volatility_grid_20_05_30", "SPY", "1m"),
        ("market_state", "SPY_market_state_grid_30_100", "SPY", "1m"),
    ]
    
    print("📁 Creating strategy-based organization...")
    
    # Create signal storages
    for strategy_type, component_id, symbol, timeframe in strategy_configs:
        strategy_dir = signals_dir / strategy_type
        
        storage = TemporalSparseStorage(
            base_dir=str(strategy_dir),
            run_id=component_id,
            timeframe=timeframe,
            source_file_path=f"./data/{symbol}_{timeframe}.csv",
            data_source_type="csv"
        )
        
        # Add a test signal with source metadata
        storage.process_signal(symbol, "long", component_id, "2023-01-01T09:30:00", 400.0, 1)
        
        # Save
        filepath = storage.save(tag=component_id)
        print(f"✅ {strategy_type}: {filepath}")
    
    # Create classifier storages  
    for classifier_type, component_id, symbol, timeframe in classifier_configs:
        classifier_dir = classifiers_dir / classifier_type
        
        storage = TemporalSparseStorage(
            base_dir=str(classifier_dir),
            run_id=component_id,
            timeframe=timeframe,
            source_file_path=f"./data/{symbol}_{timeframe}.csv",
            data_source_type="csv"
        )
        
        # Add a test classification
        storage.process_signal(symbol, "trending", component_id, "2023-01-01T09:30:00", 400.0, 1)
        
        # Save
        filepath = storage.save(tag=component_id)
        print(f"✅ {classifier_type}: {filepath}")
    
    # Show final structure
    print(f"\n📊 Final Workspace Structure:")
    print(f"workspaces/config_id/")
    print(f"├── analytics.duckdb")
    print(f"├── metadata.json")
    print(f"└── traces/")
    
    for root, dirs, files in sorted(test_workspace.walk()):
        if root == test_workspace:
            continue
            
        level = len(root.relative_to(test_workspace).parts) - 1
        indent = "    " * level
        
        if root.name in ["signals", "classifiers"]:
            print(f"{indent}├── {root.name}/")
        elif root.parent.name in ["signals", "classifiers"]:
            print(f"{indent}│   ├── {root.name}/")
            # Show files in strategy/classifier directories
            file_indent = "    " * (level + 2)
            for file in sorted(files):
                print(f"{file_indent}│   └── {file}")
    
    print(f"\n🎯 Benefits of Strategy-Based Organization:")
    print(f"  ✅ Clear strategy grouping (rsi/, momentum/, ma_crossover/, etc.)")
    print(f"  ✅ Easy to find all variants of a strategy type")
    print(f"  ✅ Separate signals/ and classifiers/ organization")
    print(f"  ✅ Source metadata preserved in each parquet file")
    print(f"  ✅ Multi-symbol support within each strategy directory")
    
    # Test reading metadata from a file
    print(f"\n📖 Testing Source Metadata Preservation:")
    import pandas as pd
    
    sample_file = signals_dir / "rsi" / "SPY_rsi_grid_7_20_70.parquet"
    if sample_file.exists():
        data = pd.read_parquet(sample_file)
        print(f"  File: {sample_file}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Source metadata:")
        print(f"    Timeframe: {data['tf'].iloc[0] if 'tf' in data.columns else 'Missing'}")
        print(f"    Source file: {data['src_file'].iloc[0] if 'src_file' in data.columns else 'Missing'}")
        print(f"    Source type: {data['src_type'].iloc[0] if 'src_type' in data.columns else 'Missing'}")
    
    # Cleanup
    shutil.rmtree(test_workspace)
    print(f"\n✅ Demo completed and cleaned up!")

if __name__ == "__main__":
    demo_restored_structure()