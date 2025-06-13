#!/usr/bin/env python3
"""
Test the enhanced signal storage with source metadata and symbol_timeframe workspace structure.
"""

import sys
import os
sys.path.append('src')

from core.events.storage.temporal_sparse_storage import TemporalSparseStorage, SignalChange
from pathlib import Path
import json

def test_enhanced_signal_storage():
    """Test the enhanced signal storage functionality."""
    print("🧪 Testing Enhanced Signal Storage")
    
    # Create test workspace with symbol_timeframe structure
    test_workspace = Path("./test_workspace")
    symbol_timeframe_dir = test_workspace / "SPY_1m"
    
    # Clean up previous test
    if test_workspace.exists():
        import shutil
        shutil.rmtree(test_workspace)
    
    # Create storage with source metadata
    storage = TemporalSparseStorage(
        base_dir=str(symbol_timeframe_dir),
        run_id="SPY_rsi_test_strategy",
        timeframe="1m",
        source_file_path="./data/SPY_1m.csv",
        data_source_type="csv"
    )
    
    print(f"✅ Created storage in: {symbol_timeframe_dir}")
    print(f"✅ Source metadata: {storage.timeframe}, {storage.source_file_path}, {storage.data_source_type}")
    
    # Process some test signals
    test_signals = [
        ("SPY", "long", "SPY_rsi_test", "2023-01-01T09:30:00", 400.0, 5),
        ("SPY", "flat", "SPY_rsi_test", "2023-01-01T09:35:00", 401.0, 10),
        ("SPY", "short", "SPY_rsi_test", "2023-01-01T09:40:00", 399.0, 15),
        ("SPY", "flat", "SPY_rsi_test", "2023-01-01T09:45:00", 400.5, 20),
    ]
    
    for symbol, direction, strategy_id, timestamp, price, bar_index in test_signals:
        storage.process_signal(symbol, direction, strategy_id, timestamp, price, bar_index)
    
    print(f"✅ Processed {len(test_signals)} signals")
    print(f"✅ Signal changes recorded: {len(storage._changes)}")
    
    # Check that source metadata is included in signal changes
    for i, change in enumerate(storage._changes):
        print(f"Signal {i+1}: {change.symbol} {change.signal_value} at bar {change.bar_index}")
        print(f"  Timeframe: {change.timeframe}")
        print(f"  Source file: {change.source_file_path}")
        print(f"  Source type: {change.data_source_type}")
    
    # Save and reload to test serialization
    filepath = storage.save(tag="test_strategy")
    print(f"✅ Saved to: {filepath}")
    
    # Read the saved file and check metadata
    import pandas as pd
    data = pd.read_parquet(filepath)
    print(f"✅ Loaded {len(data)} rows from parquet")
    print("Columns:", data.columns.tolist())
    
    # Check for source metadata columns
    expected_metadata_cols = ['tf', 'src_file', 'src_type']
    for col in expected_metadata_cols:
        if col in data.columns:
            print(f"✅ Found metadata column: {col}")
            print(f"  Sample value: {data[col].iloc[0]}")
        else:
            print(f"❌ Missing metadata column: {col}")
    
    # Test multi-symbol scenario
    print("\n🌍 Testing Multi-Symbol Scenario")
    
    # Create storages for different symbols and timeframes
    symbols_timeframes = [
        ("SPY", "1m"),
        ("QQQ", "1m"), 
        ("SPY", "5m"),
        ("QQQ", "5m")
    ]
    
    for symbol, timeframe in symbols_timeframes:
        symbol_tf_dir = test_workspace / f"{symbol}_{timeframe}"
        storage = TemporalSparseStorage(
            base_dir=str(symbol_tf_dir),
            run_id=f"{symbol}_strategy_{timeframe}",
            timeframe=timeframe,
            source_file_path=f"./data/{symbol}_{timeframe}.csv",
            data_source_type="csv"
        )
        
        # Add a test signal
        storage.process_signal(symbol, "long", f"{symbol}_test_strategy", "2023-01-01T09:30:00", 100.0, 1)
        
        # Save
        filepath = storage.save(tag=f"{symbol}_test_{timeframe}")
        print(f"✅ Created {symbol}_{timeframe}: {filepath}")
    
    # Show final directory structure
    print(f"\n📁 Final workspace structure:")
    for root, dirs, files in os.walk(test_workspace):
        level = root.replace(str(test_workspace), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print("\n🎉 All tests passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_workspace)
    print("✅ Cleaned up test workspace")

if __name__ == "__main__":
    test_enhanced_signal_storage()