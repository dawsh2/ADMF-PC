#!/usr/bin/env python3
"""
Test that the clean hierarchy works with no old directories
"""

import sys
sys.path.append('src/core/events/storage')
from temporal_sparse_storage import TemporalSparseStorage
from pathlib import Path
import shutil

def test_clean_hierarchy():
    """Test clean symbol_timeframe/signals+classifiers hierarchy."""
    print("🧪 Testing Clean Hierarchy Structure")
    
    # Clean workspace
    test_workspace = Path("./test_clean")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    
    traces_dir = test_workspace / "traces"
    
    # Test creating nested structure without any old directories
    configs = [
        ("strategy", "rsi", "SPY_rsi_test", "SPY", "1m"),
        ("strategy", "momentum", "SPY_momentum_test", "SPY", "1m"),
        ("classifier", "trend", "SPY_trend_test", "SPY", "1m"),
        ("strategy", "rsi", "QQQ_rsi_test", "QQQ", "1m"),
        ("strategy", "breakout", "SPY_breakout_test", "SPY", "5m"),
    ]
    
    for component_type, strategy_type, component_id, symbol, timeframe in configs:
        # Create nested structure exactly like MultiStrategyTracer
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
        
        # Add test signal
        storage.process_signal(symbol, "long", component_id, "2023-01-01T09:30:00", 400.0, 1)
        filepath = storage.save(tag=component_id)
        
        rel_path = Path(filepath).relative_to(test_workspace)
        print(f"✅ Created: {rel_path}")
    
    # Verify ONLY symbol_timeframe directories exist under traces/
    print(f"\n📁 Final Clean Structure:")
    
    import os
    for root, dirs, files in os.walk(test_workspace):
        level = root.replace(str(test_workspace), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # Verify no old-style directories
    old_style_dirs = [
        traces_dir / "signals",
        traces_dir / "classifiers"
    ]
    
    print(f"\n🔍 Checking for old-style directories:")
    for old_dir in old_style_dirs:
        if old_dir.exists():
            print(f"❌ Found old directory: {old_dir}")
        else:
            print(f"✅ No old directory: {old_dir}")
    
    # Check structure correctness
    expected_structure = {
        "SPY_1m": ["signals", "classifiers"],
        "QQQ_1m": ["signals"],
        "SPY_5m": ["signals"]
    }
    
    print(f"\n✅ Structure Validation:")
    for symbol_tf, expected_dirs in expected_structure.items():
        symbol_tf_dir = traces_dir / symbol_tf
        if symbol_tf_dir.exists():
            actual_dirs = [d.name for d in symbol_tf_dir.iterdir() if d.is_dir()]
            for expected_dir in expected_dirs:
                if expected_dir in actual_dirs:
                    print(f"  ✅ {symbol_tf}/{expected_dir}/")
                else:
                    print(f"  ❌ Missing {symbol_tf}/{expected_dir}/")
        else:
            print(f"  ❌ Missing {symbol_tf}/")
    
    # Cleanup
    shutil.rmtree(test_workspace)
    print(f"\n🎉 Clean hierarchy test passed!")

if __name__ == "__main__":
    test_clean_hierarchy()