#!/usr/bin/env python3
"""
Simple test for SignalChange with source metadata
"""

import sys
import os
sys.path.insert(0, 'src')

# Direct import to avoid module initialization issues
from core.events.storage.temporal_sparse_storage import SignalChange

def test_signal_change_metadata():
    """Test SignalChange with source metadata."""
    print("🧪 Testing SignalChange with Source Metadata")
    
    # Create a signal change with source metadata
    change = SignalChange(
        bar_index=10,
        timestamp="2023-01-01T09:30:00",
        symbol="SPY",
        signal_value=1,  # long
        strategy_id="SPY_rsi_test",
        price=400.0,
        timeframe="1m",
        source_file_path="./data/SPY_1m.csv",
        data_source_type="csv"
    )
    
    print(f"✅ Created SignalChange:")
    print(f"  Symbol: {change.symbol}")
    print(f"  Signal: {change.signal_value}")
    print(f"  Bar: {change.bar_index}")
    print(f"  Price: {change.price}")
    print(f"  Timeframe: {change.timeframe}")
    print(f"  Source file: {change.source_file_path}")
    print(f"  Source type: {change.data_source_type}")
    
    # Test serialization
    data_dict = change.to_dict()
    print(f"\n✅ Serialized to dict:")
    for key, value in data_dict.items():
        print(f"  {key}: {value}")
    
    # Test deserialization
    restored_change = SignalChange.from_dict(data_dict)
    print(f"\n✅ Restored from dict:")
    print(f"  Symbol: {restored_change.symbol}")
    print(f"  Timeframe: {restored_change.timeframe}")
    print(f"  Source file: {restored_change.source_file_path}")
    
    # Verify all fields match
    assert change.bar_index == restored_change.bar_index
    assert change.symbol == restored_change.symbol
    assert change.timeframe == restored_change.timeframe
    assert change.source_file_path == restored_change.source_file_path
    assert change.data_source_type == restored_change.data_source_type
    
    print(f"\n🎉 SignalChange serialization test passed!")

if __name__ == "__main__":
    test_signal_change_metadata()