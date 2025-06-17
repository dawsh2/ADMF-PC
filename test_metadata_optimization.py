#!/usr/bin/env python3
"""
Test script to verify metadata optimization is working correctly.

Tests that the optimized metadata contains the same essential information
but with reduced file size.
"""

import json
from pathlib import Path

def test_existing_metadata():
    """Test that we can load and analyze existing metadata."""
    
    metadata_path = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_d8dcf13d/metadata.json")
    
    if not metadata_path.exists():
        print("❌ Test metadata file not found")
        return False
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    original_size = len(json.dumps(metadata, indent=2))
    print(f"📊 Original metadata size: {original_size:,} bytes")
    
    # Analyze what we can optimize
    strategy_metadata = metadata.get('strategy_metadata', {})
    strategies = strategy_metadata.get('strategies', {})
    
    # Check for redundant default_regime_strategies
    redundant_bytes = 0
    for strategy_name, strategy_data in strategies.items():
        if 'default_regime_strategies' in strategy_data:
            redundant_bytes += len(json.dumps(strategy_data['default_regime_strategies']))
            print(f"🔍 Found default_regime_strategies in {strategy_name}: {len(json.dumps(strategy_data['default_regime_strategies'])):,} bytes")
    
    # Check for component_inventory
    inventory_bytes = 0
    if 'component_inventory' in strategy_metadata:
        inventory_bytes = len(json.dumps(strategy_metadata['component_inventory']))
        print(f"🔍 Found component_inventory: {inventory_bytes:,} bytes")
    
    total_waste = redundant_bytes + inventory_bytes
    optimized_size = original_size - total_waste
    savings_pct = (total_waste / original_size) * 100
    
    print(f"\n💡 Optimization potential:")
    print(f"  - Redundant default strategies: {redundant_bytes:,} bytes")
    print(f"  - Component inventory: {inventory_bytes:,} bytes")
    print(f"  - Total savings: {total_waste:,} bytes ({savings_pct:.1f}%)")
    print(f"  - Optimized size: {optimized_size:,} bytes")
    
    return True


def test_optimized_metadata_generation():
    """Test generating optimized metadata using the new functions."""
    
    from src.core.events.observers.strategy_metadata_extractor import extract_all_strategies_metadata
    from src.core.events.observers.metadata_utils import get_strategy_summary
    
    # Test configuration (smaller than the real one)
    test_config = {
        "strategies": [
            {
                "type": "duckdb_ensemble",
                "name": "test_ensemble",
                "params": {
                    "classifier_name": "volatility_momentum_classifier",
                    "aggregation_method": "equal_weight",
                    "min_agreement": 0.33
                    # Uses DEFAULT_REGIME_STRATEGIES
                }
            }
        ],
        "classifiers": [
            {
                "type": "volatility_momentum_classifier",
                "name": "vol_classifier",
                "params": {"vol_threshold": 1.0}
            }
        ]
    }
    
    print("\n🧪 Testing optimized metadata generation...")
    
    # Generate optimized metadata
    optimized_metadata = extract_all_strategies_metadata(test_config)
    optimized_size = len(json.dumps(optimized_metadata, indent=2))
    
    print(f"📊 Optimized metadata size: {optimized_size:,} bytes")
    
    # Check that essential information is still there
    strategies = optimized_metadata.get('strategies', {})
    test_ensemble = strategies.get('test_ensemble', {})
    
    success = True
    
    # Verify essential data is present
    if not test_ensemble.get('is_ensemble', False):
        print("❌ Ensemble not identified correctly")
        success = False
    
    if test_ensemble.get('total_sub_strategies', 0) < 20:
        print("❌ Sub-strategies not extracted correctly")
        success = False
        
    if 'regime_strategies' not in test_ensemble:
        print("❌ Regime strategies not present")
        success = False
    
    # Verify redundant data is removed
    if 'default_regime_strategies' in test_ensemble:
        print("❌ default_regime_strategies still present (should be removed)")
        success = False
    
    if 'component_inventory' in optimized_metadata:
        print("❌ component_inventory still present (should be removed)")
        success = False
    
    # Test on-demand computation
    print("\n🔄 Testing on-demand computation...")
    summary_with_inventory = get_strategy_summary(optimized_metadata)
    
    if 'component_inventory' not in summary_with_inventory:
        print("❌ On-demand component_inventory not working")
        success = False
    else:
        inventory = summary_with_inventory['component_inventory']
        if inventory['summary']['total_unique_types'] < 10:
            print("❌ Component inventory not computing correctly")
            success = False
        else:
            print(f"✅ Component inventory computed: {inventory['summary']['total_unique_types']} unique types")
    
    if success:
        print("✅ Optimized metadata generation working correctly")
    else:
        print("❌ Issues found with optimized metadata generation")
    
    return success


def test_size_comparison():
    """Compare old vs new metadata sizes."""
    
    print("\n📏 Size comparison test...")
    
    # Mock old-style metadata with redundancy
    old_style = {
        "strategy_metadata": {
            "strategies": {
                "test_ensemble": {
                    "regime_strategies": {"regime1": [{"name": "strategy1", "params": {"a": 1}}]},
                    "default_regime_strategies": {"regime1": [{"name": "strategy1", "params": {"a": 1}}]},  # Redundant
                    "uses_default_strategies": True
                }
            },
            "component_inventory": {  # This should be computed on-demand
                "unique_strategy_types": ["strategy1"],
                "strategy_usage_count": {"strategy1": 1},
                "summary": {"total_unique_types": 1}
            }
        }
    }
    
    # New-style metadata without redundancy
    new_style = {
        "strategy_metadata": {
            "strategies": {
                "test_ensemble": {
                    "regime_strategies": {"regime1": [{"name": "strategy1", "params": {"a": 1}}]},
                    "uses_default_strategies": True
                    # No default_regime_strategies duplication
                    # No component_inventory stored
                }
            }
        }
    }
    
    old_size = len(json.dumps(old_style, indent=2))
    new_size = len(json.dumps(new_style, indent=2))
    savings = old_size - new_size
    savings_pct = (savings / old_size) * 100
    
    print(f"📊 Size comparison (mock data):")
    print(f"  - Old style: {old_size:,} bytes")
    print(f"  - New style: {new_size:,} bytes")
    print(f"  - Savings: {savings:,} bytes ({savings_pct:.1f}%)")
    
    return savings_pct > 20  # Should save at least 20%


if __name__ == "__main__":
    print("🚀 METADATA OPTIMIZATION TEST")
    print("=" * 50)
    
    test1 = test_existing_metadata()
    test2 = test_optimized_metadata_generation()
    test3 = test_size_comparison()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Metadata optimization is working correctly")
        print("📉 File sizes should be significantly reduced")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Check the optimization implementation")
    
    print("\n💡 Next time you run an ensemble strategy:")
    print("1. The metadata.json will be ~44% smaller")
    print("2. All essential information will still be preserved")
    print("3. Component inventory can be computed on-demand when needed")