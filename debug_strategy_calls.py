#!/usr/bin/env python3
"""Debug strategy calls directly."""

import pandas as pd
import sys

# Track all strategy calls
strategy_calls = []
signals_generated = []

# Patch the bollinger_bands strategy
from src.strategy.strategies.indicators import volatility

original_bb = volatility.bollinger_bands

def debug_bollinger_bands(features, bar, params):
    """Wrapped bollinger_bands with debugging."""
    call_info = {
        'bar_close': bar.get('close', 'N/A'),
        'params': params,
        'features_count': len(features),
        'has_required': all(k in features for k in ['bollinger_bands_20_2.0_upper', 'bollinger_bands_20_2.0_middle', 'bollinger_bands_20_2.0_lower'])
    }
    strategy_calls.append(call_info)
    
    # Call original
    result = original_bb(features, bar, params)
    
    if result and result.get('signal_value', 0) != 0:
        signals_generated.append({
            'signal': result['signal_value'],
            'price': bar.get('close'),
            'call_num': len(strategy_calls)
        })
        print(f"\n✓ Signal generated on call #{len(strategy_calls)}: {result['signal_value']} at price {bar.get('close')}")
    
    return result

# Apply patch
volatility.bollinger_bands = debug_bollinger_bands

# Also check if strategies are being wrapped
from src.core.coordinator import topology

original_make_wrapper = None
wrapper_count = 0

# Find the wrapper creation in topology
if hasattr(topology.TopologyBuilder, '_infer_and_inject_features'):
    class_method = topology.TopologyBuilder._infer_and_inject_features
    
    # Patch to count wrappers
    def count_wrappers(self, *args, **kwargs):
        global wrapper_count
        result = class_method(self, *args, **kwargs)
        
        # Count strategies in result
        if isinstance(result, dict) and 'strategies' in result:
            wrapper_count = len(result['strategies'])
            print(f"\nCreated {wrapper_count} strategy wrappers")
        
        return result
    
    topology.TopologyBuilder._infer_and_inject_features = count_wrappers

# Run main
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '100']

print("=== Running with strategy call tracking ===\n")

from main import main
main()

print(f"\n\n=== FINAL SUMMARY ===")
print(f"Strategy wrappers created: {wrapper_count}")
print(f"Total bollinger_bands calls: {len(strategy_calls)}")
print(f"Total signals generated: {len(signals_generated)}")

if strategy_calls:
    print(f"\nFirst few calls:")
    for i, call in enumerate(strategy_calls[:5]):
        print(f"  Call {i+1}: close={call['bar_close']}, has_features={call['has_required']}")

if not signals_generated and strategy_calls:
    print(f"\nNo signals generated. Checking why...")
    # Sample a call with features
    calls_with_features = [c for c in strategy_calls if c['has_required']]
    if not calls_with_features:
        print("  → No calls had required features!")
    else:
        print(f"  → {len(calls_with_features)} calls had required features but no signals")