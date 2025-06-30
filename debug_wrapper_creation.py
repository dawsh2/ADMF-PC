#!/usr/bin/env python3
"""Debug strategy wrapper creation."""

# Patch the topology builder to see what's happening
from src.core.coordinator import topology
import sys

wrapper_info = []

# Get the original method
if hasattr(topology.TopologyBuilder, '_infer_and_inject_features'):
    # Find make_strategy_wrapper inside the method
    import types
    import inspect
    
    # We need to patch at module load time
    original_init = topology.TopologyBuilder.__init__
    
    def debug_init(self, *args, **kwargs):
        print("\n=== TopologyBuilder created ===")
        return original_init(self, *args, **kwargs)
    
    topology.TopologyBuilder.__init__ = debug_init

# Also patch the compiler
from src.core.coordinator import compiler

original_compile = compiler.StrategyCompiler.compile_strategies

def debug_compile(self, strategies_config, discovered_components=None):
    print(f"\n=== StrategyCompiler.compile_strategies called ===")
    print(f"  Config: {strategies_config}")
    print(f"  Components available: {list(discovered_components.keys())[:5] if discovered_components else 'None'}")
    
    result = original_compile(self, strategies_config, discovered_components)
    
    print(f"  Result: {len(result)} strategies compiled")
    for name, info in list(result.items())[:3]:
        print(f"    - {name}: {info.get('strategy_type', 'unknown')}")
    
    return result

compiler.StrategyCompiler.compile_strategies = debug_compile

# Run main
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '30']

print("=== Running with wrapper debugging ===\n")

from main import main
main()