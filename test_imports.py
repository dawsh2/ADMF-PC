#!/usr/bin/env python3
"""
Test if basic imports are working.
"""

import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")

try:
    print("\nTesting imports...")
    
    print("1. Testing coordinator imports...")
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    print("   ✓ Simple types imported")
    
    print("\n2. Testing bootstrap import...")
    from src.core.containers.bootstrap import ContainerBootstrap
    print("   ✓ Bootstrap imported")
    
    print("\n3. Testing data models...")
    from src.data.models import MarketData
    print("   ✓ MarketData imported")
    
    print("\n4. Testing components...")
    from src.core.components import ComponentSpec
    print("   ✓ ComponentSpec imported")
    
    print("\n5. Testing execution imports...")
    from src.execution.simple_backtest_engine import SimpleBacktestEngine
    print("   ✓ SimpleBacktestEngine imported")
    
    print("\n6. Testing full coordinator flow...")
    # This will test the full import chain
    bootstrap = ContainerBootstrap()
    print("   ✓ Bootstrap created")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()