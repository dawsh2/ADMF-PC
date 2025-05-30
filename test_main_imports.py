#!/usr/bin/env python3
"""
Test main.py import chain step by step.
"""

import sys

try:
    print("1. Testing simple types import...")
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    print("   ✓ Success")
    
    print("\n2. Testing bootstrap import...")
    from src.core.containers.bootstrap import ContainerBootstrap
    print("   ✓ Success")
    
    print("\n3. Creating bootstrap...")
    bootstrap = ContainerBootstrap()
    print("   ✓ Success")
    
    print("\n4. Testing coordinator creation...")
    coordinator = bootstrap.create_coordinator()
    print("   ✓ Success")
    print(f"   Coordinator type: {type(coordinator)}")
    
    print("\n✅ All imports successful! main.py should work.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()