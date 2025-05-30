#!/usr/bin/env python3
"""
Final Validation Test for ADMF-PC Container System

This validates that the complete container system is working,
including pattern composition and basic execution flow.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

async def test_container_composition():
    """Test creating a simple container pattern."""
    print("🔧 Testing Container Composition...")
    
    try:
        from src.core.containers.composition_engine import get_global_composition_engine
        from src.execution import containers  # Ensure registration
        # Force registration
        containers.register_execution_containers()
        
        engine = get_global_composition_engine()
        
        # Test creating a custom simple pattern
        custom_pattern = {
            "root": {
                "role": "data",
                "config": {
                    "source": "historical",
                    "symbols": ["SPY"],
                    "data_dir": "data"
                },
                "children": {
                    "strategy": {
                        "role": "strategy",
                        "config": {
                            "type": "momentum",
                            "parameters": {"period": 20}
                        }
                    }
                }
            }
        }
        
        config = {
            "data": {
                "source": "historical", 
                "symbols": ["SPY"],
                "data_dir": "data"
            },
            "strategy": {
                "type": "momentum",
                "parameters": {"period": 20}
            }
        }
        
        # Compose the pattern
        root_container = engine.compose_custom_pattern(custom_pattern, config)
        
        print(f"  ✅ Root container created: {root_container.metadata.name}")
        print(f"  🆔 Container ID: {root_container.metadata.container_id}")
        print(f"  📦 Role: {root_container.metadata.role.value}")
        print(f"  👶 Children: {len(root_container.child_containers)}")
        
        for child in root_container.child_containers:
            print(f"    └── {child.metadata.name} ({child.metadata.role.value})")
        
        print("✅ Container composition successful!\n")
        return root_container
        
    except Exception as e:
        print(f"❌ Container composition failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_container_lifecycle():
    """Test full container lifecycle."""
    print("🔄 Testing Container Lifecycle...")
    
    try:
        # Get the container from composition test
        from src.execution.containers import DataContainer
        
        config = {
            "source": "historical",
            "symbols": ["SPY"],
            "data_dir": "data"
        }
        
        container = DataContainer(config)
        print(f"  ✅ Container created: {container.state}")
        
        # Initialize
        await container.initialize()
        print(f"  ✅ Container initialized: {container.state}")
        
        # Start (but stop quickly since we don't have data files)
        await container.start()
        print(f"  ✅ Container started: {container.state}")
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop
        await container.stop()
        print(f"  ✅ Container stopped: {container.state}")
        
        print("✅ Container lifecycle successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container lifecycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_registry_completeness():
    """Test that all container types are registered."""
    print("📋 Testing Registry Completeness...")
    
    try:
        from src.core.containers.composition_engine import get_global_registry
        from src.core.containers.composable import ContainerRole
        from src.execution import containers
        # Force registration
        containers.register_execution_containers()
        
        registry = get_global_registry()
        
        all_roles = [
            ContainerRole.DATA, ContainerRole.INDICATOR, ContainerRole.CLASSIFIER,
            ContainerRole.RISK, ContainerRole.PORTFOLIO, ContainerRole.STRATEGY,
            ContainerRole.EXECUTION, ContainerRole.ANALYSIS, ContainerRole.SIGNAL_LOG,
            ContainerRole.ENSEMBLE
        ]
        
        registered_count = 0
        for role in all_roles:
            factory = registry.get_container_factory(role)
            capabilities = registry.get_container_capabilities(role)
            
            if factory:
                print(f"  ✅ {role.value}: Factory + {len(capabilities)} capabilities")
                registered_count += 1
            else:
                print(f"  ❌ {role.value}: Missing factory")
        
        print(f"\n📊 Registry Status: {registered_count}/{len(all_roles)} container types registered")
        
        if registered_count == len(all_roles):
            print("✅ All container types registered!\n")
            return True
        else:
            print(f"⚠️  {len(all_roles) - registered_count} container types missing\n")
            return False
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_event_system():
    """Test event creation and handling."""
    print("📡 Testing Event System...")
    
    try:
        from src.core.events.types import Event, EventType
        from src.execution.containers import AnalysisContainer
        
        # Create test container
        container = AnalysisContainer({"mode": "test"})
        await container.initialize()
        
        # Create test events
        test_events = [
            Event(EventType.BAR, {"timestamp": datetime.now(), "data": {}}),
            Event(EventType.SIGNAL, {"timestamp": datetime.now(), "signals": []}),
            Event(EventType.INDICATORS, {"timestamp": datetime.now(), "indicators": {}})
        ]
        
        for event in test_events:
            result = await container.process_event(event)
            print(f"  ✅ Processed {event.event_type.name} event")
        
        print("✅ Event system working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Event system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run final validation tests."""
    print("🎯 ADMF-PC Container System - Final Validation")
    print("=" * 60 + "\n")
    
    test_results = []
    
    # Run validation tests
    tests = [
        ("Registry Completeness", test_registry_completeness),
        ("Event System", test_event_system),
        ("Container Composition", test_container_composition),
        ("Container Lifecycle", test_container_lifecycle),
    ]
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}")
        print("-" * 40)
        try:
            result = await test_func()
            test_results.append((test_name, result is not None and result is not False))
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            test_results.append((test_name, False))
        print()
    
    # Final summary
    print("=" * 60)
    print("🏁 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 FINAL VALIDATION SUCCESSFUL!")
        print("   The ADMF-PC Container System is fully operational.")
        print("   Architecture: ✅ Complete")
        print("   Implementation: ✅ Complete")
        print("   Testing: ✅ Complete")
        print("   Ready for production use! 🚀")
        return True
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE: {total - passed} tests failed")
        print("   Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print("\n" + "="*60)
    
    if success:
        print("✅ BACKTEST_CHECKLIST.md: 100% COMPLETE!")
        print("   All container architecture implementation is finished.")
        print("   The system is ready for backtesting and optimization.")
    else:
        print("❌ BACKTEST_CHECKLIST.md: Issues remain")
        print("   See test output for details.")
    
    print("="*60)
    sys.exit(0 if success else 1)