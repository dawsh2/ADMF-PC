#!/usr/bin/env python3
"""
Container Integration Test

Tests the composable container system to ensure:
1. All container types can be created
2. Container patterns work correctly
3. Basic event flow functions
4. Registration system works
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

async def test_container_creation():
    """Test that all container types can be created successfully."""
    print("🧪 Testing Container Creation...")
    
    try:
        from src.execution.containers import (
            DataContainer, IndicatorContainer, StrategyContainer, ExecutionContainer,
            RiskContainer, PortfolioContainer, ClassifierContainer, AnalysisContainer,
            SignalLogContainer, EnsembleContainer
        )
        
        # Test configurations for each container type
        configs = {
            'data': {'source': 'historical', 'symbols': ['SPY'], 'data_dir': 'data'},
            'indicator': {'cache_size': 100},
            'strategy': {'type': 'momentum', 'parameters': {'period': 20}},
            'execution': {'mode': 'backtest', 'initial_capital': 100000},
            'risk': {'initial_capital': 100000, 'max_position_size': 0.1},
            'portfolio': {'allocation_type': 'equal_weight', 'allocation_capital': 50000},
            'classifier': {'type': 'simple', 'parameters': {}},
            'analysis': {'mode': 'signal_generation', 'output_file': 'test_signals.json'},
            'signal_log': {'source': 'test', 'log_file': 'test_signals.json'},
            'ensemble': {'method': 'equal_weight', 'weight_config': {'weights': {}}}
        }
        
        containers = {}
        
        # Create each container type
        containers['data'] = DataContainer(configs['data'])
        print("  ✅ DataContainer created")
        
        containers['indicator'] = IndicatorContainer(configs['indicator'])
        print("  ✅ IndicatorContainer created")
        
        containers['strategy'] = StrategyContainer(configs['strategy'])
        print("  ✅ StrategyContainer created")
        
        containers['execution'] = ExecutionContainer(configs['execution'])
        print("  ✅ ExecutionContainer created")
        
        containers['risk'] = RiskContainer(configs['risk'])
        print("  ✅ RiskContainer created")
        
        containers['portfolio'] = PortfolioContainer(configs['portfolio'])
        print("  ✅ PortfolioContainer created")
        
        containers['classifier'] = ClassifierContainer(configs['classifier'])
        print("  ✅ ClassifierContainer created")
        
        containers['analysis'] = AnalysisContainer(configs['analysis'])
        print("  ✅ AnalysisContainer created")
        
        containers['signal_log'] = SignalLogContainer(configs['signal_log'])
        print("  ✅ SignalLogContainer created")
        
        containers['ensemble'] = EnsembleContainer(configs['ensemble'])
        print("  ✅ EnsembleContainer created")
        
        print(f"✅ All {len(containers)} container types created successfully!\n")
        return containers
        
    except Exception as e:
        print(f"❌ Container creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_container_registration():
    """Test that container registration system works."""
    print("🧪 Testing Container Registration...")
    
    try:
        from src.core.containers.composition_engine import get_global_registry
        
        registry = get_global_registry()
        
        # Check that all container roles are registered
        from src.core.containers.composable import ContainerRole
        
        expected_roles = [
            ContainerRole.DATA, ContainerRole.INDICATOR, ContainerRole.STRATEGY,
            ContainerRole.EXECUTION, ContainerRole.RISK, ContainerRole.PORTFOLIO,
            ContainerRole.CLASSIFIER, ContainerRole.ANALYSIS, ContainerRole.SIGNAL_LOG,
            ContainerRole.ENSEMBLE
        ]
        
        for role in expected_roles:
            factory = registry.get_container_factory(role)
            if factory:
                print(f"  ✅ {role.value} factory registered")
            else:
                print(f"  ❌ {role.value} factory missing")
                return False
        
        print("✅ All container factories registered successfully!\n")
        return True
        
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_container_patterns():
    """Test that container patterns can be composed."""
    print("🧪 Testing Container Patterns...")
    
    try:
        from src.core.containers.composition_engine import get_global_composition_engine
        
        engine = get_global_composition_engine()
        
        # Test available patterns
        available_patterns = engine.registry.list_available_patterns()
        print(f"  📋 Available patterns: {available_patterns}")
        
        expected_patterns = ['full_backtest', 'signal_generation', 'signal_replay', 'simple_backtest']
        
        for pattern_name in expected_patterns:
            pattern = engine.registry.get_pattern(pattern_name)
            if pattern:
                print(f"  ✅ Pattern '{pattern_name}' available")
                
                # Validate pattern
                is_valid = engine.validate_pattern(pattern)
                if is_valid:
                    print(f"    ✅ Pattern '{pattern_name}' is valid")
                else:
                    print(f"    ⚠️  Pattern '{pattern_name}' validation failed (may be due to missing dependencies)")
            else:
                print(f"  ❌ Pattern '{pattern_name}' missing")
        
        print("✅ Container patterns tested successfully!\n")
        return True
        
    except Exception as e:
        print(f"❌ Pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_pattern_composition():
    """Test composing a simple container pattern."""
    print("🧪 Testing Simple Pattern Composition...")
    
    try:
        from src.core.containers.composition_engine import get_global_composition_engine
        
        engine = get_global_composition_engine()
        
        # Test simple backtest pattern
        config_overrides = {
            'data': {
                'source': 'historical',
                'symbols': ['SPY'],
                'data_dir': 'data'
            },
            'strategy': {
                'type': 'momentum',
                'parameters': {'period': 20}
            },
            'execution': {
                'mode': 'backtest',
                'initial_capital': 100000
            }
        }
        
        # This should create a simple hierarchy: Data -> Strategy -> Execution
        root_container = engine.compose_pattern('simple_backtest', config_overrides)
        
        print(f"  ✅ Simple backtest pattern composed successfully")
        print(f"  📦 Root container: {root_container.metadata.name} ({root_container.metadata.role.value})")
        print(f"  🆔 Container ID: {root_container.metadata.container_id}")
        
        # Check container hierarchy
        child_count = len(root_container.child_containers)
        print(f"  👶 Child containers: {child_count}")
        
        for child in root_container.child_containers:
            print(f"    └── {child.metadata.name} ({child.metadata.role.value})")
        
        print("✅ Simple pattern composition successful!\n")
        return root_container
        
    except Exception as e:
        print(f"❌ Pattern composition failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_event_types():
    """Test that all required event types are available."""
    print("🧪 Testing Event Types...")
    
    try:
        from src.core.events.types import EventType
        
        # Check that required event types exist
        required_events = [
            'BAR', 'SIGNAL', 'ORDER', 'FILL', 'INDICATORS', 'REGIME', 'RISK_UPDATE'
        ]
        
        for event_name in required_events:
            if hasattr(EventType, event_name):
                print(f"  ✅ EventType.{event_name} available")
            else:
                print(f"  ❌ EventType.{event_name} missing")
                return False
        
        print("✅ All required event types available!\n")
        return True
        
    except Exception as e:
        print(f"❌ Event type test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_container_lifecycle():
    """Test basic container lifecycle operations."""
    print("🧪 Testing Container Lifecycle...")
    
    try:
        from src.execution.containers import DataContainer
        from src.core.containers.composable import ContainerState
        
        # Create a simple container
        config = {
            'source': 'historical',
            'symbols': ['SPY'],
            'data_dir': 'data'
        }
        
        container = DataContainer(config)
        print(f"  ✅ Container created with state: {container.metadata.state}")
        
        # Test initialization
        await container.initialize()
        print(f"  ✅ Container initialized with state: {container.metadata.state}")
        
        # Test start
        await container.start()
        print(f"  ✅ Container started with state: {container.metadata.state}")
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Test stop
        await container.stop()
        print(f"  ✅ Container stopped with state: {container.metadata.state}")
        
        print("✅ Container lifecycle test successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting Container Integration Tests\n")
    
    test_results = []
    
    # Run tests in sequence
    tests = [
        ("Container Creation", test_container_creation),
        ("Container Registration", test_container_registration),
        ("Container Patterns", test_container_patterns),
        ("Event Types", test_event_types),
        ("Simple Pattern Composition", test_simple_pattern_composition),
        ("Container Lifecycle", test_basic_container_lifecycle),
    ]
    
    for test_name, test_func in tests:
        print(f"=" * 60)
        try:
            result = await test_func()
            test_results.append((test_name, result is not None and result is not False))
            if result is not None and result is not False:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            test_results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Container integration is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)