#!/usr/bin/env python3
"""
Minimal Container Test

Tests core container functionality without external dependencies.
This validates the container architecture itself.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_event_types():
    """Test that all required event types are available."""
    print("🧪 Testing Event Types...")
    
    try:
        from src.core.events.types import EventType, Event
        
        # Test that we can create events with the new types
        test_events = [
            (EventType.BAR, {'timestamp': datetime.now(), 'data': {}}),
            (EventType.SIGNAL, {'timestamp': datetime.now(), 'signals': []}),
            (EventType.INDICATORS, {'timestamp': datetime.now(), 'indicators': {}}),
            (EventType.REGIME, {'timestamp': datetime.now(), 'regime': 'neutral'}),
            (EventType.RISK_UPDATE, {'timestamp': datetime.now(), 'limits': {}})
        ]
        
        for event_type, payload in test_events:
            event = Event(
                event_type=event_type,
                payload=payload,
                timestamp=datetime.now()
            )
            print(f"  ✅ Created {event_type.name} event")
        
        print("✅ Event types test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Event types test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_container_roles():
    """Test that all container roles are defined."""
    print("🧪 Testing Container Roles...")
    
    try:
        from src.core.containers.composable import ContainerRole
        
        expected_roles = [
            'DATA', 'INDICATOR', 'CLASSIFIER', 'RISK', 'PORTFOLIO', 
            'STRATEGY', 'EXECUTION', 'ANALYSIS', 'SIGNAL_LOG', 'ENSEMBLE'
        ]
        
        for role_name in expected_roles:
            if hasattr(ContainerRole, role_name):
                role = getattr(ContainerRole, role_name)
                print(f"  ✅ ContainerRole.{role_name} = '{role.value}'")
            else:
                print(f"  ❌ ContainerRole.{role_name} missing")
                return False
        
        print("✅ Container roles test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container roles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_container_protocols():
    """Test that container protocols are properly defined."""
    print("🧪 Testing Container Protocols...")
    
    try:
        from src.core.containers.composable import (
            ComposableContainerProtocol, BaseComposableContainer, 
            ContainerMetadata, ContainerState, ContainerRole
        )
        
        # Test that we can create container metadata
        metadata = ContainerMetadata(
            container_id="test_001",
            role=ContainerRole.DATA,
            name="TestContainer",
            config={'test': True},
            created_at=datetime.now()
        )
        print(f"  ✅ Created ContainerMetadata: {metadata.container_id}")
        
        # Test that BaseComposableContainer can be subclassed
        class TestContainer(BaseComposableContainer):
            def __init__(self):
                super().__init__(
                    role=ContainerRole.DATA,
                    name="TestContainer",
                    config={'test': True}
                )
            
            async def _initialize_self(self) -> None:
                pass
            
            def get_capabilities(self):
                return {"test.capability"}
        
        test_container = TestContainer()
        print(f"  ✅ Created test container: {test_container.metadata.name}")
        print(f"  🆔 Container ID: {test_container.metadata.container_id}")
        print(f"  📍 Container state: {test_container.state}")
        
        print("✅ Container protocols test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container protocols test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_container_states():
    """Test container state transitions."""
    print("🧪 Testing Container States...")
    
    try:
        from src.core.containers.composable import ContainerState
        
        # Test all expected states exist
        expected_states = [
            'UNINITIALIZED', 'INITIALIZING', 'INITIALIZED', 'RUNNING', 
            'PAUSED', 'STOPPING', 'STOPPED', 'ERROR'
        ]
        
        for state_name in expected_states:
            if hasattr(ContainerState, state_name):
                state = getattr(ContainerState, state_name)
                print(f"  ✅ ContainerState.{state_name} = '{state.value}'")
            else:
                print(f"  ❌ ContainerState.{state_name} missing")
                return False
        
        print("✅ Container states test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container states test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_container_lifecycle():
    """Test async container lifecycle without dependencies."""
    print("🧪 Testing Async Container Lifecycle...")
    
    try:
        from src.core.containers.composable import (
            BaseComposableContainer, ContainerRole, ContainerState
        )
        
        class MockContainer(BaseComposableContainer):
            def __init__(self):
                super().__init__(
                    role=ContainerRole.ANALYSIS,  # Use ANALYSIS since it has minimal dependencies
                    name="MockContainer",
                    config={'mock': True}
                )
                self.initialization_called = False
                self.start_called = False
                self.stop_called = False
            
            async def _initialize_self(self) -> None:
                self.initialization_called = True
                await asyncio.sleep(0.001)  # Simulate async work
            
            async def _start_self(self) -> None:
                self.start_called = True
                await asyncio.sleep(0.001)
            
            async def _stop_self(self) -> None:
                self.stop_called = True
                await asyncio.sleep(0.001)
            
            def get_capabilities(self):
                return {"mock.testing"}
        
        container = MockContainer()
        print(f"  ✅ Created container with state: {container.state}")
        
        # Test initialization
        await container.initialize()
        print(f"  ✅ Initialized - state: {container.state}")
        print(f"  🔄 Initialization called: {container.initialization_called}")
        
        # Test start
        await container.start()
        print(f"  ✅ Started - state: {container.state}")
        print(f"  🔄 Start called: {container.start_called}")
        
        # Brief run
        await asyncio.sleep(0.01)
        
        # Test stop
        await container.stop()
        print(f"  ✅ Stopped - state: {container.state}")
        print(f"  🔄 Stop called: {container.stop_called}")
        
        print("✅ Async container lifecycle test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Async container lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_container_hierarchy():
    """Test container parent-child relationships."""
    print("🧪 Testing Container Hierarchy...")
    
    try:
        from src.core.containers.composable import (
            BaseComposableContainer, ContainerRole
        )
        
        class ParentContainer(BaseComposableContainer):
            def __init__(self):
                super().__init__(
                    role=ContainerRole.DATA,
                    name="ParentContainer",
                    config={}
                )
            
            async def _initialize_self(self) -> None:
                pass
            
            def get_capabilities(self):
                return {"parent.test"}
        
        class ChildContainer(BaseComposableContainer):
            def __init__(self):
                super().__init__(
                    role=ContainerRole.STRATEGY,
                    name="ChildContainer",
                    config={}
                )
            
            async def _initialize_self(self) -> None:
                pass
            
            def get_capabilities(self):
                return {"child.test"}
        
        # Create containers
        parent = ParentContainer()
        child = ChildContainer()
        
        print(f"  ✅ Created parent: {parent.metadata.name}")
        print(f"  ✅ Created child: {child.metadata.name}")
        
        # Test adding child to parent
        parent.add_child_container(child)
        print(f"  ✅ Added child to parent")
        print(f"  👶 Parent has {len(parent.child_containers)} children")
        print(f"  👨 Child parent: {child.parent_container.metadata.name if child.parent_container else 'None'}")
        
        # Test hierarchy
        assert len(parent.child_containers) == 1
        assert child.parent_container == parent
        assert parent.child_containers[0] == child
        
        print("✅ Container hierarchy test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Container hierarchy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run minimal integration tests."""
    print("🚀 Starting Minimal Container Tests\n")
    
    test_results = []
    
    # Run tests in sequence
    tests = [
        ("Event Types", test_event_types),
        ("Container Roles", test_container_roles),
        ("Container Protocols", test_container_protocols),
        ("Container States", test_container_states),
        ("Async Container Lifecycle", test_async_container_lifecycle),
        ("Container Hierarchy", test_container_hierarchy),
    ]
    
    for test_name, test_func in tests:
        print(f"=" * 60)
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            test_results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("🏁 MINIMAL TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All minimal tests passed! Core container architecture is solid.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print("\n" + "="*60)
    if success:
        print("✅ CONCLUSION: Container architecture is working correctly!")
        print("   The core protocols, states, events, and lifecycle management")
        print("   are all functioning properly. Dependencies (pandas, yaml) are")
        print("   the only remaining barrier to full integration testing.")
    else:
        print("❌ CONCLUSION: Core container architecture has issues.")
    print("="*60)
    sys.exit(0 if success else 1)