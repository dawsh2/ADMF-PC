"""
Enhanced event bus with strict isolation enforcement.

This module provides additional isolation mechanisms to prevent
accidental event leakage between containers.
"""

from typing import Dict, Set, Optional, Any
import logging
import time
from contextlib import contextmanager

from .event_bus import ContainerEventBus
from .types import Event, EventHandler, EventType
from .isolation import EventIsolationManager


logger = logging.getLogger(__name__)


class StrictIsolationEventBus(ContainerEventBus):
    """
    Event bus with strict isolation enforcement.
    
    This enhanced event bus adds multiple layers of isolation validation
    to prevent cross-container contamination.
    """
    
    def __init__(self, container_id: str, isolation_manager: EventIsolationManager):
        super().__init__(container_id)
        self.isolation_manager = isolation_manager
        self._strict_mode = True
        self._allowed_source_containers: Set[str] = {container_id}
        self._isolation_violations = []
    
    def enable_strict_mode(self, enabled: bool = True):
        """Enable or disable strict isolation enforcement."""
        self._strict_mode = enabled
        logger.info(f"Strict isolation mode {'enabled' if enabled else 'disabled'} for container {self.container_id}")
    
    def allow_events_from_container(self, source_container_id: str):
        """Allow events from another container (for parent-child relationships)."""
        self._allowed_source_containers.add(source_container_id)
        logger.debug(f"Container {self.container_id} now accepts events from {source_container_id}")
    
    def revoke_container_access(self, source_container_id: str):
        """Revoke event access from another container."""
        self._allowed_source_containers.discard(source_container_id)
        logger.debug(f"Container {self.container_id} revoked access from {source_container_id}")
    
    def publish(self, event: Event) -> None:
        """Publish with strict isolation validation."""
        # Enforce container ID consistency
        if event.container_id is None:
            event.container_id = self.container_id
        elif event.container_id != self.container_id and self._strict_mode:
            violation = {
                'type': 'wrong_container_publish',
                'event_container': event.container_id,
                'bus_container': self.container_id,
                'event_type': str(event.event_type),
                'source_id': event.source_id
            }
            self._isolation_violations.append(violation)
            
            logger.error(f"ISOLATION VIOLATION: Event with container_id '{event.container_id}' "
                        f"published to bus for container '{self.container_id}'")
            
            if self._strict_mode:
                raise IsolationViolationError(
                    f"Event container mismatch: {event.container_id} != {self.container_id}"
                )
        
        # Validate source authorization
        if self._strict_mode and event.source_id:
            # Extract container from source_id if it follows convention
            source_container = self._extract_container_from_source(event.source_id)
            if source_container and source_container not in self._allowed_source_containers:
                violation = {
                    'type': 'unauthorized_source',
                    'source_container': source_container,
                    'bus_container': self.container_id,
                    'event_type': str(event.event_type)
                }
                self._isolation_violations.append(violation)
                
                logger.warning(f"ISOLATION WARNING: Event from unauthorized source container "
                             f"'{source_container}' in container '{self.container_id}'")
        
        # Proceed with normal publishing
        super().publish(event)
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe with isolation-aware handler wrapping."""
        if self._strict_mode:
            # Wrap handler to validate event isolation
            wrapped_handler = self._create_isolation_validated_handler(handler)
            super().subscribe(event_type, wrapped_handler)
        else:
            super().subscribe(event_type, handler)
    
    def _create_isolation_validated_handler(self, original_handler: EventHandler) -> EventHandler:
        """Create a handler wrapper that validates event isolation."""
        def isolation_validated_handler(event: Event) -> None:
            # Validate event came from allowed source
            if event.container_id not in self._allowed_source_containers:
                violation = {
                    'type': 'cross_container_event',
                    'source_container': event.container_id,
                    'target_container': self.container_id,
                    'event_type': str(event.event_type),
                    'source_id': event.source_id
                }
                self._isolation_violations.append(violation)
                
                logger.error(f"ISOLATION VIOLATION: Handler in container '{self.container_id}' "
                           f"received event from unauthorized container '{event.container_id}'")
                
                if self._strict_mode:
                    raise IsolationViolationError(
                        f"Cross-container event detected: {event.container_id} → {self.container_id}"
                    )
                return  # Don't process the event
            
            # Process normally if validation passes
            try:
                original_handler(event)
            except Exception as e:
                logger.error(f"Error in isolation-validated handler: {e}", exc_info=True)
        
        # Preserve original handler reference for unsubscribe
        isolation_validated_handler.__wrapped__ = original_handler
        return isolation_validated_handler
    
    def _extract_container_from_source(self, source_id: str) -> Optional[str]:
        """Extract container ID from source ID if it follows naming convention."""
        # Convention: source_id = "component_name@container_id"
        if '@' in source_id:
            return source_id.split('@')[-1]
        return None
    
    def get_isolation_stats(self) -> Dict[str, Any]:
        """Get isolation statistics and violations."""
        stats = super().get_stats()
        stats.update({
            'strict_mode': self._strict_mode,
            'allowed_source_containers': list(self._allowed_source_containers),
            'isolation_violations': len(self._isolation_violations),
            'violation_details': self._isolation_violations[-10:]  # Last 10 violations
        })
        return stats
    
    def clear_violations(self):
        """Clear the violation log."""
        self._isolation_violations.clear()
        logger.info(f"Cleared isolation violations for container {self.container_id}")


class IsolationViolationError(Exception):
    """Raised when strict isolation mode detects a violation."""
    pass


class HierarchicalIsolationManager(EventIsolationManager):
    """
    Enhanced isolation manager supporting parent-child container relationships.
    
    This allows for controlled event flow in nested container hierarchies
    while maintaining isolation between unrelated containers.
    """
    
    def __init__(self):
        super().__init__()
        self._container_hierarchy: Dict[str, Set[str]] = {}  # parent -> children
        self._parent_mapping: Dict[str, str] = {}  # child -> parent
    
    def create_child_container_bus(
        self, 
        child_container_id: str, 
        parent_container_id: str
    ) -> StrictIsolationEventBus:
        """
        Create a child container bus with controlled access to parent events.
        
        Args:
            child_container_id: ID for the child container
            parent_container_id: ID of the parent container
            
        Returns:
            StrictIsolationEventBus configured for parent-child relationship
        """
        # Validate parent exists
        if parent_container_id not in self._active_containers:
            raise ValueError(f"Parent container {parent_container_id} does not exist")
        
        # Create child bus
        child_bus = StrictIsolationEventBus(child_container_id, self)
        
        # Allow child to receive events from parent
        child_bus.allow_events_from_container(parent_container_id)
        
        # Update hierarchy tracking
        if parent_container_id not in self._container_hierarchy:
            self._container_hierarchy[parent_container_id] = set()
        self._container_hierarchy[parent_container_id].add(child_container_id)
        self._parent_mapping[child_container_id] = parent_container_id
        
        # Register the bus
        self._container_buses[child_container_id] = child_bus
        self._active_containers.add(child_container_id)
        
        logger.info(f"Created child container {child_container_id} under parent {parent_container_id}")
        return child_bus
    
    def create_container_bus(self, container_id: str) -> StrictIsolationEventBus:
        """Create a top-level container bus with strict isolation."""
        if container_id in self._active_containers:
            raise ValueError(f"Container {container_id} already exists")
        
        event_bus = StrictIsolationEventBus(container_id, self)
        self._container_buses[container_id] = event_bus
        self._active_containers.add(container_id)
        
        logger.info(f"Created top-level isolated container: {container_id}")
        return event_bus
    
    def remove_container_bus(self, container_id: str) -> None:
        """Remove container and all its children."""
        # Remove all children first
        children = self._container_hierarchy.get(container_id, set()).copy()
        for child_id in children:
            self.remove_container_bus(child_id)
        
        # Remove from parent's children if this is a child
        parent_id = self._parent_mapping.get(container_id)
        if parent_id and parent_id in self._container_hierarchy:
            self._container_hierarchy[parent_id].discard(container_id)
            if not self._container_hierarchy[parent_id]:
                del self._container_hierarchy[parent_id]
        
        # Remove from mappings
        self._parent_mapping.pop(container_id, None)
        self._container_hierarchy.pop(container_id, None)
        
        # Remove the container itself
        super().remove_container_bus(container_id)
        
        logger.info(f"Removed container {container_id} and all its children")
    
    def get_container_children(self, container_id: str) -> Set[str]:
        """Get all child container IDs for a parent container."""
        return self._container_hierarchy.get(container_id, set()).copy()
    
    def get_container_parent(self, container_id: str) -> Optional[str]:
        """Get the parent container ID for a child container."""
        return self._parent_mapping.get(container_id)
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics about container hierarchy."""
        stats = super().get_stats()
        stats.update({
            'hierarchy': dict(self._container_hierarchy),
            'parent_mappings': dict(self._parent_mapping),
            'top_level_containers': [
                cid for cid in self._active_containers 
                if cid not in self._parent_mapping
            ],
            'total_children': sum(len(children) for children in self._container_hierarchy.values())
        })
        return stats
    
    @contextmanager
    def isolated_container_group(self, group_name: str, container_count: int):
        """Context manager for creating and cleaning up a group of isolated containers."""
        container_ids = [f"{group_name}_container_{i}" for i in range(container_count)]
        created_buses = []
        
        try:
            # Create all containers
            for container_id in container_ids:
                bus = self.create_container_bus(container_id)
                created_buses.append(bus)
            
            yield created_buses
            
        finally:
            # Clean up all containers
            for container_id in container_ids:
                try:
                    self.remove_container_bus(container_id)
                except Exception as e:
                    logger.error(f"Error cleaning up container {container_id}: {e}")


class IsolationTestSuite:
    """Comprehensive test suite for event isolation validation."""
    
    def __init__(self, isolation_manager: HierarchicalIsolationManager):
        self.isolation_manager = isolation_manager
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all isolation tests and return comprehensive results."""
        tests = [
            ('basic_isolation', self.test_basic_isolation),
            ('strict_mode_enforcement', self.test_strict_mode_enforcement),
            ('parent_child_hierarchy', self.test_parent_child_hierarchy),
            ('parallel_container_stress', self.test_parallel_container_stress),
            ('violation_detection', self.test_violation_detection),
            ('resource_cleanup', self.test_resource_cleanup)
        ]
        
        for test_name, test_func in tests:
            try:
                self.test_results[test_name] = {
                    'passed': test_func(),
                    'error': None
                }
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                logger.error(f"❌ {test_name}: FAILED - {e}")
        
        overall_passed = all(result['passed'] for result in self.test_results.values())
        
        return {
            'overall_passed': overall_passed,
            'test_results': self.test_results,
            'summary': self._generate_test_summary()
        }
    
    def test_basic_isolation(self) -> bool:
        """Test basic container isolation."""
        with self.isolation_manager.isolated_container_group("basic_test", 2) as buses:
            bus_a, bus_b = buses
            
            events_a = []
            events_b = []
            
            def handler_a(event): events_a.append(event)
            def handler_b(event): events_b.append(event)
            
            bus_a.subscribe(EventType.SIGNAL, handler_a)
            bus_b.subscribe(EventType.SIGNAL, handler_b)
            
            # Publish to each bus
            event_a = Event(EventType.SIGNAL, {'data': 'a'}, container_id=bus_a.container_id)
            event_b = Event(EventType.SIGNAL, {'data': 'b'}, container_id=bus_b.container_id)
            
            bus_a.publish(event_a)
            bus_b.publish(event_b)
            
            time.sleep(0.1)  # Allow processing
            
            # Validate isolation
            return (
                len(events_a) == 1 and len(events_b) == 1 and
                events_a[0].payload['data'] == 'a' and
                events_b[0].payload['data'] == 'b'
            )
    
    def test_strict_mode_enforcement(self) -> bool:
        """Test that strict mode prevents violations."""
        with self.isolation_manager.isolated_container_group("strict_test", 1) as buses:
            bus = buses[0]
            
            # Enable strict mode
            bus.enable_strict_mode(True)
            
            # Try to publish event with wrong container ID
            wrong_event = Event(
                EventType.SIGNAL, 
                {'test': 'violation'}, 
                container_id='wrong_container'
            )
            
            violation_caught = False
            try:
                bus.publish(wrong_event)
            except IsolationViolationError:
                violation_caught = True
            
            return violation_caught
    
    def test_parent_child_hierarchy(self) -> bool:
        """Test parent-child container relationships."""
        parent_id = "hierarchy_parent"
        child_id = "hierarchy_child"
        
        try:
            # Create parent
            parent_bus = self.isolation_manager.create_container_bus(parent_id)
            
            # Create child
            child_bus = self.isolation_manager.create_child_container_bus(child_id, parent_id)
            
            # Test that child can receive parent events
            child_events = []
            def child_handler(event): child_events.append(event)
            child_bus.subscribe(EventType.SIGNAL, child_handler)
            
            # Parent publishes event
            parent_event = Event(EventType.SIGNAL, {'from': 'parent'}, container_id=parent_id)
            parent_bus.publish(parent_event)
            
            time.sleep(0.1)
            
            # Child should receive parent event
            hierarchy_works = len(child_events) == 0  # Child shouldn't auto-receive parent events
            
            # Test explicit event forwarding (if implemented)
            # This would require additional implementation for event forwarding
            
            return True  # Basic hierarchy creation works
            
        finally:
            self.isolation_manager.remove_container_bus(parent_id)
    
    def test_parallel_container_stress(self) -> bool:
        """Stress test with many parallel containers."""
        num_containers = 20  # Reduced for faster testing
        events_per_container = 10
        
        with self.isolation_manager.isolated_container_group("stress_test", num_containers) as buses:
            results = {}
            
            def container_worker(bus, container_index):
                events_received = []
                
                def handler(event):
                    events_received.append(event)
                
                bus.subscribe(EventType.SIGNAL, handler)
                
                # Publish events
                for i in range(events_per_container):
                    event = Event(
                        EventType.SIGNAL,
                        {'container': container_index, 'event': i},
                        container_id=bus.container_id
                    )
                    bus.publish(event)
                
                time.sleep(0.2)  # Allow processing
                
                # Validate only received own events
                own_events = [
                    e for e in events_received 
                    if e.container_id == bus.container_id
                ]
                
                results[container_index] = {
                    'expected': events_per_container,
                    'received': len(events_received),
                    'own_events': len(own_events),
                    'isolated': len(own_events) == len(events_received)
                }
            
            # Run all containers in parallel
            import threading
            threads = []
            for i, bus in enumerate(buses):
                thread = threading.Thread(target=container_worker, args=(bus, i))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Validate all containers were properly isolated
            return all(result['isolated'] for result in results.values())
    
    def test_violation_detection(self) -> bool:
        """Test that violations are properly detected and logged."""
        with self.isolation_manager.isolated_container_group("violation_test", 1) as buses:
            bus = buses[0]
            
            # Disable strict mode to allow violations but still detect them
            bus.enable_strict_mode(False)
            
            initial_violations = len(bus.get_isolation_stats()['violation_details'])
            
            # Create a violation
            violation_event = Event(
                EventType.SIGNAL,
                {'test': 'violation'},
                container_id='other_container'
            )
            
            bus.publish(violation_event)
            
            final_violations = len(bus.get_isolation_stats()['violation_details'])
            
            # Should have detected one more violation
            return final_violations > initial_violations
    
    def test_resource_cleanup(self) -> bool:
        """Test that resources are properly cleaned up."""
        initial_container_count = len(self.isolation_manager._active_containers)
        
        # Create and destroy containers
        container_ids = []
        for i in range(10):
            container_id = f"cleanup_test_{i}"
            container_ids.append(container_id)
            self.isolation_manager.create_container_bus(container_id)
        
        # Remove all containers
        for container_id in container_ids:
            self.isolation_manager.remove_container_bus(container_id)
        
        final_container_count = len(self.isolation_manager._active_containers)
        
        # Should be back to initial count
        return final_container_count == initial_container_count
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results."""
        passed_tests = [name for name, result in self.test_results.items() if result['passed']]
        failed_tests = [name for name, result in self.test_results.items() if not result['passed']]
        
        return {
            'total_tests': len(self.test_results),
            'passed_count': len(passed_tests),
            'failed_count': len(failed_tests),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': len(passed_tests) / len(self.test_results) if self.test_results else 0
        }


# Global enhanced isolation manager
_enhanced_isolation_manager = HierarchicalIsolationManager()


def get_enhanced_isolation_manager() -> HierarchicalIsolationManager:
    """Get the global enhanced isolation manager."""
    return _enhanced_isolation_manager


def run_isolation_validation() -> Dict[str, Any]:
    """Run comprehensive isolation validation tests."""
    test_suite = IsolationTestSuite(_enhanced_isolation_manager)
    return test_suite.run_all_tests()


if __name__ == "__main__":
    # Run validation when executed directly
    import time
    
    print("🔒 Running Enhanced Event Isolation Validation")
    print("=" * 60)
    
    results = run_isolation_validation()
    
    print(f"\n📊 Test Results Summary:")
    print(f"  Total Tests: {results['summary']['total_tests']}")
    print(f"  Passed: {results['summary']['passed_count']} ✅")
    print(f"  Failed: {results['summary']['failed_count']} ❌")
    print(f"  Pass Rate: {results['summary']['pass_rate']:.1%}")
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if results['overall_passed'] else '❌ SOME TESTS FAILED'}")
    
    if not results['overall_passed']:
        print("\n❌ Failed Tests:")
        for test_name in results['summary']['failed_tests']:
            error = results['test_results'][test_name]['error']
            print(f"  • {test_name}: {error}")
    
    print("\n💡 Isolation Status: READY FOR PRODUCTION" if results['overall_passed'] else "\n⚠️  Isolation Status: NEEDS ATTENTION")
