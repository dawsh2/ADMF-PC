"""
Comprehensive isolation validation tests for the event bus system.

These tests validate that container isolation is working correctly and
no events leak between containers during parallel execution.
"""

import pytest
import threading
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .isolation import get_isolation_manager, validate_event_isolation
from .types import Event, EventType
from .event_bus import ContainerEventBus


class IsolationTestFramework:
    """Framework for testing event isolation between containers."""
    
    def __init__(self):
        self.isolation_manager = get_isolation_manager()
        self.test_results = {}
        self.event_log = []
        self._lock = threading.Lock()
    
    def log_event(self, container_id: str, event: Event, action: str):
        """Log events for later analysis."""
        with self._lock:
            self.event_log.append({
                'timestamp': time.time(),
                'container_id': container_id,
                'event_container_id': event.container_id,
                'event_type': str(event.event_type),
                'source_id': event.source_id,
                'action': action  # 'published' or 'received'
            })
    
    def test_basic_isolation(self) -> bool:
        """Test basic isolation between two containers."""
        container_a_id = "test_container_a"
        container_b_id = "test_container_b"
        
        # Create isolated buses
        bus_a = self.isolation_manager.create_container_bus(container_a_id)
        bus_b = self.isolation_manager.create_container_bus(container_b_id)
        
        # Track events received
        events_a = []
        events_b = []
        
        def handler_a(event: Event):
            events_a.append(event)
            self.log_event(container_a_id, event, 'received')
        
        def handler_b(event: Event):
            events_b.append(event)
            self.log_event(container_b_id, event, 'received')
        
        # Subscribe handlers
        bus_a.subscribe(EventType.SIGNAL, handler_a)
        bus_b.subscribe(EventType.SIGNAL, handler_b)
        
        # Publish events to each bus
        event_for_a = Event(
            event_type=EventType.SIGNAL,
            payload={'test': 'event_a'},
            source_id='test_source_a',
            container_id=container_a_id
        )
        
        event_for_b = Event(
            event_type=EventType.SIGNAL,
            payload={'test': 'event_b'},
            source_id='test_source_b',
            container_id=container_b_id
        )
        
        bus_a.publish(event_for_a)
        self.log_event(container_a_id, event_for_a, 'published')
        
        bus_b.publish(event_for_b)
        self.log_event(container_b_id, event_for_b, 'published')
        
        # Wait for event processing
        time.sleep(0.1)
        
        # Validate isolation
        isolation_passed = (
            len(events_a) == 1 and
            len(events_b) == 1 and
            events_a[0].container_id == container_a_id and
            events_b[0].container_id == container_b_id and
            events_a[0].payload['test'] == 'event_a' and
            events_b[0].payload['test'] == 'event_b'
        )
        
        # Cleanup
        self.isolation_manager.remove_container_bus(container_a_id)
        self.isolation_manager.remove_container_bus(container_b_id)
        
        return isolation_passed
    
    def test_parallel_execution_isolation(self, num_containers: int = 10, events_per_container: int = 100) -> bool:
        """Test isolation under parallel execution stress."""
        container_ids = [f"parallel_test_container_{i}" for i in range(num_containers)]
        container_events = {cid: [] for cid in container_ids}
        
        def container_worker(container_id: str) -> bool:
            """Worker function for each container."""
            try:
                # Create container bus
                bus = self.isolation_manager.create_container_bus(container_id)
                
                # Track events for this container
                received_events = []
                
                def event_handler(event: Event):
                    received_events.append(event)
                    self.log_event(container_id, event, 'received')
                
                # Subscribe to events
                bus.subscribe(EventType.SIGNAL, event_handler)
                
                # Publish events
                for i in range(events_per_container):
                    event = Event(
                        event_type=EventType.SIGNAL,
                        payload={'container': container_id, 'event_num': i},
                        source_id=f'source_{container_id}',
                        container_id=container_id
                    )
                    bus.publish(event)
                    self.log_event(container_id, event, 'published')
                    
                    # Small delay to allow processing
                    time.sleep(0.001)
                
                # Wait for all events to process
                time.sleep(0.5)
                
                # Validate this container only received its own events
                container_isolation_passed = all(
                    event.container_id == container_id
                    for event in received_events
                )
                
                expected_event_count = len(received_events) == events_per_container
                
                # Store results
                container_events[container_id] = received_events
                
                # Cleanup
                self.isolation_manager.remove_container_bus(container_id)
                
                return container_isolation_passed and expected_event_count
                
            except Exception as e:
                print(f"Error in container {container_id}: {e}")
                return False
        
        # Run containers in parallel
        with ThreadPoolExecutor(max_workers=num_containers) as executor:
            futures = {
                executor.submit(container_worker, container_id): container_id 
                for container_id in container_ids
            }
            
            results = {}
            for future in as_completed(futures):
                container_id = futures[future]
                try:
                    results[container_id] = future.result()
                except Exception as e:
                    print(f"Container {container_id} failed: {e}")
                    results[container_id] = False
        
        # Overall validation
        all_containers_isolated = all(results.values())
        
        # Cross-container contamination check
        total_events_expected = num_containers * events_per_container
        total_events_received = sum(len(events) for events in container_events.values())
        
        no_event_loss = total_events_received == total_events_expected
        
        return all_containers_isolated and no_event_loss
    
    def test_event_metadata_integrity(self) -> bool:
        """Test that event metadata prevents contamination."""
        container_a = "metadata_test_a"
        container_b = "metadata_test_b"
        
        bus_a = self.isolation_manager.create_container_bus(container_a)
        bus_b = self.isolation_manager.create_container_bus(container_b)
        
        contamination_detected = []
        
        def contamination_detector(expected_container: str):
            def handler(event: Event):
                if not validate_event_isolation(event, expected_container):
                    contamination_detected.append({
                        'expected': expected_container,
                        'actual': event.container_id,
                        'event': event
                    })
            return handler
        
        # Subscribe detectors
        bus_a.subscribe(EventType.SIGNAL, contamination_detector(container_a))
        bus_b.subscribe(EventType.SIGNAL, contamination_detector(container_b))
        
        # Publish correct events
        correct_event_a = Event(
            event_type=EventType.SIGNAL,
            payload={'test': 'correct_a'},
            container_id=container_a
        )
        
        correct_event_b = Event(
            event_type=EventType.SIGNAL,
            payload={'test': 'correct_b'},
            container_id=container_b
        )
        
        bus_a.publish(correct_event_a)
        bus_b.publish(correct_event_b)
        
        # Try to publish event with wrong container_id (should be caught)
        wrong_event = Event(
            event_type=EventType.SIGNAL,
            payload={'test': 'wrong'},
            container_id=container_b  # Wrong container ID
        )
        
        bus_a.publish(wrong_event)  # Publishing to bus_a but with container_b ID
        
        time.sleep(0.1)
        
        # Should detect one contamination
        contamination_correctly_detected = len(contamination_detected) == 1
        
        # Cleanup
        self.isolation_manager.remove_container_bus(container_a)
        self.isolation_manager.remove_container_bus(container_b)
        
        return contamination_correctly_detected
    
    def generate_isolation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive isolation test report."""
        report = {
            'test_results': {
                'basic_isolation': self.test_basic_isolation(),
                'parallel_execution': self.test_parallel_execution_isolation(),
                'metadata_integrity': self.test_event_metadata_integrity()
            },
            'event_log_summary': self._analyze_event_log(),
            'recommendations': []
        }
        
        # Add recommendations based on results
        if not report['test_results']['basic_isolation']:
            report['recommendations'].append(
                "CRITICAL: Basic isolation failed - review event bus implementation"
            )
        
        if not report['test_results']['parallel_execution']:
            report['recommendations'].append(
                "CRITICAL: Parallel execution isolation failed - check thread safety"
            )
        
        if not report['test_results']['metadata_integrity']:
            report['recommendations'].append(
                "WARNING: Event metadata integrity issues detected"
            )
        
        if all(report['test_results'].values()):
            report['recommendations'].append(
                "✅ All isolation tests passed - system is ready for complex workflows"
            )
        
        return report
    
    def _analyze_event_log(self) -> Dict[str, Any]:
        """Analyze the event log for patterns and issues."""
        if not self.event_log:
            return {'status': 'no_events_logged'}
        
        total_events = len(self.event_log)
        published_events = [e for e in self.event_log if e['action'] == 'published']
        received_events = [e for e in self.event_log if e['action'] == 'received']
        
        # Check for container ID mismatches
        mismatches = [
            e for e in received_events 
            if e['container_id'] != e['event_container_id']
        ]
        
        return {
            'total_events': total_events,
            'published_count': len(published_events),
            'received_count': len(received_events),
            'container_mismatches': len(mismatches),
            'mismatch_details': mismatches[:5] if mismatches else [],  # First 5 mismatches
            'unique_containers': len(set(e['container_id'] for e in self.event_log))
        }


def run_comprehensive_isolation_tests() -> Dict[str, Any]:
    """Run all isolation tests and return comprehensive report."""
    test_framework = IsolationTestFramework()
    return test_framework.generate_isolation_report()


if __name__ == "__main__":
    # Run tests when executed directly
    report = run_comprehensive_isolation_tests()
    
    print("🧪 Event Bus Isolation Test Report")
    print("=" * 50)
    
    for test_name, passed in report['test_results'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print("\n📊 Event Log Analysis:")
    log_summary = report['event_log_summary']
    if log_summary.get('status') != 'no_events_logged':
        print(f"  Total Events: {log_summary['total_events']}")
        print(f"  Published: {log_summary['published_count']}")
        print(f"  Received: {log_summary['received_count']}")
        print(f"  Container Mismatches: {log_summary['container_mismatches']}")
        print(f"  Unique Containers: {log_summary['unique_containers']}")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
