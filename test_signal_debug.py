#!/usr/bin/env python3
"""
Debug signal flow
"""

import logging
from src.core.coordinator.coordinator import Coordinator
from src.core.events.types import Event, EventType

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on key loggers
logging.getLogger('src.core.containers').setLevel(logging.INFO)
logging.getLogger('src.data').setLevel(logging.INFO)
logging.getLogger('src.strategy.strategies').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)
logging.getLogger('src.core.events.bus').setLevel(logging.DEBUG)
logging.getLogger('src.core.events.observers.multi_strategy_tracer').setLevel(logging.DEBUG)

# Test configuration
config = {
    'data': {
        'symbols': ['SPY'],
        'source': 'csv'
    },
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'max_bars': 25,  # Just enough for sma_20
    
    'strategies': [
        {
            'name': 'ma_5_20',
            'type': 'ma_crossover',
            'params': {
                'fast_period': 5,
                'slow_period': 20
            }
        }
    ],
    
    'classifiers': [],
    
    'metadata': {
        'workflow_id': 'signal_debug'
    },
    
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'use_sparse_storage': True,
            'storage': {
                'base_dir': './workspaces'
            }
        }
    }
}

# Create a custom observer to track events
class EventLogger:
    def __init__(self):
        self.events = []
    
    def on_publish(self, event: Event) -> None:
        if event.event_type in [EventType.SIGNAL.value, EventType.BAR.value]:
            self.events.append((event.event_type, event.payload.get('symbol', '?')))
            if event.event_type == EventType.SIGNAL.value:
                print(f"🎯 SIGNAL EVENT: {event.payload}")
    
    def on_delivered(self, event: Event, handler) -> None:
        pass
    
    def on_error(self, event: Event, error: Exception) -> None:
        print(f"❌ ERROR: {error}")

# Create coordinator and run
coordinator = Coordinator()

print("Running signal generation topology...")

# Hook into the topology to add our observer
from src.core.coordinator.topology import TopologyBuilder
original_build = TopologyBuilder.build_topology
event_logger = EventLogger()

def patched_build(self, topology_def):
    result = original_build(self, topology_def)
    # Add our observer to root bus if available
    if 'root_event_bus' in self.config_resolver.context:
        root_bus = self.config_resolver.context['root_event_bus']
        root_bus.attach_observer(event_logger)
        print("✅ Attached EventLogger to root bus")
    return result

TopologyBuilder.build_topology = patched_build

result = coordinator.run_topology('signal_generation', config)

print(f"\nCaptured {len(event_logger.events)} events")
signal_count = sum(1 for e in event_logger.events if e[0] == EventType.SIGNAL.value)
print(f"Signal events: {signal_count}")

print("\nResult:", result.get('success'))
if 'tracer_results' in result:
    tr = result['tracer_results']
    print(f"Tracer - signals: {tr.get('total_signals')}, bars: {tr.get('total_bars')}")