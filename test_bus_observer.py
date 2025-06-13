#!/usr/bin/env python3
"""
Debug bus observer attachment
"""

import logging
from src.core.coordinator.coordinator import Coordinator
from src.core.events.bus import EventBus

# Enable debug logging  
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on specific loggers
for logger_name in ['src.core.containers', 'src.data', 'src.strategy.strategies']:
    logging.getLogger(logger_name).setLevel(logging.INFO)

# Minimal test configuration
config = {
    'data': {'symbols': ['SPY'], 'source': 'csv'},
    'symbols': ['SPY'],
    'timeframes': ['1m'], 
    'max_bars': 5,
    'strategies': [{
        'name': 'test_ma',
        'type': 'ma_crossover',
        'params': {'fast_period': 2, 'slow_period': 3}
    }],
    'metadata': {'workflow_id': 'observer_debug'},
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {'use_sparse_storage': True}
    }
}

# Patch EventBus to track observers
original_attach = EventBus.attach_observer
observer_map = {}

def patched_attach(self, observer):
    original_attach(self, observer)
    observer_name = type(observer).__name__
    observer_map[observer_name] = self.bus_id
    print(f"✅ Attached {observer_name} to bus {self.bus_id} (observers: {len(self._observers)})")

EventBus.attach_observer = patched_attach

# Patch publish to show observer count
original_publish = EventBus.publish
publish_counts = {}

def patched_publish(self, event):
    bus_key = f"{self.bus_id}:{event.event_type}"
    publish_counts[bus_key] = publish_counts.get(bus_key, 0) + 1
    
    if event.event_type in ['SIGNAL', 'BAR'] and publish_counts[bus_key] <= 2:
        print(f"📤 Bus {self.bus_id} publishing {event.event_type} (observers: {len(self._observers)})")
        for i, obs in enumerate(self._observers):
            print(f"   Observer {i}: {type(obs).__name__}")
    
    original_publish(self, event)

EventBus.publish = patched_publish

# Run topology
coordinator = Coordinator()
print("\n🚀 Running signal_generation topology...\n")
result = coordinator.run_topology('signal_generation', config)

print("\n📊 Observer attachment summary:")
for obs_name, bus_id in observer_map.items():
    print(f"  {obs_name} -> {bus_id}")

print("\n📤 Publish counts:")
for bus_event, count in sorted(publish_counts.items()):
    print(f"  {bus_event}: {count} events")