#!/usr/bin/env python3
"""
Debug event bus configuration
"""

import logging
from src.core.coordinator.coordinator import Coordinator

# Enable debug logging  
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on bus creation
logging.getLogger('src.core.containers').setLevel(logging.DEBUG)
logging.getLogger('src.core.events.bus').setLevel(logging.DEBUG)
logging.getLogger('src.core.coordinator.topology').setLevel(logging.DEBUG)

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
    'metadata': {'workflow_id': 'bus_debug'},
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {'use_sparse_storage': True}
    }
}

# Patch to track bus IDs
from src.core.containers import container
original_init = container.Container.__init__
bus_map = {}

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    bus_map[self.name] = (self.event_bus.bus_id, id(self.event_bus))
    print(f"📦 Container '{self.name}' -> bus_id={self.event_bus.bus_id}, obj_id={id(self.event_bus)}")

container.Container.__init__ = patched_init

# Run topology
coordinator = Coordinator()
print("\n🚀 Running signal_generation topology...\n")
result = coordinator.run_topology('signal_generation', config)

print("\n📊 Bus Summary:")
for name, (bus_id, obj_id) in bus_map.items():
    print(f"  {name}: {bus_id} (obj: {obj_id})")
    
# Check if all containers share the same bus object
obj_ids = [obj_id for _, (_, obj_id) in bus_map.items()]
if len(set(obj_ids)) == 1:
    print("\n✅ All containers share the same event bus object!")
else:
    print(f"\n❌ Containers have {len(set(obj_ids))} different bus objects!")