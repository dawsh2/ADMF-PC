#!/usr/bin/env python3
"""
Debug MultiStrategyTracer attachment
"""

import logging
from src.core.coordinator.coordinator import Coordinator
from src.core.events.observers.multi_strategy_tracer import MultiStrategyTracer

# Enable debug logging  
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on tracer
logging.getLogger('src.core.containers').setLevel(logging.INFO)
logging.getLogger('src.data').setLevel(logging.INFO)

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
    'metadata': {'workflow_id': 'tracer_debug'},
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {'use_sparse_storage': True}
    }
}

# Patch MultiStrategyTracer to log events
original_on_event = MultiStrategyTracer.on_event
event_count = {'bar': 0, 'signal': 0, 'other': 0}

def patched_on_event(self, event):
    event_type = event.event_type
    print(f"🔍 MultiStrategyTracer.on_event: {event_type}")
    
    if event_type == 'BAR':
        event_count['bar'] += 1
    elif event_type == 'SIGNAL':
        event_count['signal'] += 1
        print(f"   📡 SIGNAL: {event.payload}")
    else:
        event_count['other'] += 1
        
    original_on_event(self, event)

MultiStrategyTracer.on_event = patched_on_event

# Also patch on_publish
original_on_publish = MultiStrategyTracer.on_publish

def patched_on_publish(self, event):
    print(f"🎯 MultiStrategyTracer.on_publish called with {event.event_type}")
    original_on_publish(self, event)

MultiStrategyTracer.on_publish = patched_on_publish

# Run topology
coordinator = Coordinator()
print("\n🚀 Running signal_generation topology...\n")
result = coordinator.run_topology('signal_generation', config)

print(f"\n📊 Event counts: BAR={event_count['bar']}, SIGNAL={event_count['signal']}, OTHER={event_count['other']}")

if 'tracer_results' in result:
    tr = result['tracer_results']
    print(f"\n📈 Tracer results: signals={tr.get('total_signals')}, bars={tr.get('total_bars')}")
else:
    print("\n❌ No tracer results found")