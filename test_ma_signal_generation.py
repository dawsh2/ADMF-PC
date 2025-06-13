#!/usr/bin/env python3
"""Test MA crossover signal generation with detailed output."""

import logging
import yaml
from src.core.coordinator.coordinator import Coordinator

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Filter out some noisy loggers
logging.getLogger('src.core.coordinator.config.pattern_loader').setLevel(logging.WARNING)
logging.getLogger('src.core.events.storage.hierarchical').setLevel(logging.WARNING)
logging.getLogger('src.core.containers.factory').setLevel(logging.WARNING)
logging.getLogger('src.core.events.bus').setLevel(logging.INFO)
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)
logging.getLogger('src.data.handlers').setLevel(logging.DEBUG)

# Create test config
config = {
    'symbols': ['SPY'],
    'timeframes': ['1m'],
    'data_source': 'file',
    'data_dir': './data',
    'start_date': '2024-03-26',
    'end_date': '2024-03-26',
    'max_bars': 20,  # Just 20 bars for testing
    
    'strategies': [{
        'name': 'ma_crossover',
        'type': 'ma_crossover',
        'params': {
            'fast_period': 5,
            'slow_period': 10
        }
    }],
    
    'execution': {
        'enable_event_tracing': True,
        'trace_settings': {
            'storage_backend': 'hierarchical',
            'batch_size': 10,
            'auto_flush_on_cleanup': True,
            'enable_console_output': True,
            'console_filter': ['SIGNAL']
        }
    }
}

# Create coordinator and run
coordinator = Coordinator()
result = coordinator.run_topology('signal_generation', config)

print(f"\n✅ Execution completed!")
print(f"Result keys: {result.keys() if isinstance(result, dict) else result}")