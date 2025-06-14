#!/usr/bin/env python
"""Run grid search with debug logging."""

import logging
import sys
from analytics import main

# Configure logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s [%(name)s] %(message)s',
    stream=sys.stdout
)

# Focus on key loggers
loggers_to_debug = [
    'src.strategy.state',
    'src.core.coordinator.topology',
    'src.core.events.observers.multi_strategy_tracer'
]

for logger_name in loggers_to_debug:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

# Run with a smaller number of bars for debugging
sys.argv = ['analytics.py', 'run', 'config/expansive_grid_search.yaml', '--max-bars', '10']
main()