#!/usr/bin/env python3
"""Debug the main flow to see how features are configured."""

import sys
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on specific loggers
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.components.features.hub').setLevel(logging.DEBUG)
logging.getLogger('src.core.coordinator.topology').setLevel(logging.DEBUG)
logging.getLogger('src.core.coordinator.sequencer').setLevel(logging.DEBUG)

# Run with minimal config
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '30']

print("=== Starting main.py with debug logging ===")

# Import and run
from main import main
main()