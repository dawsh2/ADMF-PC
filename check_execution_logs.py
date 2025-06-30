#!/usr/bin/env python3
"""Check execution with detailed logging."""

import logging
import sys

# Enable debug logging for strategy state
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on ComponentState
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)

# Run
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '60']

from main import main
main()