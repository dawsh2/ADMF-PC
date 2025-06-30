#!/usr/bin/env python3
"""Debug signal flow in detail."""

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to DEBUG
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.components.features.hub').setLevel(logging.DEBUG)
logging.getLogger('src.core.events').setLevel(logging.DEBUG)

# Now run a minimal test
import sys
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '30']

# Import and run main
from main import main
main()