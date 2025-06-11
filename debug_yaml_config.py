#!/usr/bin/env python3
"""
Debug YAML configuration loading.
"""

import logging
import yaml
from src.core.coordinator.coordinator import Coordinator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Load config
    config_path = "config/test_yaml_signal_gen.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded: {config}")
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Add max_bars to config
    if 'workflow' in config:
        for phase in config['workflow'].get('phases', []):
            if 'config' in phase:
                phase['config']['max_bars'] = 50
    
    # Try to execute
    try:
        result = coordinator.run_workflow(config)
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()