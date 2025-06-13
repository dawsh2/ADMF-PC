#!/usr/bin/env python3
"""
Debug script to understand why hierarchical storage isn't being set up.
"""

import os
import sys
import yaml
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Only show logs for container setup and tracing
for name in logging.root.manager.loggerDict:
    if name not in ['src.core.containers.container', '__main__']:
        logging.getLogger(name).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from src.core.coordinator import Coordinator

def main():
    """Debug hierarchical storage setup."""
    logger.info("=" * 80)
    logger.info("Debugging Hierarchical Storage Setup")
    logger.info("=" * 80)
    
    # Load config
    config_path = "config/test_hierarchical_storage.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Execution config: {config.get('execution', {})}")
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Run the workflow
    logger.info("\nRunning signal generation workflow...")
    try:
        topology_mode = config.get('topology', {}).get('mode', 'signal_generation')
        result = coordinator.run_topology(topology_mode, config)
        logger.info("Workflow completed")
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())