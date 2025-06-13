#!/usr/bin/env python3
"""
Simple test for hierarchical storage - run signal generation only.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.coordinator import Coordinator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run signal generation with hierarchical storage."""
    logger.info("=" * 80)
    logger.info("Testing Hierarchical Storage - Signal Generation")
    logger.info("=" * 80)
    
    # Load config
    config_path = "config/test_hierarchical_storage.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config: {config_path}")
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Run the workflow
    logger.info("\nRunning signal generation workflow...")
    try:
        # Extract topology mode from config
        topology_mode = config.get('topology', {}).get('mode', 'signal_generation')
        result = coordinator.run_topology(topology_mode, config)
        logger.info("Workflow completed successfully")
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1
    
    # Check if storage was created
    storage_dir = Path("./analytics_storage")
    if storage_dir.exists():
        logger.info(f"\n✅ Analytics storage directory created: {storage_dir}")
        
        # List contents
        logger.info("\nStorage contents:")
        for item in storage_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(storage_dir)
                size = item.stat().st_size
                logger.info(f"  {rel_path} ({size} bytes)")
    else:
        logger.error(f"\n❌ Analytics storage directory not found: {storage_dir}")
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test completed!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())