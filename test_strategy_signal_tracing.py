#!/usr/bin/env python3
"""Test strategy-level signal tracing."""

import sys
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.coordinator import UnifiedCoordinator
from core.events.types import EventType
import time


def main():
    """Run strategy signal tracing test."""
    
    # Load configuration
    config_path = "config/test_strategy_signal_tracing.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Testing strategy-level signal tracing with config: {config_path}")
    
    # Create coordinator
    coordinator = UnifiedCoordinator()
    
    try:
        # Run workflow
        logger.info("Starting workflow...")
        results = coordinator.run_workflow(config, topology_type='signal_generation')
        
        logger.info(f"Workflow completed. Results: {results}")
        
        # Check workspace directory for strategy signal files
        import os
        workflow_id = config.get('metadata', {}).get('workflow_id', 'strategy_signal_test')
        workspace_dir = Path(f"workspaces/{workflow_id}")
        
        if workspace_dir.exists():
            logger.info(f"\nFiles created in {workspace_dir}:")
            for root, dirs, files in os.walk(workspace_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, workspace_dir)
                    logger.info(f"  {rel_path}")
                    
                    # Check if it's a strategy signal file
                    if 'strategies' in rel_path and file.endswith('.json'):
                        logger.info(f"\n  Strategy signal file content:")
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Show first few lines
                            lines = content.split('\n')[:20]
                            for line in lines:
                                logger.info(f"    {line}")
                            if len(content.split('\n')) > 20:
                                logger.info(f"    ... ({len(content.split('\n')) - 20} more lines)")
        else:
            logger.warning(f"Workspace directory {workspace_dir} not found")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())