#!/usr/bin/env python3
"""Debug strategy-level signal tracing."""

import sys
import logging
import yaml
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.coordinator import UnifiedCoordinator
from core.containers.container import Container


def main():
    """Run strategy signal tracing test with debug output."""
    
    # Simple test config
    config = {
        'name': 'Debug Strategy Signal Tracing',
        'data': {
            'symbols': ['SPY'],
            'source': 'csv',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        },
        'portfolio': {
            'initial_capital': 100000,
            'position_sizing': 'fixed',
            'max_positions': 1
        },
        'symbols': ['SPY'],
        'timeframes': ['1m'],
        'max_bars': 20,  # Small for testing
        'features': {
            'fast_ma': {
                'feature': 'sma',
                'period': 5
            },
            'slow_ma': {
                'feature': 'sma',
                'period': 20
            }
        },
        'strategies': [
            {
                'name': 'test_ma_crossover',
                'type': 'ma_crossover',
                'params': {
                    'fast_period': 5,
                    'slow_period': 20
                }
            }
        ],
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'use_sparse_storage': True,
                'enable_console_output': False,
                'container_settings': {
                    'strategy*': {
                        'enabled': True
                    }
                }
            }
        },
        'metadata': {
            'workflow_id': 'strategy_trace_debug',
            'experiment_type': 'signal_generation'
        }
    }
    
    logger.info("Testing strategy-level signal tracing...")
    
    # Create coordinator
    coordinator = UnifiedCoordinator()
    
    try:
        # Run workflow
        logger.info("Starting workflow...")
        results = coordinator.run_workflow(config, topology_type='signal_generation_strategy_trace')
        
        logger.info(f"Workflow completed. Results: {results}")
        
        # Check workspace directory
        import os
        workflow_id = config['metadata']['workflow_id']
        workspace_dir = Path(f"workspaces/{workflow_id}")
        
        if workspace_dir.exists():
            logger.info(f"\nFiles created in {workspace_dir}:")
            for root, dirs, files in os.walk(workspace_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, workspace_dir)
                    file_size = os.path.getsize(file_path)
                    logger.info(f"  {rel_path} ({file_size} bytes)")
        else:
            logger.warning(f"Workspace directory {workspace_dir} not found")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())