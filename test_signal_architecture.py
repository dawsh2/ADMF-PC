#!/usr/bin/env python3
"""Test the updated signal generation architecture."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.coordinator import Coordinator


def main():
    """Test signal generation with single strategy container."""
    
    config = {
        'name': 'Signal Architecture Test',
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
        'max_bars': 50,
        'features': {
            'fast_ma': {'feature': 'sma', 'period': 5},
            'slow_ma': {'feature': 'sma', 'period': 20}
        },
        'strategies': [
            {
                'name': 'ma_crossover_5_20',
                'type': 'ma_crossover',
                'params': {'fast_period': 5, 'slow_period': 20}
            },
            {
                'name': 'ma_crossover_10_30',
                'type': 'ma_crossover',
                'params': {'fast_period': 10, 'slow_period': 30}
            }
        ],
        'execution': {
            'max_duration': 0.0,
            'enable_event_tracing': True,
            'trace_settings': {
                'use_sparse_storage': True,
                'enable_console_output': False,
                'container_settings': {
                    'strategy*': {'enabled': True},
                    'portfolio*': {'enabled': False}
                }
            }
        },
        'metadata': {
            'workflow_id': 'signal_arch_test',
            'experiment_type': 'signal_generation'
        }
    }
    
    logger.info("Testing signal generation architecture...")
    
    # Create coordinator
    coordinator = Coordinator()
    
    try:
        # Run signal generation
        results = coordinator.run_topology('signal_generation', config)
        
        logger.info(f"Results: {results}")
        
        # Check workspace directory
        import os
        workspace_dir = Path(f"workspaces/signal_arch_test")
        
        if workspace_dir.exists():
            logger.info(f"\nFiles created in {workspace_dir}:")
            for root, dirs, files in os.walk(workspace_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, workspace_dir)
                    logger.info(f"  {rel_path}")
        else:
            logger.warning(f"Workspace directory {workspace_dir} not found")
            # Check tmp directory
            tmp_dirs = sorted(Path("workspaces/tmp").glob("*"), key=lambda x: x.stat().st_mtime)
            if tmp_dirs:
                latest = tmp_dirs[-1]
                logger.info(f"\nLatest tmp directory: {latest}")
                for file in latest.rglob("*.json"):
                    logger.info(f"  {file.name}")
                    
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())