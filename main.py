#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components

Main entry point - parses arguments and delegates to Bootstrap.
"""

import asyncio
import argparse
import yaml
from typing import Dict, Any

# Use minimal imports to avoid deep chains
try:
    from src.core.containers.minimal_bootstrap import MinimalBootstrap as ContainerBootstrap
    USING_MINIMAL = True
except ImportError:
    from src.core.containers.bootstrap import ContainerBootstrap
    USING_MINIMAL = False

# Simple types for configuration
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class WorkflowType(str, Enum):
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE = "live"

@dataclass
class WorkflowConfig:
    workflow_type: WorkflowType
    parameters: Dict[str, Any]
    data_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    
    def dict(self):
        return {
            'workflow_type': self.workflow_type.value,
            'parameters': self.parameters,
            'data_config': self.data_config,
            'backtest_config': self.backtest_config,
            'optimization_config': self.optimization_config
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components'
    )
    
    # Core arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    # Execution mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'optimization', 'signal-generation', 'signal-replay', 'live'],
        default=None,
        help='Override execution mode from config'
    )
    
    parser.add_argument(
        '--signal-log',
        type=str,
        default=None,
        help='Path to signal log file for replay mode'
    )
    
    parser.add_argument(
        '--signal-output',
        type=str,
        default=None,
        help='Path to save generated signals (signal-generation mode)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='JSON string or file with strategy weights for signal-replay mode'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'test', 'full'],
        default=None,
        help='Dataset to use (enables reproducible train/test splits)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars (useful for testing)'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=None,
        help='Train/test split ratio when dataset is "full"'
    )
    
    # Execution arguments
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        help='Number of parallel workers for optimization'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output files'
    )
    
    # Logging arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log to file instead of console'
    )
    
    # Development arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()


def build_workflow_config(args: argparse.Namespace, base_config: Dict[str, Any]) -> WorkflowConfig:
    """Build workflow configuration from arguments and base config."""
    # Determine execution mode
    mode = args.mode or base_config.get('workflow_type', 'backtest')
    
    # Create workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType(mode if mode in ['backtest', 'optimization', 'live'] else 'backtest'),
        parameters=base_config.get('parameters', {}),
        data_config=base_config.get('data', {}),
        backtest_config=base_config.get('backtest', {}),
        optimization_config=base_config.get('optimization', {})
    )
    
    # Apply CLI overrides
    if args.dataset:
        workflow_config.data_config['dataset'] = args.dataset
    if args.bars:
        workflow_config.data_config['max_bars'] = args.bars
    if args.split_ratio:
        workflow_config.data_config['split_ratio'] = args.split_ratio
    
    if args.parallel:
        workflow_config.parameters['parallel_workers'] = args.parallel
    if args.checkpoint:
        workflow_config.parameters['checkpoint_file'] = args.checkpoint
    if args.output_dir:
        workflow_config.parameters['output_dir'] = args.output_dir
        
    if args.verbose:
        workflow_config.parameters['verbose'] = True
    if args.log_file:
        workflow_config.parameters['log_file'] = args.log_file
    if args.dry_run:
        workflow_config.parameters['dry_run'] = True
    if args.profile:
        workflow_config.parameters['profile'] = True
    
    return workflow_config


async def main():
    """Main entry point - parse args and delegate to Bootstrap."""
    args = parse_arguments()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Using {'minimal' if USING_MINIMAL else 'full'} bootstrap")
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create bootstrap
    bootstrap = ContainerBootstrap(config_path=args.config)
    bootstrap.initialize()
    
    # Build workflow configuration
    workflow_config = build_workflow_config(args, base_config)
    
    # Prepare mode-specific arguments
    mode = args.mode or base_config.get('workflow_type', 'backtest')
    mode_args = {}
    
    if mode == 'signal-generation':
        mode_args['signal_output'] = args.signal_output
    elif mode == 'signal-replay':
        mode_args['signal_log'] = args.signal_log
        mode_args['weights'] = args.weights
    
    # Execute workflow through bootstrap
    result = await bootstrap.execute_workflow(
        workflow_config=workflow_config.dict(),
        mode_override=mode if mode in ['signal-generation', 'signal-replay'] else None,
        mode_args=mode_args
    )
    
    # Shutdown
    await bootstrap.shutdown()
    
    # Log result
    if result.get('success'):
        logger.info("Workflow completed successfully")
        if result.get('results'):
            logger.info(f"Results: {result['results']}")
    else:
        logger.error("Workflow failed")
        if result.get('errors'):
            for error in result['errors']:
                logger.error(f"  - {error}")
    
    # Return appropriate exit code
    return 0 if result.get('success', False) else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code if exit_code is not None else 0)