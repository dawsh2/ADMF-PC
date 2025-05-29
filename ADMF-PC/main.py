#!/usr/bin/env python3
"""
ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components

Main entry point - parses arguments and delegates to Coordinator.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any

from src.core.coordinator import Coordinator
from src.core.containers import Bootstrap


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


async def main():
    """Main entry point - parse args and delegate to Coordinator."""
    args = parse_arguments()
    
    # Build configuration overrides from command line
    cli_config = {
        'logging': {
            'verbose': args.verbose,
            'log_file': args.log_file
        },
        'execution': {
            'dry_run': args.dry_run,
            'profile': args.profile
        }
    }
    
    # Add optional data overrides
    if any([args.dataset, args.bars, args.split_ratio]):
        cli_config['data'] = {}
        if args.dataset:
            cli_config['data']['dataset'] = args.dataset
        if args.bars:
            cli_config['data']['max_bars'] = args.bars
        if args.split_ratio:
            cli_config['data']['split_ratio'] = args.split_ratio
    
    # Add optional execution overrides
    if any([args.parallel, args.checkpoint, args.output_dir]):
        if 'execution' not in cli_config:
            cli_config['execution'] = {}
        if args.parallel:
            cli_config['execution']['parallel_workers'] = args.parallel
        if args.checkpoint:
            cli_config['execution']['checkpoint_file'] = args.checkpoint
        if args.output_dir:
            cli_config['execution']['output_dir'] = args.output_dir
    
    # Bootstrap handles everything else
    bootstrap = Bootstrap()
    coordinator = await bootstrap.create_coordinator(
        config_path=args.config,
        cli_overrides=cli_config
    )
    
    # Execute workflow (type determined by config)
    result = await coordinator.execute()
    
    # Return appropriate exit code
    return 0 if result.success else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)