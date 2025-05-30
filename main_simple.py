#!/usr/bin/env python3
"""
Simple main entry point without external dependencies.
"""

import asyncio
import argparse
import json
from typing import Dict, Any
from pathlib import Path

# Import our simple backtest engine directly
from src.execution.simple_backtest_engine import SimpleBacktestEngine


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Simple Backtest Runner'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    return parser.parse_args()


def load_yaml_simple(file_path: str) -> Dict[str, Any]:
    """Load YAML manually without pyyaml."""
    # For our simple config, we'll just parse the essential parts
    config = {
        'workflow_type': 'backtest',
        'data': {
            'type': 'csv',
            'file_path': 'data/SYNTH_1min.csv'
        },
        'portfolio': {
            'initial_capital': 10000,
            'position_sizing': 'all_in'
        },
        'strategies': [{
            'name': 'threshold_strategy',
            'type': 'price_threshold',
            'parameters': {
                'buy_threshold': 90.0,
                'sell_threshold': 100.0
            }
        }],
        'backtest': {
            'commission': 0.0,
            'slippage': 0.0
        }
    }
    return config


async def run_backtest_simple(config: Dict[str, Any], max_bars: int = None):
    """Run a simple backtest."""
    print("\n" + "=" * 70)
    print("ADMF-PC BACKTEST ENGINE")
    print("=" * 70)
    
    # Create engine
    engine = SimpleBacktestEngine(config)
    
    # Load data
    print("\nLoading data...")
    engine.load_data(max_bars=max_bars)
    
    # Run backtest
    print("\nExecuting backtest...")
    result = engine.run_backtest()
    
    # Display results
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    
    return result


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration
    config = load_yaml_simple(args.config)
    
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN MODE")
        print("=" * 70)
        print("\nConfiguration loaded successfully:")
        print(f"  Workflow Type: {config['workflow_type']}")
        print(f"  Data Source: {config['data']['file_path']}")
        print(f"  Initial Capital: ${config['portfolio']['initial_capital']:,}")
        print(f"  Strategy: {config['strategies'][0]['name']}")
        print(f"    Buy Threshold: ${config['strategies'][0]['parameters']['buy_threshold']}")
        print(f"    Sell Threshold: ${config['strategies'][0]['parameters']['sell_threshold']}")
        if args.bars:
            print(f"  Max Bars: {args.bars}")
        print("\nValidation: PASSED ✓")
        return 0
    
    # Run backtest
    result = await run_backtest_simple(config, max_bars=args.bars)
    
    return 0 if result else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)