#!/usr/bin/env python3
"""
Working main.py that properly routes through the coordinator.
"""

import asyncio
import argparse
import yaml
from typing import Dict, Any
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Dynamic Market Framework - Protocol Components'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'optimization', 'signal-generation', 'signal-replay', 'live'],
        default=None,
        help='Override execution mode from config'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars (useful for testing)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


class SimpleCoordinator:
    """Simplified coordinator that can execute workflows."""
    
    def __init__(self):
        self.logger = logger
        
    async def execute_workflow(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Execute a workflow based on configuration."""
        
        mode = args.mode or config.get('workflow_type', 'backtest')
        
        self.logger.info(f"Executing workflow in {mode} mode")
        
        if args.dry_run:
            return await self._dry_run(config, args)
            
        if mode == 'backtest':
            return await self._run_backtest(config, args)
        elif mode == 'optimization':
            return await self._run_optimization(config, args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    async def _dry_run(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Validate configuration without executing."""
        self.logger.info("Running in dry-run mode - validating configuration")
        
        # Validate required sections
        required = ['data', 'strategies']
        for section in required:
            if section not in config:
                return {
                    'success': False,
                    'errors': [f"Missing required section: {section}"]
                }
        
        # Check data configuration
        data_config = config['data']
        if 'file_path' not in data_config:
            return {
                'success': False,
                'errors': ["Missing data.file_path"]
            }
            
        # Check strategies
        strategies = config.get('strategies', [])
        if not strategies:
            return {
                'success': False,
                'errors': ["No strategies defined"]
            }
            
        self.logger.info("Configuration validation PASSED")
        return {
            'success': True,
            'message': 'Configuration is valid',
            'details': {
                'data_source': data_config['file_path'],
                'num_strategies': len(strategies),
                'max_bars': args.bars
            }
        }
    
    async def _run_backtest(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Run a backtest using the simple engine."""
        try:
            # Import here to avoid circular imports
            from src.execution.simple_backtest_engine import SimpleBacktestEngine
            
            self.logger.info("Creating backtest engine")
            engine = SimpleBacktestEngine(config)
            
            self.logger.info(f"Loading data (max_bars={args.bars})")
            engine.load_data(max_bars=args.bars)
            
            self.logger.info("Running backtest")
            result = engine.run_backtest()
            
            return {
                'success': True,
                'workflow_type': 'backtest',
                'results': {
                    'final_equity': result.final_equity,
                    'total_return': result.total_return,
                    'num_trades': result.num_trades,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown
                }
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {
                'success': False,
                'errors': [str(e)]
            }
    
    async def _run_optimization(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """Run optimization workflow."""
        self.logger.info("Running optimization workflow")
        
        # For now, just run a simple parameter sweep
        param_sets = [
            {'buy_threshold': 88, 'sell_threshold': 102},
            {'buy_threshold': 90, 'sell_threshold': 100},
            {'buy_threshold': 92, 'sell_threshold': 98},
        ]
        
        results = []
        for params in param_sets:
            # Update config
            test_config = config.copy()
            test_config['strategies'][0]['parameters'] = params
            
            # Run backtest
            backtest_result = await self._run_backtest(test_config, args)
            if backtest_result['success']:
                results.append({
                    'parameters': params,
                    'total_return': backtest_result['results']['total_return']
                })
        
        # Find best
        best = max(results, key=lambda x: x['total_return'])
        
        return {
            'success': True,
            'workflow_type': 'optimization',
            'results': {
                'tested_parameters': len(results),
                'best_parameters': best['parameters'],
                'best_return': best['total_return']
            }
        }


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create coordinator
    coordinator = SimpleCoordinator()
    
    # Execute workflow
    result = await coordinator.execute_workflow(config, args)
    
    # Display results
    if result['success']:
        logger.info("Workflow completed successfully")
        if not args.dry_run:
            print("\nResults:")
            for key, value in result.get('results', {}).items():
                if isinstance(value, float):
                    if 'return' in key or 'rate' in key:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    else:
        logger.error("Workflow failed")
        for error in result.get('errors', []):
            logger.error(f"  - {error}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)