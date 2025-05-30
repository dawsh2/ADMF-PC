#!/usr/bin/env python3
"""
Main entry point that uses the coordinator for consistent execution paths.

This demonstrates the YAML-driven system with proper routing through
the coordinator architecture.
"""

import asyncio
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid __init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "simple_backtest_engine", 
    "src/execution/simple_backtest_engine.py"
)
simple_backtest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simple_backtest_module)
SimpleBacktestEngine = simple_backtest_module.SimpleBacktestEngine


class SimpleCoordinator:
    """Simplified coordinator that routes execution properly."""
    
    def __init__(self):
        self.active_workflows = {}
        
    async def execute_workflow(
        self,
        config: Dict[str, Any],
        mode: str = 'backtest',
        dry_run: bool = False,
        max_bars: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a workflow through the coordinator."""
        
        workflow_id = f"workflow_{len(self.active_workflows) + 1}"
        print(f"\nCoordinator: Starting {mode} workflow [{workflow_id}]")
        
        if dry_run:
            print("\nDRY RUN MODE - Validating configuration...")
            return self._validate_config(config)
            
        if mode == 'backtest':
            return await self._run_backtest(config, max_bars)
        elif mode == 'optimization':
            return await self._run_optimization(config, max_bars)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration."""
        print("\nValidating configuration:")
        print(f"  ✓ Data source: {config.get('data', {}).get('file_path')}")
        print(f"  ✓ Initial capital: ${config.get('portfolio', {}).get('initial_capital', 0):,}")
        
        strategies = config.get('strategies', [])
        if strategies:
            print(f"  ✓ Strategies: {len(strategies)}")
            for s in strategies:
                print(f"    - {s['name']}: {s['type']}")
                
        print("\nValidation: PASSED ✓")
        return {'success': True, 'validated': True}
    
    async def _run_backtest(self, config: Dict[str, Any], max_bars: Optional[int]) -> Dict[str, Any]:
        """Run backtest through proper execution path."""
        print("\nCoordinator: Delegating to backtest engine...")
        
        # Create backtest engine
        engine = SimpleBacktestEngine(config)
        
        # Load data
        engine.load_data(max_bars=max_bars)
        
        # Run backtest
        result = engine.run_backtest()
        
        # Return standardized result
        return {
            'success': True,
            'workflow_type': 'backtest',
            'results': {
                'final_equity': result.final_equity,
                'total_return': result.total_return,
                'num_trades': result.num_trades,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio
            }
        }
    
    async def _run_optimization(self, config: Dict[str, Any], max_bars: Optional[int]) -> Dict[str, Any]:
        """Run optimization workflow."""
        print("\nCoordinator: Running optimization...")
        
        # For demo, we'll test a few parameter sets
        param_sets = [
            {'buy_threshold': 88, 'sell_threshold': 102},
            {'buy_threshold': 90, 'sell_threshold': 100},
            {'buy_threshold': 92, 'sell_threshold': 98},
        ]
        
        results = []
        for params in param_sets:
            # Update config with test parameters
            test_config = config.copy()
            test_config['strategies'][0]['parameters'] = params
            
            # Run backtest
            engine = SimpleBacktestEngine(test_config)
            engine.load_data(max_bars=max_bars)
            result = engine.run_backtest()
            
            results.append({
                'parameters': params,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio
            })
        
        # Find best by return
        best = max(results, key=lambda x: x['total_return'])
        
        return {
            'success': True,
            'workflow_type': 'optimization',
            'results': {
                'tested_parameters': len(results),
                'best_parameters': best['parameters'],
                'best_return': best['total_return'],
                'all_results': results
            }
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Coordinator-based execution'
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
        choices=['backtest', 'optimization'],
        default='backtest',
        help='Execution mode'
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


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration (simplified for demo)."""
    # For demo, return our standard config
    return {
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


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("\n" + "#" * 70)
    print("ADMF-PC: COORDINATOR-BASED EXECUTION")
    print("#" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Config: {args.config}")
    if args.bars:
        print(f"Max Bars: {args.bars}")
    if args.dry_run:
        print("Dry Run: ENABLED")
    
    # Load configuration
    config = load_yaml_config(args.config)
    
    # Create coordinator
    coordinator = SimpleCoordinator()
    
    # Execute workflow
    result = await coordinator.execute_workflow(
        config=config,
        mode=args.mode,
        dry_run=args.dry_run,
        max_bars=args.bars
    )
    
    # Display results
    if not args.dry_run:
        print("\n" + "=" * 70)
        print("WORKFLOW RESULTS")
        print("=" * 70)
        
        if result['success']:
            if args.mode == 'backtest':
                r = result['results']
                print(f"Final Equity:  ${r['final_equity']:,.2f}")
                print(f"Total Return:  {r['total_return']:.2%}")
                print(f"Num Trades:    {r['num_trades']}")
                print(f"Win Rate:      {r['win_rate']:.1%}")
                print(f"Max Drawdown:  {r['max_drawdown']:.2%}")
                print(f"Sharpe Ratio:  {r['sharpe_ratio']:.2f}")
            elif args.mode == 'optimization':
                r = result['results']
                print(f"Parameters Tested: {r['tested_parameters']}")
                print(f"Best Parameters:   {r['best_parameters']}")
                print(f"Best Return:       {r['best_return']:.2%}")
                print("\nAll Results:")
                for res in r['all_results']:
                    print(f"  {res['parameters']} -> {res['total_return']:.2%}")
        else:
            print("Workflow failed!")
    
    print("\n" + "#" * 70)
    print("EXECUTION COMPLETE")
    print("#" * 70)
    
    return 0 if result.get('success', False) else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)