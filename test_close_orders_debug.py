#!/usr/bin/env python3
"""Test script to debug why close orders aren't being generated at END_OF_DATA."""

import asyncio
import logging
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_close_orders.log', mode='w')
    ]
)

async def main():
    """Run a simple backtest and check for close orders."""
    # Import after logging is configured
    from src.core.coordinator.yaml_coordinator import run_backtest
    
    # Create a simple config that should generate positions
    config = {
        'backtest': {
            'start_date': '2024-01-01',
            'end_date': '2024-01-05',  # Short period
            'initial_capital': 100000,
            'data_source': 'csv',
            'csv_path': 'data/SPY.csv'
        },
        'strategies': [
            {
                'name': 'simple_momentum',
                'type': 'momentum',
                'params': {
                    'fast_period': 5,
                    'slow_period': 10,
                    'threshold': 0.01
                }
            }
        ],
        'risk': {
            'position_sizer': {
                'type': 'fixed_dollar',
                'params': {
                    'amount': 1000
                }
            },
            'risk_limits': {
                'max_position_pct': 20,
                'max_exposure_pct': 90,
                'max_positions': 10
            }
        },
        'execution': {
            'broker': 'backtest',
            'costs': {
                'commission': 1.0,
                'slippage_pct': 0.1
            }
        }
    }
    
    # Write config to temporary YAML file
    config_path = 'debug_close_orders_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run backtest
    print("Starting backtest...")
    result = await run_backtest(config_path)
    
    # Check result
    if result:
        print(f"\nBacktest completed successfully")
        print(f"Final portfolio value: ${result.get('final_value', 0):,.2f}")
        print(f"Total return: {result.get('total_return', 0):.2%}")
        
        # Check for trades
        if 'trades' in result:
            print(f"\nTotal trades: {len(result['trades'])}")
            for i, trade in enumerate(result['trades'][:5]):  # Show first 5
                print(f"  Trade {i+1}: {trade}")
        
        # Check for final positions
        if 'final_positions' in result:
            print(f"\nFinal positions: {result['final_positions']}")
    else:
        print("Backtest failed")
    
    print("\nCheck debug_close_orders.log for detailed debug output")

if __name__ == "__main__":
    asyncio.run(main())