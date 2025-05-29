"""
Example usage of the Coordinator for containerized backtests and optimizations.

This demonstrates how the Coordinator serves as the single entry point for
all high-level operations in the ADMF-PC system.
"""
import asyncio
from pathlib import Path
import yaml

from ..containers import UniversalScopedContainer
from ..components import ComponentFactory
from ..logging import StructuredLogger

from .coordinator import Coordinator
from .types import WorkflowConfig, WorkflowType


# Example 1: Simple Backtest
async def run_simple_backtest():
    """Run a simple backtest using the Coordinator."""
    
    # Create coordinator with shared services
    shared_services = {
        'market_data_provider': create_market_data_provider(),
        'indicator_library': create_indicator_library()
    }
    
    coordinator = Coordinator(shared_services=shared_services)
    
    # Define backtest configuration
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        
        # Data configuration
        data_config={
            'sources': {
                'csv': {
                    'type': 'csv',
                    'path': 'data/historical/EURUSD_1min.csv'
                }
            },
            'symbols': ['EURUSD'],
            'timeframe': '1min'
        },
        
        # Backtest-specific configuration
        backtest_config={
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            
            # Strategy configuration
            'strategy': {
                'class': 'TrendFollowingStrategy',
                'parameters': {
                    'fast_period': 10,
                    'slow_period': 30,
                    'signal_threshold': 0.02
                }
            },
            
            # Risk management
            'risk_management': {
                'position_sizer': {
                    'type': 'percent_equity',
                    'percentage': 2.0
                },
                'risk_limits': [
                    {'type': 'max_position_size', 'value': 10000},
                    {'type': 'max_drawdown', 'value': 0.15}
                ]
            }
        }
    )
    
    # Execute backtest
    result = await coordinator.execute_workflow(config)
    
    # Check results
    if result.success:
        print(f"Backtest completed successfully!")
        print(f"Metrics: {result.final_results.get('metrics', {})}")
    else:
        print(f"Backtest failed: {result.errors}")
    
    # Cleanup
    await coordinator.shutdown()


# Example 2: Optimization Workflow
async def run_optimization_workflow():
    """Run a multi-stage optimization workflow."""
    
    # Load configuration from YAML
    config_path = Path("configs/optimization_workflow.yaml")
    
    coordinator = Coordinator(
        config_path=str(config_path),
        shared_services={
            'market_data_provider': create_market_data_provider()
        }
    )
    
    # The configuration would look like this in YAML:
    yaml_config = """
workflow:
  type: "optimization"
  
  # Shared infrastructure
  parameters:
    shared_indicators:
      indicators:
        - type: "SMA"
          params: {periods: [5, 10, 20, 50]}
        - type: "RSI"
          params: {periods: [14, 21]}
  
  data_config:
    sources:
      - type: "csv"
        path: "data/historical/EURUSD_1min.csv"
    symbols: ["EURUSD"]
    timeframe: "1min"
  
  # Optimization configuration
  optimization_config:
    # Phase 1: Parameter optimization
    algorithm: "genetic"
    objective: "maximize_sharpe"
    
    parameter_space:
      fast_period: [5, 10, 15, 20]
      slow_period: [20, 30, 40, 50]
      signal_threshold: [0.01, 0.02, 0.03]
    
    # Genetic algorithm settings
    population_size: 50
    generations: 20
    mutation_rate: 0.1
    crossover_rate: 0.7
    
    # Validation
    validation:
      walk_forward: true
      training_ratio: 0.7
      min_trades: 30
"""
    
    # Parse YAML config
    workflow_config = yaml.safe_load(yaml_config)
    
    # Execute optimization
    result = await coordinator.execute_workflow(workflow_config['workflow'])
    
    if result.success:
        print(f"Optimization completed!")
        print(f"Best parameters: {result.final_results}")
        
        # Save optimized parameters
        save_optimization_results(result)
    
    await coordinator.shutdown()


# Example 3: Live Trading with Optimized Parameters
async def run_live_trading():
    """Run live trading with parameters from optimization."""
    
    # Create coordinator
    coordinator = Coordinator(
        shared_services={
            'broker_connection': create_broker_connection(),
            'market_data_provider': create_live_data_provider()
        }
    )
    
    # Load optimized parameters
    optimized_params = load_optimized_parameters("TrendFollowingStrategy")
    
    # Configure live trading
    config = WorkflowConfig(
        workflow_type=WorkflowType.LIVE_TRADING,
        
        data_config={
            'sources': {
                'live_feed': {
                    'type': 'websocket',
                    'url': 'wss://market-data.example.com'
                }
            },
            'symbols': ['EURUSD'],
            'timeframe': 'tick'
        },
        
        live_config={
            'broker': {
                'name': 'example_broker',
                'account_id': 'DEMO123',
                'api_key': '${BROKER_API_KEY}'
            },
            
            'strategy': {
                'class': 'TrendFollowingStrategy',
                'parameters': optimized_params
            },
            
            'risk_limits': {
                'max_position_size': 10000,
                'max_daily_loss': 1000,
                'max_open_positions': 5
            },
            
            'monitoring': {
                'heartbeat_interval': 60,
                'alert_channels': ['email', 'slack']
            }
        }
    )
    
    # Start live trading
    result = await coordinator.execute_workflow(config)
    
    if result.success:
        print("Live trading session started")
        
        # Monitor active workflows
        while True:
            active = await coordinator.list_active_workflows()
            if not active:
                break
                
            status = await coordinator.get_workflow_status(active[0]['workflow_id'])
            print(f"Trading status: {status}")
            
            await asyncio.sleep(10)
    
    await coordinator.shutdown()


# Example 4: Complex Multi-Stage Workflow
async def run_complex_workflow():
    """
    Run a complex workflow that combines multiple stages:
    1. Regime detection optimization
    2. Strategy parameter optimization per regime
    3. Weight optimization for ensemble
    4. Validation on out-of-sample data
    """
    
    config = {
        'workflow': {
            'type': 'optimization',
            
            'data_config': {
                'sources': {
                    'train': {
                        'type': 'csv',
                        'path': 'data/train/EURUSD_2020_2022.csv'
                    },
                    'test': {
                        'type': 'csv', 
                        'path': 'data/test/EURUSD_2023.csv'
                    }
                },
                'symbols': ['EURUSD']
            },
            
            'optimization_config': {
                'workflow_type': 'sequential',
                
                'stages': [
                    {
                        'name': 'regime_detector_optimization',
                        'component': 'RegimeDetector',
                        'algorithm': 'grid',
                        'objective': 'classification_accuracy',
                        'parameter_space': {
                            'volatility_window': [15, 20, 25],
                            'trend_threshold': [0.01, 0.015, 0.02]
                        }
                    },
                    {
                        'name': 'strategy_optimization_by_regime',
                        'component': 'TrendFollowingStrategy',
                        'algorithm': 'bayesian',
                        'objective': 'risk_adjusted_return',
                        'regime_specific': True,
                        'parameter_space': {
                            'fast_period': [5, 10, 15, 20],
                            'slow_period': [20, 30, 40, 50]
                        }
                    },
                    {
                        'name': 'ensemble_weight_optimization',
                        'component': 'EnsembleStrategy',
                        'algorithm': 'genetic',
                        'objective': 'minimize_drawdown',
                        'use_previous_results': True
                    }
                ],
                
                'validation': {
                    'type': 'walk_forward',
                    'windows': 12,
                    'test_ratio': 0.2
                }
            }
        }
    }
    
    coordinator = Coordinator(
        shared_services=create_shared_services()
    )
    
    # Execute complex workflow
    result = await coordinator.execute_workflow(config['workflow'])
    
    # Process results
    if result.success:
        print("Complex optimization completed!")
        
        # Extract results from each stage
        for phase, phase_result in result.phase_results.items():
            print(f"\nPhase {phase}:")
            print(f"  Success: {phase_result.success}")
            print(f"  Data: {phase_result.data}")
        
        # Final optimized configuration
        print(f"\nFinal configuration: {result.final_results}")
    
    await coordinator.shutdown()


# Helper functions
def create_market_data_provider():
    """Create a market data provider service."""
    return {'type': 'historical', 'ready': True}


def create_indicator_library():
    """Create indicator library service."""
    return {'indicators': ['SMA', 'EMA', 'RSI', 'MACD']}


def create_broker_connection():
    """Create broker connection service."""
    return {'connected': False, 'demo': True}


def create_live_data_provider():
    """Create live data provider service."""
    return {'type': 'live', 'connected': False}


def create_shared_services():
    """Create all shared services."""
    return {
        'market_data_provider': create_market_data_provider(),
        'indicator_library': create_indicator_library(),
        'regime_detector': {'type': 'volatility', 'ready': True}
    }


def save_optimization_results(result):
    """Save optimization results to file."""
    print(f"Saving optimization results: {result.workflow_id}")


def load_optimized_parameters(strategy_name):
    """Load previously optimized parameters."""
    # Example parameters
    return {
        'fast_period': 10,
        'slow_period': 30,
        'signal_threshold': 0.02
    }


# Main execution
if __name__ == "__main__":
    print("=== Running Simple Backtest ===")
    asyncio.run(run_simple_backtest())
    
    print("\n=== Running Optimization Workflow ===")
    asyncio.run(run_optimization_workflow())
    
    print("\n=== Running Complex Multi-Stage Workflow ===")
    asyncio.run(run_complex_workflow())