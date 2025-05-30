#!/usr/bin/env python3
"""
Test that backtests create the proper container hierarchy from BACKTEST.MD.
"""
import asyncio
import logging
from src.core.coordinator import get_coordinator
from src.core.containers.bootstrap import ContainerBootstrap
from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_backtest_hierarchy():
    """Test that backtest creates proper container hierarchy."""
    
    # Create bootstrap
    bootstrap = ContainerBootstrap()
    
    # Create workflow config
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config={
            'symbols': ['AAPL', 'GOOGL'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        backtest_config={
            'initial_capital': 1000000,
            'execution': {
                'slippage_model': 'fixed',
                'commission_model': 'percentage'
            }
        },
        parameters={
            'indicators': [
                {'name': 'SMA_20', 'type': 'SMA', 'parameters': {'period': 20}},
                {'name': 'RSI_14', 'type': 'RSI', 'parameters': {'period': 14}}
            ],
            'classifiers': [
                {
                    'type': 'hmm',
                    'parameters': {'n_states': 3},
                    'risk_profiles': [
                        {
                            'name': 'conservative',
                            'capital_allocation': 300000,
                            'risk_parameters': {
                                'max_position_size': 0.02,
                                'max_total_exposure': 0.1
                            },
                            'strategies': [
                                {
                                    'name': 'momentum',
                                    'class': 'MomentumStrategy',
                                    'parameters': {
                                        'fast_period': 10,
                                        'slow_period': 30
                                    }
                                }
                            ]
                        },
                        {
                            'name': 'aggressive',
                            'capital_allocation': 700000,
                            'risk_parameters': {
                                'max_position_size': 0.05,
                                'max_total_exposure': 0.3
                            },
                            'strategies': [
                                {
                                    'name': 'breakout',
                                    'class': 'BreakoutStrategy',
                                    'parameters': {
                                        'breakout_period': 20
                                    }
                                }
                            ]
                        }
                    ]
                },
                {
                    'type': 'pattern',
                    'parameters': {'lookback_period': 50},
                    'risk_profiles': [
                        {
                            'name': 'balanced',
                            'capital_allocation': 500000,
                            'risk_parameters': {
                                'max_position_size': 0.03,
                                'max_total_exposure': 0.2
                            },
                            'strategies': [
                                {
                                    'name': 'mean_reversion',
                                    'class': 'MeanReversionStrategy',
                                    'parameters': {
                                        'entry_zscore': 2.0,
                                        'exit_zscore': 0.5
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    
    # Create coordinator
    Coordinator = get_coordinator()
    coordinator = bootstrap.create_coordinator()
    
    # Execute workflow
    result = await coordinator.execute_workflow(config)
    
    print("\n=== Backtest Workflow Result ===")
    print(f"Success: {result.success}")
    print(f"Workflow ID: {result.workflow_id}")
    
    if result.errors:
        print(f"Errors: {result.errors}")
    
    if result.results:
        # Check container structure
        container_structure = result.results.get('container_structure', {})
        print("\n=== Container Structure ===")
        print(f"Main Container: {container_structure.get('main_container')}")
        print(f"Components: {container_structure.get('components')}")
        
        print("\n=== Classifier Containers ===")
        classifiers = container_structure.get('classifiers', {})
        for name, info in classifiers.items():
            print(f"\n{name}:")
            print(f"  Type: {info['type']}")
            print(f"  Risk Profiles: {info['risk_profiles']}")
    
    # Test that we can get the container
    container_id = result.metadata.get('container_id')
    if container_id:
        container_manager = coordinator.container_manager
        
        # Try to get the container
        if container_id in container_manager.active_containers:
            container = container_manager.active_containers[container_id]
            
            print(f"\n=== Container Details ===")
            print(f"Container ID: {container.container_id}")
            print(f"Container Type: {container.container_type}")
            print(f"State: {container.state}")
            
            # Check components
            print(f"\nComponents: {list(container._components.keys())}")
            
            # Check sub-containers
            print(f"\nSub-containers: {list(container._subcontainers.keys())}")
            
            # Check event bus
            print(f"\nEvent Bus Subscriptions: {container.event_bus.get_subscription_count()}")

if __name__ == "__main__":
    asyncio.run(test_backtest_hierarchy())