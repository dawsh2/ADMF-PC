#!/usr/bin/env python3
"""
Test the BacktestContainerFactory directly to verify container hierarchy.
"""
import asyncio
import logging
from src.execution.backtest_container_factory import (
    BacktestContainerFactory, 
    BacktestContainerConfig,
    ClassifierConfig,
    RiskProfileConfig
)
from src.strategy.components import IndicatorConfig, IndicatorType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_container_factory():
    """Test that factory creates proper container hierarchy."""
    
    print("=== Testing BacktestContainerFactory ===\n")
    
    # Create configuration
    config = BacktestContainerConfig(
        container_id="test_backtest_001",
        data_config={
            'symbols': ['AAPL', 'GOOGL'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        indicator_configs=[
            IndicatorConfig(
                name='SMA_20',
                indicator_type=IndicatorType.TREND,
                parameters={'period': 20}
            ),
            IndicatorConfig(
                name='RSI_14',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'period': 14}
            )
        ],
        classifiers=[
            ClassifierConfig(
                type='hmm',
                parameters={'n_states': 3},
                risk_profiles=[
                    RiskProfileConfig(
                        name='conservative',
                        capital_allocation=300000,
                        risk_parameters={
                            'max_position_size': 0.02,
                            'max_total_exposure': 0.1
                        },
                        strategies=[
                            {
                                'name': 'momentum',
                                'class': 'MomentumStrategy',
                                'parameters': {
                                    'fast_period': 10,
                                    'slow_period': 30
                                }
                            }
                        ]
                    ),
                    RiskProfileConfig(
                        name='aggressive',
                        capital_allocation=700000,
                        risk_parameters={
                            'max_position_size': 0.05,
                            'max_total_exposure': 0.3
                        },
                        strategies=[
                            {
                                'name': 'breakout',
                                'class': 'BreakoutStrategy',
                                'parameters': {
                                    'breakout_period': 20
                                }
                            }
                        ]
                    )
                ]
            ),
            ClassifierConfig(
                type='pattern',
                parameters={'lookback_period': 50},
                risk_profiles=[
                    RiskProfileConfig(
                        name='balanced',
                        capital_allocation=500000,
                        risk_parameters={
                            'max_position_size': 0.03,
                            'max_total_exposure': 0.2
                        },
                        strategies=[
                            {
                                'name': 'mean_reversion',
                                'class': 'MeanReversionStrategy',
                                'parameters': {
                                    'entry_zscore': 2.0,
                                    'exit_zscore': 0.5
                                }
                            }
                        ]
                    )
                ]
            )
        ],
        execution_config={
            'slippage_model': 'fixed',
            'commission_model': 'percentage'
        }
    )
    
    # Create container using factory
    print("Creating container with factory...")
    container = BacktestContainerFactory.create_instance(config)
    
    print(f"\n=== Container Created ===")
    print(f"Container ID: {container.container_id}")
    print(f"Container Type: {container.container_type}")
    print(f"State: {container.state}")
    
    # Check components
    print(f"\n=== Main Components ===")
    for name, component in container._components.items():
        print(f"- {name}: {type(component).__name__}")
    
    # Check sub-containers (classifiers)
    print(f"\n=== Classifier Containers ===")
    for name, sub_container in container._subcontainers.items():
        print(f"\n{name}:")
        print(f"  Container ID: {sub_container.container_id}")
        
        # Check classifier components
        if hasattr(sub_container, '_components'):
            print(f"  Components:")
            for comp_name, comp in sub_container._components.items():
                print(f"    - {comp_name}: {type(comp).__name__}")
        
        # Check risk & portfolio containers
        if hasattr(sub_container, 'risk_portfolio_containers'):
            print(f"  Risk & Portfolio Containers:")
            for risk_name, risk_container in sub_container.risk_portfolio_containers.items():
                print(f"    {risk_name}:")
                
                # Show components in risk container
                if hasattr(risk_container, '_components'):
                    for comp_name, comp in risk_container._components.items():
                        print(f"      - {comp_name}: {type(comp).__name__}")
    
    # Initialize and start
    print("\n=== Initializing Container Hierarchy ===")
    await container.initialize()
    
    print("\n=== Starting Container Hierarchy ===")
    await container.start()
    
    print(f"\n=== Container State After Start ===")
    print(f"State: {container.state}")
    
    # Check event bus
    print(f"\n=== Event Bus ===")
    print(f"Subscriptions: {container.event_bus.get_subscription_count()}")
    
    # Stop and cleanup
    print("\n=== Stopping Container ===")
    await container.stop()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_container_factory())