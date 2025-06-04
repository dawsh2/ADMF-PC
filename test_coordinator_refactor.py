#!/usr/bin/env python3
"""
Test script to verify the refactored Coordinator delegation pattern.
"""

import asyncio
import logging
from src.core.coordinator.coordinator import Coordinator
from src.core.types.workflow import WorkflowConfig, WorkflowType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_single_phase_workflow():
    """Test single-phase workflow delegation to WorkflowManager."""
    logger.info("=== Testing Single-Phase Workflow ===")
    
    # Create coordinator
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_communication=True
    )
    
    # Create simple backtest config
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'container_pattern': 'simple_backtest'
        },
        data_config={
            'source': 'csv',
            'file_path': 'data/SPY.csv',
            'symbols': ['SPY'],
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        },
        backtest_config={
            'initial_capital': 100000,
            'commission': 0.001,
            'strategies': [{
                'type': 'momentum',
                'parameters': {'period': 20}
            }]
        }
    )
    
    try:
        # Execute workflow - should delegate to WorkflowManager
        result = await coordinator.execute_workflow(config)
        
        logger.info(f"Workflow completed: success={result.success}")
        logger.info(f"Correlation ID: {result.metadata.get('correlation_id')}")
        
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        
        return result
        
    finally:
        await coordinator.shutdown()


async def test_multi_phase_workflow():
    """Test multi-phase workflow delegation to Sequencer."""
    logger.info("\n=== Testing Multi-Phase Workflow ===")
    
    # Create coordinator
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_phase_management=True
    )
    
    # Create multi-phase config
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        parameters={
            'phases': [
                {
                    'name': 'strategy_discovery',
                    'type': 'optimization',
                    'strategies': [
                        {'type': 'momentum', 'parameters': {'period': 20}},
                        {'type': 'mean_reversion', 'parameters': {'period': 14}}
                    ],
                    'optimization': {
                        'parameters': {
                            'momentum.period': [10, 20, 30],
                            'mean_reversion.period': [7, 14, 21]
                        },
                        'objective': 'sharpe_ratio'
                    }
                },
                {
                    'name': 'risk_optimization',
                    'type': 'backtest',
                    'inherit_best_from': 'strategy_discovery',
                    'risk': {
                        'position_sizers': [
                            {'type': 'fixed', 'size': 10000},
                            {'type': 'percentage', 'percentage': 5.0}
                        ]
                    }
                }
            ]
        },
        data_config={
            'source': 'csv',
            'file_path': 'data/SPY.csv',
            'symbols': ['SPY'],
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        }
    )
    
    try:
        # Execute workflow - should delegate to Sequencer
        result = await coordinator.execute_workflow(config)
        
        logger.info(f"Multi-phase workflow completed: success={result.success}")
        logger.info(f"Correlation ID: {result.metadata.get('correlation_id')}")
        logger.info(f"Phases completed: {result.data.get('phase_results', {}).keys() if result.data else 'None'}")
        
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        
        return result
        
    finally:
        await coordinator.shutdown()


async def test_analytics_integration():
    """Test analytics storage integration."""
    logger.info("\n=== Testing Analytics Integration ===")
    
    # Create coordinator
    coordinator = Coordinator(enable_composable_containers=True)
    
    # Simple config
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={'test': True},
        data_config={'source': 'synthetic'},
        backtest_config={'initial_capital': 100000}
    )
    
    try:
        # Execute workflow
        result = await coordinator.execute_workflow(config)
        
        logger.info(f"Analytics test completed: success={result.success}")
        
        # Check if analytics stored the correlation ID
        correlation_id = result.metadata.get('correlation_id')
        if correlation_id:
            logger.info(f"✅ Correlation ID generated: {correlation_id}")
        else:
            logger.error("❌ No correlation ID found")
        
        return result
        
    finally:
        await coordinator.shutdown()


async def main():
    """Run all tests."""
    logger.info("Starting Coordinator Refactoring Tests\n")
    
    # Test single-phase workflow
    single_result = await test_single_phase_workflow()
    
    # Test multi-phase workflow
    multi_result = await test_multi_phase_workflow()
    
    # Test analytics integration
    analytics_result = await test_analytics_integration()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Single-phase test: {'✅ PASSED' if single_result.success else '❌ FAILED'}")
    logger.info(f"Multi-phase test: {'✅ PASSED' if multi_result.success else '❌ FAILED'}")
    logger.info(f"Analytics test: {'✅ PASSED' if analytics_result.success else '❌ FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())