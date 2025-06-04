"""
Example demonstrating how the event communication system integrates with
the WorkflowCoordinator to enable cross-container communication during workflows.

This example shows:
1. Setting up a coordinator with communication
2. Configuring communication adapters for different container types
3. Running workflows that utilize the communication layer
4. Monitoring communication metrics during execution
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_workflow_with_communication():
    """Create and execute a workflow with cross-container communication."""
    
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.types import WorkflowConfig, WorkflowType
    
    # Create coordinator with all features enabled
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_communication=True
    )
    
    logger.info(f"🚀 Created coordinator: {coordinator.coordinator_id}")
    
    try:
        # Configure communication layer with multiple adapters
        communication_config = {
            'adapters': [
                {
                    'name': 'data_flow_pipeline',
                    'type': 'pipeline',
                    'containers': ['data_container', 'indicator_container', 'strategy_container'],
                    'buffer_size': 10000,
                    'timeout_ms': 1000,
                    'log_level': 'INFO'
                },
                {
                    'name': 'signal_pipeline',
                    'type': 'pipeline',
                    'containers': ['strategy_container', 'risk_container', 'execution_container'],
                    'buffer_size': 5000,
                    'timeout_ms': 500,
                    'enable_compression': True,
                    'log_level': 'INFO'
                }
            ]
        }
        
        # Setup communication
        logger.info("\n📡 Setting up communication layer...")
        success = await coordinator.setup_communication(communication_config)
        
        if not success:
            logger.error("Failed to setup communication layer")
            return
        
        # Create a comprehensive workflow configuration
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            data_config={
                'type': 'csv',
                'path': 'data/SPY.csv',
                'symbols': ['SPY'],
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            backtest_config={
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.0005,
                'position_sizing': {
                    'method': 'fixed_fraction',
                    'fraction': 0.02
                }
            },
            strategy_config={
                'strategies': [
                    {
                        'name': 'momentum_strategy',
                        'type': 'momentum',
                        'parameters': {
                            'fast_period': 10,
                            'slow_period': 30,
                            'rsi_period': 14,
                            'rsi_overbought': 70,
                            'rsi_oversold': 30
                        }
                    }
                ],
                'indicators': [
                    {'type': 'sma', 'period': 10},
                    {'type': 'sma', 'period': 30},
                    {'type': 'rsi', 'period': 14},
                    {'type': 'volume_profile'}
                ]
            },
            risk_config={
                'max_position_size': 0.1,
                'max_portfolio_heat': 0.06,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'max_correlation': 0.7
            },
            parameters={
                # Enable container patterns for composable execution
                'container_pattern': 'tiered_communication',
                
                # Configure event routing
                'event_routing': {
                    'enable_cross_container': True,
                    'event_tiers': {
                        'data_events': 'fast',
                        'signal_events': 'standard',
                        'order_events': 'reliable'
                    }
                },
                
                # Disable reporting for this example
                'reporting': {'enabled': False}
            }
        )
        
        # Monitor communication before workflow
        initial_status = await coordinator.get_system_status()
        logger.info("\n📊 Initial Communication Status:")
        log_communication_status(initial_status.get('communication', {}))
        
        # Execute the workflow
        logger.info("\n🏃 Executing workflow with communication tracking...")
        start_time = datetime.now()
        
        result = await coordinator.execute_workflow(
            config=workflow_config,
            workflow_id="comm_test_workflow"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log workflow results
        logger.info(f"\n✅ Workflow completed in {execution_time:.2f} seconds")
        logger.info(f"Success: {result.success}")
        logger.info(f"Workflow ID: {result.workflow_id}")
        
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        if result.warnings:
            logger.warning(f"Warnings: {result.warnings}")
        
        # Get final communication metrics
        final_status = await coordinator.get_system_status()
        logger.info("\n📊 Final Communication Metrics:")
        log_communication_status(final_status.get('communication', {}))
        
        # Calculate communication statistics
        comm_initial = initial_status.get('communication', {})
        comm_final = final_status.get('communication', {})
        
        if isinstance(comm_initial, dict) and isinstance(comm_final, dict):
            events_processed = (
                comm_final.get('total_events_sent', 0) - 
                comm_initial.get('total_events_sent', 0) +
                comm_final.get('total_events_received', 0) - 
                comm_initial.get('total_events_received', 0)
            )
            
            logger.info(f"\n📈 Communication Summary:")
            logger.info(f"  - Total events processed: {events_processed}")
            logger.info(f"  - Events per second: {events_processed / execution_time:.2f}")
            logger.info(f"  - Final error rate: {comm_final.get('overall_error_rate', 0):.2%}")
            logger.info(f"  - System health: {comm_final.get('overall_health', 'unknown')}")
            
            # Log latency percentiles if available
            if 'latency_percentiles' in comm_final:
                latencies = comm_final['latency_percentiles']
                logger.info(f"\n⏱️  Latency Statistics:")
                logger.info(f"  - P50: {latencies.get('p50', 0):.2f}ms")
                logger.info(f"  - P95: {latencies.get('p95', 0):.2f}ms")
                logger.info(f"  - P99: {latencies.get('p99', 0):.2f}ms")
                logger.info(f"  - Max: {latencies.get('max', 0):.2f}ms")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("\n🧹 Shutting down coordinator...")
        await coordinator.shutdown()
        logger.info("✅ Shutdown complete")


def log_communication_status(comm_status: Dict[str, Any]):
    """Helper to log communication status in a readable format."""
    
    if not isinstance(comm_status, dict):
        logger.info(f"  Communication: {comm_status}")
        return
    
    if 'error' in comm_status:
        logger.error(f"  Communication Error: {comm_status['error']}")
        return
    
    if comm_status.get('status') == 'disabled':
        logger.info("  Communication: Disabled")
        return
    
    # Log basic metrics
    logger.info(f"  - Total Adapters: {comm_status.get('total_adapters', 0)}")
    logger.info(f"  - Active Adapters: {comm_status.get('active_adapters', 0)}")
    logger.info(f"  - Connected: {comm_status.get('connected_adapters', 0)}")
    logger.info(f"  - Events Sent: {comm_status.get('total_events_sent', 0)}")
    logger.info(f"  - Events Received: {comm_status.get('total_events_received', 0)}")
    logger.info(f"  - Error Rate: {comm_status.get('overall_error_rate', 0):.2%}")
    logger.info(f"  - Health: {comm_status.get('overall_health', 'unknown')}")
    
    # Log adapter status
    adapter_status = comm_status.get('adapter_status', {})
    if adapter_status:
        logger.info("  - Adapter Status:")
        for name, status in adapter_status.items():
            logger.info(f"    • {name}: {status}")


async def test_communication_patterns():
    """Test different communication patterns and configurations."""
    
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.types import WorkflowConfig, WorkflowType
    
    patterns = [
        {
            'name': 'Pipeline Pattern',
            'config': {
                'adapters': [{
                    'name': 'sequential_pipeline',
                    'type': 'pipeline',
                    'containers': [],
                    'buffer_size': 1000
                }]
            }
        },
        # Future patterns can be added here as they're implemented:
        # {
        #     'name': 'Broadcast Pattern',
        #     'config': {
        #         'adapters': [{
        #             'name': 'broadcast_adapter',
        #             'type': 'broadcast',
        #             'containers': [],
        #             'fanout_strategy': 'parallel'
        #         }]
        #     }
        # },
        # {
        #     'name': 'Hierarchical Pattern',
        #     'config': {
        #         'adapters': [{
        #             'name': 'hierarchical_adapter',
        #             'type': 'hierarchical',
        #             'containers': [],
        #             'hierarchy_levels': 3
        #         }]
        #     }
        # }
    ]
    
    for pattern in patterns:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {pattern['name']}")
        logger.info(f"{'='*60}")
        
        coordinator = Coordinator(enable_communication=True)
        
        try:
            # Setup communication with pattern
            success = await coordinator.setup_communication(pattern['config'])
            logger.info(f"Setup success: {success}")
            
            if success:
                # Get status
                status = await coordinator.get_system_status()
                comm = status.get('communication', {})
                logger.info(f"Active adapters: {comm.get('adapter_status', {})}")
            
        finally:
            await coordinator.shutdown()


async def main():
    """Run the communication workflow examples."""
    
    logger.info("=" * 80)
    logger.info("Coordinator Communication Workflow Example")
    logger.info("=" * 80)
    
    # Run main workflow example
    await create_workflow_with_communication()
    
    # Test different patterns
    logger.info("\n\n" + "=" * 80)
    logger.info("Testing Communication Patterns")
    logger.info("=" * 80)
    await test_communication_patterns()
    
    logger.info("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())