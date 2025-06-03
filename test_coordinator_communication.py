"""
Test script demonstrating the integration of event communication with WorkflowCoordinator.

This example shows how to:
1. Create a coordinator with communication enabled
2. Setup communication adapters
3. Execute workflows with communication tracking
4. Get system status including communication metrics
5. Properly shutdown with cleanup
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_coordinator_with_communication():
    """Test the coordinator with communication system integration."""
    
    # Import coordinator
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    
    # Create coordinator with communication enabled
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_communication=True
    )
    
    logger.info(f"Created coordinator: {coordinator.coordinator_id}")
    
    try:
        # Setup communication with a pipeline adapter
        communication_config = {
            'adapters': [
                {
                    'name': 'test_pipeline',
                    'type': 'pipeline',
                    'containers': [],  # Will be auto-populated
                    'log_level': 'INFO',
                    'buffer_size': 1000,
                    'timeout_ms': 5000
                }
            ]
        }
        
        # Setup communication layer
        success = await coordinator.setup_communication(communication_config)
        if success:
            logger.info("✅ Communication layer setup successful")
        else:
            logger.warning("❌ Communication layer setup failed")
        
        # Get initial system status
        status = await coordinator.get_system_status()
        logger.info("Initial system status:")
        logger.info(f"  - Coordinator ID: {status['coordinator_id']}")
        logger.info(f"  - Communication enabled: {status['communication_enabled']}")
        logger.info(f"  - Communication status: {status['communication']}")
        
        # Create a simple workflow config
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            data_config={
                'type': 'csv',
                'path': 'data/SPY.csv'
            },
            backtest_config={
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'initial_capital': 100000
            },
            parameters={
                'reporting': {
                    'enabled': False  # Disable for this test
                }
            }
        )
        
        # Execute workflow
        logger.info("\n🚀 Executing test workflow...")
        result = await coordinator.execute_workflow(workflow_config)
        
        logger.info(f"Workflow result: success={result.success}")
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        if result.warnings:
            logger.warning(f"Warnings: {result.warnings}")
        
        # Get system status after workflow
        status = await coordinator.get_system_status()
        logger.info("\nSystem status after workflow:")
        logger.info(f"  - Active workflows: {status['active_workflows']}")
        
        # Check communication metrics
        if 'communication' in status and isinstance(status['communication'], dict):
            comm = status['communication']
            if 'error' not in comm:
                logger.info("\n📊 Communication Metrics:")
                logger.info(f"  - Total adapters: {comm.get('total_adapters', 0)}")
                logger.info(f"  - Active adapters: {comm.get('active_adapters', 0)}")
                logger.info(f"  - Events sent: {comm.get('total_events_sent', 0)}")
                logger.info(f"  - Events received: {comm.get('total_events_received', 0)}")
                logger.info(f"  - Error rate: {comm.get('overall_error_rate', 0):.2%}")
                logger.info(f"  - Events/sec: {comm.get('events_per_second', 0):.2f}")
                logger.info(f"  - Health: {comm.get('overall_health', 'unknown')}")
                
                # Show adapter status
                adapter_status = comm.get('adapter_status', {})
                if adapter_status:
                    logger.info("\n📡 Adapter Status:")
                    for name, status in adapter_status.items():
                        logger.info(f"  - {name}: {status}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    finally:
        # Always cleanup
        logger.info("\n🧹 Shutting down coordinator...")
        await coordinator.shutdown()
        logger.info("✅ Coordinator shutdown complete")


async def test_communication_configuration():
    """Test different communication configurations."""
    
    from src.core.coordinator.coordinator import Coordinator
    
    # Test 1: Default communication (auto-configuration)
    logger.info("\n=== Test 1: Default Communication ===")
    coordinator1 = Coordinator(enable_communication=True)
    
    try:
        # Setup with default config (None)
        success = await coordinator1.setup_communication()
        logger.info(f"Default setup success: {success}")
        
        status = await coordinator1.get_system_status()
        logger.info(f"Default adapters: {status['communication'].get('total_adapters', 0)}")
        
    finally:
        await coordinator1.shutdown()
    
    # Test 2: Disabled communication
    logger.info("\n=== Test 2: Disabled Communication ===")
    coordinator2 = Coordinator(enable_communication=False)
    
    try:
        # Try to setup (should return False)
        success = await coordinator2.setup_communication()
        logger.info(f"Disabled setup success: {success}")
        
        status = await coordinator2.get_system_status()
        logger.info(f"Communication status: {status['communication']}")
        
    finally:
        await coordinator2.shutdown()
    
    # Test 3: Multiple adapters
    logger.info("\n=== Test 3: Multiple Adapters ===")
    coordinator3 = Coordinator(enable_communication=True)
    
    try:
        multi_config = {
            'adapters': [
                {
                    'name': 'data_pipeline',
                    'type': 'pipeline',
                    'containers': [],
                    'buffer_size': 5000
                },
                {
                    'name': 'signal_pipeline',
                    'type': 'pipeline',
                    'containers': [],
                    'buffer_size': 1000,
                    'enable_compression': True
                }
            ]
        }
        
        success = await coordinator3.setup_communication(multi_config)
        logger.info(f"Multi-adapter setup success: {success}")
        
        status = await coordinator3.get_system_status()
        comm = status.get('communication', {})
        logger.info(f"Total adapters: {comm.get('total_adapters', 0)}")
        logger.info(f"Adapter status: {comm.get('adapter_status', {})}")
        
    finally:
        await coordinator3.shutdown()


async def main():
    """Run all tests."""
    logger.info("Starting Coordinator Communication Integration Tests")
    logger.info("=" * 60)
    
    # Test basic integration
    await test_coordinator_with_communication()
    
    # Test different configurations
    await test_communication_configuration()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())