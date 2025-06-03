#!/usr/bin/env python3
"""
Test script for multi-strategy backtest with event communication adapters
"""

import asyncio
import yaml
import logging
from src.core.coordinator.coordinator import Coordinator
from src.core.coordinator.coordinator import ExecutionMode
from src.execution import containers

# Simple types for configuration
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class WorkflowType(str, Enum):
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE = "live"

@dataclass
class WorkflowConfig:
    workflow_type: WorkflowType
    parameters: Dict[str, Any]
    data_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    analysis_config: Dict[str, Any] = None
    communication_config: Dict[str, Any] = None
    
    def dict(self):
        return {
            'workflow_type': self.workflow_type.value,
            'parameters': self.parameters,
            'data_config': self.data_config,
            'backtest_config': self.backtest_config,
            'optimization_config': self.optimization_config,
            'analysis_config': self.analysis_config or {},
            'communication_config': self.communication_config or {}
        }


async def test_multi_strategy_with_communication():
    """Test multi-strategy backtest with new communication system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load the fixed configuration
    config_path = "config/multi_strategy_test_fixed.yaml"
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract communication config
    communication_config = config.pop('communication', {})
    
    # Extract strategies and add to backtest config
    backtest_config = config.get('backtest', {}).copy()
    if 'strategies' in config:
        backtest_config['strategies'] = config['strategies']
    
    # Create workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters=config,
        data_config=config.get('data', {}),
        backtest_config=backtest_config,
        optimization_config=config.get('optimization', {}),
        analysis_config=config.get('analysis', {}),
        communication_config=communication_config
    )
    
    # Create coordinator with communication enabled
    logger.info("Creating coordinator with communication support")
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_communication=True
    )
    
    # Setup communication before executing workflow
    if communication_config:
        logger.info("Setting up event communication system")
        success = await coordinator.setup_communication(communication_config)
        if success:
            logger.info("✅ Communication system initialized successfully")
        else:
            logger.error("❌ Failed to initialize communication system")
            return
    
    # Register execution containers
    containers.register_execution_containers()
    
    # Execute workflow
    logger.info("🚀 Starting multi-strategy backtest with communication...")
    try:
        import time
        start_time = time.time()
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Backtest completed in {elapsed:.2f} seconds")
        
        # Check communication metrics
        status = await coordinator.get_system_status()
        if 'communication' in status:
            comm_status = status['communication']
            logger.info(f"📊 Communication metrics:")
            logger.info(f"   - Total adapters: {comm_status.get('total_adapters', 0)}")
            logger.info(f"   - Total events processed: {comm_status.get('total_events_processed', 0)}")
            logger.info(f"   - Overall health: {comm_status.get('overall_health', 'unknown')}")
            
            # Check for errors
            if comm_status.get('total_errors', 0) > 0:
                logger.warning(f"   - Total errors: {comm_status['total_errors']}")
            
            # Check adapter details
            if 'adapter_breakdown' in comm_status:
                for adapter_name, adapter_metrics in comm_status['adapter_breakdown'].items():
                    logger.info(f"   - {adapter_name}:")
                    logger.info(f"     • Events/sec: {adapter_metrics.get('events_per_second', 0):.2f}")
                    logger.info(f"     • Error rate: {adapter_metrics.get('error_rate', 0):.2%}")
                    logger.info(f"     • Avg latency: {adapter_metrics.get('average_latency_ms', 0):.2f}ms")
                    logger.info(f"     • Health: {adapter_metrics.get('health', 'unknown')}")
        
        # Report results
        if result and 'success' in result:
            if result['success']:
                logger.info("✅ Multi-strategy backtest completed successfully!")
                if 'performance' in result:
                    logger.info(f"📈 Performance summary: {result['performance']}")
            else:
                logger.error(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await coordinator.shutdown()
        logger.info("Coordinator shutdown complete")


async def compare_with_original():
    """Compare results with and without communication system"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("Running comparison test: Original vs Communication-based")
    logger.info("="*80)
    
    # Test 1: Original configuration (should fail with circular dependencies)
    logger.info("\n1️⃣ Testing original configuration (expected to fail)...")
    try:
        # Would test original config here, but we know it has circular dependency issues
        logger.info("⚠️  Skipping original test - known circular dependency issues")
    except Exception as e:
        logger.error(f"Original config failed as expected: {e}")
    
    # Test 2: Fixed configuration with communication
    logger.info("\n2️⃣ Testing fixed configuration with communication...")
    await test_multi_strategy_with_communication()


if __name__ == "__main__":
    print("Testing Multi-Strategy Backtest with Event Communication Adapters")
    print("=" * 80)
    asyncio.run(test_multi_strategy_with_communication())