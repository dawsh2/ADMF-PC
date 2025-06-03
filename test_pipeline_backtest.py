#!/usr/bin/env python3
"""
Test script for multi-strategy backtest with pipeline communication adapters.
This version uses the new containers that work with pipeline adapters.
"""

import asyncio
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.coordinator import Coordinator, ExecutionMode
from src.core.coordinator.composable_workflow_manager_pipeline import ComposableWorkflowManagerPipeline
from src.execution import containers_pipeline

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


async def test_pipeline_backtest():
    """Test multi-strategy backtest with pipeline communication system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Register pipeline-enabled containers
    logger.info("Registering pipeline-enabled containers")
    containers_pipeline.register_execution_containers()
    
    # Load configuration
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
    
    # Setup communication
    if communication_config:
        logger.info("Setting up pipeline communication system")
        success = await coordinator.setup_communication(communication_config)
        if success:
            logger.info("✅ Pipeline communication system initialized")
        else:
            logger.error("❌ Failed to initialize communication system")
            return
    
    # Use pipeline workflow manager
    coordinator._workflow_managers[ExecutionMode.COMPOSABLE] = ComposableWorkflowManagerPipeline(coordinator)
    
    # Execute workflow
    logger.info("🚀 Starting multi-strategy backtest with pipeline communication...")
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
            else:
                logger.info("   - No errors in communication!")
            
            # Check adapter details
            if 'adapter_breakdown' in comm_status:
                for adapter_name, adapter_metrics in comm_status['adapter_breakdown'].items():
                    logger.info(f"   - {adapter_name}:")
                    logger.info(f"     • Events/sec: {adapter_metrics.get('events_per_second', 0):.2f}")
                    logger.info(f"     • Error rate: {adapter_metrics.get('error_rate', 0):.2%}")
                    logger.info(f"     • Avg latency: {adapter_metrics.get('average_latency_ms', 0):.2f}ms")
                    logger.info(f"     • Health: {adapter_metrics.get('health', 'unknown')}")
        
        # Report results
        if result:
            logger.info("✅ Multi-strategy backtest completed successfully!")
            logger.info(f"📊 Result summary:")
            logger.info(f"   - Success: {result.get('success', False)}")
            logger.info(f"   - Container ID: {result.get('container_id', 'N/A')}")
            
            if 'portfolio' in result:
                portfolio = result['portfolio']
                logger.info(f"   - Portfolio:")
                logger.info(f"     • Cash: ${portfolio.get('cash', 0):,.2f}")
                logger.info(f"     • Positions: {portfolio.get('positions', 0)}")
                logger.info(f"     • Total Value: ${portfolio.get('total_value', 0):,.2f}")
            
            if 'total_trades' in result:
                logger.info(f"   - Total trades: {result['total_trades']}")
                
            if 'performance' in result:
                perf = result['performance']
                logger.info(f"   - Performance:")
                for key, value in perf.items():
                    logger.info(f"     • {key}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await coordinator.shutdown()
        logger.info("Coordinator shutdown complete")


async def compare_communication_methods():
    """Compare old hybrid interface vs new pipeline communication"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("Comparing communication methods: Hybrid vs Pipeline")
    logger.info("="*80)
    
    # Test with pipeline communication
    logger.info("\n🔧 Testing with Pipeline Communication (no circular dependencies expected)...")
    await test_pipeline_backtest()
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline communication test complete!")
    logger.info("Expected result: No circular dependency warnings, clean event flow")
    logger.info("="*80)


if __name__ == "__main__":
    print("Testing Multi-Strategy Backtest with Pipeline Communication")
    print("=" * 80)
    print("This version uses pipeline adapters to eliminate circular dependencies")
    print("=" * 80)
    asyncio.run(test_pipeline_backtest())