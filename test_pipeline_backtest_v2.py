#!/usr/bin/env python3
"""
Test script for multi-strategy backtest with pipeline communication adapters.
This version properly integrates with the coordinator's workflow management.
"""

import asyncio
import yaml
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Monkey patch to use pipeline containers before imports
import src.execution.containers_pipeline as containers_pipeline
import src.execution.containers as original_containers

# Replace the original containers module with pipeline version
sys.modules['src.execution.containers'] = containers_pipeline

from src.core.coordinator.coordinator import Coordinator, ExecutionMode

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
            
            # Configure pipeline with actual container order
            if hasattr(coordinator, 'communication_layer') and coordinator.communication_layer:
                logger.info("Configuring pipeline order for multi-strategy backtest")
                # Pipeline order will be set up by the workflow manager
        else:
            logger.error("❌ Failed to initialize communication system")
            return
    
    # Execute workflow - let coordinator handle the execution mode
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
            
            # Get detailed adapter metrics
            if 'adapters' in comm_status:
                adapter_metrics = list(comm_status['adapters'].values())[0] if comm_status['adapters'] else {}
                events_processed = adapter_metrics.get('events_processed', 0)
                logger.info(f"   - Total events processed: {events_processed}")
                
                if events_processed > 0:
                    logger.info("   ✅ Pipeline adapter successfully processed events!")
                else:
                    logger.warning("   ⚠️  No events processed through pipeline adapter")
            
            logger.info(f"   - Overall health: {comm_status.get('overall_health', 'unknown')}")
            
            # Check for errors
            total_errors = sum(a.get('errors', 0) for a in comm_status.get('adapters', {}).values())
            if total_errors > 0:
                logger.warning(f"   - Total errors: {total_errors}")
            else:
                logger.info("   - ✅ No errors in communication!")
            
            # Check adapter details
            if 'adapters' in comm_status:
                for adapter_name, adapter_metrics in comm_status['adapters'].items():
                    logger.info(f"   - {adapter_name}:")
                    logger.info(f"     • Events processed: {adapter_metrics.get('events_processed', 0)}")
                    logger.info(f"     • Events/sec: {adapter_metrics.get('events_per_second', 0):.2f}")
                    logger.info(f"     • Error rate: {adapter_metrics.get('error_rate', 0):.2%}")
                    logger.info(f"     • Avg latency: {adapter_metrics.get('average_latency_ms', 0):.2f}ms")
        
        # Report results
        if result:
            logger.info("📊 Result summary:")
            logger.info(f"   - Success: {result.success}")
            logger.info(f"   - Workflow ID: {result.workflow_id}")
            
            if result.data:
                data = result.data
                if 'portfolio' in data:
                    portfolio = data['portfolio']
                    logger.info(f"   - Portfolio:")
                    logger.info(f"     • Cash: ${portfolio.get('cash', 0):,.2f}")
                    logger.info(f"     • Positions: {portfolio.get('positions', 0)}")
                    logger.info(f"     • Total Value: ${portfolio.get('total_value', 0):,.2f}")
                
                if 'total_trades' in data:
                    logger.info(f"   - Total trades: {data['total_trades']}")
                    
                if 'performance' in data:
                    perf = data['performance']
                    logger.info(f"   - Performance:")
                    for key, value in perf.items():
                        logger.info(f"     • {key}: {value}")
            
            if result.errors:
                logger.error(f"   - Errors: {result.errors}")
        
    except Exception as e:
        logger.error(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await coordinator.shutdown()
        logger.info("Coordinator shutdown complete")


if __name__ == "__main__":
    print("Testing Multi-Strategy Backtest with Pipeline Communication")
    print("=" * 80)
    print("This version uses pipeline adapters to eliminate circular dependencies")
    print("=" * 80)
    asyncio.run(test_pipeline_backtest())