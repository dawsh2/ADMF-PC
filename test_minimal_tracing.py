#!/usr/bin/env python3
"""
Minimal test to verify event tracing in single-phase workflows.
"""

import asyncio
import logging

from src.core.coordinator.topology import WorkflowManager
from src.core.types.workflow import WorkflowConfig, WorkflowType, ExecutionContext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Enable debug for tracing
logging.getLogger('src.core.events.tracing').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_minimal_tracing():
    """Test minimal workflow with tracing."""
    
    logger.info("Starting minimal tracing test")
    
    # Create minimal config with tracing enabled
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        parameters={
            'mode': 'backtest',
            'symbols': ['SPY'],
            'start_date': '2024-01-01',
            'end_date': '2024-01-02',  # Just 2 days
            'tracing': {
                'enabled': True,
                'max_events': 1000
            },
            'backtest': {
                'data': {
                    'source': 'csv',
                    'file_path': './data/SPY.csv'
                },
                'features': {
                    'indicators': [
                        {'name': 'sma_20', 'type': 'sma', 'period': 20}
                    ]
                },
                'strategies': [
                    {'type': 'momentum', 'name': 'test_momentum'}
                ],
                'risk_profiles': [
                    {'type': 'conservative'}
                ],
                'portfolio': {
                    'initial_capital': 100000
                }
            }
        }
    )
    
    # Create execution context
    context = ExecutionContext(
        workflow_id='test_trace_001',
        workflow_type=WorkflowType.BACKTEST,
        metadata={'test': 'minimal_tracing'}
    )
    
    # Create workflow manager and execute
    manager = WorkflowManager(container_id='test_manager')
    
    try:
        # Execute workflow
        result = await manager.execute(config, context)
        
        logger.info(f"Workflow completed: success={result.success}")
        
        # Check if manager has event_tracer
        logger.info(f"Manager attributes: {[attr for attr in dir(manager) if 'trace' in attr]}")
        if hasattr(manager, 'event_tracer'):
            logger.info(f"Manager has event_tracer: {manager.event_tracer}")
            logger.info(f"Event tracer type: {type(manager.event_tracer)}")
            
            # Try to get summary directly from tracer
            if manager.event_tracer:
                direct_summary = manager.event_tracer.get_summary()
                logger.info(f"Direct summary from tracer: {direct_summary}")
            
            summary = manager.get_trace_summary()
            if summary:
                logger.info(f"🔍 Trace Summary:")
                logger.info(f"  Total events: {summary.get('total_events', 0)}")
                logger.info(f"  Event types: {summary.get('event_types', {})}")
                logger.info(f"  Source containers: {list(summary.get('source_containers', {}).keys())}")
            else:
                logger.warning("No trace summary available")
        else:
            logger.warning("Manager does not have event_tracer attribute")
            
        # Check the result for trace summary
        if 'trace_summary' in result.final_results:
            logger.info("✅ Trace summary found in final_results")
        else:
            logger.warning("⚠️ No trace summary in final_results")
            
        if result.metadata and 'trace_summary' in result.metadata:
            logger.info("✅ Trace summary found in metadata")
        else:
            logger.warning("⚠️ No trace summary in metadata")
            
    finally:
        await manager.cleanup()
        
    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(test_minimal_tracing())