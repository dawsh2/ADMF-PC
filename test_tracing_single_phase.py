#!/usr/bin/env python3
"""
Test to verify tracing works for single-phase workflows.
"""

import asyncio
import logging
from pathlib import Path

from src.core.coordinator.coordinator import Coordinator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_tracing_single_phase():
    """Test tracing in single-phase workflow."""
    
    logger.info("🚀 Starting single-phase workflow with tracing enabled")
    
    # Initialize coordinator
    coordinator = Coordinator(
        enable_composable_containers=True,
        enable_communication=True,
        enable_yaml=True
    )
    
    try:
        # Execute workflow from YAML config with tracing enabled
        yaml_path = Path("config/test_tracing_backtest.yaml")
        result = await coordinator.execute_yaml_workflow(str(yaml_path))
        
        # Check if execution was successful
        if result.success:
            logger.info("✅ Workflow completed successfully")
            
            # Check if trace summary is in the results
            if 'trace_summary' in result.data:
                trace_summary = result.data['trace_summary']
                logger.info(f"🔍 Event Trace Summary:")
                logger.info(f"  - Total events: {trace_summary.get('total_events', 0)}")
                logger.info(f"  - Event types: {trace_summary.get('event_types', {})}")
                logger.info(f"  - Source containers: {list(trace_summary.get('source_containers', {}).keys())}")
                logger.info(f"  - Event chains: {trace_summary.get('event_chains', 0)}")
                logger.info(f"  - Max chain depth: {trace_summary.get('max_chain_depth', 0)}")
            else:
                logger.warning("⚠️ No trace summary found in results")
            
            # Print basic results
            if 'best_combination' in result.data:
                best = result.data['best_combination']
                if best:
                    logger.info(f"📊 Best combination: {best.get('combo_id')}")
                    logger.info(f"  - Final value: ${best.get('final_value', 0):,.2f}")
                    logger.info(f"  - Total return: {best.get('total_return', 0):.2%}")
                    logger.info(f"  - Sharpe ratio: {best.get('sharpe_ratio', 0):.2f}")
        else:
            logger.error(f"❌ Workflow failed: {result.errors}")
            
            # Check if trace summary is in metadata (added on failure)
            if result.metadata and 'trace_summary' in result.metadata:
                trace_summary = result.metadata['trace_summary']
                logger.info(f"🔍 Event Trace Summary (from metadata):")
                logger.info(f"  - Total events before failure: {trace_summary.get('total_events', 0)}")
                logger.info(f"  - Event types: {trace_summary.get('event_types', {})}")
    
    finally:
        await coordinator.shutdown()
        logger.info("🏁 Test completed")


if __name__ == "__main__":
    asyncio.run(test_tracing_single_phase())