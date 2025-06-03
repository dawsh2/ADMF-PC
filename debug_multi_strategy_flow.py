"""Debug script to investigate multi-strategy event flow issue."""

import asyncio
import logging
import yaml
from typing import Dict, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patch various components to add debugging

# 1. Patch IndicatorContainer to log all events
from src.execution.containers_pipeline import IndicatorContainer
original_indicator_receive = IndicatorContainer.receive_event
original_indicator_handle_bar = IndicatorContainer._handle_bar_event

def debug_indicator_receive(self, event):
    logger.info(f"🔍 IndicatorContainer.receive_event: {event.event_type}, payload keys: {list(event.payload.keys())}")
    original_indicator_receive(self, event)

async def debug_indicator_handle_bar(self, event):
    logger.info(f"📊 IndicatorContainer._handle_bar_event: Processing BAR for {event.payload.get('symbol')}")
    result = await original_indicator_handle_bar(self, event)
    logger.info(f"✅ IndicatorContainer._handle_bar_event: Complete")
    return result

IndicatorContainer.receive_event = debug_indicator_receive
IndicatorContainer._handle_bar_event = debug_indicator_handle_bar

# 2. Patch StrategyContainer to log all events
from src.execution.containers_pipeline import StrategyContainer
original_strategy_receive = StrategyContainer.receive_event

def debug_strategy_receive(self, event):
    logger.info(f"🎯 StrategyContainer.receive_event: {event.event_type}, payload keys: {list(event.payload.keys())}")
    if event.event_type.value == "INDICATORS":
        indicators = event.payload.get('indicators', {})
        logger.info(f"   Indicators received: {list(indicators.keys())}")
    original_strategy_receive(self, event)

StrategyContainer.receive_event = debug_strategy_receive

# 2.5 Patch EventBus publish to show all published events
from src.core.events.event_bus import EventBus
original_eventbus_publish = EventBus.publish

def debug_eventbus_publish(self, event):
    logger.info(f"📤 EventBus.publish: {event.event_type} from container {getattr(self, '_container_name', 'Unknown')}")
    if event.event_type.value == "INDICATORS":
        logger.info(f"   INDICATORS event payload keys: {list(event.payload.keys())}")
    original_eventbus_publish(self, event)

EventBus.publish = debug_eventbus_publish

# 3. Patch pipeline adapter to trace connections
from src.core.communication.pipeline_adapter_protocol import PipelineAdapter
original_setup_pipeline = PipelineAdapter.setup_pipeline
original_start = PipelineAdapter.start

def debug_setup_pipeline(self, container_list):
    logger.info(f"🔗 PipelineAdapter.setup_pipeline called with {len(container_list)} containers:")
    for i, container in enumerate(container_list):
        logger.info(f"   {i}: {container.metadata.name} (role: {container.metadata.role.value})")
    original_setup_pipeline(self, container_list)

def debug_start(self):
    logger.info(f"🚀 PipelineAdapter.start called, setting up {len(self.connections)} connections:")
    for i, (source, target) in enumerate(self.connections):
        logger.info(f"   Connection {i}: {source.name} -> {target.name}")
    original_start(self)

PipelineAdapter.setup_pipeline = debug_setup_pipeline
PipelineAdapter.start = debug_start

# 4. Patch event bus to trace subscriptions
from src.core.events.event_bus import EventBus
original_subscribe = EventBus.subscribe
original_publish = EventBus.publish

def debug_subscribe(self, event_type, handler):
    logger.info(f"🔔 EventBus.subscribe: {event_type} -> {handler}")
    original_subscribe(self, event_type, handler)

def debug_publish(self, event):
    logger.info(f"📮 EventBus.publish: {event.event_type} (source: {event.source_id})")
    original_publish(self, event)

EventBus.subscribe = debug_subscribe
EventBus.publish = debug_publish

async def test_multi_strategy():
    """Test the multi-strategy configuration with debugging."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("STARTING MULTI-STRATEGY DEBUG TEST")
    logger.info("=" * 80)
    
    # Import main components
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    
    # Create coordinator
    coordinator = Coordinator(enable_composable_containers=True)
    
    # Build workflow config
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config=config['data'],
        backtest_config={
            **config.get('backtest', {}),
            'strategies': config.get('strategies', [])  # Add strategies to backtest_config
        },
        parameters={
            'strategies': config.get('strategies', []),
            'risk': config.get('risk', {}),
            'container_pattern': 'simple_backtest'
        }
    )
    
    # Add signal aggregation to parameters
    if 'signal_aggregation' in config:
        workflow_config.parameters['signal_aggregation'] = config['signal_aggregation']
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        logger.info("🚀 Starting workflow execution...")
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("=" * 80)
        logger.info("WORKFLOW EXECUTION COMPLETE")
        logger.info(f"Success: {result.success}")
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_strategy())