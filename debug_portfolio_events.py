"""Debug script to trace portfolio event handling."""

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

# Patch PortfolioContainer to log all events
from src.execution.containers_pipeline import PortfolioContainer
original_portfolio_receive = PortfolioContainer.receive_event
original_portfolio_handle_fill = PortfolioContainer._handle_fill_event if hasattr(PortfolioContainer, '_handle_fill_event') else None

def debug_portfolio_receive(self, event):
    logger.info(f"💼 PortfolioContainer.receive_event: {event.event_type}, payload keys: {list(event.payload.keys())}")
    if event.event_type.value == "FILL":
        fill = event.payload
        logger.info(f"   FILL details: {fill.get('symbol')} {fill.get('side')} {fill.get('quantity')} @ {fill.get('price')}")
    original_portfolio_receive(self, event)

PortfolioContainer.receive_event = debug_portfolio_receive

# Patch RiskContainer to log ORDER and FILL routing
from src.execution.containers_pipeline import RiskContainer
original_risk_receive = RiskContainer.receive_event

def debug_risk_receive(self, event):
    logger.info(f"⚠️ RiskContainer.receive_event: {event.event_type}, payload keys: {list(event.payload.keys())}")
    if event.event_type.value == "FILL":
        logger.info(f"   RiskContainer received FILL - forwarding to portfolio state update")
    original_risk_receive(self, event)

RiskContainer.receive_event = debug_risk_receive

# Patch ExecutionContainer to trace FILL generation
from src.execution.containers_pipeline import ExecutionContainer
original_exec_handle_order = ExecutionContainer._handle_order_event

async def debug_handle_order(self, event):
    logger.info(f"🎯 ExecutionContainer._handle_order_event called")
    result = await original_exec_handle_order(self, event)
    logger.info(f"🎯 ExecutionContainer._handle_order_event completed")
    return result

ExecutionContainer._handle_order_event = debug_handle_order

# Patch the forward handler creation to log event forwarding
from src.core.communication import helpers
original_create_forward = helpers.create_forward_handler

def debug_create_forward_handler(adapter, target):
    def forward_event(event):
        logger.info(f"📤 FORWARDING: {event.event_type} to {target.name}")
        if event.event_type.value == "FILL":
            logger.info(f"   FILL being forwarded from {getattr(event, 'source_id', 'unknown')} to {target.name}")
        target.receive_event(event)
    return forward_event

helpers.create_forward_handler = debug_create_forward_handler

# Also check if portfolio state is being updated
from src.risk.portfolio_state import PortfolioState
if hasattr(PortfolioState, 'process_fill'):
    original_process_fill = PortfolioState.process_fill
    
    def debug_process_fill(self, fill_event):
        logger.info(f"📊 PortfolioState.process_fill called with fill: {fill_event}")
        result = original_process_fill(self, fill_event)
        logger.info(f"📊 PortfolioState after fill: cash={self.cash_balance}, positions={len(self.positions)}")
        return result
    
    PortfolioState.process_fill = debug_process_fill

async def test_portfolio_events():
    """Test portfolio event handling."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("STARTING PORTFOLIO EVENT DEBUG TEST")
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
            'strategies': config.get('strategies', [])
        },
        parameters={
            'strategies': config.get('strategies', []),
            'risk': config.get('risk', {}),
            'container_pattern': 'simple_backtest'
        }
    )
    
    # Limit to fewer bars for debugging
    workflow_config.data_config['max_bars'] = 20
    
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
        logger.info(f"Portfolio result: {result.data.get('portfolio') if result.data else 'No data'}")
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_portfolio_events())