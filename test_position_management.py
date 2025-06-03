"""Test position management with detailed logging."""

import asyncio
import logging
import yaml
from decimal import Decimal

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track what's happening
position_tracker = {
    'fills': [],
    'positions': {},
    'portfolio_events': 0,
    'risk_checks': []
}

# Patch RiskContainer to log portfolio state
from src.execution.containers_pipeline import RiskContainer
original_get_portfolio = RiskContainer._get_portfolio_state

def debug_get_portfolio_state(self):
    """Debug portfolio state retrieval."""
    state = original_get_portfolio(self)
    
    # Log what we got
    if hasattr(state, 'get_all_positions'):
        positions = state.get_all_positions()
        cash = state.get_cash_balance()
        logger.info(f"🔍 RiskContainer._get_portfolio_state: Cash=${cash:.2f}, Positions={len(positions)}")
        for symbol, pos in positions.items():
            logger.info(f"   Position: {symbol} = {pos.quantity} shares")
    else:
        logger.warning("🔍 RiskContainer._get_portfolio_state: Got temporary/empty state")
    
    return state

RiskContainer._get_portfolio_state = debug_get_portfolio_state

# Patch portfolio event handling
original_handle_portfolio = RiskContainer._handle_portfolio_event

async def debug_handle_portfolio_event(self, event):
    """Debug portfolio event handling."""
    position_tracker['portfolio_events'] += 1
    logger.info(f"📊 RiskContainer received PORTFOLIO event #{position_tracker['portfolio_events']}")
    
    # Call original
    result = await original_handle_portfolio(self, event)
    
    # Check what was cached
    if self._cached_portfolio_state:
        positions = self._cached_portfolio_state.get_all_positions()
        logger.info(f"   ✅ Cached portfolio state with {len(positions)} positions")
    else:
        logger.warning("   ❌ No portfolio state cached!")
    
    return result

RiskContainer._handle_portfolio_event = debug_handle_portfolio_event

# Patch risk limit checking
original_check_limits = RiskContainer._check_risk_limits

def debug_check_risk_limits(self, signal):
    """Debug risk limit checking."""
    symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
    side = signal.side if hasattr(signal, 'side') else signal.get('side', 'UNKNOWN')
    
    logger.info(f"🚦 Checking risk limits for {symbol} {side} signal")
    
    # Call original
    result = original_check_limits(self, signal)
    
    position_tracker['risk_checks'].append({
        'symbol': symbol,
        'side': side,
        'result': result
    })
    
    logger.info(f"   Result: {'✅ APPROVED' if result else '❌ REJECTED'}")
    
    return result

RiskContainer._check_risk_limits = debug_check_risk_limits

# Also check if PORTFOLIO events are being routed
from src.core.communication.pipeline_adapter_protocol import PipelineAdapter
original_setup_reverse = PipelineAdapter._setup_reverse_routing

def debug_setup_reverse_routing(self):
    """Debug reverse routing setup."""
    logger.info("🔄 Setting up reverse routing in pipeline adapter")
    
    # Call original
    original_setup_reverse(self)
    
    # Log what was set up
    logger.info("   ✅ Reverse routing configured")

PipelineAdapter._setup_reverse_routing = debug_setup_reverse_routing

async def test_position_management():
    """Test position management."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("TESTING POSITION MANAGEMENT")
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
    
    # Use just 20 bars for testing
    workflow_config.data_config['max_bars'] = 20
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("POSITION MANAGEMENT SUMMARY")
        logger.info("=" * 80)
        
        # Summary
        logger.info(f"\nPortfolio events received by RiskContainer: {position_tracker['portfolio_events']}")
        logger.info(f"Risk checks performed: {len(position_tracker['risk_checks'])}")
        
        # Show risk check results
        approved = sum(1 for r in position_tracker['risk_checks'] if r['result'])
        rejected = sum(1 for r in position_tracker['risk_checks'] if not r['result'])
        logger.info(f"Risk checks approved: {approved}")
        logger.info(f"Risk checks rejected: {rejected}")
        
        # Final result
        portfolio = result.data.get('portfolio', {}) if result.data else {}
        logger.info(f"\nFinal equity: ${portfolio.get('total_equity', 0):.2f}")
        logger.info(f"Final positions: {portfolio.get('positions', 0)}")
        
        # Diagnosis
        if position_tracker['portfolio_events'] == 0:
            logger.error("\n❌ ISSUE: RiskContainer never received portfolio state updates!")
            logger.error("   This explains why position limits aren't working.")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_position_management())