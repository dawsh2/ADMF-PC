"""Debug cash flow tracking for portfolio."""

import asyncio
import logging
import yaml
from decimal import Decimal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Track all cash changes
cash_flows = []
initial_cash = 100000

def track_cash_flow(description, amount, balance_after):
    """Track a cash flow event."""
    cash_flows.append({
        'description': description,
        'amount': float(amount),
        'balance_after': float(balance_after),
        'timestamp': datetime.now()
    })
    logger.info(f"💰 {description}: ${amount:,.2f} → Balance: ${balance_after:,.2f}")

# Patch PortfolioContainer to track cash changes
from src.execution.containers_pipeline import PortfolioContainer
original_handle_fill = PortfolioContainer._handle_fill_event

async def tracked_handle_fill(self, event):
    """Track fills with cash flow."""
    fill = event.payload.get('fill')
    if fill:
        # Get cash before
        cash_before = float(self.portfolio_state._cash_balance)
        
        # Call original
        result = await original_handle_fill(self, event)
        
        # Get cash after
        cash_after = float(self.portfolio_state._cash_balance)
        cash_change = cash_after - cash_before
        
        # Track the flow
        side = "SELL" if hasattr(fill, 'side') and str(fill.side).find('SELL') != -1 else "BUY"
        desc = f"{side} {fill.quantity} {fill.symbol} @ ${fill.price}"
        track_cash_flow(desc, cash_change, cash_after)
        
        return result
    return await original_handle_fill(self, event)

PortfolioContainer._handle_fill_event = tracked_handle_fill

# Also track END_OF_DATA closing
original_close_all = PortfolioContainer._close_all_positions_directly

async def tracked_close_all(self, reason="manual"):
    """Track position closing."""
    if self.portfolio_state:
        cash_before = float(self.portfolio_state._cash_balance)
        positions_before = dict(self.portfolio_state.get_all_positions())
        
        # Call original
        await original_close_all(self, reason)
        
        cash_after = float(self.portfolio_state._cash_balance)
        cash_change = cash_after - cash_before
        
        # Track the flow
        for symbol, pos in positions_before.items():
            if hasattr(pos, 'quantity') and pos.quantity != 0:
                track_cash_flow(
                    f"Close {symbol} {pos.quantity} shares @ {getattr(pos, 'current_price', pos.average_price)}", 
                    cash_change, 
                    cash_after
                )

PortfolioContainer._close_all_positions_directly = tracked_close_all

async def run_debug():
    """Run with cash flow tracking."""
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("CASH FLOW AUDIT")
    logger.info("=" * 80)
    logger.info(f"Initial Cash: ${initial_cash:,.2f}\n")
    
    # Track initial
    track_cash_flow("Initial Capital", 0, initial_cash)
    
    # Import and run
    from src.core.coordinator.coordinator import Coordinator
    from src.core.coordinator.simple_types import WorkflowConfig, WorkflowType
    
    coordinator = Coordinator(enable_composable_containers=True)
    
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
    
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("CASH FLOW SUMMARY")
        logger.info("=" * 80)
        
        total_change = 0
        for i, flow in enumerate(cash_flows[1:], 1):  # Skip initial
            logger.info(f"{i}. {flow['description']}")
            logger.info(f"   Amount: ${flow['amount']:,.2f}")
            logger.info(f"   Balance: ${flow['balance_after']:,.2f}")
            total_change += flow['amount']
        
        logger.info(f"\nTotal Cash Change: ${total_change:,.2f}")
        logger.info(f"Expected Final: ${initial_cash + total_change:,.2f}")
        
        if result and 'portfolio' in result:
            actual_final = result['portfolio']['cash_balance']
            logger.info(f"Actual Final: ${actual_final:,.2f}")
            logger.info(f"Discrepancy: ${actual_final - (initial_cash + total_change):,.2f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_debug())