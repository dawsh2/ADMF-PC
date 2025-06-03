"""Detailed P&L audit to understand excessive returns."""

import asyncio
import logging
import yaml
from decimal import Decimal
from collections import defaultdict

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track all trades
trades = []
positions = defaultdict(lambda: {'quantity': 0, 'cost_basis': 0})

# Patch portfolio to track trades
from src.execution.containers_pipeline import PortfolioContainer
original_handle_fill = PortfolioContainer._handle_fill_event

async def audit_handle_fill(self, event):
    """Audit fill handling."""
    fill_data = event.payload.get('fill', {})
    order_data = event.payload.get('order', {})
    
    symbol = fill_data.get('symbol', 'UNKNOWN')
    side = fill_data.get('side', 'UNKNOWN')
    quantity = float(fill_data.get('quantity', 0))
    price = float(fill_data.get('price', 0))
    commission = float(fill_data.get('commission', 0))
    
    # Track position before
    pos_before = positions[symbol].copy()
    
    # Call original
    result = await original_handle_fill(self, event)
    
    # Calculate P&L for this trade
    if 'SELL' in str(side):
        # Closing or shorting
        if pos_before['quantity'] > 0:
            # Closing long position
            avg_cost = pos_before['cost_basis'] / pos_before['quantity'] if pos_before['quantity'] > 0 else 0
            realized_pnl = (price - avg_cost) * min(quantity, pos_before['quantity']) - commission
            trade_type = "CLOSE_LONG"
        else:
            # Opening short
            realized_pnl = -commission
            trade_type = "OPEN_SHORT"
    else:
        # BUY
        if pos_before['quantity'] < 0:
            # Closing short position
            avg_cost = abs(pos_before['cost_basis'] / pos_before['quantity']) if pos_before['quantity'] < 0 else 0
            realized_pnl = (avg_cost - price) * min(quantity, abs(pos_before['quantity'])) - commission
            trade_type = "CLOSE_SHORT"
        else:
            # Opening long
            realized_pnl = -commission
            trade_type = "OPEN_LONG"
    
    # Update position tracking
    if 'BUY' in str(side):
        positions[symbol]['quantity'] += quantity
        positions[symbol]['cost_basis'] += quantity * price
    else:
        positions[symbol]['quantity'] -= quantity
        positions[symbol]['cost_basis'] -= quantity * price
    
    trade_info = {
        'timestamp': event.timestamp,
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price,
        'commission': commission,
        'trade_type': trade_type,
        'realized_pnl': realized_pnl,
        'position_after': positions[symbol]['quantity']
    }
    trades.append(trade_info)
    
    logger.info(f"TRADE: {trade_type} {symbol} {quantity:.2f} @ ${price:.2f}")
    logger.info(f"  Commission: ${commission:.2f}, Realized P&L: ${realized_pnl:.2f}")
    logger.info(f"  Position: {pos_before['quantity']:.2f} -> {positions[symbol]['quantity']:.2f}")
    
    return result

PortfolioContainer._handle_fill_event = audit_handle_fill

# Also track position closing
original_close = PortfolioContainer._close_position

def audit_close_position(self, symbol: str, quantity: float, price: float, reason: str):
    """Audit position closing."""
    pos_before = positions[symbol].copy()
    
    # Call original
    result = original_close(self, symbol, quantity, price, reason)
    
    # Calculate closing P&L
    if pos_before['quantity'] != 0:
        avg_cost = pos_before['cost_basis'] / pos_before['quantity']
        realized_pnl = (price - avg_cost) * quantity if pos_before['quantity'] > 0 else (avg_cost - price) * quantity
        
        logger.info(f"CLOSE: {reason} {symbol} {quantity:.2f} @ ${price:.2f}")
        logger.info(f"  Avg cost: ${avg_cost:.2f}, Realized P&L: ${realized_pnl:.2f}")
        
        trade_info = {
            'timestamp': None,
            'symbol': symbol,
            'side': 'SELL' if pos_before['quantity'] > 0 else 'BUY',
            'quantity': quantity,
            'price': price,
            'commission': 0,
            'trade_type': f'CLOSE_{reason}',
            'realized_pnl': realized_pnl,
            'position_after': 0
        }
        trades.append(trade_info)
    
    return result

PortfolioContainer._close_position = audit_close_position

async def detailed_pnl_audit():
    """Run detailed P&L audit."""
    
    # Load config
    with open('config/multi_strategy_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("DETAILED P&L AUDIT")
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
    
    # Use 50 bars like the problem case
    workflow_config.data_config['max_bars'] = 50
    
    # Execute workflow
    try:
        from src.core.coordinator.coordinator import ExecutionMode
        
        result = await coordinator.execute_workflow(
            workflow_config, 
            execution_mode=ExecutionMode.COMPOSABLE
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("TRADE SUMMARY")
        logger.info("=" * 80)
        
        # Analyze trades
        total_realized_pnl = sum(t['realized_pnl'] for t in trades)
        total_commission = sum(t['commission'] for t in trades)
        winning_trades = [t for t in trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in trades if t['realized_pnl'] < 0]
        
        logger.info(f"\nTotal trades: {len(trades)}")
        logger.info(f"Winning trades: {len(winning_trades)}")
        logger.info(f"Losing trades: {len(losing_trades)}")
        logger.info(f"Total realized P&L: ${total_realized_pnl:.2f}")
        logger.info(f"Total commissions: ${total_commission:.2f}")
        
        # Show largest winners and losers
        if winning_trades:
            sorted_winners = sorted(winning_trades, key=lambda x: x['realized_pnl'], reverse=True)
            logger.info(f"\nTop 3 winning trades:")
            for t in sorted_winners[:3]:
                logger.info(f"  {t['trade_type']} {t['symbol']} {t['quantity']:.2f} @ ${t['price']:.2f}: +${t['realized_pnl']:.2f}")
        
        if losing_trades:
            sorted_losers = sorted(losing_trades, key=lambda x: x['realized_pnl'])
            logger.info(f"\nTop 3 losing trades:")
            for t in sorted_losers[:3]:
                logger.info(f"  {t['trade_type']} {t['symbol']} {t['quantity']:.2f} @ ${t['price']:.2f}: -${abs(t['realized_pnl']):.2f}")
        
        # Final portfolio
        portfolio = result.data.get('portfolio', {}) if result.data else {}
        initial_capital = 100000
        final_equity = portfolio.get('total_equity', initial_capital)
        returns = ((final_equity - initial_capital) / initial_capital) * 100
        
        logger.info(f"\nFINAL RESULTS:")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Final equity: ${final_equity:,.2f}")
        logger.info(f"Return: {returns:.2f}%")
        logger.info(f"Return per trade: {returns/len(trades) if trades else 0:.2f}%")
        
        # Check for issues
        if abs(returns) > 5:
            logger.warning(f"\n⚠️ WARNING: {returns:.2f}% return in 50 minutes is unrealistic!")
            
            # Calculate annualized return
            minutes_in_year = 252 * 6.5 * 60  # Trading days * hours * minutes
            periods_per_year = minutes_in_year / 50
            annualized = ((1 + returns/100) ** periods_per_year - 1) * 100
            logger.warning(f"Annualized return would be: {annualized:,.0f}%")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(detailed_pnl_audit())