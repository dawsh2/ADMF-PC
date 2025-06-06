#!/usr/bin/env python3
"""
Complete test of EVENT_FLOW_ARCHITECTURE with execution.

This tests the full flow:
1. Symbol containers broadcast FEATURES events
2. Portfolio containers process features and generate orders
3. Execution container processes orders and generates fills
4. Portfolio containers receive fills and update positions
"""

import asyncio
import logging
from datetime import datetime
from src.core.types.trading import Bar
from src.core.types.events import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_complete_event_flow():
    """Test the complete event flow including execution."""
    
    # 1. Create all containers
    from src.core.containers.symbol_timeframe_container import SymbolTimeframeContainer
    from src.core.containers.portfolio_container import PortfolioContainer
    from src.core.containers.execution_container import ExecutionContainer
    from src.core.communication.factory import AdapterFactory
    
    # Create symbol container
    symbol_container = SymbolTimeframeContainer(
        symbol='SPY',
        timeframe='1d',
        data_config={'source': 'csv', 'file': 'data/SPY.csv'},
        feature_config={
            'indicators': [
                {'name': 'sma_20', 'type': 'sma', 'period': 20},
                {'name': 'sma_50', 'type': 'sma', 'period': 50},
                {'name': 'rsi', 'type': 'rsi', 'period': 14}
            ]
        },
        container_id='SPY_1d'
    )
    
    # Create portfolio container
    portfolio_container = PortfolioContainer(
        combo_id='c0001',
        strategy_params={
            'type': 'momentum',
            'sma_period': 20,
            'rsi_threshold_long': 50,  # Relaxed thresholds
            'rsi_threshold_short': 50
        },
        risk_params={
            'type': 'conservative',
            'max_position_value': 50000,
            'max_position_percent': 0.1,
            'max_drawdown': 0.15
        },
        initial_capital=100000,
        container_id='portfolio_c0001'
    )
    
    # Create execution container
    execution_container = ExecutionContainer(
        execution_config={
            'fill_probability': 1.0,  # Always fill for testing
            'partial_fill_probability': 0.0,  # No partial fills
            'slippage': {
                'base': 0.0001,
                'volume_factor': 0.00001,
                'random_factor': 0.0
            },
            'commission': {
                'per_share': 0.01,
                'minimum': 1.0,
                'maximum': 10.0
            }
        },
        container_id='execution'
    )
    
    # 2. Set up stateless services
    from src.strategy.strategies.momentum import momentum_strategy
    # Risk validators are used by RiskServiceAdapter, not directly by portfolio
    
    portfolio_container.set_strategy_service(momentum_strategy)
    # Risk validation happens through RiskServiceAdapter, not directly in portfolio
    
    # 3. Wire up communication
    adapter_factory = AdapterFactory()
    
    # Symbol → Portfolio broadcast (FEATURES)
    feature_broadcast = adapter_factory.create_adapter(
        name='feature_broadcast',
        config={
            'type': 'broadcast',
            'source': 'SPY_1d',
            'targets': ['portfolio_c0001']
        }
    )
    
    # Portfolio → Execution (ORDERS)
    order_routing = adapter_factory.create_adapter(
        name='order_routing',
        config={
            'type': 'selective', 
            'source': 'portfolio_c0001',
            'route_by_type': {
                EventType.ORDER: 'execution'
            },
            'routing_rules': [],  # Not needed with route_by_type
            'default_target': None  # Don't route non-ORDER events
        }
    )
    
    # Execution → Portfolio broadcast (FILLS)
    fill_broadcast = adapter_factory.create_adapter(
        name='fill_broadcast',
        config={
            'type': 'broadcast',
            'source': 'execution',
            'targets': ['portfolio_c0001']
        }
    )
    
    # Wire containers
    containers = {
        'SPY_1d': symbol_container,
        'portfolio_c0001': portfolio_container,
        'execution': execution_container
    }
    
    for adapter in [feature_broadcast, order_routing, fill_broadcast]:
        adapter.setup(containers)
        adapter.start()
    
    # 4. Initialize and start containers
    logger.info("Initializing containers...")
    for container in containers.values():
        await container.initialize()
    
    logger.info("Starting containers...")
    for container in containers.values():
        await container.start()
    
    # 5. Let the system process
    logger.info("Processing data...")
    await asyncio.sleep(1)  # Initial processing
    
    # Check interim state
    logger.info(f"Interim check - Orders created: {portfolio_container._orders_created}")
    
    await asyncio.sleep(2)  # More processing time
    
    # 6. Collect and display results
    logger.info("\n" + "="*60)
    logger.info("COMPLETE EVENT FLOW TEST RESULTS")
    logger.info("="*60)
    
    # Symbol container stats
    symbol_state = symbol_container.get_state_info()
    logger.info(f"\nSymbol Container (SPY_1d):")
    logger.info(f"  Bars processed: {symbol_state['bars_processed']}")
    logger.info(f"  Features broadcasted: {symbol_state['features_broadcasted']}")
    
    # Portfolio container stats
    portfolio_state = {
        'cash': portfolio_container.portfolio_state.cash,
        'total_value': portfolio_container.portfolio_state.total_value,
        'positions': len(portfolio_container.portfolio_state.positions)
    }
    logger.info(f"\nPortfolio Container (c0001):")
    logger.info(f"  Features received: {portfolio_container._features_received}")
    logger.info(f"  Signals generated: {portfolio_container._signals_generated}")
    logger.info(f"  Orders created: {portfolio_container._orders_created}")
    logger.info(f"  Cash: ${portfolio_state['cash']:.2f}")
    logger.info(f"  Total value: ${portfolio_state['total_value']:.2f}")
    logger.info(f"  Positions: {portfolio_state['positions']}")
    
    # Execution container stats
    exec_stats = execution_container.get_execution_stats()
    logger.info(f"\nExecution Container:")
    logger.info(f"  Orders received: {exec_stats['orders_received']}")
    logger.info(f"  Orders filled: {exec_stats['orders_filled']}")
    logger.info(f"  Fill rate: {exec_stats['fill_rate']:.1%}")
    logger.info(f"  Total volume: ${exec_stats['total_volume']:.2f}")
    logger.info(f"  Total commission: ${exec_stats['total_commission']:.2f}")
    
    # Show some positions if any
    if portfolio_container.portfolio_state.positions:
        logger.info(f"\nActive Positions:")
        for symbol, position in portfolio_container.portfolio_state.positions.items():
            logger.info(f"  {symbol}: {position.quantity} shares @ ${position.avg_price:.2f}")
            logger.info(f"    Market value: ${position.market_value:.2f}")
            logger.info(f"    Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    # 7. Stop containers
    logger.info("\nStopping containers...")
    for container in containers.values():
        await container.stop()
    
    # Stop adapters
    for adapter in [feature_broadcast, order_routing, fill_broadcast]:
        adapter.stop()
    
    logger.info("\nTest completed!")
    
    # Return success if we processed data and generated some activity
    return (symbol_state['bars_processed'] > 0 and 
            portfolio_container._features_received > 0 and
            exec_stats['orders_received'] == portfolio_container._orders_created)


if __name__ == "__main__":
    success = asyncio.run(test_complete_event_flow())
    logger.info(f"\nTEST {'PASSED' if success else 'FAILED'}")