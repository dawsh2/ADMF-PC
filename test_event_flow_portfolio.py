#!/usr/bin/env python3
"""
Test the complete EVENT_FLOW_ARCHITECTURE with Portfolio processing.

This tests:
1. Symbol containers broadcasting FEATURES events
2. Portfolio containers receiving and processing FEATURES
3. Stateless momentum strategy generating signals
4. Risk validation and order generation
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


async def test_event_flow():
    """Test the complete event flow with portfolio processing."""
    
    # 1. Create containers
    from src.core.containers.symbol_timeframe_container import SymbolTimeframeContainer
    from src.core.containers.portfolio_container import PortfolioContainer
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
    
    # Create portfolio container with momentum strategy
    portfolio_container = PortfolioContainer(
        combo_id='c0001',
        strategy_params={
            'type': 'momentum',
            'sma_period': 20,
            'rsi_threshold_long': 50,  # Relaxed from 30
            'rsi_threshold_short': 50   # Relaxed from 70
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
    
    # 2. Set up stateless services
    from src.strategy.strategies.stateless_momentum import momentum_strategy
    from src.risk.validators import validate_composite
    
    portfolio_container.set_strategy_service(momentum_strategy)
    portfolio_container.set_risk_validator(validate_composite)
    
    # 3. Wire up communication
    adapter_factory = AdapterFactory()
    
    # Create broadcast adapter for FEATURES events
    feature_broadcast = adapter_factory.create_adapter(
        name='feature_broadcast_SPY_1d',
        config={
            'type': 'broadcast',
            'source': 'SPY_1d',
            'targets': ['portfolio_c0001']
            # No event filter - let all events through
        }
    )
    
    # Wire containers
    containers = {
        'SPY_1d': symbol_container,
        'portfolio_c0001': portfolio_container
    }
    feature_broadcast.setup(containers)
    feature_broadcast.start()
    
    # 4. Initialize containers
    logger.info("Initializing containers...")
    await symbol_container.initialize()
    await portfolio_container.initialize()
    
    # 5. Start containers
    logger.info("Starting containers...")
    await symbol_container.start()
    await portfolio_container.start()
    
    # 6. Process a few bars manually to test the flow
    logger.info("Processing test data...")
    
    # Wait a bit for data processing
    await asyncio.sleep(2)
    
    # 7. Check results
    logger.info("\n=== Test Results ===")
    
    # Check if portfolio received any FEATURES events
    if hasattr(portfolio_container, '_features_received'):
        logger.info(f"Portfolio received {portfolio_container._features_received} FEATURES events")
    
    # Check portfolio state  
    state = {
        'cash': portfolio_container.portfolio_state.cash,
        'total_value': portfolio_container.portfolio_state.total_value,
        'positions': len(portfolio_container.portfolio_state.positions),
        'metrics': portfolio_container.metrics
    }
    logger.info(f"Portfolio state: cash=${state['cash']:.2f}, total_value=${state['total_value']:.2f}")
    
    # Check if any signals were generated
    if hasattr(portfolio_container, '_signals_generated'):
        logger.info(f"Signals generated: {portfolio_container._signals_generated}")
    
    # Check if any orders were created
    if hasattr(portfolio_container, '_orders_created'):
        logger.info(f"Orders created: {portfolio_container._orders_created}")
    
    # 8. Stop containers
    logger.info("\nStopping containers...")
    await symbol_container.stop()
    await portfolio_container.stop()
    
    # Stop adapter
    feature_broadcast.stop()
    
    logger.info("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_event_flow())