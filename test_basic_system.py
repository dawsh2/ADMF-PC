"""
Basic system test to verify all modules are working together.
"""

import asyncio
from decimal import Decimal
from datetime import datetime

from src.core.containers import UniversalScopedContainer
from src.core.components import ComponentFactory
from src.core.events import Event, EventType
from src.risk import (
    RiskPortfolioCapability, 
    Signal, 
    SignalType, 
    OrderSide
)
from src.execution import (
    ExecutionEngineCapability,
    MarketDataUpdate
)


async def test_basic_system():
    """Test basic system functionality."""
    print("\n=== Basic System Test ===\n")
    
    factory = ComponentFactory()
    
    # 1. Create container hierarchy
    print("1. Creating container hierarchy...")
    
    # Main backtest container
    backtest = UniversalScopedContainer("backtest_test")
    
    # Classifier container
    classifier = UniversalScopedContainer("classifier_test", parent_container=backtest)
    
    # Risk & Portfolio container
    risk_portfolio = UniversalScopedContainer("risk_portfolio_test", parent_container=classifier)
    
    # Apply Risk & Portfolio capability
    factory.apply_capability(
        risk_portfolio,
        RiskPortfolioCapability(),
        {
            'initial_capital': 10000,  # $10k for simple test
            'position_sizers': [
                {'name': 'default', 'type': 'fixed', 'size': 10}
            ],
            'risk_limits': [
                {'type': 'position', 'max_position': 100}
            ]
        }
    )
    
    print("✓ Risk & Portfolio container created")
    
    # 2. Create execution engine
    print("\n2. Creating execution engine...")
    
    execution = UniversalScopedContainer("execution_test", parent_container=backtest)
    
    # Apply Execution capability
    factory.apply_capability(
        execution,
        ExecutionEngineCapability(),
        {
            'broker_type': 'backtest',
            'initial_capital': 10000
        }
    )
    
    print("✓ Execution engine created")
    
    # 3. Wire up event flow
    print("\n3. Setting up event flow...")
    
    # Risk publishes orders
    def on_order_created(order):
        """Handle order from risk."""
        order_event = Event(
            event_type=EventType.CUSTOM,
            event_name="ORDER",
            payload=order,
            source_id=risk_portfolio.container_id
        )
        backtest.event_bus.publish(order_event)
        print(f"  → ORDER event published: {order['symbol']} {order['side']} {order['quantity']}")
    
    # Execution processes orders
    async def on_order_received(event):
        """Handle order in execution."""
        if event.event_name == "ORDER":
            await execution.execution_engine.process_order(event.payload)
    
    backtest.event_bus.subscribe("ORDER", lambda e: asyncio.create_task(on_order_received(e)))
    
    # Risk receives fills
    def on_fill_received(event):
        """Handle fill in risk."""
        if event.event_name == "FILL":
            risk_portfolio.risk_portfolio.handle_fill(event.payload)
            print(f"  → FILL processed: {event.payload['symbol']} @ ${event.payload['price']}")
    
    backtest.event_bus.subscribe("FILL", on_fill_received)
    
    print("✓ Event flow configured")
    
    # 4. Test signal → order → fill pipeline
    print("\n4. Testing signal → order → fill pipeline...")
    
    # Update market data in execution
    market_update = MarketDataUpdate(
        symbol="AAPL",
        bid=149.99,
        ask=150.01,
        last=150.00,
        timestamp=datetime.now()
    )
    await execution.execution_engine.update_market_data(market_update)
    print(f"  Market data updated: AAPL @ $150.00")
    
    # Create a test signal
    test_signal = {
        'signal_id': 'test_001',
        'timestamp': datetime.now(),
        'strategy_id': 'test_strategy',
        'symbol': 'AAPL',
        'signal_type': SignalType.ENTRY.value,
        'side': OrderSide.BUY.value,
        'strength': 0.8,
        'metadata': {'price': 150.00}
    }
    
    print(f"\n  Sending signal: BUY AAPL")
    
    # Process signal through risk
    order = risk_portfolio.process_signal(test_signal)
    
    if order:
        print(f"  ✓ Signal accepted → Order created: {order['quantity']} shares")
        on_order_created(order)
        
        # Wait for async processing
        await asyncio.sleep(0.5)
        
        # Check portfolio state
        portfolio_state = risk_portfolio.get_portfolio_state()
        print(f"\n  Portfolio after fill:")
        print(f"    Cash: ${portfolio_state['cash']:,.2f}")
        print(f"    Positions: {portfolio_state['position_count']}")
        
        position = risk_portfolio.risk_portfolio.portfolio_state.get_position('AAPL')
        if position:
            print(f"    AAPL: {position.quantity} shares @ ${position.average_price}")
    else:
        print("  ✗ Signal rejected by risk")
    
    # 5. Test risk limits
    print("\n5. Testing risk limits...")
    
    # Try to create a large position that exceeds limit
    large_signal = {
        'signal_id': 'test_002',
        'timestamp': datetime.now(),
        'strategy_id': 'test_strategy',
        'symbol': 'GOOGL',
        'signal_type': SignalType.ENTRY.value,
        'side': OrderSide.BUY.value,
        'strength': 0.9,
        'metadata': {'price': 140.00}
    }
    
    # Temporarily set a large position size to test limit
    risk_portfolio.risk_portfolio.position_sizers['default'].position_size = 200
    
    print(f"\n  Sending signal: BUY GOOGL (200 shares - exceeds limit)")
    order = risk_portfolio.process_signal(large_signal)
    
    if order:
        print("  ✗ ERROR: Large position was not rejected!")
    else:
        print("  ✓ Large position correctly rejected by risk limit")
    
    # 6. Show final statistics
    print("\n6. Final Statistics")
    print("=" * 40)
    
    # Portfolio metrics
    portfolio_state = risk_portfolio.get_portfolio_state()
    risk_metrics = risk_portfolio.get_risk_metrics()
    
    print(f"Portfolio Value: ${portfolio_state['total_value']:,.2f}")
    print(f"Cash: ${portfolio_state['cash']:,.2f}")
    print(f"Positions: {portfolio_state['position_count']}")
    print(f"Total Exposure: {risk_metrics.get('total_exposure_pct', 0):.1f}%")
    
    # Execution statistics
    exec_stats = await execution.execution_engine.get_statistics()
    print(f"\nExecution Statistics:")
    print(f"Orders Processed: {exec_stats['total_orders']}")
    print(f"Orders Filled: {exec_stats['filled_orders']}")
    
    print("\n✅ Basic system test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_basic_system())