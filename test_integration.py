"""
Integration test bringing together all core modules.

This test demonstrates the complete signal → order → fill pipeline:
1. Classifier determines regime
2. Strategies generate signals 
3. Risk & Portfolio converts to orders
4. Execution engine processes orders and generates fills
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List

from src.core.containers import UniversalScopedContainer
from src.core.components import ComponentFactory
from src.core.events import Event, EventType
from src.risk import RiskPortfolioCapability, Signal, SignalType, OrderSide
from src.execution import ExecutionEngineCapability, MarketDataUpdate
from src.strategy.regime import RegimeClassifier


class DummyStrategy:
    """Simple strategy for testing that generates signals."""
    
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.signal_count = 0
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Signal:
        """Generate a signal based on market data."""
        symbol = market_data['symbol']
        if symbol not in self.symbols:
            return None
            
        # Simple logic: buy if price < 100, sell if price > 200
        price = market_data['close']
        
        if price < 100:
            side = OrderSide.BUY
            strength = 0.8
        elif price > 200:
            side = OrderSide.SELL
            strength = 0.7
        else:
            return None
        
        self.signal_count += 1
        
        return {
            'signal_id': f"{self.name}_{self.signal_count}",
            'timestamp': datetime.now(),
            'strategy_id': self.name,
            'symbol': symbol,
            'signal_type': SignalType.ENTRY.value,
            'side': side.value,
            'strength': strength,
            'metadata': {
                'price': price,
                'reason': 'price_threshold'
            }
        }


async def test_full_pipeline():
    """Test the complete trading pipeline."""
    
    print("\n=== Full Pipeline Integration Test ===\n")
    
    # Create component factory
    factory = ComponentFactory()
    
    # 1. Create Backtest Container
    backtest_container = UniversalScopedContainer("backtest_001")
    
    # 2. Create Classifier Container (simplified - no actual classification)
    classifier_container = UniversalScopedContainer(
        "classifier_hmm",
        parent_container=backtest_container
    )
    
    # 3. Create Risk & Portfolio Container
    risk_container = UniversalScopedContainer(
        "risk_conservative", 
        parent_container=classifier_container
    )
    
    # Apply Risk & Portfolio capability
    factory.apply_capability(
        risk_container,
        RiskPortfolioCapability(),
        {
            'initial_capital': 100000,
            'position_sizers': [
                {
                    'name': 'default',
                    'type': 'fixed',
                    'size': 100
                }
            ],
            'risk_limits': [
                {
                    'type': 'position',
                    'max_position': 1000
                },
                {
                    'type': 'exposure',
                    'max_exposure_pct': 20
                }
            ]
        }
    )
    
    # 4. Create Execution Container
    execution_container = UniversalScopedContainer(
        "execution_engine",
        parent_container=backtest_container
    )
    
    # Apply Execution capability
    factory.apply_capability(
        execution_container,
        ExecutionEngineCapability(),
        {
            'broker_type': 'backtest',
            'initial_capital': 100000,
            'slippage': {
                'type': 'fixed',
                'value': 0.01  # $0.01 per share
            },
            'commission': {
                'type': 'per_share',
                'value': 0.005  # $0.005 per share
            }
        }
    )
    
    # 5. Create strategies within Risk container
    strategies = [
        DummyStrategy("momentum", ["AAPL", "GOOGL"]),
        DummyStrategy("mean_reversion", ["SPY", "QQQ"])
    ]
    
    # 6. Wire up event flow
    # Risk container subscribes to strategy signals
    async def handle_strategy_signal(signal):
        """Process signal through risk."""
        if signal:
            order = risk_container.process_signal(signal)
            if order:
                print(f"✓ Order created: {order['symbol']} {order['side']} {order['quantity']} shares")
                # Publish ORDER event
                order_event = Event(
                    event_type=EventType.CUSTOM,
                    event_name="ORDER",
                    payload=order,
                    source_id=risk_container.container_id
                )
                backtest_container.event_bus.publish(order_event)
            else:
                print(f"✗ Signal rejected by risk: {signal['symbol']}")
    
    # Execution subscribes to ORDER events
    def handle_order(event: Event):
        """Process order through execution."""
        if event.event_name == "ORDER":
            asyncio.create_task(
                execution_container.execution_engine.process_order(event.payload)
            )
    
    backtest_container.event_bus.subscribe("ORDER", handle_order)
    
    # Execution publishes FILL events
    def handle_fill(event: Event):
        """Process fill back to risk."""
        if event.event_name == "FILL":
            fill = event.payload
            print(f"✓ Fill received: {fill['symbol']} @ ${fill['price']:.2f}")
            # Update risk container's portfolio
            risk_container.risk_portfolio.handle_fill(fill)
    
    backtest_container.event_bus.subscribe("FILL", handle_fill)
    
    # 7. Simulate market data and process
    print("Processing market data...\n")
    
    market_data_samples = [
        {'symbol': 'AAPL', 'close': 95.0},   # Should trigger BUY
        {'symbol': 'SPY', 'close': 450.0},   # No signal
        {'symbol': 'GOOGL', 'close': 85.0},  # Should trigger BUY
        {'symbol': 'QQQ', 'close': 210.0},   # Should trigger SELL
        {'symbol': 'AAPL', 'close': 98.0},   # Another BUY (may hit limit)
    ]
    
    for data in market_data_samples:
        print(f"\nMarket update: {data['symbol']} @ ${data['close']}")
        
        # Update execution engine's market data
        market_update = MarketDataUpdate(
            symbol=data['symbol'],
            bid=data['close'] - 0.01,
            ask=data['close'] + 0.01,
            last=data['close'],
            timestamp=datetime.now()
        )
        await execution_container.execution_engine.update_market_data(market_update)
        
        # Generate signals from strategies
        for strategy in strategies:
            signal = await strategy.process_market_data(data)
            if signal:
                print(f"  Signal from {strategy.name}: {signal['side']} {signal['symbol']}")
                await handle_strategy_signal(signal)
        
        # Allow async operations to complete
        await asyncio.sleep(0.1)
    
    # 8. Show final state
    print("\n=== Final Portfolio State ===")
    portfolio_state = risk_container.get_portfolio_state()
    print(f"Cash: ${portfolio_state['cash']:,.2f}")
    print(f"Positions: {portfolio_state['position_count']}")
    print(f"Total Value: ${portfolio_state['total_value']:,.2f}")
    
    print("\n=== Risk Metrics ===")
    risk_metrics = risk_container.get_risk_metrics()
    print(f"Total Exposure: {risk_metrics.get('total_exposure_pct', 0):.1f}%")
    print(f"Max Position: {risk_metrics.get('max_position_pct', 0):.1f}%")
    
    print("\n=== Execution Statistics ===")
    exec_stats = await execution_container.execution_engine.get_statistics()
    print(f"Orders Processed: {exec_stats['total_orders']}")
    print(f"Orders Filled: {exec_stats['filled_orders']}")
    print(f"Total Commission: ${exec_stats['total_commission']:.2f}")
    print(f"Total Slippage: ${exec_stats['total_slippage']:.2f}")


async def test_multi_classifier_setup():
    """Test multiple classifiers with different risk profiles."""
    
    print("\n=== Multi-Classifier Setup Test ===\n")
    
    # This demonstrates the architecture we defined:
    # Backtest Container
    # ├── Classifier 1 (HMM)
    # │   ├── Risk & Portfolio 1 (Conservative)
    # │   │   ├── Strategy A
    # │   │   └── Strategy B
    # │   └── Risk & Portfolio 2 (Aggressive)
    # │       └── Strategy C
    # └── Classifier 2 (Pattern)
    #     └── Risk & Portfolio 3 (Balanced)
    #         └── Strategy D
    
    factory = ComponentFactory()
    
    # Create main backtest container
    backtest = UniversalScopedContainer("backtest_multi")
    
    # Create execution engine at backtest level (shared)
    execution = UniversalScopedContainer("execution", parent_container=backtest)
    factory.apply_capability(
        execution,
        ExecutionEngineCapability(),
        {'broker_type': 'backtest', 'initial_capital': 300000}
    )
    
    # Classifier 1: HMM
    classifier1 = UniversalScopedContainer("classifier_hmm", parent_container=backtest)
    
    # Risk Profile 1: Conservative (under HMM)
    risk1 = UniversalScopedContainer("risk_conservative", parent_container=classifier1)
    factory.apply_capability(
        risk1,
        RiskPortfolioCapability(),
        {
            'initial_capital': 100000,
            'position_sizers': [{'name': 'default', 'type': 'percentage', 'percentage': 1.0}],
            'risk_limits': [{'type': 'exposure', 'max_exposure_pct': 10}]
        }
    )
    
    # Risk Profile 2: Aggressive (under HMM)
    risk2 = UniversalScopedContainer("risk_aggressive", parent_container=classifier1)
    factory.apply_capability(
        risk2,
        RiskPortfolioCapability(),
        {
            'initial_capital': 100000,
            'position_sizers': [{'name': 'default', 'type': 'percentage', 'percentage': 5.0}],
            'risk_limits': [{'type': 'exposure', 'max_exposure_pct': 50}]
        }
    )
    
    # Classifier 2: Pattern
    classifier2 = UniversalScopedContainer("classifier_pattern", parent_container=backtest)
    
    # Risk Profile 3: Balanced (under Pattern)
    risk3 = UniversalScopedContainer("risk_balanced", parent_container=classifier2)
    factory.apply_capability(
        risk3,
        RiskPortfolioCapability(),
        {
            'initial_capital': 100000,
            'position_sizers': [{'name': 'default', 'type': 'percentage', 'percentage': 2.5}],
            'risk_limits': [{'type': 'exposure', 'max_exposure_pct': 25}]
        }
    )
    
    print("Container Hierarchy:")
    print("Backtest")
    print("├── Execution Engine (shared)")
    print("├── Classifier HMM")
    print("│   ├── Risk Conservative ($100k)")
    print("│   └── Risk Aggressive ($100k)")
    print("└── Classifier Pattern")
    print("    └── Risk Balanced ($100k)")
    
    print("\nTotal Capital Allocated: $300,000")
    
    # Each risk container manages its own strategies and portfolio
    # All share the same execution engine
    # This allows testing different approaches in parallel


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(test_full_pipeline())
    asyncio.run(test_multi_classifier_setup())