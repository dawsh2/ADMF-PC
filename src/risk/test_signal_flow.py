"""Test signal flow and processing components."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List

from .protocols import (
    Signal,
    SignalType,
    OrderSide,
    PortfolioStateProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
)
from .signal_flow import SignalFlowManager, MultiSymbolSignalFlow
from .portfolio_state import PortfolioState
from .position_sizing import PercentagePositionSizer
from .risk_limits import MaxPositionLimit, MaxDrawdownLimit


async def test_basic_signal_flow():
    """Test basic signal flow from collection to order generation."""
    print("\n=== Testing Basic Signal Flow ===\n")
    
    # Create components
    portfolio_state = PortfolioState(
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    
    position_sizer = PercentagePositionSizer(
        percentage=Decimal("0.02")  # 2% per position
    )
    
    risk_limits = [
        MaxPositionLimit(max_position=Decimal("5000")),
        MaxDrawdownLimit(
            max_drawdown_pct=Decimal("10"),
            reduce_at_pct=Decimal("8")
        )
    ]
    
    # Create flow manager
    flow_manager = SignalFlowManager(
        enable_caching=True,
        enable_validation=True,
        enable_aggregation=True,
        aggregation_method="weighted_average"
    )
    
    # Register strategies
    flow_manager.register_strategy("momentum", weight=Decimal("0.7"))
    flow_manager.register_strategy("mean_reversion", weight=Decimal("0.3"))
    
    # Generate test signals
    signals = [
        Signal(
            signal_id="sig_001",
            strategy_id="momentum",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.85"),
            timestamp=datetime.now(),
            metadata={"confidence": 0.85}
        ),
        Signal(
            signal_id="sig_002",
            strategy_id="mean_reversion",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.65"),
            timestamp=datetime.now(),
            metadata={"confidence": 0.65}
        ),
        Signal(
            signal_id="sig_003",
            strategy_id="momentum",
            symbol="GOOGL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.SELL,
            strength=Decimal("-0.75"),
            timestamp=datetime.now(),
            metadata={"confidence": 0.75}
        ),
    ]
    
    # Collect signals
    print("Collecting signals...")
    for signal in signals:
        await flow_manager.collect_signal(signal)
        print(f"  - {signal.strategy_id}: {signal.symbol} {signal.side.value}")
    
    # Process signals
    print("\nProcessing signals...")
    market_data = {
        "prices": {
            "AAPL": Decimal("150.00"),
            "GOOGL": Decimal("2800.00")
        },
        "timestamp": datetime.now()
    }
    
    orders = await flow_manager.process_signals(
        portfolio_state=portfolio_state,
        position_sizer=position_sizer,
        risk_limits=risk_limits,
        market_data=market_data
    )
    
    print(f"\nGenerated {len(orders)} orders:")
    for order in orders:
        print(f"  - {order.symbol}: {order.side.value} {order.quantity} @ {order.order_type.value}")
    
    # Show statistics
    stats = flow_manager.get_statistics()
    print(f"\nFlow Statistics:")
    print(f"  - Signals received: {stats['total_signals_received']}")
    print(f"  - Orders generated: {stats['total_orders_generated']}")
    print(f"  - Approval rate: {stats['approval_rate']}")


async def test_multi_symbol_flow():
    """Test multi-symbol signal flow with different classifiers."""
    print("\n=== Testing Multi-Symbol Signal Flow ===\n")
    
    # Create multi-symbol flow manager
    multi_flow = MultiSymbolSignalFlow()
    
    # Create flow managers for different classifiers
    tech_flow = multi_flow.create_flow_manager(
        classifier_id="tech_stocks",
        config={
            "enable_aggregation": True,
            "aggregation_method": "majority_vote"
        }
    )
    
    energy_flow = multi_flow.create_flow_manager(
        classifier_id="energy_stocks",
        config={
            "enable_aggregation": True,
            "aggregation_method": "unanimous"
        }
    )
    
    # Map symbols to classifiers
    tech_symbols = ["AAPL", "GOOGL", "MSFT"]
    energy_symbols = ["XOM", "CVX", "COP"]
    
    for symbol in tech_symbols:
        multi_flow.map_symbol_to_classifier(symbol, "tech_stocks")
    for symbol in energy_symbols:
        multi_flow.map_symbol_to_classifier(symbol, "energy_stocks")
    
    # Register strategies
    for flow_manager in [tech_flow, energy_flow]:
        flow_manager.register_strategy("momentum")
        flow_manager.register_strategy("mean_reversion")
        flow_manager.register_strategy("breakout")
    
    # Generate test signals for multiple symbols
    test_signals = [
        # Tech signals
        Signal(
            signal_id=f"tech_{i}",
            strategy_id=strat,
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            strength=Decimal(str(0.5 + i * 0.1)),
            timestamp=datetime.now(),
            metadata={"sector": "tech"}
        )
        for i, (symbol, strat) in enumerate([
            ("AAPL", "momentum"),
            ("AAPL", "mean_reversion"),
            ("GOOGL", "momentum"),
            ("MSFT", "breakout"),
        ])
    ] + [
        # Energy signals
        Signal(
            signal_id=f"energy_{i}",
            strategy_id=strat,
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,  # All agree for unanimous test
            strength=Decimal("0.7"),
            timestamp=datetime.now(),
            metadata={"sector": "energy"}
        )
        for i, (symbol, strat) in enumerate([
            ("XOM", "momentum"),
            ("XOM", "mean_reversion"),
            ("XOM", "breakout"),
        ])
    ]
    
    # Route signals
    print("Routing signals to classifiers...")
    for signal in test_signals:
        await multi_flow.route_signal(signal)
        print(f"  - {signal.symbol} → {signal.strategy_id}")
    
    # Create portfolio states and components for each classifier
    portfolio_states = {
        "tech_stocks": PortfolioState(
            initial_capital=Decimal("50000"),
            base_currency="USD"
        ),
        "energy_stocks": PortfolioState(
            initial_capital=Decimal("50000"),
            base_currency="USD"
        )
    }
    
    position_sizers = {
        classifier_id: PercentagePositionSizer(percentage=Decimal("0.02"))
        for classifier_id in ["tech_stocks", "energy_stocks"]
    }
    
    risk_limits = {
        classifier_id: [MaxPositionLimit(max_position=Decimal("5000"))]
        for classifier_id in ["tech_stocks", "energy_stocks"]
    }
    
    market_data = {
        "prices": {
            "AAPL": Decimal("150.00"),
            "GOOGL": Decimal("2800.00"),
            "MSFT": Decimal("400.00"),
            "XOM": Decimal("110.00"),
            "CVX": Decimal("150.00"),
            "COP": Decimal("120.00"),
        },
        "timestamp": datetime.now()
    }
    
    # Process all signals
    print("\nProcessing signals by classifier...")
    orders_by_classifier = await multi_flow.process_all_signals(
        portfolio_states=portfolio_states,
        position_sizers=position_sizers,
        risk_limits=risk_limits,
        market_data=market_data
    )
    
    # Show results
    for classifier_id, orders in orders_by_classifier.items():
        print(f"\n{classifier_id}: {len(orders)} orders")
        for order in orders:
            print(f"  - {order.symbol}: {order.side.value} {order.quantity}")
    
    # Show statistics
    all_stats = multi_flow.get_statistics()
    print("\nClassifier Statistics:")
    for classifier_id, stats in all_stats.items():
        print(f"\n{classifier_id}:")
        print(f"  - Signals: {stats['total_signals_received']}")
        print(f"  - Orders: {stats['total_orders_generated']}")
        print(f"  - Approval: {stats['approval_rate']}")


async def test_signal_validation_and_caching():
    """Test signal validation and caching features."""
    print("\n=== Testing Signal Validation & Caching ===\n")
    
    flow_manager = SignalFlowManager(
        enable_caching=True,
        enable_validation=True
    )
    
    flow_manager.register_strategy("test_strategy")
    
    # Test validation - invalid signal
    invalid_signal = Signal(
        signal_id="invalid_001",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("1.5"),  # Invalid: > 1
        timestamp=datetime.now(),
        metadata={}
    )
    
    print("Testing invalid signal (strength > 1)...")
    try:
        await flow_manager.collect_signal(invalid_signal)
    except:
        pass  # Expected to fail validation
    
    stats = flow_manager.get_statistics()
    print(f"  - Signals rejected: {stats['signals_rejected']}")
    
    # Test caching - duplicate signals
    valid_signal = Signal(
        signal_id="valid_001",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={}
    )
    
    print("\nTesting duplicate signal detection...")
    await flow_manager.collect_signal(valid_signal)
    print("  - First signal collected")
    
    # Try same signal again (different ID but same content)
    duplicate_signal = Signal(
        signal_id="valid_002",  # Different ID
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={}
    )
    
    await flow_manager.collect_signal(duplicate_signal)
    print("  - Duplicate signal rejected")
    
    stats = flow_manager.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  - Total signals: {stats['total_signals_received']}")
    print(f"  - Rejected: {stats['signals_rejected']}")
    if 'cache' in stats:
        print(f"  - Cache hit rate: {stats['cache']['hit_rate']}")


async def main():
    """Run all tests."""
    await test_basic_signal_flow()
    await test_multi_symbol_flow()
    await test_signal_validation_and_caching()


if __name__ == "__main__":
    asyncio.run(main())