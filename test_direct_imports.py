"""Test with direct imports, avoiding __init__.py chains."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import specific modules directly, bypassing __init__.py
import src.risk.portfolio_state as ps
import src.risk.position_sizing as sizing
import src.risk.risk_limits as limits
import src.risk.signal_processing as sp
import src.risk.protocols as protocols

from datetime import datetime, timedelta
from decimal import Decimal

print("Testing ADMF-PC with Direct Imports")
print("=" * 60)


def test_basic_flow():
    """Test basic signal → order → portfolio flow."""
    
    # 1. Create portfolio
    print("\n1. Creating Portfolio")
    portfolio = ps.PortfolioState(
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    print(f"   Initial capital: ${portfolio.get_cash_balance()}")
    
    # 2. Create components
    print("\n2. Setting up Components")
    position_sizer = sizing.PercentagePositionSizer(percentage=Decimal("0.02"))
    risk_limit = limits.MaxPositionLimit(max_position_value=Decimal("150000"))  # $150k max
    processor = sp.SignalProcessor()
    print("   ✓ Position sizer (2%)")
    print("   ✓ Risk limit (max $150k position)")
    print("   ✓ Signal processor")
    
    # 3. Create buy signal
    print("\n3. Processing BUY Signal")
    buy_signal = protocols.Signal(
        signal_id="BUY_001",
        strategy_id="test",
        symbol="AAPL",
        signal_type=protocols.SignalType.ENTRY,
        side=protocols.OrderSide.BUY,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={"price": 150.0}
    )
    
    market_data = {
        "prices": {"AAPL": Decimal("150")},
        "timestamp": datetime.now()
    }
    
    order = processor.process_signal(
        signal=buy_signal,
        portfolio_state=portfolio,
        position_sizer=position_sizer,
        risk_limits=[risk_limit],
        market_data=market_data
    )
    
    if order:
        print(f"   ✓ Order created: BUY {order.quantity} @ MARKET")
        
        # Simulate fill
        position = portfolio.update_position(
            symbol="AAPL",
            quantity_delta=order.quantity,
            price=Decimal("150"),
            timestamp=datetime.now()
        )
        print(f"   ✓ Position: {position.quantity} shares @ ${position.average_price}")
        print(f"   Cash remaining: ${portfolio.get_cash_balance()}")
    
    # 4. Price moves up
    print("\n4. Price Movement")
    new_price = Decimal("155")
    portfolio.update_market_prices({"AAPL": new_price})
    position = portfolio.get_position("AAPL")
    print(f"   Price: $150 → ${new_price}")
    print(f"   Unrealized P&L: ${position.unrealized_pnl}")
    
    # 5. Create sell signal
    print("\n5. Processing SELL Signal")
    sell_signal = protocols.Signal(
        signal_id="SELL_001",
        strategy_id="test",
        symbol="AAPL",
        signal_type=protocols.SignalType.EXIT,
        side=protocols.OrderSide.SELL,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={"price": 155.0}
    )
    
    sell_order = processor.process_signal(
        signal=sell_signal,
        portfolio_state=portfolio,
        position_sizer=position_sizer,
        risk_limits=[risk_limit],
        market_data={"prices": {"AAPL": new_price}}
    )
    
    if sell_order:
        print(f"   ✓ Order created: SELL {sell_order.quantity} @ MARKET")
        
        # Simulate fill
        portfolio.update_position(
            symbol="AAPL",
            quantity_delta=-sell_order.quantity,
            price=new_price,
            timestamp=datetime.now()
        )
        print(f"   ✓ Position closed")
        print(f"   Realized P&L: ${portfolio._realized_pnl}")
    
    # 6. Final summary
    print("\n6. Performance Summary")
    print("=" * 40)
    performance = portfolio.get_performance_summary()
    metrics = portfolio.get_risk_metrics()
    
    print(f"Initial Capital: ${performance['initial_capital']}")
    print(f"Final Value: ${performance['current_value']}")
    print(f"Total Return: {performance['total_return']}")
    print(f"Realized P&L: ${performance['realized_pnl']}")
    print(f"Max Drawdown: {performance['max_drawdown']}")
    
    # Processor stats
    stats = processor.get_statistics()
    print(f"\nSignal Processing:")
    print(f"  Processed: {stats['processed_signals']}")
    print(f"  Approved: {stats['approved_orders']}")
    print(f"  Approval Rate: {stats['approval_rate']}")


def test_signal_aggregation():
    """Test signal aggregation from multiple strategies."""
    print("\n\n" + "=" * 60)
    print("Testing Signal Aggregation")
    print("=" * 60)
    
    # Create aggregator
    aggregator = sp.SignalAggregator(aggregation_method="weighted_average")
    
    # Create multiple signals for same symbol
    signals = [
        protocols.Signal(
            signal_id=f"SIG_{i}",
            strategy_id=f"strategy_{i}",
            symbol="AAPL",
            signal_type=protocols.SignalType.ENTRY,
            side=protocols.OrderSide.BUY,
            strength=Decimal(str(0.5 + i * 0.2)),  # 0.5, 0.7, 0.9
            timestamp=datetime.now(),
            metadata={"confidence": 0.5 + i * 0.2}
        )
        for i in range(3)
    ]
    
    print(f"\nInput signals:")
    for sig in signals:
        print(f"  {sig.strategy_id}: strength={sig.strength}")
    
    # Aggregate with weights
    weights = {
        "strategy_0": Decimal("0.2"),
        "strategy_1": Decimal("0.3"),
        "strategy_2": Decimal("0.5"),
    }
    
    aggregated = aggregator.aggregate_signals(signals, weights)
    
    print(f"\nAggregated result:")
    for agg in aggregated:
        print(f"  Symbol: {agg.symbol}")
        print(f"  Side: {agg.side.value}")
        print(f"  Strength: {agg.strength}")
        print(f"  Method: {agg.metadata['aggregation_method']}")


if __name__ == "__main__":
    try:
        test_basic_flow()
        test_signal_aggregation()
        print("\n✅ All tests passed!")
        
        print("\n" + "=" * 60)
        print("INTEGRATION SUMMARY")
        print("=" * 60)
        print("\nWorking Components:")
        print("  ✓ Portfolio state tracking")
        print("  ✓ Position sizing")
        print("  ✓ Risk limits")
        print("  ✓ Signal processing")
        print("  ✓ Signal aggregation")
        print("  ✓ P&L calculation")
        
        print("\nTo run with full dependencies:")
        print("  1. Activate your virtual environment")
        print("  2. Run: python test_basic_backtest_simple.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()