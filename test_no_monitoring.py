"""Test that avoids the monitoring module which requires numpy."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing ADMF-PC Integration (No External Dependencies)")
print("=" * 60)

# Import directly what we need, avoiding the monitoring chain
from src.risk.portfolio_state import PortfolioState
from src.risk.position_sizing import PercentagePositionSizer
from src.risk.risk_limits import MaxPositionLimit
from src.risk.signal_processing import SignalProcessor
from src.risk.protocols import (
    Signal, SignalType, OrderSide, Order
)


async def test_integration():
    """Test basic integration without numpy dependencies."""
    
    # 1. Create Portfolio
    print("\n1. Creating Portfolio")
    portfolio = PortfolioState(
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    print(f"   Initial capital: ${portfolio.get_cash_balance()}")
    
    # 2. Create Risk Components
    print("\n2. Setting up Risk Management")
    position_sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
    risk_limits = [MaxPositionLimit(max_position=Decimal("1000"))]
    signal_processor = SignalProcessor()
    print("   Position sizing: 2% per trade")
    print("   Max position: 1000 shares")
    
    # 3. Create test signals
    print("\n3. Creating Test Signals")
    signals = [
        Signal(
            signal_id="TEST_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={"price": 150.0, "reason": "test_buy"}
        ),
        Signal(
            signal_id="TEST_002",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            side=OrderSide.SELL,
            strength=Decimal("0.8"),
            timestamp=datetime.now() + timedelta(days=1),
            metadata={"price": 155.0, "reason": "test_sell"}
        )
    ]
    print(f"   Created {len(signals)} signals")
    
    # 4. Process signals
    print("\n4. Processing Signals")
    orders = []
    
    for signal in signals:
        print(f"\n   Processing {signal.signal_type.value} signal for {signal.symbol}")
        
        # Update market data
        market_data = {
            "prices": {signal.symbol: Decimal(str(signal.metadata["price"]))},
            "timestamp": signal.timestamp
        }
        
        # Update portfolio prices
        portfolio.update_market_prices(market_data["prices"])
        
        # Process signal
        order = signal_processor.process_signal(
            signal=signal,
            portfolio_state=portfolio,
            position_sizer=position_sizer,
            risk_limits=risk_limits,
            market_data=market_data
        )
        
        if order:
            orders.append(order)
            print(f"   ✓ Order created: {order.side.value} {order.quantity} shares")
            
            # Simulate fill
            if order.side == OrderSide.BUY:
                position = portfolio.update_position(
                    symbol=order.symbol,
                    quantity_delta=order.quantity,
                    price=market_data["prices"][order.symbol],
                    timestamp=signal.timestamp
                )
            else:
                position = portfolio.update_position(
                    symbol=order.symbol,
                    quantity_delta=-order.quantity,
                    price=market_data["prices"][order.symbol],
                    timestamp=signal.timestamp
                )
            
            print(f"   Position updated: {position.quantity if position.quantity != 0 else 'Closed'}")
        else:
            print("   ✗ Order rejected")
    
    # 5. Show results
    print("\n5. Final Results")
    print("=" * 40)
    
    performance = portfolio.get_performance_summary()
    print(f"Initial Capital: ${performance['initial_capital']}")
    print(f"Final Value: ${performance['current_value']}")
    print(f"Total Return: {performance['total_return']}")
    print(f"Realized P&L: ${performance['realized_pnl']}")
    
    # Signal processor stats
    processor_stats = signal_processor.get_statistics()
    print(f"\nSignal Processing:")
    print(f"  Processed: {processor_stats['processed_signals']}")
    print(f"  Approved: {processor_stats['approved_orders']}")
    print(f"  Rejected: {processor_stats['rejected_signals']}")
    print(f"  Approval Rate: {processor_stats['approval_rate']}")
    
    print("\n✅ Integration test completed successfully!")


async def test_risk_limits():
    """Test that risk limits work correctly."""
    print("\n\n" + "=" * 60)
    print("Testing Risk Limits")
    print("=" * 60)
    
    # Small portfolio to test limits
    portfolio = PortfolioState(initial_capital=Decimal("10000"))
    position_sizer = PercentagePositionSizer(percentage=Decimal("0.50"))  # 50%!
    risk_limits = [MaxPositionLimit(max_position=Decimal("50"))]  # Max 50 shares
    processor = SignalProcessor()
    
    signal = Signal(
        signal_id="LIMIT_TEST",
        strategy_id="test",
        symbol="EXPENSIVE",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.9"),
        timestamp=datetime.now(),
        metadata={"price": 200.0}  # $200/share
    )
    
    print(f"\nTesting large position request:")
    print(f"  Portfolio: ${portfolio.get_cash_balance()}")
    print(f"  Position sizing: 50% = ${portfolio.get_cash_balance() * Decimal('0.5')}")
    print(f"  Share price: $200")
    print(f"  Requested shares: {portfolio.get_cash_balance() * Decimal('0.5') / 200}")
    print(f"  Risk limit: Max 50 shares")
    
    order = processor.process_signal(
        signal=signal,
        portfolio_state=portfolio,
        position_sizer=position_sizer,
        risk_limits=risk_limits,
        market_data={"prices": {"EXPENSIVE": Decimal("200")}}
    )
    
    if order:
        print(f"\n✓ Order approved but limited to: {order.quantity} shares")
    else:
        print("\n✗ Order rejected by risk limits")


if __name__ == "__main__":
    print("\nNote: This test avoids numpy/pandas dependencies")
    print("For full testing with your venv, activate it first:")
    print("  source /path/to/your/venv/bin/activate")
    print("  python test_basic_backtest_simple.py\n")
    
    asyncio.run(test_integration())
    asyncio.run(test_risk_limits())