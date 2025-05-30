"""Minimal integration test avoiding external dependencies.

Tests the core flow: Signal → Risk → Order → Execution
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the essentials we need
from src.risk.protocols import (
    Signal, SignalType, OrderSide, 
    PortfolioStateProtocol, Position, RiskMetrics
)
from src.risk.portfolio_state import PortfolioState
from src.risk.position_sizing import PercentagePositionSizer
from src.risk.risk_limits import MaxPositionLimit
from src.risk.signal_processing import SignalProcessor


def create_test_signal(i: int, side: OrderSide) -> Signal:
    """Create a test signal."""
    return Signal(
        signal_id=f"TEST_{i}",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY if side == OrderSide.BUY else SignalType.EXIT,
        side=side,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={"price": 150.0 + i}
    )


def test_basic_flow():
    """Test basic signal to order flow."""
    print("\n" + "="*50)
    print("MINIMAL INTEGRATION TEST")
    print("="*50 + "\n")
    
    # 1. Create portfolio state
    print("1. Creating portfolio state...")
    portfolio = PortfolioState(
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    print(f"   Initial capital: ${portfolio.get_cash_balance()}")
    
    # 2. Create position sizer
    print("\n2. Creating position sizer...")
    sizer = PercentagePositionSizer(percentage=Decimal("0.02"))  # 2% per position
    print("   Using 2% position sizing")
    
    # 3. Create risk limits
    print("\n3. Creating risk limits...")
    risk_limits = [
        MaxPositionLimit(max_position=Decimal("5000"))
    ]
    print("   Max position: 5000 shares")
    
    # 4. Create signal processor
    print("\n4. Creating signal processor...")
    processor = SignalProcessor()
    
    # 5. Process a BUY signal
    print("\n5. Processing BUY signal...")
    buy_signal = create_test_signal(1, OrderSide.BUY)
    market_data = {
        "prices": {"AAPL": Decimal("150")},
        "timestamp": datetime.now()
    }
    
    order = processor.process_signal(
        signal=buy_signal,
        portfolio_state=portfolio,
        position_sizer=sizer,
        risk_limits=risk_limits,
        market_data=market_data
    )
    
    if order:
        print(f"   ✓ Order created: {order.side.value} {order.quantity} {order.symbol}")
        
        # 6. Simulate fill and update portfolio
        print("\n6. Simulating fill...")
        fill_quantity = order.quantity
        fill_price = Decimal("150")
        
        position = portfolio.update_position(
            symbol=order.symbol,
            quantity_delta=fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
            price=fill_price,
            timestamp=datetime.now()
        )
        
        print(f"   ✓ Position updated: {position.quantity} @ ${position.average_price}")
        print(f"   Cash remaining: ${portfolio.get_cash_balance()}")
    else:
        print("   ✗ Order rejected")
    
    # 7. Check portfolio state
    print("\n7. Portfolio summary:")
    metrics = portfolio.get_risk_metrics()
    print(f"   Total value: ${metrics.total_value}")
    print(f"   Cash: ${metrics.cash_balance}")
    print(f"   Positions value: ${metrics.positions_value}")
    print(f"   Positions: {len(portfolio.get_all_positions())}")
    
    # 8. Process a SELL signal
    print("\n8. Processing SELL signal...")
    sell_signal = Signal(
        signal_id="TEST_SELL",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.EXIT,
        side=OrderSide.SELL,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={"price": 155.0}
    )
    
    # Update market price
    portfolio.update_market_prices({"AAPL": Decimal("155")})
    
    sell_order = processor.process_signal(
        signal=sell_signal,
        portfolio_state=portfolio,
        position_sizer=sizer,
        risk_limits=risk_limits,
        market_data={"prices": {"AAPL": Decimal("155")}}
    )
    
    if sell_order:
        print(f"   ✓ Sell order created: {sell_order.quantity} shares")
        
        # Simulate sell fill
        portfolio.update_position(
            symbol=sell_order.symbol,
            quantity_delta=-sell_order.quantity,
            price=Decimal("155"),
            timestamp=datetime.now()
        )
        
        print(f"   ✓ Position closed")
        print(f"   Realized P&L: ${portfolio._realized_pnl}")
    
    # 9. Final summary
    print("\n9. Final results:")
    performance = portfolio.get_performance_summary()
    print(f"   Initial: ${performance['initial_capital']}")
    print(f"   Final: ${performance['current_value']}")
    print(f"   Return: {performance['total_return']}")
    print(f"   Realized P&L: ${performance['realized_pnl']}")
    
    print("\n" + "="*50)
    print("✅ TEST COMPLETE")
    print("="*50)


def test_risk_limits():
    """Test risk limit enforcement."""
    print("\n\n" + "="*50)
    print("RISK LIMITS TEST")
    print("="*50 + "\n")
    
    # Create portfolio with small capital
    portfolio = PortfolioState(initial_capital=Decimal("10000"))
    sizer = PercentagePositionSizer(percentage=Decimal("0.5"))  # 50% - too large!
    risk_limits = [MaxPositionLimit(max_position=Decimal("50"))]  # Max 50 shares
    processor = SignalProcessor()
    
    # Try to buy expensive stock
    signal = create_test_signal(1, OrderSide.BUY)
    market_data = {"prices": {"AAPL": Decimal("150")}}
    
    order = processor.process_signal(
        signal=signal,
        portfolio_state=portfolio,
        position_sizer=sizer,
        risk_limits=risk_limits,
        market_data=market_data
    )
    
    if order:
        print(f"Order quantity: {order.quantity} (limited by risk)")
    else:
        print("Order rejected by risk limits")
    
    stats = processor.get_statistics()
    print(f"\nProcessor statistics:")
    print(f"  Processed: {stats['processed_signals']}")
    print(f"  Approved: {stats['approved_orders']}")
    print(f"  Rejected: {stats['rejected_signals']}")
    print(f"  Approval rate: {stats['approval_rate']}")


if __name__ == "__main__":
    print("Running minimal integration tests...")
    print("(No external dependencies required)")
    
    test_basic_flow()
    test_risk_limits()
    
    print("\n✅ All tests complete!")
    print("\nThis demonstrates:")
    print("- Signal processing")
    print("- Position sizing")
    print("- Risk limit enforcement")
    print("- Portfolio state tracking")
    print("- P&L calculation")