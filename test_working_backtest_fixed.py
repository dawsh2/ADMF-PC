
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import random
import logging

logging.basicConfig(level=logging.INFO)

from src.risk import RiskPortfolioContainer, PercentagePositionSizer, MaxPositionLimit
from src.risk.protocols import Signal, SignalType, OrderSide
from src.risk.signal_flow import SignalFlowManager

async def simple_test():
    print("\nSimple Integration Test")
    print("=" * 40)
    
    # Create portfolio
    portfolio = RiskPortfolioContainer(
        name="Test",
        initial_capital=Decimal("100000")
    )
    
    # Direct assignment to avoid logging issues
    portfolio._position_sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
    portfolio._risk_limits = [MaxPositionLimit(max_position_value=Decimal("50000"))]
    
    print(f"Initial capital: $100,000")
    print(f"Position sizing: 2%")
    
    # Create signals
    signals = [
        Signal(
            signal_id="BUY1",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={"price": 150.0}
        ),
        Signal(
            signal_id="SELL1",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            side=OrderSide.SELL,
            strength=Decimal("0.8"),
            timestamp=datetime.now() + timedelta(days=1),
            metadata={"price": 155.0}
        )
    ]
    
    # Process signals
    flow = SignalFlowManager(enable_validation=False)
    flow.register_strategy("test")
    
    for i, signal in enumerate(signals):
        print(f"\nProcessing signal {i+1}: {signal.signal_type.value}")
        
        await flow.collect_signal(signal)
        
        market_data = {
            "prices": {"AAPL": Decimal(str(signal.metadata["price"]))},
            "timestamp": signal.timestamp
        }
        
        portfolio.update_market_data(market_data)
        
        orders = await flow.process_signals(
            portfolio_state=portfolio.get_portfolio_state(),
            position_sizer=portfolio._position_sizer,
            risk_limits=portfolio._risk_limits,
            market_data=market_data
        )
        
        if orders:
            order = orders[0]
            print(f"  Order: {order.side.value} {order.quantity} shares")
            
            # Simulate fill
            fill = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": "buy" if order.side == OrderSide.BUY else "sell",
                "quantity": order.quantity,
                "price": market_data["prices"][order.symbol],
                "timestamp": signal.timestamp,
                "commission": Decimal("1.0")
            }
            
            portfolio.update_fills([fill])
            print(f"  Filled @ ${fill['price']}")
    
    # Results
    state = portfolio.get_portfolio_state()
    metrics = state.get_risk_metrics()
    
    print(f"\nFinal Results:")
    print(f"  Total value: ${metrics.total_value}")
    print(f"  Cash: ${metrics.cash_balance}")
    print(f"  Realized P&L: ${metrics.realized_pnl}")
    
    return metrics.total_value > 100000  # Should have profit

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    print(f"\n{'✅ PASSED' if success else '❌ FAILED'}")
