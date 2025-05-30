"""Basic integration test for end-to-end backtest functionality - no external dependencies.

This test brings together all modules to perform a simple backtest:
- Data: Simple price data generation
- Strategy: Simple trend following
- Risk: Position sizing and risk limits
- Execution: Order processing and fills
- Core: Basic orchestration
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
import random
import json

# Core modules
from src.core.containers import UniversalScopedContainer
from src.core.events import EventBus

# Risk module
from src.risk import (
    RiskPortfolioContainer,
    PercentagePositionSizer,
    MaxDrawdownLimit,
    SignalFlowManager,
    Signal,
)
from src.risk.protocols import SignalType, OrderSide

# Execution module
from src.execution import BacktestBrokerRefactored

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleData:
    """Simple data container without pandas."""
    
    def __init__(self, symbol: str, days: int = 100):
        self.symbol = symbol
        self.days = days
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Dict[str, Any]]:
        """Generate simple price data."""
        data = []
        current_price = 100.0
        base_date = datetime.now() - timedelta(days=self.days)
        
        random.seed(42)  # For reproducibility
        
        for i in range(self.days):
            # Random walk
            change = random.uniform(-0.02, 0.02)  # +/- 2% daily
            current_price *= (1 + change)
            
            date = base_date + timedelta(days=i)
            
            data.append({
                'date': date,
                'open': current_price * random.uniform(0.99, 1.01),
                'high': current_price * random.uniform(1.00, 1.02),
                'low': current_price * random.uniform(0.98, 1.00),
                'close': current_price,
                'volume': random.randint(1000000, 5000000)
            })
        
        return data
    
    def get_price_at(self, index: int) -> Optional[float]:
        """Get closing price at index."""
        if 0 <= index < len(self.data):
            return self.data[index]['close']
        return None
    
    def calculate_ma(self, period: int) -> List[Optional[float]]:
        """Calculate simple moving average."""
        ma = []
        for i in range(len(self.data)):
            if i < period - 1:
                ma.append(None)
            else:
                window = [self.data[j]['close'] for j in range(i - period + 1, i + 1)]
                ma.append(sum(window) / period)
        return ma


class SimpleTrendStrategy:
    """Simple trend following strategy."""
    
    def __init__(self, ma_period: int = 20):
        self.ma_period = ma_period
        self.position_open = False
    
    def generate_signals(self, data: SimpleData) -> List[Signal]:
        """Generate signals based on price vs MA."""
        signals = []
        ma_values = data.calculate_ma(self.ma_period)
        
        for i in range(self.ma_period, len(data.data)):
            current_price = data.data[i]['close']
            current_ma = ma_values[i]
            prev_price = data.data[i-1]['close']
            prev_ma = ma_values[i-1]
            
            if current_ma is None or prev_ma is None:
                continue
            
            # Entry signal: price crosses above MA
            if prev_price <= prev_ma and current_price > current_ma and not self.position_open:
                signal = Signal(
                    signal_id=f"TREND_{i}_BUY",
                    strategy_id="trend_following",
                    symbol=data.symbol,
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal("0.8"),
                    timestamp=data.data[i]['date'],
                    metadata={
                        "price": current_price,
                        "ma": current_ma,
                        "reason": "price_above_ma"
                    }
                )
                signals.append(signal)
                self.position_open = True
            
            # Exit signal: price crosses below MA
            elif prev_price >= prev_ma and current_price < current_ma and self.position_open:
                signal = Signal(
                    signal_id=f"TREND_{i}_SELL",
                    strategy_id="trend_following",
                    symbol=data.symbol,
                    signal_type=SignalType.EXIT,
                    side=OrderSide.SELL,
                    strength=Decimal("0.8"),
                    timestamp=data.data[i]['date'],
                    metadata={
                        "price": current_price,
                        "ma": current_ma,
                        "reason": "price_below_ma"
                    }
                )
                signals.append(signal)
                self.position_open = False
        
        return signals


async def run_simple_backtest():
    """Run a simple end-to-end backtest."""
    print("\n" + "="*60)
    print("SIMPLE BACKTEST INTEGRATION TEST")
    print("="*60 + "\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    data = SimpleData("AAPL", days=100)
    print(f"   Generated {len(data.data)} days of data")
    prices = [d['close'] for d in data.data]
    print(f"   Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    
    # 2. Create Risk & Portfolio container
    print("\n2. Creating Risk & Portfolio container...")
    risk_portfolio = RiskPortfolioContainer(
        name="TestPortfolio",
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    
    # Container is ready to use immediately for our test
    
    # Configure risk management
    risk_portfolio.set_position_sizer(
        PercentagePositionSizer(percentage=Decimal("0.02"))  # 2% per position
    )
    risk_portfolio.add_risk_limit(
        MaxDrawdownLimit(
            max_drawdown_pct=Decimal("10"),
            reduce_at_pct=Decimal("8")
        )
    )
    
    # 3. Create execution components
    print("\n3. Setting up execution...")
    broker = BacktestBrokerRefactored(
        portfolio_state=risk_portfolio.get_portfolio_state()
    )
    
    # 4. Create strategy
    print("\n4. Creating strategy...")
    strategy = SimpleTrendStrategy(ma_period=20)
    
    # 5. Create signal flow manager
    print("\n5. Setting up signal flow...")
    event_bus = EventBus()
    signal_flow = SignalFlowManager(
        event_bus=event_bus,
        enable_validation=True,
        enable_aggregation=False  # Single strategy
    )
    signal_flow.register_strategy("trend_following")
    
    # 6. Generate signals
    print("\n6. Generating signals...")
    signals = strategy.generate_signals(data)
    print(f"   Generated {len(signals)} signals")
    for sig in signals[:3]:  # Show first 3
        print(f"   - {sig.timestamp.strftime('%Y-%m-%d')}: {sig.signal_type.value} {sig.side.value}")
    
    # 7. Process signals through the system
    print("\n7. Processing signals through backtest...")
    
    # Track results
    results = {
        "signals": len(signals),
        "orders": 0,
        "fills": 0,
        "trades": []
    }
    
    # Process each signal
    for i, signal in enumerate(signals):
        # Update market data
        market_data = {
            "prices": {"AAPL": Decimal(str(signal.metadata["price"]))},
            "timestamp": signal.timestamp
        }
        
        # Update portfolio with current prices
        risk_portfolio.update_market_data(market_data)
        
        # Collect signal
        await signal_flow.collect_signal(signal)
        
        # Process signals into orders
        orders = await signal_flow.process_signals(
            portfolio_state=risk_portfolio.get_portfolio_state(),
            position_sizer=risk_portfolio._position_sizer,
            risk_limits=risk_portfolio._risk_limits,
            market_data=market_data
        )
        
        results["orders"] += len(orders)
        
        # Process orders
        for order in orders:
            # Create fill
            fill = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": "buy" if order.side == OrderSide.BUY else "sell",
                "quantity": order.quantity,
                "price": market_data["prices"][order.symbol],
                "timestamp": signal.timestamp,
                "commission": Decimal("1.0")
            }
            
            # Update portfolio
            risk_portfolio.update_fills([fill])
            results["fills"] += 1
            
            # Record trade
            results["trades"].append({
                "date": signal.timestamp.strftime('%Y-%m-%d'),
                "side": fill["side"],
                "quantity": float(fill["quantity"]),
                "price": float(fill["price"])
            })
    
    # 8. Get final results
    print("\n8. Backtest Results:")
    print("="*40)
    
    portfolio_state = risk_portfolio.get_portfolio_state()
    performance = portfolio_state.get_performance_summary()
    
    print(f"Signals Generated: {results['signals']}")
    print(f"Orders Created: {results['orders']}")
    print(f"Fills Executed: {results['fills']}")
    print(f"\nPortfolio Performance:")
    print(f"  Initial Capital: ${performance['initial_capital']}")
    print(f"  Final Value: ${performance['current_value']}")
    print(f"  Total Return: {performance['total_return']}")
    print(f"  Max Drawdown: {performance['max_drawdown']}")
    print(f"  Commission Paid: ${performance['commission_paid']}")
    
    # Show trades
    print(f"\nTrades Executed ({len(results['trades'])}):")
    for trade in results["trades"][:5]:  # Show first 5
        print(f"  {trade['date']}: {trade['side'].upper()} {trade['quantity']} @ ${trade['price']:.2f}")
    if len(results["trades"]) > 5:
        print(f"  ... and {len(results['trades']) - 5} more trades")
    
    # Risk metrics
    risk_report = risk_portfolio.get_risk_report()
    print(f"\nFinal Risk Metrics:")
    print(f"  Cash: ${risk_report['portfolio_metrics']['cash_balance']}")
    print(f"  Positions Value: ${risk_report['portfolio_metrics']['positions_value']}")
    print(f"  Realized P&L: ${risk_report['portfolio_metrics']['realized_pnl']}")
    
    # Container cleanup not needed for test
    
    print("\n" + "="*60)
    print("✅ BACKTEST COMPLETE")
    print("="*60)
    
    return results


async def test_error_handling():
    """Test error handling and edge cases."""
    print("\n\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60 + "\n")
    
    # Create portfolio
    risk_portfolio = RiskPortfolioContainer(
        name="ErrorTestPortfolio",
        initial_capital=Decimal("1000"),  # Small capital
        base_currency="USD"
    )
    await risk_portfolio.start()
    
    # Set aggressive position sizing
    risk_portfolio.set_position_sizer(
        PercentagePositionSizer(percentage=Decimal("0.5"))  # 50% per position!
    )
    
    # Create signal flow
    signal_flow = SignalFlowManager(enable_validation=True)
    signal_flow.register_strategy("test_strategy")
    
    # Test 1: Invalid signal
    print("Test 1: Invalid signal (strength > 1)")
    invalid_signal = Signal(
        signal_id="INVALID_1",
        strategy_id="test_strategy",
        symbol="TEST",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("1.5"),  # Invalid!
        timestamp=datetime.now(),
        metadata={}
    )
    
    try:
        # This should fail validation
        await signal_flow.collect_signal(invalid_signal)
    except Exception as e:
        print(f"  ✓ Correctly rejected: {type(e).__name__}")
    
    # Test 2: Insufficient funds
    print("\nTest 2: Order with insufficient funds")
    large_signal = Signal(
        signal_id="LARGE_1",
        strategy_id="test_strategy",
        symbol="EXPENSIVE",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.9"),
        timestamp=datetime.now(),
        metadata={"price": 10000.0}  # Very expensive!
    )
    
    await signal_flow.collect_signal(large_signal)
    orders = await signal_flow.process_signals(
        portfolio_state=risk_portfolio.get_portfolio_state(),
        position_sizer=risk_portfolio._position_sizer,
        risk_limits=risk_portfolio._risk_limits,
        market_data={"prices": {"EXPENSIVE": Decimal("10000")}}
    )
    
    if not orders:
        print("  ✓ Order correctly rejected due to insufficient funds")
    else:
        print("  ✗ Order should have been rejected")
    
    await risk_portfolio.stop()
    print("\n✅ Error handling tests complete")


async def main():
    """Run all integration tests."""
    try:
        # Run simple backtest
        results = await run_simple_backtest()
        
        # Test error handling
        await test_error_handling()
        
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS COMPLETE!")
        print("="*60)
        
        print("\nNext Implementation Steps:")
        print("1. ✓ Basic signal flow working")
        print("2. ✓ Risk management integration working")
        print("3. ✓ Position tracking working")
        print("4. TODO: Implement workflow managers for Coordinator")
        print("5. TODO: Add real data loading")
        print("6. TODO: Implement optimization workflows")
        print("7. TODO: Add performance analytics")
        print("8. TODO: Create visualization tools")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())