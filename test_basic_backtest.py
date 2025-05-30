"""Basic integration test for end-to-end backtest functionality.

This test brings together all modules to perform a simple backtest:
- Data: Load price data
- Strategy: Simple moving average crossover
- Risk: Position sizing and risk limits
- Execution: Order processing and fills
- Core: Coordinator orchestration
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Core modules
from src.core.coordinator import Coordinator
from src.core.containers import UniversalScopedContainer, Bootstrap
from src.core.events import EventBus
from src.core.components import ComponentFactory

# Data module
from src.data import DataLoader, DataHandler

# Strategy module - we'll create a simple one inline
from src.strategy.protocols import StrategyProtocol

# Risk module
from src.risk import (
    RiskPortfolioContainer,
    PercentagePositionSizer,
    MaxDrawdownLimit,
    SignalFlowManager,
    Signal,
    SignalType,
    OrderSide,
)

# Execution module
from src.execution import BacktestBrokerRefactored, DefaultExecutionEngine

# Logging
from src.core.logging.structured import get_logger

logger = get_logger(__name__)


class SimpleMovingAverageCrossover:
    """Simple MA crossover strategy for testing."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position = None
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate signals based on MA crossover."""
        signals = []
        
        # Calculate moving averages
        data['MA_fast'] = data['close'].rolling(self.fast_period).mean()
        data['MA_slow'] = data['close'].rolling(self.slow_period).mean()
        
        # Skip warmup period
        data = data.iloc[self.slow_period:]
        
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            curr_row = data.iloc[i]
            
            # Check for crossover
            prev_fast = prev_row['MA_fast']
            prev_slow = prev_row['MA_slow']
            curr_fast = curr_row['MA_fast']
            curr_slow = curr_row['MA_slow']
            
            # Bullish crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signal = Signal(
                    signal_id=f"SMA_{i}_BUY",
                    strategy_id="sma_crossover",
                    symbol=data.index.name or "TEST",
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal("0.8"),
                    timestamp=curr_row.name,
                    metadata={
                        "fast_ma": float(curr_fast),
                        "slow_ma": float(curr_slow),
                        "price": float(curr_row['close'])
                    }
                )
                signals.append(signal)
                self.position = "long"
            
            # Bearish crossover
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                if self.position == "long":
                    # Exit long
                    signal = Signal(
                        signal_id=f"SMA_{i}_EXIT",
                        strategy_id="sma_crossover",
                        symbol=data.index.name or "TEST",
                        signal_type=SignalType.EXIT,
                        side=OrderSide.SELL,
                        strength=Decimal("0.8"),
                        timestamp=curr_row.name,
                        metadata={
                            "fast_ma": float(curr_fast),
                            "slow_ma": float(curr_slow),
                            "price": float(curr_row['close'])
                        }
                    )
                    signals.append(signal)
                    self.position = None
        
        return signals


def generate_sample_data(symbol: str = "TEST", days: int = 100) -> pd.DataFrame:
    """Generate sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # 0.05% daily return, 2% volatility
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.005, 0.005, days)),
        'high': price * (1 + np.random.uniform(0, 0.01, days)),
        'low': price * (1 + np.random.uniform(-0.01, 0, days)),
        'close': price,
        'volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    df.index.name = symbol
    return df


async def run_basic_backtest():
    """Run a basic end-to-end backtest."""
    print("\n" + "="*60)
    print("BASIC BACKTEST INTEGRATION TEST")
    print("="*60 + "\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    data = generate_sample_data("AAPL", days=100)
    print(f"   Generated {len(data)} days of data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 2. Create core infrastructure
    print("\n2. Setting up core infrastructure...")
    event_bus = EventBus()
    
    # 3. Create Risk & Portfolio container
    print("\n3. Creating Risk & Portfolio container...")
    risk_portfolio = RiskPortfolioContainer(
        name="TestPortfolio",
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    
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
    
    # 4. Create execution components
    print("\n4. Setting up execution...")
    broker = BacktestBrokerRefactored(
        portfolio_state=risk_portfolio.get_portfolio_state()
    )
    
    # 5. Create strategy
    print("\n5. Creating strategy...")
    strategy = SimpleMovingAverageCrossover(fast_period=10, slow_period=20)
    
    # 6. Create signal flow manager
    print("\n6. Setting up signal flow...")
    signal_flow = SignalFlowManager(
        event_bus=event_bus,
        enable_validation=True,
        enable_aggregation=False  # Single strategy
    )
    signal_flow.register_strategy("sma_crossover")
    
    # 7. Generate signals
    print("\n7. Generating signals...")
    signals = strategy.generate_signals(data)
    print(f"   Generated {len(signals)} signals")
    
    # 8. Process signals through the system
    print("\n8. Processing signals through backtest...")
    
    # Track results
    results = {
        "signals": len(signals),
        "orders": 0,
        "fills": 0,
        "positions": [],
        "pnl": []
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
        
        # Process orders through broker
        for order in orders:
            # Submit order
            order_dict = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": "BUY" if order.side == OrderSide.BUY else "SELL",
                "order_type": "MARKET",
                "quantity": float(order.quantity),
                "price": None
            }
            
            # Create fill (simplified - in real system this would go through market sim)
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
        
        # Track position
        position = risk_portfolio.get_portfolio_state().get_position("AAPL")
        if position:
            results["positions"].append({
                "timestamp": signal.timestamp,
                "quantity": float(position.quantity),
                "avg_price": float(position.average_price),
                "current_price": float(position.current_price),
                "unrealized_pnl": float(position.unrealized_pnl)
            })
    
    # 9. Get final results
    print("\n9. Backtest Results:")
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
    print(f"  Positions Held: {performance['positions_count']}")
    
    # Show risk report
    risk_report = risk_portfolio.get_risk_report()
    print(f"\nRisk Metrics:")
    print(f"  Cash Balance: ${risk_report['portfolio_metrics']['cash_balance']}")
    print(f"  Positions Value: ${risk_report['portfolio_metrics']['positions_value']}")
    print(f"  Unrealized P&L: ${risk_report['portfolio_metrics']['unrealized_pnl']}")
    print(f"  Realized P&L: ${risk_report['portfolio_metrics']['realized_pnl']}")
    
    # Signal flow statistics
    flow_stats = signal_flow.get_statistics()
    print(f"\nSignal Flow Statistics:")
    print(f"  Signals Received: {flow_stats['total_signals_received']}")
    print(f"  Orders Generated: {flow_stats['total_orders_generated']}")
    print(f"  Approval Rate: {flow_stats['approval_rate']}")
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    
    return results


async def test_with_coordinator():
    """Test using the Coordinator for orchestration."""
    print("\n\n" + "="*60)
    print("TESTING WITH COORDINATOR")
    print("="*60 + "\n")
    
    # Create configuration for coordinator
    config = {
        "workflow_type": "backtest",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "source": "sample"  # Use sample data generator
        },
        "strategies": [{
            "name": "sma_crossover",
            "class": "SimpleMovingAverageCrossover",
            "params": {
                "fast_period": 10,
                "slow_period": 20
            }
        }],
        "risk": {
            "initial_capital": 100000,
            "position_sizing": {
                "type": "percentage",
                "percentage": 2.0
            },
            "risk_limits": [{
                "type": "max_drawdown",
                "max_drawdown_pct": 10,
                "reduce_at_pct": 8
            }]
        },
        "execution": {
            "broker": "backtest",
            "commission": 1.0,
            "slippage": 0.0
        }
    }
    
    # Create coordinator
    print("Creating Coordinator...")
    coordinator = Coordinator()
    
    # Note: The coordinator would need workflow managers implemented
    # For now, we'll note this as the next step
    print("\nNOTE: Full coordinator integration requires workflow managers")
    print("This would be the next implementation step")


async def main():
    """Run all integration tests."""
    # Run basic backtest
    results = await run_basic_backtest()
    
    # Test with coordinator (placeholder)
    await test_with_coordinator()
    
    print("\n✅ Basic integration test complete!")
    print("\nNext steps:")
    print("1. Implement workflow managers for Coordinator")
    print("2. Add market simulation for realistic fills")
    print("3. Implement actual strategy classes")
    print("4. Add data loading from files/APIs")
    print("5. Create performance reporting")


if __name__ == "__main__":
    asyncio.run(main())