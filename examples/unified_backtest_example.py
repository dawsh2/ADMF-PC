#!/usr/bin/env python3
"""
Example of using the unified backtest engine from the execution module.

This demonstrates:
1. How backtesting now lives in the execution module
2. Integration with risk module for position management
3. Event-driven architecture
4. Decimal precision for financial calculations
"""

from datetime import datetime
from decimal import Decimal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import from execution module (backtest now lives here)
from src.execution import UnifiedBacktestEngine, BacktestConfig
from src.risk.protocols import Signal, SignalType, OrderSide
from src.data.protocols import DataLoader
import pandas as pd


class SimpleMovingAverageCrossStrategy:
    """Simple MA crossover strategy for demonstration."""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.positions = {}
        self.data_buffer = {}
        
    def initialize(self, context):
        """Initialize strategy with context."""
        self.symbols = context.get('symbols', [])
        for symbol in self.symbols:
            self.data_buffer[symbol] = []
            self.positions[symbol] = 0
    
    def generate_signals(self, market_data):
        """Generate signals based on MA crossover."""
        signals = []
        timestamp = market_data['timestamp']
        
        for symbol in self.symbols:
            if symbol in market_data:
                # Add to buffer
                price = market_data[symbol]['close']
                self.data_buffer[symbol].append(price)
                
                # Keep buffer size limited
                if len(self.data_buffer[symbol]) > self.slow_period:
                    self.data_buffer[symbol].pop(0)
                
                # Check if we have enough data
                if len(self.data_buffer[symbol]) >= self.slow_period:
                    # Calculate moving averages
                    fast_ma = sum(self.data_buffer[symbol][-self.fast_period:]) / self.fast_period
                    slow_ma = sum(self.data_buffer[symbol]) / len(self.data_buffer[symbol])
                    
                    # Generate signals
                    if fast_ma > slow_ma and self.positions[symbol] <= 0:
                        # Buy signal
                        signal = Signal(
                            signal_id=f"BUY_{symbol}_{timestamp}",
                            strategy_id="ma_cross",
                            symbol=symbol,
                            signal_type=SignalType.ENTRY,
                            side=OrderSide.BUY,
                            strength=Decimal("0.8"),
                            timestamp=timestamp,
                            metadata={
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'reason': 'golden_cross'
                            }
                        )
                        signals.append(signal)
                        self.positions[symbol] = 1
                        
                    elif fast_ma < slow_ma and self.positions[symbol] > 0:
                        # Sell signal
                        signal = Signal(
                            signal_id=f"SELL_{symbol}_{timestamp}",
                            strategy_id="ma_cross",
                            symbol=symbol,
                            signal_type=SignalType.EXIT,
                            side=OrderSide.SELL,
                            strength=Decimal("1.0"),
                            timestamp=timestamp,
                            metadata={
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'reason': 'death_cross'
                            }
                        )
                        signals.append(signal)
                        self.positions[symbol] = 0
        
        return signals


class MockDataLoader:
    """Mock data loader for example."""
    
    def load(self, symbol, start, end, frequency):
        """Generate mock price data."""
        import numpy as np
        
        # Generate daily timestamps
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Generate random walk prices
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, 0.02, len(date_range))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        df = pd.DataFrame(data, index=date_range)
        return df


def main():
    """Run unified backtest example."""
    print("=== Unified Backtest Engine Example ===\n")
    
    # 1. Configure backtest
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("100000"),
        symbols=["AAPL", "GOOGL", "MSFT"],
        frequency="1d",
        commission=Decimal("0.001"),  # 0.1%
        slippage=Decimal("0.0005")     # 0.05%
    )
    
    print("Backtest Configuration:")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Initial Capital: ${config.initial_capital:,}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Commission: {config.commission:.2%}")
    print(f"  Slippage: {config.slippage:.2%}")
    print()
    
    # 2. Create strategy
    strategy = SimpleMovingAverageCrossStrategy(fast_period=20, slow_period=50)
    
    # 3. Create data loader
    data_loader = MockDataLoader()
    
    # 4. Create and run backtest engine
    print("Running backtest...")
    engine = UnifiedBacktestEngine(config)
    
    # Optional: Add custom risk limits
    from src.risk.risk_limits import MaxDrawdownLimit, MaxPositionLimit
    engine.risk_portfolio.add_risk_limit(
        MaxDrawdownLimit(
            max_drawdown_pct=Decimal("20"),  # 20% max drawdown
            reduce_at_pct=Decimal("15")       # Start reducing at 15%
        )
    )
    engine.risk_portfolio.add_risk_limit(
        MaxPositionLimit(max_position_value=Decimal("50000"))  # Max $50k per position
    )
    
    # Run the backtest
    results = engine.run(strategy, data_loader)
    
    # 5. Display results
    print("\n=== Backtest Results ===")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annualized Return: {results.annualized_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Volatility: {results.volatility:.2%}")
    print()
    
    print("Trade Statistics:")
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Average Win: ${results.avg_win:.2f}")
    print(f"Average Loss: ${results.avg_loss:.2f}")
    print()
    
    print("Final Portfolio:")
    print(f"Initial Capital: ${results.initial_capital:,.2f}")
    print(f"Final Equity: ${results.final_equity:,.2f}")
    print(f"Total P&L: ${results.final_equity - results.initial_capital:,.2f}")
    
    # 6. Show position details
    if results.positions:
        print("\nOpen Positions:")
        for position in results.positions:
            print(f"  {position['symbol']}: {position['quantity']} shares @ ${position['unrealized_pnl']:,.2f} P&L")
    
    # 7. Show recent trades
    if results.trades:
        print(f"\nRecent Trades (last 5):")
        for trade in results.trades[-5:]:
            print(f"  {trade['symbol']}: {trade['quantity']} shares, "
                  f"Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}, "
                  f"P&L: ${trade['pnl']:.2f}")
    
    # 8. Risk metrics
    print(f"\nRisk Metrics:")
    print(f"95% VaR: ${results.var_95:.2f}")
    print(f"95% CVaR: ${results.cvar_95:.2f}")
    
    print("\n✅ Backtest completed successfully!")
    print("\nKey advantages of the unified engine:")
    print("- Single source of truth for positions (Risk module)")
    print("- Realistic order execution through ExecutionEngine")
    print("- Decimal precision for accurate calculations")
    print("- Full integration with risk limits and position sizing")
    print("- Event-driven architecture for monitoring")


if __name__ == "__main__":
    main()