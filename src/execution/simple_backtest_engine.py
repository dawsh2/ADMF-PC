"""Simple backtest engine for testing with synthetic data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Simple trade representation."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float = 0.0
    status: str = 'open'  # 'open' or 'closed'


@dataclass
class BacktestResult:
    """Results from a backtest."""
    trades: List[Trade]
    final_equity: float
    total_return: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: pd.Series


class SimpleBacktestEngine:
    """Simple backtest engine for YAML-driven execution."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.data_config = config.get('data', {})
        self.portfolio_config = config.get('portfolio', {})
        self.strategies = config.get('strategies', [])
        
        # Portfolio state
        self.initial_capital = self.portfolio_config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.position = 0
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # Data
        self.data: Optional[pd.DataFrame] = None
        self.current_idx = 0
    
    def load_data(self, max_bars: Optional[int] = None, dataset: Optional[str] = None) -> pd.DataFrame:
        """Load data using SimpleDataLoader."""
        from ..data.simple_loader import SimpleDataLoader
        
        # Create data loader
        loader = SimpleDataLoader(self.data_config)
        
        # Get dataset (train, test, or full)
        dataset = dataset or self.data_config.get('dataset', 'full')
        self.data = loader.get_dataset(dataset, max_bars)
        
        return self.data
    
    def run_backtest(self) -> BacktestResult:
        """Run the backtest."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\nRunning backtest...")
        
        # For each bar
        for idx in range(len(self.data)):
            self.current_idx = idx
            bar = self.data.iloc[idx]
            
            # Generate signals from strategies
            signal = self._get_signal(bar)
            
            # Execute trades based on signal
            if signal == 'buy' and self.position == 0:
                self._open_position(bar, 'long')
            elif signal == 'sell' and self.position > 0:
                self._close_position(bar)
            
            # Update equity
            self._update_equity(bar)
        
        # Close any open positions at end
        if self.position > 0:
            self._close_position(self.data.iloc[-1])
        
        # Calculate results
        return self._calculate_results()
    
    def _get_signal(self, bar) -> Optional[str]:
        """Get trading signal from strategies."""
        # For now, implement simple threshold strategy
        for strategy in self.strategies:
            if strategy['type'] == 'price_threshold':
                params = strategy.get('parameters', {})
                buy_threshold = params.get('buy_threshold', 90)
                sell_threshold = params.get('sell_threshold', 100)
                
                if bar['close'] <= buy_threshold and self.position == 0:
                    return 'buy'
                elif bar['close'] >= sell_threshold and self.position > 0:
                    return 'sell'
        
        return None
    
    def _open_position(self, bar, side='long'):
        """Open a position."""
        # Calculate position size
        position_size = self.cash  # All-in for simple test
        shares = int(position_size / bar['close'])
        
        if shares > 0:
            trade = Trade(
                entry_time=bar.name,
                entry_price=bar['close'],
                quantity=shares,
                side=side,
                exit_time=None,
                exit_price=None
            )
            self.trades.append(trade)
            
            # Update portfolio
            self.position = shares
            self.cash -= shares * bar['close']
            
            print(f"OPEN {side.upper()}: {shares} shares @ ${bar['close']:.2f} at {bar.name}")
    
    def _close_position(self, bar):
        """Close current position."""
        if self.position == 0 or not self.trades:
            return
        
        # Get open trade
        trade = next((t for t in reversed(self.trades) if t.status == 'open'), None)
        if not trade:
            return
        
        # Close trade
        trade.exit_time = bar.name
        trade.exit_price = bar['close']
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        trade.status = 'closed'
        
        # Update portfolio
        self.cash += self.position * bar['close']
        self.position = 0
        
        print(f"CLOSE: {trade.quantity} shares @ ${bar['close']:.2f} at {bar.name}")
        print(f"  P&L: ${trade.pnl:.2f} ({trade.pnl/self.initial_capital*100:.1f}%)")
    
    def _update_equity(self, bar):
        """Update current equity value."""
        if self.position > 0:
            self.equity = self.cash + self.position * bar['close']
        else:
            self.equity = self.cash
        
        self.equity_curve.append({
            'timestamp': bar.name,
            'equity': self.equity,
            'cash': self.cash,
            'position_value': self.position * bar['close'] if self.position > 0 else 0
        })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        # Filter closed trades
        closed_trades = [t for t in self.trades if t.status == 'closed']
        
        # Basic metrics
        num_trades = len(closed_trades)
        if num_trades == 0:
            print("\nNo trades executed!")
            return BacktestResult(
                trades=self.trades,
                final_equity=self.equity,
                total_return=0.0,
                num_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                equity_curve=pd.DataFrame(self.equity_curve).set_index('timestamp')['equity']
            )
        
        # Calculate metrics
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Total return
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        equity_series = equity_df['equity']
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified)
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            trades=self.trades,
            final_equity=self.equity,
            total_return=total_return,
            num_trades=num_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            equity_curve=equity_series
        )


def run_simple_backtest(config_path: str, max_bars: Optional[int] = None) -> BacktestResult:
    """Run a simple backtest from YAML config."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create engine
    engine = SimpleBacktestEngine(config)
    
    # Load data
    engine.load_data(max_bars=max_bars)
    
    # Run backtest
    result = engine.run_backtest()
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:    ${engine.initial_capital:,.2f}")
    print(f"Final Equity:       ${result.final_equity:,.2f}")
    print(f"Total Return:       {result.total_return:.2%}")
    print(f"Number of Trades:   {result.num_trades}")
    print(f"Win Rate:           {result.win_rate:.1%}")
    print(f"Average Win:        ${result.avg_win:,.2f}")
    print(f"Average Loss:       ${result.avg_loss:,.2f}")
    print(f"Max Drawdown:       {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    
    return result