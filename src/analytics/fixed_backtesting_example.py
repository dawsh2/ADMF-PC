"""
Example of properly implemented backtesting without look-ahead bias.

This demonstrates:
1. Correct feature calculation with proper lags
2. Realistic signal execution timing
3. Proper transaction cost modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProperFeatureCalculator:
    """Calculate features with proper timing to avoid look-ahead bias."""
    
    def __init__(self):
        self.feature_history = {}
        
    def calculate_sma(self, prices: pd.Series, period: int, lag: int = 1) -> pd.Series:
        """
        Calculate SMA with proper lag.
        
        Args:
            prices: Price series
            period: SMA period
            lag: Number of bars to lag (default 1 = use previous bar's SMA)
        """
        # Calculate SMA normally
        sma = prices.rolling(window=period, min_periods=period).mean()
        
        # CRITICAL: Lag the result so we can't see current bar's SMA
        sma_lagged = sma.shift(lag)
        
        return sma_lagged
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14, lag: int = 1) -> pd.Series:
        """Calculate RSI with proper lag."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # CRITICAL: Lag the result
        rsi_lagged = rsi.shift(lag)
        
        return rsi_lagged
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0, lag: int = 1) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with proper lag."""
        sma = prices.rolling(window=period, min_periods=period).mean()
        std = prices.rolling(window=period, min_periods=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # CRITICAL: Lag all results
        return {
            'middle': sma.shift(lag),
            'upper': upper_band.shift(lag),
            'lower': lower_band.shift(lag)
        }


class ProperSignalGenerator:
    """Generate signals with realistic timing constraints."""
    
    def __init__(self, entry_delay: int = 1):
        """
        Args:
            entry_delay: Bars to wait before entering position (default 1)
        """
        self.entry_delay = entry_delay
        
    def generate_ma_crossover_signals(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """
        Generate MA crossover signals with proper timing.
        
        Note: The MAs should already be lagged from the feature calculator.
        """
        signals = pd.Series(0, index=fast_ma.index)
        
        # Generate raw signals
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        # Additional delay for execution (signal seen at bar close, execute at next open)
        signals_executable = signals.shift(self.entry_delay)
        
        return signals_executable
    
    def generate_mean_reversion_signals(self, prices: pd.Series, 
                                      bollinger: Dict[str, pd.Series]) -> pd.Series:
        """Generate mean reversion signals with Bollinger Bands."""
        signals = pd.Series(0, index=prices.index)
        
        # Note: Bollinger bands already lagged, comparing with current price is OK
        # Buy when price touches lower band
        signals[prices < bollinger['lower']] = 1
        
        # Sell when price touches upper band  
        signals[prices > bollinger['upper']] = -1
        
        # Apply execution delay
        signals_executable = signals.shift(self.entry_delay)
        
        return signals_executable


class RealisticBacktester:
    """Backtest with realistic execution and costs."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 position_size: float = 1.0,
                 commission_rate: float = 0.001,  # 10 bps
                 slippage_rate: float = 0.0005):  # 5 bps
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series) -> Dict:
        """
        Run realistic backtest.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with position signals (-1, 0, 1)
        """
        results = []
        capital = self.initial_capital
        position = 0
        shares = 0
        
        for i in range(len(data)):
            if pd.isna(signals.iloc[i]):
                continue
                
            current_signal = signals.iloc[i]
            
            # Use next bar's open for execution (realistic)
            if i < len(data) - 1:
                execution_price = data.iloc[i + 1]['open']
            else:
                continue
                
            # Position changes
            if position != current_signal:
                # Exit current position
                if position != 0:
                    # Calculate exit proceeds
                    exit_value = shares * execution_price
                    
                    # Apply slippage (adverse price movement)
                    if position > 0:  # Selling
                        exit_value *= (1 - self.slippage_rate)
                    else:  # Buying to cover
                        exit_value *= (1 + self.slippage_rate)
                        
                    # Apply commission
                    commission = abs(exit_value) * self.commission_rate
                    
                    # Update capital
                    if position > 0:
                        capital = capital - (shares * entry_price) + exit_value - commission
                    else:  # Short position
                        capital = capital + (shares * entry_price) - exit_value - commission
                        
                    # Record trade
                    pnl = capital - self.initial_capital
                    results.append({
                        'exit_time': data.index[i + 1],
                        'exit_price': execution_price,
                        'pnl': pnl,
                        'capital': capital
                    })
                    
                # Enter new position
                if current_signal != 0:
                    # Calculate position size
                    position_value = capital * self.position_size
                    
                    # Apply slippage on entry
                    if current_signal > 0:  # Buying
                        entry_price = execution_price * (1 + self.slippage_rate)
                    else:  # Selling short
                        entry_price = execution_price * (1 - self.slippage_rate)
                        
                    # Calculate shares
                    shares = position_value / entry_price
                    
                    # Apply commission
                    commission = position_value * self.commission_rate
                    capital -= commission
                    
                    # Update position
                    position = current_signal
                    
                else:
                    position = 0
                    shares = 0
                    
        # Calculate final metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        if results:
            returns = pd.Series([r['pnl'] / self.initial_capital for r in results])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = self._calculate_max_drawdown([r['capital'] for r in results])
        else:
            sharpe = 0
            max_dd = 0
            
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(results),
            'final_capital': capital,
            'trades': results
        }
    
    def _calculate_max_drawdown(self, capital_series: List[float]) -> float:
        """Calculate maximum drawdown from capital series."""
        if not capital_series:
            return 0
            
        running_max = capital_series[0]
        max_dd = 0
        
        for capital in capital_series:
            running_max = max(running_max, capital)
            drawdown = (capital - running_max) / running_max
            max_dd = min(max_dd, drawdown)
            
        return abs(max_dd)


def demonstrate_proper_backtesting():
    """Demonstrate the difference between proper and improper backtesting."""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.001, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, len(prices))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.0002, len(prices)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.0002, len(prices)))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(prices))
    }, index=dates)
    
    # Initialize components
    feature_calc = ProperFeatureCalculator()
    signal_gen = ProperSignalGenerator()
    backtester = RealisticBacktester()
    
    # Calculate features WITH PROPER LAGS
    fast_ma = feature_calc.calculate_sma(data['close'], period=10, lag=1)
    slow_ma = feature_calc.calculate_sma(data['close'], period=20, lag=1)
    
    # Generate signals WITH EXECUTION DELAY
    signals = signal_gen.generate_ma_crossover_signals(fast_ma, slow_ma)
    
    # Run backtest WITH REALISTIC COSTS
    results = backtester.run_backtest(data, signals)
    
    print("Proper Backtesting Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Now show what happens with look-ahead bias
    # WRONG: No lag on features
    fast_ma_wrong = data['close'].rolling(10).mean()  # No lag!
    slow_ma_wrong = data['close'].rolling(20).mean()  # No lag!
    
    # WRONG: No execution delay
    signals_wrong = pd.Series(0, index=data.index)
    signals_wrong[fast_ma_wrong > slow_ma_wrong] = 1
    signals_wrong[fast_ma_wrong < slow_ma_wrong] = -1
    
    # Run backtest without proper adjustments
    results_wrong = backtester.run_backtest(data, signals_wrong)
    
    print("\nImproper Backtesting Results (with look-ahead bias):")
    print(f"Total Return: {results_wrong['total_return']:.2%}")
    print(f"Sharpe Ratio: {results_wrong['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results_wrong['max_drawdown']:.2%}")
    print(f"Number of Trades: {results_wrong['num_trades']}")
    
    print("\nDifference:")
    print(f"Return inflation from look-ahead bias: {results_wrong['total_return'] - results['total_return']:.2%}")
    
    return results, results_wrong


if __name__ == "__main__":
    results_proper, results_improper = demonstrate_proper_backtesting()