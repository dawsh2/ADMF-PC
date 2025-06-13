"""
Signal Reconstruction and Performance Analysis

Reconstructs full trading history from sparse signal storage
and calculates comprehensive performance metrics.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    symbol: str
    strategy_id: str
    entry_time: str
    exit_time: str
    bars_held: int
    
    @property
    def pnl(self) -> float:
        """Calculate P&L for the trade."""
        if self.direction == 'long':
            return self.exit_price - self.entry_price
        else:  # short
            return self.entry_price - self.exit_price
    
    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        return (self.pnl / self.entry_price) * 100
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


class SignalReconstructor:
    """Reconstructs full trading history from sparse signals."""
    
    def __init__(self, sparse_signal_file: str, market_data_file: Optional[str] = None):
        """
        Initialize reconstructor.
        
        Args:
            sparse_signal_file: Path to sparse signal JSON file
            market_data_file: Optional path to market data CSV for price lookup
        """
        self.signal_file = Path(sparse_signal_file)
        self.market_data_file = Path(market_data_file) if market_data_file else None
        
        # Load sparse signals
        with open(self.signal_file, 'r') as f:
            self.signal_data = json.load(f)
        
        # Load market data if provided
        self.market_data = None
        if self.market_data_file and self.market_data_file.exists():
            self.market_data = pd.read_csv(self.market_data_file)
            logger.info(f"Loaded {len(self.market_data)} bars of market data")
    
    def reconstruct_signals(self) -> Dict[int, Dict[str, int]]:
        """
        Reconstruct full signal array from sparse changes.
        
        Returns:
            Dict mapping bar_index to {strategy_id: signal_value}
        """
        changes = self.signal_data['changes']
        total_bars = self.signal_data['metadata']['total_bars']
        
        # Initialize signal array
        signals = {}
        
        # Group changes by strategy
        strategy_changes = {}
        for change in changes:
            strategy_id = f"{change['sym']}_{change['strat']}"
            if strategy_id not in strategy_changes:
                strategy_changes[strategy_id] = []
            strategy_changes[strategy_id].append(change)
        
        # Reconstruct signals for each bar
        for bar_idx in range(total_bars):
            signals[bar_idx] = {}
            
            # Find active signal for each strategy at this bar
            for strategy_id, changes_list in strategy_changes.items():
                # Sort by bar index
                sorted_changes = sorted(changes_list, key=lambda x: x['idx'])
                
                # Find the most recent change before or at this bar
                current_signal = 0  # Default flat
                for change in sorted_changes:
                    if change['idx'] <= bar_idx:
                        current_signal = change['val']
                    else:
                        break
                
                if current_signal != 0:  # Only store non-flat signals
                    signals[bar_idx][strategy_id] = current_signal
        
        return signals
    
    def extract_trades(self) -> List[Trade]:
        """
        Extract completed trades from signal changes.
        
        Returns:
            List of Trade objects
        """
        trades = []
        changes = self.signal_data['changes']
        
        # Group by strategy
        strategy_changes = {}
        for change in changes:
            strategy_id = f"{change['sym']}_{change['strat']}"
            if strategy_id not in strategy_changes:
                strategy_changes[strategy_id] = []
            strategy_changes[strategy_id].append(change)
        
        # Extract trades for each strategy
        for strategy_id, changes_list in strategy_changes.items():
            # Sort by bar index
            sorted_changes = sorted(changes_list, key=lambda x: x['idx'])
            
            # Find entry/exit pairs
            i = 0
            while i < len(sorted_changes):
                change = sorted_changes[i]
                
                # Look for position entry (1 or -1)
                if change['val'] != 0:
                    entry_bar = change['idx']
                    entry_price = change['px'] if change['px'] > 0 else self._get_price(entry_bar)
                    entry_time = change['ts']
                    direction = 'long' if change['val'] == 1 else 'short'
                    
                    # Look for exit (change to different value)
                    exit_bar = None
                    exit_price = None
                    exit_time = None
                    
                    # Check next changes
                    for j in range(i + 1, len(sorted_changes)):
                        next_change = sorted_changes[j]
                        if next_change['val'] != change['val']:
                            exit_bar = next_change['idx']
                            exit_price = next_change['px'] if next_change['px'] > 0 else self._get_price(exit_bar)
                            exit_time = next_change['ts']
                            i = j - 1  # Will be incremented at loop end
                            break
                    
                    # If no exit found, use last bar
                    if exit_bar is None:
                        exit_bar = self.signal_data['metadata']['total_bars'] - 1
                        exit_price = self._get_price(exit_bar)
                        exit_time = datetime.now().isoformat()
                    
                    # Create trade if we have valid prices
                    if entry_price and exit_price:
                        trade = Trade(
                            entry_bar=entry_bar,
                            exit_bar=exit_bar,
                            entry_price=float(entry_price),
                            exit_price=float(exit_price),
                            direction=direction,
                            symbol=change['sym'],
                            strategy_id=change['strat'],
                            entry_time=entry_time,
                            exit_time=exit_time,
                            bars_held=exit_bar - entry_bar
                        )
                        trades.append(trade)
                
                i += 1
        
        return trades
    
    def _get_price(self, bar_index: int) -> Optional[float]:
        """Get price for a specific bar index."""
        if self.market_data is not None and bar_index < len(self.market_data):
            # Try different column names
            for col in ['Close', 'close', 'CLOSE']:
                if col in self.market_data.columns:
                    return float(self.market_data.iloc[bar_index][col])
        return None
    
    def calculate_performance_metrics(self, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from trades.
        
        Args:
            initial_capital: Starting capital for calculations
            
        Returns:
            Dictionary of performance metrics
        """
        trades = self.extract_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_bars_held': 0.0,
                'by_strategy': {}
            }
        
        # Overall metrics
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        
        # Calculate consecutive wins/losses
        results = [t.is_winner for t in trades]
        max_consecutive_wins = self._max_consecutive(results, True)
        max_consecutive_losses = self._max_consecutive(results, False)
        
        # Per-strategy breakdown
        by_strategy = {}
        strategy_trades = {}
        for trade in trades:
            if trade.strategy_id not in strategy_trades:
                strategy_trades[trade.strategy_id] = []
            strategy_trades[trade.strategy_id].append(trade)
        
        for strategy_id, strat_trades in strategy_trades.items():
            strat_winners = [t for t in strat_trades if t.is_winner]
            strat_losers = [t for t in strat_trades if not t.is_winner]
            
            by_strategy[strategy_id] = {
                'total_trades': len(strat_trades),
                'winners': len(strat_winners),
                'losers': len(strat_losers),
                'win_rate': len(strat_winners) / len(strat_trades) if strat_trades else 0,
                'total_pnl': sum(t.pnl for t in strat_trades),
                'avg_pnl': sum(t.pnl for t in strat_trades) / len(strat_trades) if strat_trades else 0,
                'avg_bars_held': sum(t.bars_held for t in strat_trades) / len(strat_trades) if strat_trades else 0
            }
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) if trades else 0.0,
            'total_pnl': total_pnl,
            'total_return': (total_pnl / initial_capital) * 100,
            'avg_winner': gross_profit / len(winners) if winners else 0.0,
            'avg_loser': -gross_loss / len(losers) if losers else 0.0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_bars_held': sum(t.bars_held for t in trades) / len(trades) if trades else 0.0,
            'largest_winner': max(t.pnl for t in trades) if trades else 0.0,
            'largest_loser': min(t.pnl for t in trades) if trades else 0.0,
            'by_strategy': by_strategy,
            'trades': [self._trade_to_dict(t) for t in trades]  # Include trade details
        }
    
    def _max_consecutive(self, results: List[bool], target: bool) -> int:
        """Find maximum consecutive occurrences of target value."""
        max_count = 0
        current_count = 0
        
        for result in results:
            if result == target:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert trade to dictionary for JSON serialization."""
        return {
            'entry_bar': trade.entry_bar,
            'exit_bar': trade.exit_bar,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'direction': trade.direction,
            'symbol': trade.symbol,
            'strategy_id': trade.strategy_id,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'bars_held': trade.bars_held,
            'is_winner': trade.is_winner
        }
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Complete performance analysis
        """
        # Get all data
        metadata = self.signal_data['metadata']
        performance = self.calculate_performance_metrics()
        
        # Build report
        report = {
            'run_id': metadata.get('run_id', 'unknown'),
            'metadata': metadata,
            'performance_metrics': performance,
            'signal_efficiency': {
                'total_bars': metadata['total_bars'],
                'signal_changes': metadata['total_changes'],
                'compression_ratio': metadata.get('compression_ratio', 0),
                'signals_per_bar': metadata['total_changes'] / metadata['total_bars'] if metadata['total_bars'] > 0 else 0
            }
        }
        
        # Add existing performance data if available
        if 'performance' in self.signal_data:
            report['live_performance'] = self.signal_data['performance']
        
        # Save if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved performance report to {output_file}")
        
        return report


def analyze_sparse_signal_file(signal_file: str, market_data: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a sparse signal file.
    
    Args:
        signal_file: Path to sparse signal JSON
        market_data: Optional path to market data CSV
        
    Returns:
        Performance analysis results
    """
    reconstructor = SignalReconstructor(signal_file, market_data)
    return reconstructor.generate_performance_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python signal_reconstruction.py <sparse_signal_file> [market_data_file]")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    market_data = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Analyze the signal file
    report = analyze_sparse_signal_file(signal_file, market_data)
    
    # Print summary
    metrics = report['performance_metrics']
    print(f"\nPerformance Summary for {report['run_id']}:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winners: {metrics['winning_trades']} ({metrics['win_rate']*100:.1f}%)")
    print(f"Losers: {metrics['losing_trades']}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Print per-strategy breakdown
    if metrics['by_strategy']:
        print("\nPer-Strategy Performance:")
        for strategy_id, stats in metrics['by_strategy'].items():
            print(f"\n{strategy_id}:")
            print(f"  Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Total P&L: ${stats['total_pnl']:.2f}")
            print(f"  Avg Bars Held: {stats['avg_bars_held']:.1f}")