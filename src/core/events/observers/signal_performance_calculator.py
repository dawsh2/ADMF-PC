"""
Signal-only performance calculator for evaluating strategy performance without trade execution.

This module provides tools to calculate performance metrics directly from signals,
simulating trade outcomes using entry/exit signal prices.
"""

from typing import Dict, Any, Optional, List, Tuple, DefaultDict
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass, field
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SignalPair:
    """Represents a matched entry/exit signal pair."""
    entry_signal: Dict[str, Any]
    exit_signal: Optional[Dict[str, Any]] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    @property
    def is_closed(self) -> bool:
        """Check if this pair has both entry and exit."""
        return self.exit_signal is not None
    
    @property
    def pnl(self) -> Optional[float]:
        """Calculate P&L if pair is closed."""
        if not self.is_closed:
            return None
            
        entry_price = self.entry_signal.get('price', 0)
        exit_price = self.exit_signal.get('price', 0)
        
        if entry_price == 0:
            return None
            
        # Calculate based on signal direction
        direction = self.entry_signal.get('direction', 'long')
        if direction == 'long':
            return exit_price - entry_price
        else:  # short
            return entry_price - exit_price
    
    @property
    def pnl_pct(self) -> Optional[float]:
        """Calculate percentage P&L if pair is closed."""
        if not self.is_closed:
            return None
            
        entry_price = self.entry_signal.get('price', 0)
        if entry_price == 0:
            return None
            
        pnl = self.pnl
        if pnl is None:
            return None
            
        return pnl / entry_price
    
    @property
    def holding_period(self) -> Optional[float]:
        """Calculate holding period in seconds."""
        if not self.is_closed or not self.entry_time or not self.exit_time:
            return None
        return (self.exit_time - self.entry_time).total_seconds()


class SignalPairMatcher:
    """
    Matches entry and exit signals to create signal pairs for performance analysis.
    
    Handles multiple strategies and symbols, tracking open positions and matching
    appropriate exit signals.
    """
    
    def __init__(self):
        """Initialize the signal pair matcher."""
        # Track open positions by strategy and symbol
        # Structure: {(strategy_name, symbol): deque[SignalPair]}
        self.open_positions: DefaultDict[Tuple[str, str], deque] = defaultdict(deque)
        
        # Track all signal pairs
        # Structure: {strategy_name: List[SignalPair]}
        self.signal_pairs: DefaultDict[str, List[SignalPair]] = defaultdict(list)
        
        # Track unmatched exit signals
        self.unmatched_exits: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.matched_pairs = 0
        
    def process_signal(self, signal: Dict[str, Any], timestamp: Optional[datetime] = None) -> Optional[SignalPair]:
        """
        Process a signal and attempt to create or close a signal pair.
        
        Args:
            signal: Signal dictionary with type, direction, price, etc.
            timestamp: Optional timestamp for the signal
            
        Returns:
            Completed SignalPair if an exit matched an entry, None otherwise
        """
        strategy_name = signal.get('strategy_name', 'default')
        symbol = signal.get('symbol', 'UNKNOWN')
        signal_type = signal.get('signal_type', 'entry')
        direction = signal.get('direction', 'long')
        
        key = (strategy_name, symbol)
        
        if signal_type == 'entry':
            self.total_entries += 1
            
            # Create new signal pair
            pair = SignalPair(
                entry_signal=signal,
                entry_time=timestamp or datetime.now()
            )
            
            # Add to open positions
            self.open_positions[key].append(pair)
            
            logger.debug(f"Opened position for {strategy_name}/{symbol}: {direction} @ {signal.get('price')}")
            return None
            
        elif signal_type == 'exit' or signal_type == 'close':
            self.total_exits += 1
            
            # Find matching open position
            open_queue = self.open_positions[key]
            
            if not open_queue:
                # No open position to close
                self.unmatched_exits.append(signal)
                logger.warning(f"Exit signal for {strategy_name}/{symbol} with no open position")
                return None
            
            # Match with oldest open position (FIFO)
            pair = open_queue.popleft()
            
            # Complete the pair
            pair.exit_signal = signal
            pair.exit_time = timestamp or datetime.now()
            
            # Add to completed pairs
            self.signal_pairs[strategy_name].append(pair)
            self.matched_pairs += 1
            
            logger.debug(f"Closed position for {strategy_name}/{symbol}: "
                        f"P&L = {pair.pnl_pct:.2%}" if pair.pnl_pct else "P&L = N/A")
            
            return pair
            
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return None
    
    def get_open_positions(self, strategy_name: Optional[str] = None) -> List[SignalPair]:
        """Get all open positions, optionally filtered by strategy."""
        open_pairs = []
        
        for (strat, symbol), positions in self.open_positions.items():
            if strategy_name is None or strat == strategy_name:
                open_pairs.extend(positions)
                
        return open_pairs
    
    def get_closed_pairs(self, strategy_name: Optional[str] = None) -> List[SignalPair]:
        """Get all closed signal pairs, optionally filtered by strategy."""
        if strategy_name:
            return self.signal_pairs.get(strategy_name, [])
        
        # Return all pairs
        all_pairs = []
        for pairs in self.signal_pairs.values():
            all_pairs.extend(pairs)
        return all_pairs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics."""
        return {
            'total_entries': self.total_entries,
            'total_exits': self.total_exits,
            'matched_pairs': self.matched_pairs,
            'open_positions': sum(len(positions) for positions in self.open_positions.values()),
            'unmatched_exits': len(self.unmatched_exits),
            'match_rate': self.matched_pairs / self.total_exits if self.total_exits > 0 else 0
        }


class SignalOnlyPerformance:
    """
    Calculates performance metrics from signal pairs without actual trade execution.
    
    Simulates trade outcomes using signal entry/exit prices to compute metrics like:
    - Win rate
    - Average return
    - Profit factor
    - Sharpe ratio
    - Maximum drawdown
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize performance calculator.
        
        Args:
            initial_capital: Starting capital for calculations
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Return series for advanced metrics
        self.returns: List[float] = []
        self.equity_curve: List[float] = [initial_capital]
        
        # Drawdown tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Per-strategy metrics
        self.strategy_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'returns': []
        })
        
    def process_signal_pair(self, pair: SignalPair) -> Dict[str, Any]:
        """
        Process a completed signal pair and update metrics.
        
        Args:
            pair: Completed SignalPair with entry and exit
            
        Returns:
            Trade result dictionary
        """
        if not pair.is_closed:
            logger.warning("Cannot process unclosed signal pair")
            return {}
        
        # Calculate P&L
        pnl = pair.pnl
        pnl_pct = pair.pnl_pct
        
        if pnl is None or pnl_pct is None:
            logger.warning("Cannot calculate P&L for signal pair")
            return {}
        
        # Assume fixed position size for now (can be enhanced)
        position_size = self.current_capital * 0.02  # 2% of capital
        pnl_amount = position_size * pnl_pct
        
        # Update overall metrics
        self.total_trades += 1
        self.total_pnl += pnl_amount
        self.current_capital += pnl_amount
        self.equity_curve.append(self.current_capital)
        
        if pnl_amount > 0:
            self.winning_trades += 1
            self.gross_profit += pnl_amount
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl_amount)
        
        # Track return
        ret = pnl_amount / (self.current_capital - pnl_amount)
        self.returns.append(ret)
        
        # Update drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update strategy-specific metrics
        strategy_name = pair.entry_signal.get('strategy_name', 'default')
        strat_metrics = self.strategy_metrics[strategy_name]
        strat_metrics['trades'] += 1
        strat_metrics['total_pnl'] += pnl_amount
        strat_metrics['returns'].append(ret)
        
        if pnl_amount > 0:
            strat_metrics['wins'] += 1
            strat_metrics['gross_profit'] += pnl_amount
        else:
            strat_metrics['losses'] += 1
            strat_metrics['gross_loss'] += abs(pnl_amount)
        
        # Return trade result
        return {
            'strategy_name': strategy_name,
            'symbol': pair.entry_signal.get('symbol'),
            'entry_price': pair.entry_signal.get('price'),
            'exit_price': pair.exit_signal.get('price'),
            'direction': pair.entry_signal.get('direction'),
            'pnl': pnl_amount,
            'pnl_pct': pnl_pct,
            'position_size': position_size,
            'holding_period': pair.holding_period,
            'entry_time': pair.entry_time,
            'exit_time': pair.exit_time
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return comprehensive performance metrics."""
        metrics = {
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'total_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'final_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf'),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown
        }
        
        # Calculate average win/loss
        if self.winning_trades > 0:
            metrics['avg_win'] = self.gross_profit / self.winning_trades
        else:
            metrics['avg_win'] = 0
            
        if self.losing_trades > 0:
            metrics['avg_loss'] = self.gross_loss / self.losing_trades
        else:
            metrics['avg_loss'] = 0
        
        # Calculate Sharpe ratio if we have returns
        if len(self.returns) > 1:
            import math
            mean_return = statistics.mean(self.returns)
            std_return = statistics.stdev(self.returns)
            
            # Annualize (assuming daily returns)
            annual_return = mean_return * 252
            annual_std = std_return * math.sqrt(252)
            
            metrics['sharpe_ratio'] = annual_return / annual_std if annual_std > 0 else 0
            metrics['mean_return'] = mean_return
            metrics['return_std'] = std_return
        else:
            metrics['sharpe_ratio'] = 0
            metrics['mean_return'] = 0
            metrics['return_std'] = 0
        
        # Add per-strategy breakdown
        metrics['strategy_breakdown'] = {}
        for strategy_name, strat_data in self.strategy_metrics.items():
            if strat_data['trades'] > 0:
                win_rate = strat_data['wins'] / strat_data['trades']
                profit_factor = (strat_data['gross_profit'] / strat_data['gross_loss'] 
                               if strat_data['gross_loss'] > 0 else float('inf'))
                
                metrics['strategy_breakdown'][strategy_name] = {
                    'trades': strat_data['trades'],
                    'win_rate': win_rate,
                    'total_pnl': strat_data['total_pnl'],
                    'profit_factor': profit_factor,
                    'avg_return': statistics.mean(strat_data['returns']) if strat_data['returns'] else 0
                }
        
        return metrics
    
    def get_equity_curve(self) -> List[float]:
        """Get the equity curve."""
        return self.equity_curve.copy()
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.current_capital = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.returns = []
        self.equity_curve = [self.initial_capital]
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.strategy_metrics.clear()