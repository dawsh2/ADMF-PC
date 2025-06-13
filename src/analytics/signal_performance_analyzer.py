"""
Signal Performance Analyzer

Analyzes stored signals from hierarchical storage to calculate performance metrics
without requiring actual trade execution. Works with the existing sparse storage system.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class SignalPerformanceAnalyzer:
    """
    Analyzes signal performance from stored signal data.
    
    Reads signals from hierarchical storage and calculates hypothetical
    performance metrics by pairing entry/exit signals.
    """
    
    def __init__(self, workspace_path: Path):
        """
        Initialize analyzer with workspace path.
        
        Args:
            workspace_path: Path to workspace directory containing signal data
        """
        self.workspace_path = Path(workspace_path)
        self.signals_df = None
        self.signal_pairs = []
        self.performance_metrics = {}
        
    def load_signals(self, container_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load signals from storage.
        
        Args:
            container_id: Optional specific container to load
            
        Returns:
            DataFrame with all signals
        """
        all_signals = []
        
        if container_id:
            # Load specific container
            container_path = self.workspace_path / container_id
            signals_file = container_path / 'signals.parquet'
            
            if signals_file.exists():
                df = pd.read_parquet(signals_file)
                df['container_id'] = container_id
                all_signals.append(df)
        else:
            # Load all containers
            for container_path in self.workspace_path.iterdir():
                if container_path.is_dir():
                    signals_file = container_path / 'signals.parquet'
                    if signals_file.exists():
                        df = pd.read_parquet(signals_file)
                        df['container_id'] = container_path.name
                        all_signals.append(df)
        
        if all_signals:
            self.signals_df = pd.concat(all_signals, ignore_index=True)
            self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
            self.signals_df = self.signals_df.sort_values('timestamp')
            
            logger.info(f"Loaded {len(self.signals_df)} signals from {len(all_signals)} containers")
        else:
            self.signals_df = pd.DataFrame()
            logger.warning("No signals found in storage")
            
        return self.signals_df
    
    def load_signal_events(self, container_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load full signal events from event storage (includes price data).
        
        Args:
            container_id: Optional specific container to load
            
        Returns:
            DataFrame with signal events including prices
        """
        all_events = []
        
        if container_id:
            containers = [container_id]
        else:
            # Get all portfolio containers
            containers = [d.name for d in self.workspace_path.iterdir() if d.is_dir()]
        
        for cid in containers:
            events_file = self.workspace_path / cid / 'events.parquet'
            if events_file.exists():
                df = pd.read_parquet(events_file)
                # Filter for SIGNAL events
                signal_events = df[df['event_type'] == 'SIGNAL'].copy()
                
                if not signal_events.empty:
                    # Parse payload JSON
                    signal_events['payload_data'] = signal_events['payload'].apply(json.loads)
                    
                    # Extract signal details
                    signal_events['symbol'] = signal_events['payload_data'].apply(lambda x: x.get('symbol'))
                    signal_events['direction'] = signal_events['payload_data'].apply(lambda x: x.get('direction'))
                    signal_events['price'] = signal_events['payload_data'].apply(lambda x: x.get('price', 0))
                    signal_events['strategy_name'] = signal_events['payload_data'].apply(lambda x: x.get('strategy_name', 'default'))
                    signal_events['signal_strength'] = signal_events['payload_data'].apply(lambda x: x.get('signal_strength', 0))
                    
                    signal_events['container_id'] = cid
                    all_events.append(signal_events)
        
        if all_events:
            self.signals_df = pd.concat(all_events, ignore_index=True)
            self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
            self.signals_df = self.signals_df.sort_values('timestamp')
            
            logger.info(f"Loaded {len(self.signals_df)} signal events with price data")
        else:
            self.signals_df = pd.DataFrame()
            
        return self.signals_df
    
    def pair_signals(self) -> List[Dict[str, Any]]:
        """
        Pair entry and exit signals.
        
        Uses implicit exit logic: opposite direction signal closes previous position.
        
        Returns:
            List of signal pairs
        """
        if self.signals_df is None or self.signals_df.empty:
            logger.warning("No signals loaded")
            return []
        
        self.signal_pairs = []
        open_positions = {}  # (strategy, symbol) -> signal
        
        for _, signal in self.signals_df.iterrows():
            strategy = signal.get('strategy_name', 'default')
            symbol = signal.get('symbol', 'UNKNOWN')
            direction = signal.get('direction', 'long')
            
            key = (strategy, symbol)
            
            if key in open_positions:
                # We have an open position
                open_signal = open_positions[key]
                open_direction = open_signal.get('direction', 'long')
                
                # Check if this is a reverse signal (implicit exit)
                if open_direction != direction:
                    # Close the position
                    pair = {
                        'strategy': strategy,
                        'symbol': symbol,
                        'entry_time': open_signal['timestamp'],
                        'exit_time': signal['timestamp'],
                        'entry_price': open_signal.get('price', 0),
                        'exit_price': signal.get('price', 0),
                        'direction': open_direction,
                        'holding_period': (signal['timestamp'] - open_signal['timestamp']).total_seconds(),
                        'entry_signal': open_signal.to_dict(),
                        'exit_signal': signal.to_dict()
                    }
                    
                    # Calculate P&L
                    if pair['entry_price'] > 0:
                        if open_direction == 'long':
                            pair['pnl'] = pair['exit_price'] - pair['entry_price']
                            pair['pnl_pct'] = pair['pnl'] / pair['entry_price']
                        else:  # short
                            pair['pnl'] = pair['entry_price'] - pair['exit_price']
                            pair['pnl_pct'] = pair['pnl'] / pair['entry_price']
                    else:
                        pair['pnl'] = 0
                        pair['pnl_pct'] = 0
                    
                    self.signal_pairs.append(pair)
                    
                    # Remove closed position
                    del open_positions[key]
                    
                    # Open new position with current signal
                    open_positions[key] = signal
                else:
                    # Same direction signal - could update or ignore
                    # For now, ignore (keep first position)
                    pass
            else:
                # No open position - open new one
                open_positions[key] = signal
        
        logger.info(f"Created {len(self.signal_pairs)} signal pairs, "
                   f"{len(open_positions)} positions remain open")
        
        return self.signal_pairs
    
    def calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from signal pairs.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.signal_pairs:
            logger.warning("No signal pairs to analyze")
            return {}
        
        # Convert to DataFrame for easier analysis
        pairs_df = pd.DataFrame(self.signal_pairs)
        
        # Overall metrics
        total_trades = len(pairs_df)
        winning_trades = len(pairs_df[pairs_df['pnl'] > 0])
        losing_trades = len(pairs_df[pairs_df['pnl'] < 0])
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': pairs_df['pnl'].sum(),
            'avg_pnl': pairs_df['pnl'].mean(),
            'avg_pnl_pct': pairs_df['pnl_pct'].mean(),
            'best_trade_pnl': pairs_df['pnl'].max(),
            'worst_trade_pnl': pairs_df['pnl'].min(),
            'avg_holding_period': pairs_df['holding_period'].mean(),
        }
        
        # Calculate profit factor
        gross_profit = pairs_df[pairs_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(pairs_df[pairs_df['pnl'] < 0]['pnl'].sum())
        metrics['gross_profit'] = gross_profit
        metrics['gross_loss'] = gross_loss
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (assuming daily returns)
        if len(pairs_df) > 1:
            returns = pairs_df['pnl_pct'].values
            metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Per-strategy breakdown
        strategy_metrics = {}
        for strategy in pairs_df['strategy'].unique():
            strategy_df = pairs_df[pairs_df['strategy'] == strategy]
            strategy_wins = len(strategy_df[strategy_df['pnl'] > 0])
            
            strategy_metrics[strategy] = {
                'trades': len(strategy_df),
                'win_rate': strategy_wins / len(strategy_df) if len(strategy_df) > 0 else 0,
                'avg_pnl': strategy_df['pnl'].mean(),
                'avg_pnl_pct': strategy_df['pnl_pct'].mean(),
                'total_pnl': strategy_df['pnl'].sum()
            }
        
        metrics['strategy_breakdown'] = strategy_metrics
        
        # Per-symbol breakdown
        symbol_metrics = {}
        for symbol in pairs_df['symbol'].unique():
            symbol_df = pairs_df[pairs_df['symbol'] == symbol]
            symbol_wins = len(symbol_df[symbol_df['pnl'] > 0])
            
            symbol_metrics[symbol] = {
                'trades': len(symbol_df),
                'win_rate': symbol_wins / len(symbol_df) if len(symbol_df) > 0 else 0,
                'avg_pnl': symbol_df['pnl'].mean(),
                'total_pnl': symbol_df['pnl'].sum()
            }
        
        metrics['symbol_breakdown'] = symbol_metrics
        
        self.performance_metrics = metrics
        return metrics
    
    def save_analysis(self, output_path: Optional[Path] = None) -> Path:
        """
        Save analysis results.
        
        Args:
            output_path: Optional output path, defaults to workspace/analysis/
            
        Returns:
            Path where results were saved
        """
        if output_path is None:
            output_path = self.workspace_path / 'analysis' / 'signal_performance.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'workspace': str(self.workspace_path),
            'signal_count': len(self.signals_df) if self.signals_df is not None else 0,
            'pair_count': len(self.signal_pairs),
            'performance_metrics': self.performance_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save signal pairs as parquet for further analysis
        if self.signal_pairs:
            pairs_df = pd.DataFrame(self.signal_pairs)
            pairs_path = output_path.parent / 'signal_pairs.parquet'
            pairs_df.to_parquet(pairs_path)
        
        logger.info(f"Saved analysis to {output_path}")
        return output_path
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Formatted summary string
        """
        if not self.performance_metrics:
            return "No performance metrics calculated yet"
        
        m = self.performance_metrics
        
        report = f"""
Signal Performance Analysis Summary
==================================

Overall Performance:
-------------------
Total Trades: {m['total_trades']}
Win Rate: {m['win_rate']:.1%}
Profit Factor: {m['profit_factor']:.2f}
Sharpe Ratio: {m['sharpe_ratio']:.2f}

Average P&L: {m['avg_pnl']:.2f}
Average P&L %: {m['avg_pnl_pct']:.2%}
Total P&L: {m['total_pnl']:.2f}

Best Trade: {m['best_trade_pnl']:.2f}
Worst Trade: {m['worst_trade_pnl']:.2f}
Avg Holding Period: {m['avg_holding_period']/3600:.1f} hours

Strategy Breakdown:
------------------"""
        
        for strategy, stats in m.get('strategy_breakdown', {}).items():
            report += f"\n{strategy}:"
            report += f"\n  Trades: {stats['trades']}"
            report += f"\n  Win Rate: {stats['win_rate']:.1%}"
            report += f"\n  Avg P&L %: {stats['avg_pnl_pct']:.2%}"
            report += f"\n  Total P&L: {stats['total_pnl']:.2f}"
        
        return report


def analyze_signal_performance(workspace_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze signal performance from a workspace.
    
    Args:
        workspace_path: Path to workspace directory
        
    Returns:
        Performance metrics dictionary
    """
    analyzer = SignalPerformanceAnalyzer(Path(workspace_path))
    
    # Load signal events (includes prices)
    analyzer.load_signal_events()
    
    # Pair signals
    analyzer.pair_signals()
    
    # Calculate performance
    metrics = analyzer.calculate_performance()
    
    # Save results
    analyzer.save_analysis()
    
    # Print summary
    print(analyzer.get_summary_report())
    
    return metrics