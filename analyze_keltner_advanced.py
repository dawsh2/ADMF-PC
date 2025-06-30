#!/usr/bin/env python3
"""
Advanced analysis of swing pivot strategies with stop loss and EOD exit analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict, List, Tuple
import json
from datetime import datetime, time

class AdvancedKeltnerAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_5m_1m"
        self.signals_path = self.traces_path / "signals" / "keltner_bands"
        
        # Market hours (assuming US market)
        self.market_open_bar = 78  # 9:30 AM at 5-min bars (78 bars from midnight)
        self.market_close_bar = 156  # 4:00 PM at 5-min bars
        self.bars_per_day = 78  # 6.5 hours * 12 bars/hour
        
    def load_signal_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single parquet signal file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def calculate_trade_returns_with_stops(self, signals_df: pd.DataFrame, 
                                         stop_loss_pct: float = None,
                                         force_eod_exit: bool = True,
                                         execution_cost_bps: float = 0.5) -> Dict:
        """
        Calculate returns with optional stop loss and EOD exit.
        
        Args:
            signals_df: DataFrame with sparse signals
            stop_loss_pct: Stop loss percentage (e.g., 0.01 for 1%)
            force_eod_exit: Force exit at end of day
            execution_cost_bps: Execution cost in basis points
        """
        if signals_df.empty or len(signals_df) < 2:
            return {
                'num_trades': 0,
                'total_return': 0,
                'log_return': 0,
                'avg_return_per_trade': 0,
                'avg_return_per_trade_bps': 0,
                'trades': [],
                'stopped_trades': 0,
                'eod_exits': 0
            }
        
        # Sort by index
        signals_df = signals_df.sort_values('idx')
        
        trades = []
        current_position = None
        stopped_trades = 0
        eod_exits = 0
        
        # Convert execution cost from bps to multiplier
        execution_cost_multiplier = 1 - (execution_cost_bps / 10000)
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            # Calculate time of day
            bar_of_day = bar_idx % self.bars_per_day
            
            # Check for EOD exit
            if force_eod_exit and current_position is not None and bar_of_day >= self.market_close_bar - 1:
                # Force exit at EOD
                exit_price = price
                trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                trade_return *= execution_cost_multiplier
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': bar_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'signal': current_position['signal'],
                    'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                    'net_return': trade_return,
                    'duration_bars': bar_idx - current_position['entry_idx'],
                    'exit_type': 'eod',
                    'return_bps': trade_return * 10000
                })
                
                eod_exits += 1
                current_position = None
                continue
            
            # Handle position changes
            if current_position is None and signal != 0:
                # Opening new position
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_price': price,
                    'signal': signal,
                    'entry_time': row.get('ts', bar_idx),
                    'highest_price': price if signal > 0 else float('inf'),
                    'lowest_price': price if signal < 0 else float('inf')
                }
            elif current_position is not None:
                # Update high/low for stop loss calculation
                if current_position['signal'] > 0:
                    current_position['lowest_price'] = min(current_position['lowest_price'], price)
                else:
                    current_position['highest_price'] = max(current_position['highest_price'], price)
                
                # Check stop loss
                if stop_loss_pct is not None:
                    if current_position['signal'] > 0:  # Long position
                        stop_price = current_position['entry_price'] * (1 - stop_loss_pct)
                        if price <= stop_price:
                            # Stop loss hit
                            exit_price = stop_price
                            trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                            trade_return *= execution_cost_multiplier
                            
                            trades.append({
                                'entry_idx': current_position['entry_idx'],
                                'exit_idx': bar_idx,
                                'entry_price': current_position['entry_price'],
                                'exit_price': exit_price,
                                'signal': current_position['signal'],
                                'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                                'net_return': trade_return,
                                'duration_bars': bar_idx - current_position['entry_idx'],
                                'exit_type': 'stop_loss',
                                'return_bps': trade_return * 10000
                            })
                            
                            stopped_trades += 1
                            current_position = None
                            continue
                    else:  # Short position
                        stop_price = current_position['entry_price'] * (1 + stop_loss_pct)
                        if price >= stop_price:
                            # Stop loss hit
                            exit_price = stop_price
                            trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                            trade_return *= execution_cost_multiplier
                            
                            trades.append({
                                'entry_idx': current_position['entry_idx'],
                                'exit_idx': bar_idx,
                                'entry_price': current_position['entry_price'],
                                'exit_price': exit_price,
                                'signal': current_position['signal'],
                                'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                                'net_return': trade_return,
                                'duration_bars': bar_idx - current_position['entry_idx'],
                                'exit_type': 'stop_loss',
                                'return_bps': trade_return * 10000
                            })
                            
                            stopped_trades += 1
                            current_position = None
                            continue
                
                # Handle regular signal-based exits
                if signal == 0:
                    # Closing position
                    exit_price = price
                    trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                    trade_return *= execution_cost_multiplier
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': bar_idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'signal': current_position['signal'],
                        'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                        'net_return': trade_return,
                        'duration_bars': bar_idx - current_position['entry_idx'],
                        'exit_type': 'signal',
                        'return_bps': trade_return * 10000
                    })
                    
                    current_position = None
                elif signal != 0 and signal != current_position['signal']:
                    # Flipping position
                    exit_price = price
                    trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                    trade_return *= execution_cost_multiplier
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': bar_idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'signal': current_position['signal'],
                        'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                        'net_return': trade_return,
                        'duration_bars': bar_idx - current_position['entry_idx'],
                        'exit_type': 'flip',
                        'return_bps': trade_return * 10000
                    })
                    
                    # Open new position
                    current_position = {
                        'entry_idx': bar_idx,
                        'entry_price': price,
                        'signal': signal,
                        'entry_time': row.get('ts', bar_idx),
                        'highest_price': price if signal > 0 else float('inf'),
                        'lowest_price': price if signal < 0 else float('inf')
                    }
        
        # Calculate total returns
        total_log_return = sum(trade['net_return'] for trade in trades)
        total_return = np.exp(total_log_return) - 1
        avg_return_per_trade = total_log_return / len(trades) if trades else 0
        avg_return_per_trade_bps = avg_return_per_trade * 10000
        
        return {
            'num_trades': len(trades),
            'total_return': total_return,
            'log_return': total_log_return,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_return_per_trade_bps': avg_return_per_trade_bps,
            'trades': trades,
            'stopped_trades': stopped_trades,
            'eod_exits': eod_exits,
            'signal_exits': sum(1 for t in trades if t['exit_type'] == 'signal'),
            'flip_exits': sum(1 for t in trades if t['exit_type'] == 'flip')
        }
    
    def analyze_stop_loss_impact(self, signals_df: pd.DataFrame, 
                               stop_losses: List[float] = None) -> pd.DataFrame:
        """Analyze impact of different stop loss levels."""
        if stop_losses is None:
            stop_losses = [None, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]
        
        results = []
        
        for stop_loss in stop_losses:
            perf = self.calculate_trade_returns_with_stops(
                signals_df, 
                stop_loss_pct=stop_loss,
                force_eod_exit=True
            )
            
            results.append({
                'stop_loss_pct': stop_loss,
                'stop_loss_bps': stop_loss * 10000 if stop_loss else None,
                'num_trades': perf['num_trades'],
                'total_return': perf['total_return'],
                'avg_return_per_trade_bps': perf['avg_return_per_trade_bps'],
                'stopped_trades': perf['stopped_trades'],
                'stop_rate': perf['stopped_trades'] / perf['num_trades'] if perf['num_trades'] > 0 else 0,
                'eod_exits': perf['eod_exits'],
                'signal_exits': perf['signal_exits']
            })
        
        return pd.DataFrame(results)
    
    def analyze_all_strategies_with_filters(self, 
                                          min_return_per_trade_bps: float = 1.0,
                                          min_trades_per_day: float = 2.0,
                                          stop_loss_pct: float = None,
                                          limit: int = None) -> pd.DataFrame:
        """Analyze all strategies with filtering criteria."""
        results = []
        
        # Get all parquet files
        signal_files = sorted(list(self.signals_path.glob("*.parquet")))
        
        if limit:
            signal_files = signal_files[:limit]
        
        print(f"Found {len(signal_files)} strategy files to analyze")
        
        # Estimate total bars (you may need to adjust based on actual data)
        total_trading_days = 252  # Approximate
        
        for i, filepath in enumerate(signal_files):
            if i % 100 == 0:
                print(f"Processing strategy {i}/{len(signal_files)}...")
            
            # Extract strategy ID
            strategy_id = filepath.stem
            
            # Load signals
            signals_df = self.load_signal_file(filepath)
            
            if signals_df.empty:
                continue
            
            # Calculate performance with stops and EOD exits
            perf = self.calculate_trade_returns_with_stops(
                signals_df,
                stop_loss_pct=stop_loss_pct,
                force_eod_exit=True
            )
            
            # Calculate trades per day
            trades_per_day = perf['num_trades'] / total_trading_days
            
            # Get trade statistics
            if perf['trades']:
                winning_trades = [t for t in perf['trades'] if t['net_return'] > 0]
                losing_trades = [t for t in perf['trades'] if t['net_return'] < 0]
                
                win_rate = len(winning_trades) / len(perf['trades'])
                avg_win_bps = np.mean([t['return_bps'] for t in winning_trades]) if winning_trades else 0
                avg_loss_bps = np.mean([t['return_bps'] for t in losing_trades]) if losing_trades else 0
            else:
                win_rate = 0
                avg_win_bps = 0
                avg_loss_bps = 0
            
            result = {
                'strategy_id': strategy_id,
                'num_signals': len(signals_df),
                'num_trades': perf['num_trades'],
                'trades_per_day': trades_per_day,
                'total_return': perf['total_return'],
                'avg_return_per_trade_bps': perf['avg_return_per_trade_bps'],
                'win_rate': win_rate,
                'avg_win_bps': avg_win_bps,
                'avg_loss_bps': avg_loss_bps,
                'stopped_trades': perf['stopped_trades'],
                'stop_rate': perf['stopped_trades'] / perf['num_trades'] if perf['num_trades'] > 0 else 0,
                'eod_exits': perf['eod_exits'],
                'eod_exit_rate': perf['eod_exits'] / perf['num_trades'] if perf['num_trades'] > 0 else 0,
                'meets_criteria': (
                    perf['avg_return_per_trade_bps'] >= min_return_per_trade_bps and
                    trades_per_day >= min_trades_per_day
                )
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_filtered_report(self, results_df: pd.DataFrame, 
                               min_return_per_trade_bps: float = 1.0,
                               min_trades_per_day: float = 2.0) -> str:
        """Generate report focused on strategies meeting criteria."""
        report = []
        report.append(f"# Filtered Keltner Channel Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Workspace: {self.workspace_path}")
        report.append("")
        
        # Filter strategies meeting criteria
        filtered_df = results_df[results_df['meets_criteria']]
        
        report.append("## Filter Criteria")
        report.append(f"- Minimum return per trade: {min_return_per_trade_bps} bps")
        report.append(f"- Minimum trades per day: {min_trades_per_day}")
        report.append("")
        
        report.append("## Results Summary")
        report.append(f"- Total strategies analyzed: {len(results_df)}")
        report.append(f"- Strategies meeting criteria: {len(filtered_df)} ({len(filtered_df)/len(results_df)*100:.1f}%)")
        report.append("")
        
        if len(filtered_df) > 0:
            report.append("## Qualified Strategies Performance")
            report.append(f"- Average return per trade: {filtered_df['avg_return_per_trade_bps'].mean():.2f} bps")
            report.append(f"- Average trades per day: {filtered_df['trades_per_day'].mean():.1f}")
            report.append(f"- Average total return: {filtered_df['total_return'].mean()*100:.2f}%")
            report.append(f"- Average win rate: {filtered_df['win_rate'].mean()*100:.1f}%")
            report.append(f"- Average stop rate: {filtered_df['stop_rate'].mean()*100:.1f}%")
            report.append(f"- Average EOD exit rate: {filtered_df['eod_exit_rate'].mean()*100:.1f}%")
            report.append("")
            
            # Top performers by return per trade
            report.append("## Top 10 by Return per Trade (bps)")
            top_by_rpt = filtered_df.nlargest(10, 'avg_return_per_trade_bps')
            for idx, row in top_by_rpt.iterrows():
                report.append(f"- {row['strategy_id']}: {row['avg_return_per_trade_bps']:.2f} bps "
                            f"({row['trades_per_day']:.1f} trades/day, "
                            f"{row['win_rate']*100:.1f}% win rate)")
            report.append("")
            
            # Distribution of returns per trade
            report.append("## Return per Trade Distribution (Qualified Strategies)")
            report.append(f"- 90th percentile: {filtered_df['avg_return_per_trade_bps'].quantile(0.9):.2f} bps")
            report.append(f"- 75th percentile: {filtered_df['avg_return_per_trade_bps'].quantile(0.75):.2f} bps")
            report.append(f"- Median: {filtered_df['avg_return_per_trade_bps'].median():.2f} bps")
            report.append(f"- 25th percentile: {filtered_df['avg_return_per_trade_bps'].quantile(0.25):.2f} bps")
            report.append(f"- 10th percentile: {filtered_df['avg_return_per_trade_bps'].quantile(0.1):.2f} bps")
        
        return "\n".join(report)


def main():
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    else:
        workspace_path = "/Users/daws/ADMF-PC/workspaces/signal_generation_a2d31737"
    
    print(f"Analyzing workspace: {workspace_path}")
    analyzer = AdvancedKeltnerAnalyzer(workspace_path)
    
    # First, analyze a sample strategy to find optimal stop loss
    print("\n=== Stop Loss Analysis (Sample Strategy) ===")
    sample_file = list(analyzer.signals_path.glob("*.parquet"))[0]
    signals_df = analyzer.load_signal_file(sample_file)
    stop_loss_analysis = analyzer.analyze_stop_loss_impact(signals_df)
    print(stop_loss_analysis)
    
    # Analyze all strategies with filters
    print("\n=== Analyzing All Strategies with Filters ===")
    results_df = analyzer.analyze_all_strategies_with_filters(
        min_return_per_trade_bps=1.0,
        min_trades_per_day=2.0,
        stop_loss_pct=0.005,  # 50 bps stop loss
        limit=None
    )
    
    # Save results
    output_name = f"advanced_analysis_{Path(workspace_path).name}"
    results_df.to_csv(f"{output_name}.csv", index=False)
    print(f"\nSaved detailed results to {output_name}.csv")
    
    # Generate and save report
    report = analyzer.generate_filtered_report(results_df)
    
    with open(f"{output_name}_report.md", "w") as f:
        f.write(report)
    
    print(f"Saved report to {output_name}_report.md")
    print("\n" + "="*50)
    print(report)
    
    # Additional analysis
    print("\n=== Additional Statistics ===")
    qualified = results_df[results_df['meets_criteria']]
    if len(qualified) > 0:
        print(f"Qualified strategies: {len(qualified)}/{len(results_df)}")
        print(f"Best return per trade: {qualified['avg_return_per_trade_bps'].max():.2f} bps")
        print(f"Most active qualified: {qualified['trades_per_day'].max():.1f} trades/day")
        
        # Save qualified strategies
        qualified.to_csv(f"{output_name}_qualified_strategies.csv", index=False)
        print(f"\nSaved {len(qualified)} qualified strategies to {output_name}_qualified_strategies.csv")


if __name__ == "__main__":
    main()