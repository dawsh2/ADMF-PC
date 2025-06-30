#!/usr/bin/env python3
"""
Analyze swing pivot bounce zones strategies using sparse trace analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict, List, Tuple
import json
from datetime import datetime

class SwingPivotAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_5m_1m"
        self.signals_path = self.traces_path / "signals" / "swing_pivot_bounce_zones"
        
    def load_signal_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single parquet signal file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def calculate_trade_returns(self, signals_df: pd.DataFrame, 
                              execution_cost_multiplier: float = 0.999) -> Dict:
        """Calculate returns from sparse signal data."""
        if signals_df.empty or len(signals_df) < 2:
            return {
                'num_trades': 0,
                'total_return': 0,
                'log_return': 0,
                'trades': []
            }
        
        # Sort by index to ensure chronological order
        signals_df = signals_df.sort_values('idx')
        
        trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            # Handle position changes
            if current_position is None and signal != 0:
                # Opening new position
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_price': price,
                    'signal': signal,
                    'entry_time': row.get('ts', bar_idx)
                }
            elif current_position is not None and signal == 0:
                # Closing position
                exit_price = price
                trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                
                # Apply execution costs
                trade_return *= execution_cost_multiplier
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': bar_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'signal': current_position['signal'],
                    'gross_return': np.log(exit_price / current_position['entry_price']) * current_position['signal'],
                    'net_return': trade_return,
                    'duration_bars': bar_idx - current_position['entry_idx']
                })
                
                current_position = None
            elif current_position is not None and signal != 0 and signal != current_position['signal']:
                # Flipping position (close and open opposite)
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
                    'duration_bars': bar_idx - current_position['entry_idx']
                })
                
                # Open new position
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_price': price,
                    'signal': signal,
                    'entry_time': row.get('ts', bar_idx)
                }
        
        # Calculate total returns
        total_log_return = sum(trade['net_return'] for trade in trades)
        total_return = np.exp(total_log_return) - 1
        
        return {
            'num_trades': len(trades),
            'total_return': total_return,
            'log_return': total_log_return,
            'trades': trades
        }
    
    def analyze_all_strategies(self, limit: int = None) -> pd.DataFrame:
        """Analyze all strategy files in the workspace."""
        results = []
        
        # Get all parquet files
        signal_files = sorted(list(self.signals_path.glob("*.parquet")))
        
        if limit:
            signal_files = signal_files[:limit]
        
        print(f"Found {len(signal_files)} strategy files to analyze")
        
        for i, filepath in enumerate(signal_files):
            if i % 100 == 0:
                print(f"Processing strategy {i}/{len(signal_files)}...")
            
            # Extract strategy ID from filename
            strategy_id = filepath.stem
            
            # Load signals
            signals_df = self.load_signal_file(filepath)
            
            if signals_df.empty:
                continue
            
            # Calculate performance
            perf = self.calculate_trade_returns(signals_df)
            
            # Get additional metadata from signals
            first_signal = signals_df.iloc[0] if len(signals_df) > 0 else None
            last_signal = signals_df.iloc[-1] if len(signals_df) > 0 else None
            
            result = {
                'strategy_id': strategy_id,
                'num_signals': len(signals_df),
                'num_trades': perf['num_trades'],
                'total_return': perf['total_return'],
                'log_return': perf['log_return'],
                'first_bar_idx': first_signal['idx'] if first_signal is not None else None,
                'last_bar_idx': last_signal['idx'] if last_signal is not None else None,
                'avg_trade_duration': np.mean([t['duration_bars'] for t in perf['trades']]) if perf['trades'] else 0,
                'win_rate': sum(1 for t in perf['trades'] if t['net_return'] > 0) / len(perf['trades']) if perf['trades'] else 0,
                'avg_win': np.mean([t['net_return'] for t in perf['trades'] if t['net_return'] > 0]) if any(t['net_return'] > 0 for t in perf['trades']) else 0,
                'avg_loss': np.mean([t['net_return'] for t in perf['trades'] if t['net_return'] < 0]) if any(t['net_return'] < 0 for t in perf['trades']) else 0,
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """Generate a summary report of the analysis."""
        report = []
        report.append(f"# Swing Pivot Bounce Zones Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Workspace: {self.workspace_path}")
        report.append("")
        
        report.append("## Overview")
        report.append(f"- Total strategies analyzed: {len(results_df)}")
        report.append(f"- Total trades across all strategies: {results_df['num_trades'].sum()}")
        report.append(f"- Average trades per strategy: {results_df['num_trades'].mean():.1f}")
        report.append("")
        
        report.append("## Performance Summary")
        report.append(f"- Average return per strategy: {results_df['total_return'].mean()*100:.2f}%")
        report.append(f"- Median return per strategy: {results_df['total_return'].median()*100:.2f}%")
        report.append(f"- Best performing strategy: {results_df['total_return'].max()*100:.2f}%")
        report.append(f"- Worst performing strategy: {results_df['total_return'].min()*100:.2f}%")
        report.append(f"- Strategies with positive returns: {(results_df['total_return'] > 0).sum()} ({(results_df['total_return'] > 0).mean()*100:.1f}%)")
        report.append("")
        
        report.append("## Trade Statistics")
        report.append(f"- Average win rate: {results_df['win_rate'].mean()*100:.1f}%")
        report.append(f"- Average trade duration: {results_df['avg_trade_duration'].mean():.1f} bars")
        report.append(f"- Average winning trade: {results_df['avg_win'].mean()*100:.2f}%")
        report.append(f"- Average losing trade: {results_df['avg_loss'].mean()*100:.2f}%")
        report.append("")
        
        # Top 10 strategies
        report.append("## Top 10 Performing Strategies")
        top_10 = results_df.nlargest(10, 'total_return')
        for idx, row in top_10.iterrows():
            report.append(f"- {row['strategy_id']}: {row['total_return']*100:.2f}% ({row['num_trades']} trades, {row['win_rate']*100:.1f}% win rate)")
        report.append("")
        
        # Bottom 10 strategies
        report.append("## Bottom 10 Performing Strategies")
        bottom_10 = results_df.nsmallest(10, 'total_return')
        for idx, row in bottom_10.iterrows():
            report.append(f"- {row['strategy_id']}: {row['total_return']*100:.2f}% ({row['num_trades']} trades, {row['win_rate']*100:.1f}% win rate)")
        
        return "\n".join(report)


def main():
    # Analyze the swing pivot workspace
    workspace_path = "/Users/daws/ADMF-PC/workspaces/signal_generation_a2d31737"
    
    print(f"Analyzing workspace: {workspace_path}")
    analyzer = SwingPivotAnalyzer(workspace_path)
    
    # Analyze all strategies (or limit for testing)
    results_df = analyzer.analyze_all_strategies(limit=None)
    
    # Save results
    results_df.to_csv("swing_pivot_analysis_results.csv", index=False)
    print(f"\nSaved detailed results to swing_pivot_analysis_results.csv")
    
    # Generate and save summary report
    report = analyzer.generate_summary_report(results_df)
    
    with open("swing_pivot_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"Saved summary report to swing_pivot_analysis_report.md")
    print("\n" + "="*50)
    print(report)


if __name__ == "__main__":
    main()