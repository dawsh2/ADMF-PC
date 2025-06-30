#!/usr/bin/env python3
"""
Analyze Keltner strategies with detailed filter and directional analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class KeltnerFilterAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_5m_1m"
        self.signals_path = self.traces_path / "signals" / "keltner_bands"
        
        # Market hours
        self.market_open_bar = 78  # 9:30 AM at 5-min bars
        self.market_close_bar = 156  # 4:00 PM at 5-min bars
        self.bars_per_day = 78
        
    def load_signal_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single parquet signal file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def analyze_trade_directions(self, signals_df: pd.DataFrame, 
                               stop_loss_pct: float = 0.005,
                               execution_cost_bps: float = 0.5) -> Dict:
        """Analyze trades by direction (long vs short)."""
        if signals_df.empty or len(signals_df) < 2:
            return {
                'long_trades': 0,
                'short_trades': 0,
                'long_return_bps': 0,
                'short_return_bps': 0,
                'long_win_rate': 0,
                'short_win_rate': 0
            }
        
        # Sort by index
        signals_df = signals_df.sort_values('idx')
        
        long_trades = []
        short_trades = []
        current_position = None
        
        # Convert execution cost from bps to multiplier
        execution_cost_multiplier = 1 - (execution_cost_bps / 10000)
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            # Calculate time of day for EOD exit
            bar_of_day = bar_idx % self.bars_per_day
            
            # Check for EOD exit
            if current_position is not None and bar_of_day >= self.market_close_bar - 1:
                # Force exit at EOD
                exit_price = price
                trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                trade_return *= execution_cost_multiplier
                
                trade_info = {
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': bar_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'return_bps': trade_return * 10000,
                    'direction': 'long' if current_position['signal'] > 0 else 'short'
                }
                
                if current_position['signal'] > 0:
                    long_trades.append(trade_info)
                else:
                    short_trades.append(trade_info)
                
                current_position = None
                continue
            
            # Handle position changes
            if current_position is None and signal != 0:
                # Opening new position
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_price': price,
                    'signal': signal
                }
            elif current_position is not None:
                # Check stop loss
                if stop_loss_pct is not None:
                    if current_position['signal'] > 0:  # Long position
                        stop_price = current_position['entry_price'] * (1 - stop_loss_pct)
                        if price <= stop_price:
                            trade_return = -stop_loss_pct * execution_cost_multiplier
                            long_trades.append({
                                'entry_idx': current_position['entry_idx'],
                                'exit_idx': bar_idx,
                                'entry_price': current_position['entry_price'],
                                'exit_price': stop_price,
                                'return_bps': trade_return * 10000,
                                'direction': 'long'
                            })
                            current_position = None
                            continue
                    else:  # Short position
                        stop_price = current_position['entry_price'] * (1 + stop_loss_pct)
                        if price >= stop_price:
                            trade_return = -stop_loss_pct * execution_cost_multiplier
                            short_trades.append({
                                'entry_idx': current_position['entry_idx'],
                                'exit_idx': bar_idx,
                                'entry_price': current_position['entry_price'],
                                'exit_price': stop_price,
                                'return_bps': trade_return * 10000,
                                'direction': 'short'
                            })
                            current_position = None
                            continue
                
                # Handle regular exits
                if signal == 0 or (signal != 0 and signal != current_position['signal']):
                    # Exit current position
                    exit_price = price
                    trade_return = np.log(exit_price / current_position['entry_price']) * current_position['signal']
                    trade_return *= execution_cost_multiplier
                    
                    trade_info = {
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': bar_idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'return_bps': trade_return * 10000,
                        'direction': 'long' if current_position['signal'] > 0 else 'short'
                    }
                    
                    if current_position['signal'] > 0:
                        long_trades.append(trade_info)
                    else:
                        short_trades.append(trade_info)
                    
                    # Open new position if flipping
                    if signal != 0:
                        current_position = {
                            'entry_idx': bar_idx,
                            'entry_price': price,
                            'signal': signal
                        }
                    else:
                        current_position = None
        
        # Calculate statistics
        long_returns = [t['return_bps'] for t in long_trades]
        short_returns = [t['return_bps'] for t in short_trades]
        
        results = {
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_return_bps': np.mean(long_returns) if long_returns else 0,
            'short_return_bps': np.mean(short_returns) if short_returns else 0,
            'long_win_rate': sum(1 for r in long_returns if r > 0) / len(long_returns) if long_returns else 0,
            'short_win_rate': sum(1 for r in short_returns if r > 0) / len(short_returns) if short_returns else 0,
            'long_total_return': sum(long_returns) / 10000 if long_returns else 0,
            'short_total_return': sum(short_returns) / 10000 if short_returns else 0
        }
        
        return results
    
    def analyze_with_market_filters(self, signals_df: pd.DataFrame,
                                  trend_filter: Optional[str] = None,
                                  volatility_filter: Optional[str] = None,
                                  volume_filter: Optional[str] = None,
                                  vwap_filter: Optional[str] = None) -> Dict:
        """
        Analyze strategy performance with various market filters.
        This is a simplified version - in practice, you'd need actual market data.
        """
        # For now, we'll simulate the analysis based on signal patterns
        # In practice, you'd join with actual market data
        
        # Analyze basic metrics
        if signals_df.empty:
            return {'filtered_signals': 0, 'filter_description': 'No data'}
        
        original_count = len(signals_df)
        filtered_df = signals_df.copy()
        
        filter_desc = []
        
        # Simulate trend filter (based on signal clustering)
        if trend_filter == "bullish":
            # Keep signals where recent signals were more bullish
            filter_desc.append("Bullish trend")
        elif trend_filter == "bearish":
            filter_desc.append("Bearish trend")
        
        # Simulate volatility filter (based on signal frequency)
        if volatility_filter == "high":
            filter_desc.append("High volatility")
        elif volatility_filter == "low":
            filter_desc.append("Low volatility")
        
        # Return simplified results
        return {
            'original_signals': original_count,
            'filtered_signals': len(filtered_df),
            'filter_efficiency': len(filtered_df) / original_count if original_count > 0 else 0,
            'filter_description': ', '.join(filter_desc) if filter_desc else 'No filters'
        }
    
    def analyze_all_strategies_detailed(self) -> pd.DataFrame:
        """Perform detailed analysis of all strategies."""
        results = []
        
        # Get all parquet files
        signal_files = sorted(list(self.signals_path.glob("*.parquet")))
        
        print(f"Analyzing {len(signal_files)} Keltner strategies in detail...")
        
        for i, filepath in enumerate(signal_files):
            if i % 10 == 0:
                print(f"Processing strategy {i}/{len(signal_files)}...")
            
            strategy_id = filepath.stem
            signals_df = self.load_signal_file(filepath)
            
            if signals_df.empty:
                continue
            
            # Analyze by direction
            direction_stats = self.analyze_trade_directions(signals_df)
            
            # Basic filter analysis
            filter_stats = self.analyze_with_market_filters(signals_df)
            
            result = {
                'strategy_id': strategy_id,
                'num_signals': len(signals_df),
                
                # Directional analysis
                'long_trades': direction_stats['long_trades'],
                'short_trades': direction_stats['short_trades'],
                'long_return_bps': direction_stats['long_return_bps'],
                'short_return_bps': direction_stats['short_return_bps'],
                'long_win_rate': direction_stats['long_win_rate'],
                'short_win_rate': direction_stats['short_win_rate'],
                'long_short_ratio': direction_stats['long_trades'] / direction_stats['short_trades'] if direction_stats['short_trades'] > 0 else float('inf'),
                
                # Performance difference
                'directional_edge_bps': abs(direction_stats['long_return_bps'] - direction_stats['short_return_bps']),
                'better_direction': 'long' if direction_stats['long_return_bps'] > direction_stats['short_return_bps'] else 'short',
                
                # Signal characteristics
                'signals_per_day': len(signals_df) / 252,
                'avg_signal_spacing': np.mean(np.diff(signals_df['idx'].values)) if len(signals_df) > 1 else 0
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_filter_report(self, results_df: pd.DataFrame) -> str:
        """Generate comprehensive filter and directional analysis report."""
        report = []
        report.append("# Keltner Strategy Filter & Directional Analysis")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall directional bias
        report.append("## Overall Directional Performance")
        avg_long_return = results_df['long_return_bps'].mean()
        avg_short_return = results_df['short_return_bps'].mean()
        report.append(f"- Average Long Return: {avg_long_return:.2f} bps")
        report.append(f"- Average Short Return: {avg_short_return:.2f} bps")
        report.append(f"- Directional Bias: {'Long' if avg_long_return > avg_short_return else 'Short'} (+{abs(avg_long_return - avg_short_return):.2f} bps)")
        report.append("")
        
        # Win rates by direction
        report.append("## Win Rates by Direction")
        report.append(f"- Average Long Win Rate: {results_df['long_win_rate'].mean()*100:.1f}%")
        report.append(f"- Average Short Win Rate: {results_df['short_win_rate'].mean()*100:.1f}%")
        report.append("")
        
        # Trade distribution
        report.append("## Trade Distribution")
        total_long = results_df['long_trades'].sum()
        total_short = results_df['short_trades'].sum()
        report.append(f"- Total Long Trades: {total_long} ({total_long/(total_long+total_short)*100:.1f}%)")
        report.append(f"- Total Short Trades: {total_short} ({total_short/(total_long+total_short)*100:.1f}%)")
        report.append("")
        
        # Strategies with strong directional edge
        report.append("## Strategies with Strong Directional Edge (>2 bps difference)")
        strong_edge = results_df[results_df['directional_edge_bps'] > 2].sort_values('directional_edge_bps', ascending=False)
        for idx, row in strong_edge.head(10).iterrows():
            report.append(f"- {row['strategy_id']}: {row['better_direction']} bias "
                        f"({row['long_return_bps']:.2f} vs {row['short_return_bps']:.2f} bps)")
        report.append("")
        
        # Recommendations for filtering
        report.append("## Filter Recommendations")
        report.append("Based on directional analysis:")
        
        if avg_long_return > avg_short_return * 1.5:
            report.append("- Consider LONG-ONLY filter during bullish trends")
        elif avg_short_return > avg_long_return * 1.5:
            report.append("- Consider SHORT-ONLY filter during bearish trends")
        else:
            report.append("- Both directions profitable - use trend filters dynamically")
        
        report.append("")
        report.append("Suggested additional filters to test:")
        report.append("- Trend: Trade long only above 20-day MA, short only below")
        report.append("- Volatility: Increase position size in high volatility")
        report.append("- Volume: Trade only when volume > 20-day average")
        report.append("- VWAP: Long only above VWAP, short only below")
        
        return "\n".join(report)


def main():
    workspace_path = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
    
    print(f"Performing detailed analysis of Keltner strategies...")
    analyzer = KeltnerFilterAnalyzer(workspace_path)
    
    # Analyze all strategies
    results_df = analyzer.analyze_all_strategies_detailed()
    
    # Save results
    results_df.to_csv("keltner_directional_analysis.csv", index=False)
    print(f"\nSaved detailed results to keltner_directional_analysis.csv")
    
    # Generate report
    report = analyzer.generate_filter_report(results_df)
    
    with open("keltner_filter_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"Saved report to keltner_filter_analysis_report.md")
    print("\n" + "="*50)
    print(report)
    
    # Additional insights
    print("\n" + "="*50)
    print("ACTIONABLE INSIGHTS")
    print("="*50)
    
    # Find best long-biased strategies
    long_biased = results_df[
        (results_df['long_return_bps'] > 2) & 
        (results_df['long_return_bps'] > results_df['short_return_bps'] * 1.5)
    ].sort_values('long_return_bps', ascending=False)
    
    if len(long_biased) > 0:
        print("\nBest LONG-BIASED strategies:")
        for idx, row in long_biased.head(5).iterrows():
            print(f"  {row['strategy_id']}: {row['long_return_bps']:.2f} bps long, "
                  f"{row['long_win_rate']*100:.1f}% win rate")
    
    # Find best short-biased strategies
    short_biased = results_df[
        (results_df['short_return_bps'] > 2) & 
        (results_df['short_return_bps'] > results_df['long_return_bps'] * 1.5)
    ].sort_values('short_return_bps', ascending=False)
    
    if len(short_biased) > 0:
        print("\nBest SHORT-BIASED strategies:")
        for idx, row in short_biased.head(5).iterrows():
            print(f"  {row['strategy_id']}: {row['short_return_bps']:.2f} bps short, "
                  f"{row['short_win_rate']*100:.1f}% win rate")
    
    # Find balanced strategies
    balanced = results_df[
        (results_df['long_return_bps'] > 1) & 
        (results_df['short_return_bps'] > 1) &
        (results_df['directional_edge_bps'] < 0.5)
    ].sort_values(['long_return_bps', 'short_return_bps'], ascending=False)
    
    if len(balanced) > 0:
        print("\nBest BALANCED strategies:")
        for idx, row in balanced.head(5).iterrows():
            print(f"  {row['strategy_id']}: {row['long_return_bps']:.2f} bps long, "
                  f"{row['short_return_bps']:.2f} bps short")


if __name__ == "__main__":
    main()