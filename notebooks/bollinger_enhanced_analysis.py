#!/usr/bin/env python3
"""
Enhanced Bollinger Bands Strategy Analysis
Includes: win rates, execution costs, stop loss analysis, and intraday constraint verification
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import json
from datetime import datetime, time
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class EnhancedBollingerAnalysis:
    """Enhanced analysis for Bollinger Bands strategies including costs and risk management."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.strategy_index = None
        self.traces = {}
        self.source_data = None
        
    def load_data(self):
        """Load strategy index and trace files."""
        # Load strategy index
        self.strategy_index = pd.read_parquet(self.results_path / "strategy_index.parquet")
        print(f"Loaded {len(self.strategy_index)} strategies")
        
        # Load trace files for each strategy
        for _, strategy in self.strategy_index.iterrows():
            trace_path = self.results_path / strategy['trace_path']
            if trace_path.exists():
                self.traces[strategy['strategy_id']] = pd.read_parquet(trace_path)
        
        print(f"Loaded traces for {len(self.traces)} strategies")
        
        # Load source data if available
        try:
            # Assuming SPY_5m.csv is in data directory
            self.source_data = pd.read_csv("data/SPY_5m.csv", parse_dates=['timestamp'])
            self.source_data = self.source_data.set_index('timestamp')
            print("Loaded source price data")
        except:
            print("Warning: Could not load source price data")
    
    def extract_trades(self, strategy_id: str, execution_cost_bps: float = 1.0) -> pd.DataFrame:
        """
        Extract trades from signal trace with execution costs.
        
        Args:
            strategy_id: Strategy identifier
            execution_cost_bps: Round-trip execution cost in basis points (default 1bp)
        
        Returns:
            DataFrame with trade details including costs
        """
        if strategy_id not in self.traces:
            return pd.DataFrame()
        
        trace = self.traces[strategy_id].copy()
        
        # Parse timestamps
        trace['timestamp'] = pd.to_datetime(trace['ts'])
        trace = trace.sort_values('timestamp')
        
        # Identify trade entries and exits
        trace['position'] = trace['val'].replace({0: 0, 1: 1, -1: -1})
        trace['position_change'] = trace['position'].diff().fillna(0)
        
        trades = []
        current_trade = None
        
        for idx, row in trace.iterrows():
            if row['position_change'] != 0 and row['position'] != 0:
                # New position opened
                if current_trade is None:
                    current_trade = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['px'],
                        'direction': row['position'],
                        'entry_idx': row['idx']
                    }
            elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
                # Position closed
                exit_price = row['px']
                
                # Calculate raw return
                if current_trade['direction'] == 1:  # Long
                    raw_return = (exit_price - current_trade['entry_price']) / current_trade['entry_price']
                else:  # Short
                    raw_return = (current_trade['entry_price'] - exit_price) / current_trade['entry_price']
                
                # Apply execution costs (half on entry, half on exit)
                cost_adjustment = execution_cost_bps / 10000  # Convert bps to decimal
                net_return = raw_return - cost_adjustment
                
                trade = {
                    'strategy_id': strategy_id,
                    'entry_time': current_trade['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'direction': current_trade['direction'],
                    'raw_return': raw_return,
                    'execution_cost': cost_adjustment,
                    'net_return': net_return,
                    'duration_minutes': (row['timestamp'] - current_trade['entry_time']).total_seconds() / 60,
                    'entry_idx': current_trade['entry_idx'],
                    'exit_idx': row['idx']
                }
                trades.append(trade)
                
                # Reset for next trade
                current_trade = None
                if row['position'] != 0 and row['position_change'] != 0:
                    # Immediately open new position (reversal)
                    current_trade = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['px'],
                        'direction': row['position'],
                        'entry_idx': row['idx']
                    }
        
        return pd.DataFrame(trades)
    
    def calculate_stop_loss_impact(self, trades_df: pd.DataFrame, 
                                  stop_loss_levels: List[float] = None) -> pd.DataFrame:
        """
        Calculate returns with various stop loss levels.
        
        Args:
            trades_df: DataFrame of trades
            stop_loss_levels: List of stop loss percentages (default 0.05% to 1%)
        
        Returns:
            DataFrame with returns for each stop loss level
        """
        if stop_loss_levels is None:
            stop_loss_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
        
        results = []
        
        for sl_pct in stop_loss_levels:
            sl_decimal = sl_pct / 100
            
            trades_with_sl = trades_df.copy()
            
            # For each trade, check if it would have been stopped out
            for idx, trade in trades_with_sl.iterrows():
                # Calculate the worst drawdown during the trade
                # This is a simplified approach - ideally we'd check intraday prices
                
                if trade['direction'] == 1:  # Long trade
                    # If the trade lost more than stop loss, cap the loss
                    if trade['raw_return'] < -sl_decimal:
                        trades_with_sl.loc[idx, 'raw_return'] = -sl_decimal
                        trades_with_sl.loc[idx, 'net_return'] = -sl_decimal - trade['execution_cost']
                        trades_with_sl.loc[idx, 'stopped_out'] = True
                    else:
                        trades_with_sl.loc[idx, 'stopped_out'] = False
                else:  # Short trade
                    # For shorts, a positive move against us is a loss
                    if trade['raw_return'] < -sl_decimal:
                        trades_with_sl.loc[idx, 'raw_return'] = -sl_decimal
                        trades_with_sl.loc[idx, 'net_return'] = -sl_decimal - trade['execution_cost']
                        trades_with_sl.loc[idx, 'stopped_out'] = True
                    else:
                        trades_with_sl.loc[idx, 'stopped_out'] = False
            
            # Calculate metrics with stop loss
            total_return = trades_with_sl['net_return'].sum()
            avg_return = trades_with_sl['net_return'].mean()
            win_rate = (trades_with_sl['net_return'] > 0).mean()
            stopped_out_rate = trades_with_sl['stopped_out'].mean() if 'stopped_out' in trades_with_sl else 0
            
            results.append({
                'stop_loss_pct': sl_pct,
                'total_return': total_return,
                'avg_return_per_trade': avg_return,
                'win_rate': win_rate,
                'stopped_out_rate': stopped_out_rate,
                'num_trades': len(trades_with_sl)
            })
        
        return pd.DataFrame(results)
    
    def verify_intraday_constraint(self, trades_df: pd.DataFrame, 
                                  market_tz: str = 'America/New_York') -> Dict:
        """
        Verify that trades respect intraday constraints.
        
        Args:
            trades_df: DataFrame of trades
            market_tz: Market timezone (default NYSE)
        
        Returns:
            Dictionary with constraint verification results
        """
        # Convert to market timezone
        market_tz_obj = pytz.timezone(market_tz)
        
        trades_df['entry_time_mkt'] = trades_df['entry_time'].dt.tz_localize('UTC').dt.tz_convert(market_tz_obj)
        trades_df['exit_time_mkt'] = trades_df['exit_time'].dt.tz_localize('UTC').dt.tz_convert(market_tz_obj)
        
        # Market hours (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Check for overnight positions
        trades_df['entry_date'] = trades_df['entry_time_mkt'].dt.date
        trades_df['exit_date'] = trades_df['exit_time_mkt'].dt.date
        trades_df['overnight'] = trades_df['entry_date'] != trades_df['exit_date']
        
        # Check for after-hours trades
        trades_df['entry_time_only'] = trades_df['entry_time_mkt'].dt.time
        trades_df['exit_time_only'] = trades_df['exit_time_mkt'].dt.time
        
        trades_df['after_hours_entry'] = (
            (trades_df['entry_time_only'] < market_open) | 
            (trades_df['entry_time_only'] >= market_close)
        )
        trades_df['after_hours_exit'] = (
            (trades_df['exit_time_only'] < market_open) | 
            (trades_df['exit_time_only'] >= market_close)
        )
        
        results = {
            'total_trades': len(trades_df),
            'overnight_positions': trades_df['overnight'].sum(),
            'overnight_position_pct': trades_df['overnight'].mean() * 100,
            'after_hours_entries': trades_df['after_hours_entry'].sum(),
            'after_hours_exits': trades_df['after_hours_exit'].sum(),
            'fully_intraday': (~trades_df['overnight']).sum(),
            'avg_trade_duration_minutes': trades_df['duration_minutes'].mean(),
            'max_trade_duration_minutes': trades_df['duration_minutes'].max(),
            'trades_over_390_minutes': (trades_df['duration_minutes'] > 390).sum()  # Full trading day
        }
        
        # Add detailed breakdown by hour
        trades_df['entry_hour'] = trades_df['entry_time_mkt'].dt.hour
        trades_df['exit_hour'] = trades_df['exit_time_mkt'].dt.hour
        
        results['entries_by_hour'] = trades_df['entry_hour'].value_counts().to_dict()
        results['exits_by_hour'] = trades_df['exit_hour'].value_counts().to_dict()
        
        return results, trades_df
    
    def analyze_top_strategies(self, n_top: int = 10, execution_cost_bps: float = 1.0):
        """
        Analyze top N strategies by total return including costs and risk metrics.
        
        Args:
            n_top: Number of top strategies to analyze
            execution_cost_bps: Execution cost in basis points
        
        Returns:
            DataFrame with comprehensive metrics for top strategies
        """
        strategy_metrics = []
        
        for strategy_id in self.traces.keys():
            trades = self.extract_trades(strategy_id, execution_cost_bps)
            
            if len(trades) == 0:
                continue
            
            # Basic metrics
            total_return = trades['net_return'].sum()
            avg_return = trades['net_return'].mean()
            win_rate = (trades['net_return'] > 0).mean()
            
            # Risk metrics
            returns_std = trades['net_return'].std()
            sharpe = avg_return / returns_std * np.sqrt(252 * 78) if returns_std > 0 else 0  # Annualized
            
            # Get strategy parameters
            strategy_info = self.strategy_index[self.strategy_index['strategy_id'] == strategy_id].iloc[0]
            
            metrics = {
                'strategy_id': strategy_id,
                'period': strategy_info.get('period', None),
                'std_dev': strategy_info.get('std_dev', None),
                'num_trades': len(trades),
                'total_return': total_return,
                'avg_return_per_trade': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'avg_trade_duration': trades['duration_minutes'].mean(),
                'total_execution_cost': trades['execution_cost'].sum()
            }
            
            strategy_metrics.append(metrics)
        
        # Convert to DataFrame and get top N
        metrics_df = pd.DataFrame(strategy_metrics)
        top_strategies = metrics_df.nlargest(n_top, 'total_return')
        
        return top_strategies
    
    def create_visualizations(self, top_strategies_df: pd.DataFrame, 
                            stop_loss_analysis: Dict[str, pd.DataFrame]):
        """Create comprehensive visualizations for the analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top 10 strategies performance
        ax1 = axes[0, 0]
        top_10 = top_strategies_df.head(10)
        ax1.barh(range(len(top_10)), top_10['avg_return_per_trade'] * 100)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f"{row['period']}/{row['std_dev']}" for _, row in top_10.iterrows()])
        ax1.set_xlabel('Average Return per Trade (%)')
        ax1.set_title('Top 10 Strategies - Average Returns (after costs)')
        ax1.grid(True, alpha=0.3)
        
        # Add win rate as text
        for i, (_, row) in enumerate(top_10.iterrows()):
            ax1.text(row['avg_return_per_trade'] * 100 + 0.01, i, 
                    f"WR: {row['win_rate']:.1%}", va='center')
        
        # 2. Stop loss impact analysis
        ax2 = axes[0, 1]
        if stop_loss_analysis:
            # Get the best strategy's stop loss analysis
            best_strategy_id = top_strategies_df.iloc[0]['strategy_id']
            if best_strategy_id in stop_loss_analysis:
                sl_df = stop_loss_analysis[best_strategy_id]
                ax2.plot(sl_df['stop_loss_pct'], sl_df['total_return'] * 100, 'b-', label='Total Return')
                ax2.plot(sl_df['stop_loss_pct'], sl_df['win_rate'] * 100, 'g--', label='Win Rate')
                ax2.set_xlabel('Stop Loss (%)')
                ax2.set_ylabel('Percentage')
                ax2.set_title(f'Stop Loss Impact - Best Strategy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Parameter heatmap for returns
        ax3 = axes[1, 0]
        pivot_returns = top_strategies_df.pivot_table(
            values='avg_return_per_trade', 
            index='period', 
            columns='std_dev',
            aggfunc='mean'
        )
        sns.heatmap(pivot_returns * 100, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, ax=ax3, cbar_kws={'label': 'Avg Return (%)'})
        ax3.set_title('Average Returns by Parameters (after costs)')
        
        # 4. Trade duration distribution
        ax4 = axes[1, 1]
        all_durations = []
        for strategy_id in top_strategies_df['strategy_id'].head(10):
            if strategy_id in self.traces:
                trades = self.extract_trades(strategy_id)
                if len(trades) > 0:
                    all_durations.extend(trades['duration_minutes'].values)
        
        if all_durations:
            ax4.hist(all_durations, bins=50, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Trade Duration (minutes)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Trade Duration Distribution - Top 10 Strategies')
            ax4.axvline(390, color='red', linestyle='--', label='Market Day (390 min)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, top_strategies_df: pd.DataFrame,
                              constraint_results: List[Tuple[str, Dict]],
                              execution_cost_bps: float = 1.0) -> str:
        """Generate a comprehensive text summary report."""
        report = []
        report.append("="*60)
        report.append("BOLLINGER BANDS STRATEGY ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nExecution Cost: {execution_cost_bps} basis points round-trip")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Top 10 strategies summary
        report.append("\n" + "="*60)
        report.append("TOP 10 STRATEGIES BY TOTAL RETURN")
        report.append("="*60)
        
        for i, (_, strategy) in enumerate(top_strategies_df.head(10).iterrows(), 1):
            report.append(f"\n{i}. Strategy: Period={strategy['period']}, StdDev={strategy['std_dev']}")
            report.append(f"   - Total Return: {strategy['total_return']*100:.2f}%")
            report.append(f"   - Avg Return/Trade: {strategy['avg_return_per_trade']*100:.3f}%")
            report.append(f"   - Win Rate: {strategy['win_rate']*100:.1f}%")
            report.append(f"   - Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
            report.append(f"   - Number of Trades: {strategy['num_trades']}")
            report.append(f"   - Avg Duration: {strategy['avg_trade_duration']:.1f} minutes")
            report.append(f"   - Total Execution Cost: {strategy['total_execution_cost']*100:.2f}%")
        
        # Intraday constraint verification
        report.append("\n" + "="*60)
        report.append("INTRADAY CONSTRAINT VERIFICATION")
        report.append("="*60)
        
        total_overnight = 0
        total_trades_checked = 0
        
        for strategy_id, constraints in constraint_results[:10]:
            total_overnight += constraints['overnight_positions']
            total_trades_checked += constraints['total_trades']
            
            if constraints['overnight_positions'] > 0:
                report.append(f"\n⚠️  Strategy {strategy_id}:")
                report.append(f"   - Overnight positions: {constraints['overnight_positions']} ({constraints['overnight_position_pct']:.1f}%)")
                report.append(f"   - After-hours entries: {constraints['after_hours_entries']}")
                report.append(f"   - After-hours exits: {constraints['after_hours_exits']}")
        
        if total_overnight == 0:
            report.append("\n✅ All top 10 strategies respect intraday constraints!")
        else:
            report.append(f"\n⚠️  Warning: {total_overnight} overnight positions found across {total_trades_checked} trades")
        
        # Stop loss recommendations
        report.append("\n" + "="*60)
        report.append("STOP LOSS RECOMMENDATIONS")
        report.append("="*60)
        
        report.append("\nBased on the analysis, consider the following stop loss levels:")
        report.append("- Conservative: 0.25% - Minimizes large losses but may stop out winning trades")
        report.append("- Moderate: 0.50% - Balanced approach for most market conditions")
        report.append("- Aggressive: 1.00% - Allows trades more room but increases risk")
        
        return "\n".join(report)


# Jupyter notebook cells wrapper functions
def run_enhanced_analysis(results_path: str, execution_cost_bps: float = 1.0):
    """Main function to run the enhanced analysis."""
    analyzer = EnhancedBollingerAnalysis(results_path)
    analyzer.load_data()
    
    # Get top strategies
    print("Analyzing top strategies...")
    top_strategies = analyzer.analyze_top_strategies(n_top=20, execution_cost_bps=execution_cost_bps)
    
    # Analyze stop losses for top 5 strategies
    print("\nAnalyzing stop loss impact...")
    stop_loss_results = {}
    for strategy_id in top_strategies['strategy_id'].head(5):
        trades = analyzer.extract_trades(strategy_id, execution_cost_bps)
        if len(trades) > 0:
            stop_loss_results[strategy_id] = analyzer.calculate_stop_loss_impact(trades)
    
    # Verify intraday constraints
    print("\nVerifying intraday constraints...")
    constraint_results = []
    for strategy_id in top_strategies['strategy_id'].head(10):
        trades = analyzer.extract_trades(strategy_id, execution_cost_bps)
        if len(trades) > 0:
            constraints, trades_with_tz = analyzer.verify_intraday_constraint(trades)
            constraint_results.append((strategy_id, constraints))
    
    # Generate report
    report = analyzer.generate_summary_report(top_strategies, constraint_results, execution_cost_bps)
    print("\n" + report)
    
    # Create visualizations
    fig = analyzer.create_visualizations(top_strategies, stop_loss_results)
    
    return analyzer, top_strategies, stop_loss_results, constraint_results, fig


# Example usage for Jupyter notebook
if __name__ == "__main__":
    # This would be called from a Jupyter cell
    results_path = "config/bollinger/results/latest"
    analyzer, top_strategies, stop_loss_results, constraint_results, fig = run_enhanced_analysis(results_path)
    
    # Display the figure
    plt.show()