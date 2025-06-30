#!/usr/bin/env python3
"""
Corrected stop loss analysis that properly tracks price movement during trades.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict, List, Tuple
import json

class CorrectedStopLossAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_5m_1m"
        self.signals_path = self.traces_path / "signals" / "keltner_bands"
        
        # Market hours
        self.market_open_bar = 78
        self.market_close_bar = 156
        self.bars_per_day = 78
        
    def load_signal_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single parquet signal file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def simulate_trades_with_stops(self, signals_df: pd.DataFrame, 
                                 stop_loss_pct: float,
                                 execution_cost_bps: float = 0.5) -> Dict:
        """
        Simulate trades with proper stop loss logic.
        This version correctly tracks price movements between signals.
        """
        if signals_df.empty or len(signals_df) < 2:
            return self._empty_results()
        
        # Sort by index
        signals_df = signals_df.sort_values('idx').reset_index(drop=True)
        
        trades = []
        current_position = None
        
        # Convert execution cost
        execution_cost_multiplier = 1 - (execution_cost_bps / 10000)
        
        # Track statistics
        stopped_trades = 0
        stopped_winners = 0
        stopped_losers = 0
        
        # Process each signal
        for i in range(len(signals_df)):
            current_bar = signals_df.iloc[i]
            signal = current_bar['val']
            current_price = current_bar['px']
            bar_idx = current_bar['idx']
            
            # If we have a position, check if stop would have been hit
            if current_position is not None:
                # Calculate unrealized P&L
                if current_position['direction'] == 'long':
                    unrealized_pnl = (current_price / current_position['entry_price'] - 1)
                    stop_price = current_position['entry_price'] * (1 - stop_loss_pct)
                    
                    # Check if stop hit
                    if current_price <= stop_price or unrealized_pnl <= -stop_loss_pct:
                        # Stop hit - close position
                        exit_price = stop_price
                        trade_return = -stop_loss_pct * execution_cost_multiplier
                        
                        # Was this trade ever profitable?
                        max_profit = (current_position['high_water_mark'] / current_position['entry_price'] - 1)
                        was_winner = max_profit > 0
                        
                        trades.append({
                            'entry_idx': current_position['entry_idx'],
                            'exit_idx': bar_idx,
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'return_bps': trade_return * 10000,
                            'exit_type': 'stop_loss',
                            'direction': 'long',
                            'was_winner_before_stop': was_winner,
                            'max_profit_before_stop': max_profit * 10000
                        })
                        
                        stopped_trades += 1
                        if was_winner:
                            stopped_winners += 1
                        else:
                            stopped_losers += 1
                        
                        current_position = None
                        continue
                    
                    # Update high water mark
                    current_position['high_water_mark'] = max(current_position['high_water_mark'], current_price)
                    
                else:  # Short position
                    unrealized_pnl = (current_position['entry_price'] / current_price - 1)
                    stop_price = current_position['entry_price'] * (1 + stop_loss_pct)
                    
                    # Check if stop hit
                    if current_price >= stop_price or unrealized_pnl <= -stop_loss_pct:
                        # Stop hit - close position
                        exit_price = stop_price
                        trade_return = -stop_loss_pct * execution_cost_multiplier
                        
                        # Was this trade ever profitable?
                        max_profit = (current_position['entry_price'] / current_position['low_water_mark'] - 1)
                        was_winner = max_profit > 0
                        
                        trades.append({
                            'entry_idx': current_position['entry_idx'],
                            'exit_idx': bar_idx,
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'return_bps': trade_return * 10000,
                            'exit_type': 'stop_loss',
                            'direction': 'short',
                            'was_winner_before_stop': was_winner,
                            'max_profit_before_stop': max_profit * 10000
                        })
                        
                        stopped_trades += 1
                        if was_winner:
                            stopped_winners += 1
                        else:
                            stopped_losers += 1
                        
                        current_position = None
                        continue
                    
                    # Update low water mark
                    current_position['low_water_mark'] = min(current_position['low_water_mark'], current_price)
            
            # Check for EOD exit
            bar_of_day = bar_idx % self.bars_per_day
            if current_position is not None and bar_of_day >= self.market_close_bar - 1:
                # Force exit at EOD
                exit_price = current_price
                if current_position['direction'] == 'long':
                    trade_return = np.log(exit_price / current_position['entry_price'])
                else:
                    trade_return = -np.log(exit_price / current_position['entry_price'])
                trade_return *= execution_cost_multiplier
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': bar_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'return_bps': trade_return * 10000,
                    'exit_type': 'eod',
                    'direction': current_position['direction']
                })
                
                current_position = None
                continue
            
            # Handle signal-based actions
            if signal != 0:
                # Close existing position if we have one
                if current_position is not None:
                    # Exit at current price
                    exit_price = current_price
                    if current_position['direction'] == 'long':
                        trade_return = np.log(exit_price / current_position['entry_price'])
                    else:
                        trade_return = -np.log(exit_price / current_position['entry_price'])
                    trade_return *= execution_cost_multiplier
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': bar_idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'return_bps': trade_return * 10000,
                        'exit_type': 'signal',
                        'direction': current_position['direction']
                    })
                
                # Open new position
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_price': current_price,
                    'direction': 'long' if signal > 0 else 'short',
                    'high_water_mark': current_price,
                    'low_water_mark': current_price
                }
            elif signal == 0 and current_position is not None:
                # Close position
                exit_price = current_price
                if current_position['direction'] == 'long':
                    trade_return = np.log(exit_price / current_position['entry_price'])
                else:
                    trade_return = -np.log(exit_price / current_position['entry_price'])
                trade_return *= execution_cost_multiplier
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': bar_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'return_bps': trade_return * 10000,
                    'exit_type': 'signal',
                    'direction': current_position['direction']
                })
                
                current_position = None
        
        # Calculate statistics
        if trades:
            returns = [t['return_bps'] for t in trades]
            winning_trades = [t for t in trades if t['return_bps'] > 0]
            losing_trades = [t for t in trades if t['return_bps'] < 0]
            
            total_log_return = sum(r / 10000 for r in returns)
            
            # Additional stop analysis
            stop_trades = [t for t in trades if t['exit_type'] == 'stop_loss']
            winners_stopped = [t for t in stop_trades if t.get('was_winner_before_stop', False)]
            
            results = {
                'num_trades': len(trades),
                'total_return': np.exp(total_log_return) - 1,
                'avg_return_per_trade_bps': np.mean(returns),
                'win_rate': len(winning_trades) / len(trades),
                'stopped_trades': stopped_trades,
                'stop_rate': stopped_trades / len(trades),
                'stopped_winners': stopped_winners,
                'stopped_losers': stopped_losers,
                'pct_stopped_were_winners': stopped_winners / stopped_trades if stopped_trades > 0 else 0,
                'avg_win_bps': np.mean([t['return_bps'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss_bps': np.mean([t['return_bps'] for t in losing_trades]) if losing_trades else 0,
                'avg_max_profit_when_stopped': np.mean([t.get('max_profit_before_stop', 0) for t in winners_stopped]) if winners_stopped else 0
            }
        else:
            results = self._empty_results()
        
        return results
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'num_trades': 0,
            'total_return': 0,
            'avg_return_per_trade_bps': 0,
            'win_rate': 0,
            'stopped_trades': 0,
            'stop_rate': 0,
            'stopped_winners': 0,
            'stopped_losers': 0,
            'pct_stopped_were_winners': 0,
            'avg_win_bps': 0,
            'avg_loss_bps': 0,
            'avg_max_profit_when_stopped': 0
        }
    
    def test_stop_losses(self, signals_df: pd.DataFrame, 
                        stop_losses: List[float] = None) -> pd.DataFrame:
        """Test various stop loss levels."""
        if stop_losses is None:
            stop_losses = [0.0001, 0.0002, 0.0003, 0.0005,      # 1-5 bps
                          0.001, 0.002, 0.003, 0.005,           # 10-50 bps
                          0.0075, 0.01, 0.02, 0.05]             # 75 bps - 5%
        
        results = []
        
        # Also test without stop
        no_stop_result = self.simulate_trades_with_stops(signals_df, stop_loss_pct=float('inf'))
        results.append({
            'stop_loss_bps': 'None',
            'stop_loss_pct': None,
            **no_stop_result
        })
        
        for stop_loss in stop_losses:
            result = self.simulate_trades_with_stops(signals_df, stop_loss_pct=stop_loss)
            results.append({
                'stop_loss_bps': stop_loss * 10000,
                'stop_loss_pct': stop_loss,
                **result
            })
        
        return pd.DataFrame(results)


def main():
    workspace = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
    
    print("=== Corrected Stop Loss Analysis for Keltner Strategy_4 ===\n")
    
    analyzer = CorrectedStopLossAnalyzer(workspace)
    strategy_file = analyzer.signals_path / "SPY_5m_compiled_strategy_4.parquet"
    
    if strategy_file.exists():
        signals_df = analyzer.load_signal_file(strategy_file)
        print(f"Loaded {len(signals_df)} signals\n")
        
        # Test stop losses
        results_df = analyzer.test_stop_losses(signals_df)
        
        # Save results
        results_df.to_csv("keltner_strategy4_corrected_stops.csv", index=False)
        
        # Print summary
        print(f"{'Stop Loss':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Stop Rate':<12} {'Winners Stopped':<18} {'% Stops were Winners':<20}")
        print("-" * 95)
        
        for idx, row in results_df.iterrows():
            stop_str = f"{row['stop_loss_bps']:.1f} bps" if pd.notna(row['stop_loss_pct']) else "None"
            print(f"{stop_str:<12} {row['avg_return_per_trade_bps']:>10.2f} "
                  f"{row['win_rate']*100:>8.1f}% {row['stop_rate']*100:>11.1f}% "
                  f"{row['stopped_winners']:>17} {row['pct_stopped_were_winners']*100:>19.1f}%")
        
        # Find optimal
        valid_stops = results_df[results_df['stop_loss_pct'].notna()]
        if len(valid_stops) > 0:
            optimal = valid_stops.loc[valid_stops['avg_return_per_trade_bps'].idxmax()]
            baseline = results_df[results_df['stop_loss_bps'] == 'None'].iloc[0]
            
            print(f"\n=== Analysis Summary ===")
            print(f"Baseline (no stop): {baseline['avg_return_per_trade_bps']:.2f} bps/trade")
            print(f"Optimal stop: {optimal['stop_loss_bps']:.1f} bps")
            print(f"Optimal performance: {optimal['avg_return_per_trade_bps']:.2f} bps/trade")
            print(f"Improvement: {optimal['avg_return_per_trade_bps'] - baseline['avg_return_per_trade_bps']:.2f} bps")
            print(f"\nAt optimal stop level:")
            print(f"- Stop rate: {optimal['stop_rate']*100:.1f}%")
            print(f"- Winners stopped: {optimal['stopped_winners']} ({optimal['pct_stopped_were_winners']*100:.1f}% of all stops)")
            print(f"- Average max profit before stop: {optimal.get('avg_max_profit_when_stopped', 0):.1f} bps")
            
            # Reality check
            tight_stops = valid_stops[valid_stops['stop_loss_bps'] <= 10]
            if len(tight_stops) > 0:
                print(f"\n=== Reality Check (Stops ≤ 10 bps) ===")
                avg_pct_winners = tight_stops['pct_stopped_were_winners'].mean()
                print(f"Average % of stops that were winners: {avg_pct_winners*100:.1f}%")
                print(f"This should be >30% for realistic stop behavior")
    else:
        print("Strategy_4 file not found")


if __name__ == "__main__":
    main()