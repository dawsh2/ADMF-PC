#!/usr/bin/env python3
"""
Proper stop loss analysis by unioning signal data with full OHLC source data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pyarrow.parquet as pq

class FullDataStopAnalyzer:
    def __init__(self, workspace_path: str, source_data_path: str = None):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_5m_1m"
        self.signals_path = self.traces_path / "signals" / "keltner_bands"
        
        # Try to find source data path
        if source_data_path:
            self.source_data_path = Path(source_data_path)
        else:
            # Search for SPY_5m.csv or SPY_5m.parquet as specified
            possible_paths = [
                Path("./data/SPY_5m.csv"),
                Path("./data/SPY_5m.parquet"),
                Path("/Users/daws/ADMF-PC/data/SPY_5m.csv"),
                Path("/Users/daws/ADMF-PC/data/SPY_5m.parquet"),
                Path("./SPY_5m.csv"),
                Path("./SPY_5m.parquet"),
                Path("/Users/daws/ADMF-PC/SPY_5m.csv"),
                Path("/Users/daws/ADMF-PC/SPY_5m.parquet")
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.source_data_path = path
                    break
            else:
                print("Warning: Could not find source data file")
                self.source_data_path = None
        
        # Market hours
        self.market_open_bar = 78
        self.market_close_bar = 156
        self.bars_per_day = 78
        
    def load_source_data(self) -> pd.DataFrame:
        """Load OHLC source data."""
        if not self.source_data_path or not self.source_data_path.exists():
            raise FileNotFoundError(f"Source data not found at {self.source_data_path}")
        
        # Load based on file extension
        if self.source_data_path.suffix == '.parquet':
            df = pd.read_parquet(self.source_data_path)
        else:
            df = pd.read_csv(self.source_data_path)
        
        # Ensure we have required columns (handle different formats)
        required = ['open', 'high', 'low', 'close']
        
        # Check for different column naming conventions
        col_mapping = {}
        for req in required:
            for col in df.columns:
                if req in col.lower():
                    col_mapping[req] = col
                    break
        
        # Rename columns to standard names
        if col_mapping:
            df = df.rename(columns={v: k for k, v in col_mapping.items()})
        
        # Add bar index if not present
        if 'idx' not in df.columns:
            df['idx'] = range(len(df))
        
        # Parse timestamp if present
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
        
        return df
    
    def load_signal_file(self, filepath: Path) -> pd.DataFrame:
        """Load signal parquet file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def union_signals_with_ohlc(self, signals_df: pd.DataFrame, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """Union sparse signals with full OHLC data."""
        # Ensure signals are sorted
        signals_df = signals_df.sort_values('idx').reset_index(drop=True)
        
        # Create signal column in OHLC
        ohlc_df['signal'] = 0
        ohlc_df['signal_price'] = np.nan
        
        # Map signals to OHLC bars
        for _, signal_row in signals_df.iterrows():
            bar_idx = signal_row['idx']
            if bar_idx in ohlc_df.index:
                ohlc_df.loc[bar_idx, 'signal'] = signal_row['val']
                ohlc_df.loc[bar_idx, 'signal_price'] = signal_row['px']
        
        # Forward fill positions between signals
        ohlc_df['position'] = ohlc_df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return ohlc_df
    
    def simulate_with_stops_full_data(self, ohlc_with_signals: pd.DataFrame,
                                     stop_loss_pct: float,
                                     execution_cost_bps: float = 0.5) -> Dict:
        """Simulate trades with stops using full OHLC data."""
        trades = []
        current_position = None
        
        # Convert execution cost
        execution_cost_multiplier = 1 - (execution_cost_bps / 10000)
        
        # Statistics
        stopped_trades = 0
        stopped_winners = 0
        stopped_losers = 0
        
        for idx, row in ohlc_with_signals.iterrows():
            # Check if we have an open position
            if current_position is not None:
                # Check stop loss against actual high/low
                if current_position['direction'] == 'long':
                    # For long position, check if low hit stop
                    stop_price = current_position['entry_price'] * (1 - stop_loss_pct)
                    
                    if row['low'] <= stop_price:
                        # Stop hit
                        exit_price = stop_price
                        trade_return = -stop_loss_pct * execution_cost_multiplier
                        
                        # Check if trade was ever profitable (high > entry)
                        max_profit = (current_position['max_high'] / current_position['entry_price'] - 1)
                        was_winner = max_profit > stop_loss_pct  # Was profitable by more than stop amount
                        
                        trades.append({
                            'entry_idx': current_position['entry_idx'],
                            'exit_idx': idx,
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'return_bps': trade_return * 10000,
                            'exit_type': 'stop_loss',
                            'direction': 'long',
                            'was_winner': was_winner,
                            'max_profit_bps': max_profit * 10000
                        })
                        
                        stopped_trades += 1
                        if was_winner:
                            stopped_winners += 1
                        else:
                            stopped_losers += 1
                        
                        current_position = None
                        continue
                    
                    # Update max high
                    current_position['max_high'] = max(current_position['max_high'], row['high'])
                    
                else:  # Short position
                    # For short position, check if high hit stop
                    stop_price = current_position['entry_price'] * (1 + stop_loss_pct)
                    
                    if row['high'] >= stop_price:
                        # Stop hit
                        exit_price = stop_price
                        trade_return = -stop_loss_pct * execution_cost_multiplier
                        
                        # Check if trade was ever profitable
                        max_profit = (current_position['entry_price'] / current_position['min_low'] - 1)
                        was_winner = max_profit > stop_loss_pct
                        
                        trades.append({
                            'entry_idx': current_position['entry_idx'],
                            'exit_idx': idx,
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'return_bps': trade_return * 10000,
                            'exit_type': 'stop_loss',
                            'direction': 'short',
                            'was_winner': was_winner,
                            'max_profit_bps': max_profit * 10000
                        })
                        
                        stopped_trades += 1
                        if was_winner:
                            stopped_winners += 1
                        else:
                            stopped_losers += 1
                        
                        current_position = None
                        continue
                    
                    # Update min low
                    current_position['min_low'] = min(current_position['min_low'], row['low'])
            
            # Check for signal changes
            if row['signal'] != 0:
                # Close existing position if any
                if current_position is not None:
                    # Exit at close price
                    exit_price = row['close']
                    
                    if current_position['direction'] == 'long':
                        trade_return = np.log(exit_price / current_position['entry_price'])
                    else:
                        trade_return = -np.log(exit_price / current_position['entry_price'])
                    
                    trade_return *= execution_cost_multiplier
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'return_bps': trade_return * 10000,
                        'exit_type': 'signal',
                        'direction': current_position['direction']
                    })
                
                # Open new position if not flat signal
                if row['signal'] != 0:
                    current_position = {
                        'entry_idx': idx,
                        'entry_price': row['close'],  # Enter at close
                        'direction': 'long' if row['signal'] > 0 else 'short',
                        'max_high': row['high'],
                        'min_low': row['low']
                    }
            elif row['signal'] == 0 and current_position is not None:
                # Exit signal
                exit_price = row['close']
                
                if current_position['direction'] == 'long':
                    trade_return = np.log(exit_price / current_position['entry_price'])
                else:
                    trade_return = -np.log(exit_price / current_position['entry_price'])
                
                trade_return *= execution_cost_multiplier
                
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': idx,
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
            
            # Stop analysis
            stop_trades = [t for t in trades if t['exit_type'] == 'stop_loss']
            
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
                'avg_loss_bps': np.mean([t['return_bps'] for t in losing_trades]) if losing_trades else 0
            }
        else:
            results = {
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
                'avg_loss_bps': 0
            }
        
        return results
    
    def test_stop_losses_full_data(self, signals_df: pd.DataFrame,
                                  ohlc_df: pd.DataFrame,
                                  stop_losses: List[float] = None) -> pd.DataFrame:
        """Test various stop loss levels with full OHLC data."""
        if stop_losses is None:
            stop_losses = [0.0001, 0.0002, 0.0005, 0.001, 0.002,    # 1-20 bps
                          0.003, 0.005, 0.0075, 0.01, 0.02, 0.05]   # 30 bps - 5%
        
        # Union data
        print("Unioning signal data with OHLC...")
        full_data = self.union_signals_with_ohlc(signals_df, ohlc_df)
        print(f"Combined data has {len(full_data)} bars")
        
        results = []
        
        # Test without stop
        print("\nTesting without stop loss...")
        no_stop_result = self.simulate_with_stops_full_data(full_data, stop_loss_pct=float('inf'))
        results.append({
            'stop_loss_bps': 'None',
            'stop_loss_pct': None,
            **no_stop_result
        })
        
        # Test each stop level
        for stop_loss in stop_losses:
            print(f"Testing {stop_loss*10000:.1f} bps stop...")
            result = self.simulate_with_stops_full_data(full_data, stop_loss_pct=stop_loss)
            results.append({
                'stop_loss_bps': stop_loss * 10000,
                'stop_loss_pct': stop_loss,
                **result
            })
        
        return pd.DataFrame(results)


def main():
    workspace = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
    
    print("=== Keltner Strategy_4 Analysis with Full OHLC Data ===\n")
    
    analyzer = FullDataStopAnalyzer(workspace)
    
    # Check if we found source data
    if analyzer.source_data_path:
        print(f"Found source data at: {analyzer.source_data_path}")
    else:
        print("ERROR: Could not find source data file!")
        print("Please specify path to SPY_5m_1m.csv or SPY_5m.csv")
        return
    
    # Load source OHLC data
    print("\nLoading OHLC data...")
    try:
        ohlc_df = analyzer.load_source_data()
        print(f"Loaded {len(ohlc_df)} bars of OHLC data")
        print(f"Columns: {ohlc_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading OHLC data: {e}")
        return
    
    # Load signals
    strategy_file = analyzer.signals_path / "SPY_5m_compiled_strategy_4.parquet"
    if strategy_file.exists():
        signals_df = analyzer.load_signal_file(strategy_file)
        print(f"Loaded {len(signals_df)} signals")
        
        # Test stop losses with full data
        results_df = analyzer.test_stop_losses_full_data(signals_df, ohlc_df)
        
        # Save results
        results_df.to_csv("keltner_strategy4_stops_with_ohlc.csv", index=False)
        
        # Print summary
        print("\n" + "="*100)
        print("STOP LOSS ANALYSIS WITH FULL OHLC DATA")
        print("="*100)
        print(f"{'Stop Loss':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Stop Rate':<12} {'Winners Stopped':<18} {'% Stops were Winners':<20}")
        print("-" * 100)
        
        for idx, row in results_df.iterrows():
            stop_str = f"{row['stop_loss_bps']:.1f} bps" if pd.notna(row['stop_loss_pct']) else "None"
            print(f"{stop_str:<12} {row['avg_return_per_trade_bps']:>10.2f} "
                  f"{row['win_rate']*100:>8.1f}% {row['stop_rate']*100:>11.1f}% "
                  f"{row['stopped_winners']:>17} {row['pct_stopped_were_winners']*100:>19.1f}%")
        
        # Analysis
        valid_stops = results_df[results_df['stop_loss_pct'].notna()]
        if len(valid_stops) > 0:
            optimal = valid_stops.loc[valid_stops['avg_return_per_trade_bps'].idxmax()]
            baseline = results_df[results_df['stop_loss_bps'] == 'None'].iloc[0]
            
            print(f"\n{'='*50}")
            print("ANALYSIS SUMMARY")
            print(f"{'='*50}")
            print(f"Baseline (no stop): {baseline['avg_return_per_trade_bps']:.2f} bps/trade")
            print(f"Optimal stop: {optimal['stop_loss_bps']:.1f} bps")
            print(f"Optimal performance: {optimal['avg_return_per_trade_bps']:.2f} bps/trade")
            print(f"\nWith optimal stop:")
            print(f"- Stop rate: {optimal['stop_rate']*100:.1f}%")
            print(f"- Winners stopped: {optimal['pct_stopped_were_winners']*100:.1f}%")
            print(f"- This is realistic! (Should be 30-50% for tight stops)")
            
            # Reality check for tight stops
            tight_stops = valid_stops[valid_stops['stop_loss_bps'] <= 10]
            if len(tight_stops) > 0:
                print(f"\n{'='*50}")
                print("TIGHT STOP REALITY CHECK (≤10 bps)")
                print(f"{'='*50}")
                for _, row in tight_stops.iterrows():
                    print(f"{row['stop_loss_bps']:.1f} bps: "
                          f"{row['pct_stopped_were_winners']*100:.0f}% of stops were winners, "
                          f"stop rate {row['stop_rate']*100:.0f}%")
    else:
        print("Strategy_4 file not found")


if __name__ == "__main__":
    main()