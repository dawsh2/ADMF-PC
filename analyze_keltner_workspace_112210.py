#!/usr/bin/env python3
"""
Comprehensive analysis of Keltner workspace 20250622_112210
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add path for imports
sys.path.append('/Users/daws/ADMF-PC')

class KeltnerWorkspaceAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.signals_path = self.workspace_path / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"
        self.metadata_path = self.workspace_path / "metadata.json"
        
        # Load metadata if available
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def analyze_strategy(self, signals_df: pd.DataFrame) -> dict:
        """Analyze a single strategy's performance."""
        if signals_df.empty or len(signals_df) < 2:
            return None
        
        signals_df = signals_df.sort_values('idx').reset_index(drop=True)
        
        trades = []
        long_trades = []
        short_trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            
            if signal != 0:
                if current_position is not None:
                    # Close existing position
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price'])
                        long_trades.append(ret * 10000)
                    else:
                        ret = -np.log(price / current_position['entry_price'])
                        short_trades.append(ret * 10000)
                    trades.append(ret * 10000)
                
                # Open new position
                current_position = {
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                # Exit signal
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price'])
                    long_trades.append(ret * 10000)
                else:
                    ret = -np.log(price / current_position['entry_price'])
                    short_trades.append(ret * 10000)
                trades.append(ret * 10000)
                current_position = None
        
        if not trades:
            return None
        
        # Apply execution costs
        exec_mult = 1 - (0.5 / 10000)
        trades = [t * exec_mult for t in trades]
        long_trades = [t * exec_mult for t in long_trades]
        short_trades = [t * exec_mult for t in short_trades]
        
        # Calculate metrics
        total_return = np.exp(sum(t/10000 for t in trades)) - 1
        wins = [t for t in trades if t > 0]
        
        result = {
            'signals': len(signals_df),
            'trades': len(trades),
            'avg_return_bps': np.mean(trades),
            'total_return': total_return,
            'win_rate': len(wins) / len(trades) if trades else 0,
            'max_win_bps': max(trades) if trades else 0,
            'max_loss_bps': min(trades) if trades else 0,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades)
        }
        
        # Long/short breakdown
        if long_trades:
            result['long_avg_bps'] = np.mean(long_trades)
            result['long_win_rate'] = len([t for t in long_trades if t > 0]) / len(long_trades)
        else:
            result['long_avg_bps'] = 0
            result['long_win_rate'] = 0
            
        if short_trades:
            result['short_avg_bps'] = np.mean(short_trades)
            result['short_win_rate'] = len([t for t in short_trades if t > 0]) / len(short_trades)
        else:
            result['short_avg_bps'] = 0
            result['short_win_rate'] = 0
        
        return result
    
    def analyze_all_strategies(self):
        """Analyze all strategies in the workspace."""
        results = []
        
        strategy_files = sorted(self.signals_path.glob("SPY_5m_compiled_strategy_*.parquet"))
        
        for strategy_file in strategy_files:
            try:
                strategy_num = int(strategy_file.stem.split('_')[-1])
                signals_df = pd.read_parquet(strategy_file)
                
                result = self.analyze_strategy(signals_df)
                if result:
                    result['strategy'] = strategy_num
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {strategy_file.name}: {e}")
        
        return pd.DataFrame(results)

def test_stops_with_ohlc(signals_df: pd.DataFrame, ohlc_path: str, stop_losses: list):
    """Test stop losses using full OHLC data."""
    from analyze_keltner_with_full_data import FullDataStopAnalyzer
    
    analyzer = FullDataStopAnalyzer("dummy", ohlc_path)
    
    # Load OHLC data
    ohlc_df = analyzer.load_source_data()
    
    # Union with signals
    full_data = analyzer.union_signals_with_ohlc(signals_df, ohlc_df)
    
    results = []
    for stop_loss in stop_losses:
        result = analyzer.simulate_with_stops_full_data(full_data, stop_loss_pct=stop_loss)
        results.append({
            'stop_loss_bps': stop_loss * 10000,
            **result
        })
    
    return pd.DataFrame(results)

def main():
    workspace = "/Users/daws/ADMF-PC/configs/optimize_keltner_with_filters/20250622_112210"
    
    print("=== Fresh Analysis of Keltner Workspace 20250622_112210 ===\n")
    
    analyzer = KeltnerWorkspaceAnalyzer(workspace)
    
    # Analyze all strategies
    print("Analyzing all strategies...")
    results_df = analyzer.analyze_all_strategies()
    
    if results_df.empty:
        print("No valid strategies found!")
        return
    
    # Sort by performance
    results_df = results_df.sort_values('avg_return_bps', ascending=False)
    
    # Print summary
    print(f"\nFound {len(results_df)} strategies")
    print(f"Average performance: {results_df['avg_return_bps'].mean():.2f} bps/trade")
    print(f"Best strategy: {results_df.iloc[0]['avg_return_bps']:.2f} bps/trade")
    print(f"Worst strategy: {results_df.iloc[-1]['avg_return_bps']:.2f} bps/trade")
    
    # Top 10 strategies
    print("\n=== Top 10 Strategies ===")
    print(f"{'Strategy':<10} {'Trades':<8} {'RPT (bps)':<12} {'Win Rate':<10} {'Long RPT':<12} {'Short RPT':<12}")
    print("-" * 65)
    
    for idx, row in results_df.head(10).iterrows():
        print(f"{row['strategy']:<10} {row['trades']:<8} {row['avg_return_bps']:>10.2f} "
              f"{row['win_rate']*100:>8.1f}% {row['long_avg_bps']:>10.2f} {row['short_avg_bps']:>10.2f}")
    
    # Analyze best strategy with stops
    best_strategy = int(results_df.iloc[0]['strategy'])
    print(f"\n=== Analyzing Best Strategy ({best_strategy}) with Stop Losses ===")
    
    # Load the best strategy
    strategy_file = analyzer.signals_path / f"SPY_5m_compiled_strategy_{best_strategy}.parquet"
    signals_df = pd.read_parquet(strategy_file)
    
    # Test with stops
    stop_losses = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    try:
        stop_results = test_stops_with_ohlc(signals_df, None, stop_losses)
        
        print(f"\n{'Stop Loss':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Stop Rate':<12} {'Winners Stopped':<15}")
        print("-" * 60)
        
        for idx, row in stop_results.iterrows():
            print(f"{row['stop_loss_bps']:.1f} bps".ljust(12) + 
                  f"{row['avg_return_per_trade_bps']:>10.2f} "
                  f"{row['win_rate']*100:>8.1f}% {row['stop_rate']*100:>11.1f}% "
                  f"{row['pct_stopped_were_winners']*100:>14.1f}%")
    except Exception as e:
        print(f"Could not test stops: {e}")
    
    # Long/Short analysis
    print("\n=== Long vs Short Performance (All Strategies) ===")
    long_avg = results_df['long_avg_bps'].mean()
    short_avg = results_df['short_avg_bps'].mean()
    
    print(f"Average Long Performance: {long_avg:.2f} bps/trade")
    print(f"Average Short Performance: {short_avg:.2f} bps/trade")
    print(f"Long/Short Ratio: {long_avg/short_avg:.2f}x" if short_avg != 0 else "N/A")
    
    # Save detailed results
    results_df.to_csv("keltner_112210_analysis.csv", index=False)
    print(f"\nDetailed results saved to keltner_112210_analysis.csv")
    
    # Final recommendations
    print("\n=== RECOMMENDATIONS ===")
    if results_df.iloc[0]['avg_return_bps'] > 1.0:
        print("✓ Found profitable strategies!")
        print(f"✓ Best strategy: {best_strategy} with {results_df.iloc[0]['avg_return_bps']:.2f} bps/trade")
        print(f"✓ Trade frequency: {results_df.iloc[0]['trades']} trades")
        
        if long_avg > short_avg * 2:
            print("✓ Consider long-only implementation (2x+ better than shorts)")
    else:
        print("✗ No strategies exceed 1 bps/trade threshold")
        print("✗ Further optimization or different approach needed")

if __name__ == "__main__":
    main()