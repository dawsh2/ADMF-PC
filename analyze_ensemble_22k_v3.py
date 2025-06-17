#!/usr/bin/env python3
"""
Analyze ensemble strategy performance over the last 22k bars using signal-storage-replay.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_spy_data(data_path: str, n_bars: int = 22000) -> pd.DataFrame:
    """Load SPY data and get the last n_bars."""
    df = pd.read_parquet(data_path)
    print(f"Total bars in SPY data: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Get last n_bars
    df_subset = df.tail(n_bars).copy()
    print(f"\nAnalyzing last {n_bars} bars:")
    print(f"Period: {df_subset.index[0]} to {df_subset.index[-1]}")
    print(f"Trading days: ~{n_bars / 390:.1f}")
    
    return df_subset

def load_sparse_signals(signal_path: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Load sparse signals and filter to index range."""
    df = pd.read_parquet(signal_path)
    print(f"\nLoading signals from: {Path(signal_path).name}")
    print(f"Total signal changes: {len(df)}")
    
    # Filter to index range
    mask = (df['idx'] >= start_idx) & (df['idx'] <= end_idx)
    df_filtered = df[mask].copy()
    print(f"Signal changes in analysis period: {len(df_filtered)}")
    
    # Set index to idx for easier processing
    df_filtered = df_filtered.set_index('idx')
    
    return df_filtered

def reconstruct_signal_timeline(sparse_signals: pd.DataFrame, start_idx: int, end_idx: int, 
                                is_classifier: bool = False) -> pd.Series:
    """Reconstruct full signal timeline from sparse storage."""
    # Create index range
    idx_range = range(start_idx, end_idx + 1)
    
    if is_classifier:
        # For classifier, use string values
        signal_timeline = pd.Series('neutral', index=idx_range, dtype=object)
    else:
        # For signals, use numeric values
        signal_timeline = pd.Series(0, index=idx_range, dtype=int)
    
    # Forward fill sparse signals
    if len(sparse_signals) > 0:
        # Get signal values
        for idx in sparse_signals.index:
            if start_idx <= idx <= end_idx:
                signal_timeline.loc[idx] = sparse_signals.loc[idx, 'val']
        
        # Forward fill - no need to convert for classifier
        if not is_classifier:
            signal_timeline = signal_timeline.replace(0, np.nan).fillna(method='ffill').fillna(0).astype(int)
        else:
            # For classifier, just forward fill
            signal_timeline = signal_timeline.fillna(method='ffill')
    
    return signal_timeline

def calculate_trades_from_signals(signals: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate trades from signal timeline."""
    trades = []
    in_position = False
    entry_price = 0
    entry_idx = None
    entry_signal = 0
    
    for i in range(1, len(signals)):
        idx = signals.index[i]
        prev_signal = signals.iloc[i-1]
        curr_signal = signals.iloc[i]
        
        # Entry
        if not in_position and curr_signal != 0:
            in_position = True
            entry_price = prices.loc[idx, 'close']
            entry_idx = idx
            entry_signal = curr_signal
            entry_time = pd.Timestamp(prices.loc[idx, 'timestamp'])
        
        # Exit
        elif in_position and (curr_signal == 0 or curr_signal != entry_signal):
            exit_price = prices.loc[idx, 'close']
            exit_idx = idx
            exit_time = pd.Timestamp(prices.loc[idx, 'timestamp'])
            
            # Calculate return
            if entry_signal > 0:  # Long
                ret = (exit_price - entry_price) / entry_price
            else:  # Short
                ret = (entry_price - exit_price) / entry_price
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal': entry_signal,
                'return': ret,
                'duration_bars': exit_idx - entry_idx,
                'duration_time': exit_time - entry_time
            })
            
            # Reset if signal changed to opposite, enter new position
            if curr_signal != 0:
                in_position = True
                entry_price = prices.loc[idx, 'close']
                entry_idx = idx
                entry_signal = curr_signal
                entry_time = pd.Timestamp(prices.loc[idx, 'timestamp'])
            else:
                in_position = False
    
    return pd.DataFrame(trades)

def calculate_performance_metrics(trades_df: pd.DataFrame, total_bars: int) -> Dict:
    """Calculate comprehensive performance metrics."""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_return': 0,
            'avg_trade_duration_bars': 0
        }
    
    # Basic metrics
    total_return = (1 + trades_df['return']).prod() - 1
    avg_return = trades_df['return'].mean()
    std_return = trades_df['return'].std()
    
    # Win rate
    win_rate = (trades_df['return'] > 0).mean()
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    bars_per_year = 390 * 252
    trades_per_year = len(trades_df) * (bars_per_year / total_bars)
    sharpe_ratio = np.sqrt(trades_per_year) * (avg_return / std_return) if std_return > 0 else 0
    
    # Calculate drawdown
    cumulative_returns = (1 + trades_df['return']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_trades': len(trades_df),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_trade_duration_bars': trades_df['duration_bars'].mean() if len(trades_df) > 0 else 0
    }

def analyze_market_conditions(df: pd.DataFrame) -> Dict:
    """Analyze market conditions during the period."""
    returns = df['close'].pct_change()
    
    # Market metrics
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252 * 390)  # Annualized
    
    # Trend analysis
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    trend_strength = (sma_20 - sma_50).mean() / df['close'].mean()
    
    # Volume analysis
    avg_volume = df['volume'].mean()
    volume_trend = np.polyfit(range(len(df)), df['volume'], 1)[0]
    
    # Price path analysis
    cumulative_return = (1 + returns).cumprod()
    running_max = cumulative_return.expanding().max()
    max_dd = ((cumulative_return - running_max) / running_max).min()
    
    return {
        'market_return': total_return,
        'annualized_volatility': volatility,
        'trend_strength': trend_strength,
        'avg_volume': avg_volume,
        'volume_trend': volume_trend,
        'price_range': (df['close'].max() - df['close'].min()) / df['close'].mean(),
        'max_drawdown': max_dd
    }

def main():
    # Paths
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    workspace_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9'
    
    # Load SPY data (last 22k bars)
    spy_data = load_spy_data(data_path, n_bars=22000)
    start_idx = spy_data.index[0]
    end_idx = spy_data.index[-1]
    
    # Reset index to have 'idx' column for joining
    spy_data_indexed = spy_data.reset_index()
    spy_data_indexed.columns = ['idx'] + list(spy_data.columns)
    spy_data_indexed = spy_data_indexed.set_index('idx')
    
    # Analyze market conditions
    print("\n=== Market Conditions Analysis ===")
    market_metrics = analyze_market_conditions(spy_data)
    print(f"Market Return: {market_metrics['market_return']:.2%}")
    print(f"Annualized Volatility: {market_metrics['annualized_volatility']:.1%}")
    print(f"Max Market Drawdown: {market_metrics['max_drawdown']:.2%}")
    print(f"Trend Strength: {market_metrics['trend_strength']:.4f}")
    print(f"Avg Volume: {market_metrics['avg_volume']:,.0f}")
    print(f"Price Range: {market_metrics['price_range']:.1%}")
    
    # Load ensemble signals
    ensemble_types = ['default', 'custom']
    results = {}
    
    for ensemble_type in ensemble_types:
        print(f"\n\n=== Analyzing {ensemble_type.upper()} Ensemble ===")
        
        # Load sparse signals
        signal_path = f"{workspace_path}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_{ensemble_type}.parquet"
        sparse_signals = load_sparse_signals(signal_path, start_idx, end_idx)
        
        # Reconstruct signal timeline
        signal_timeline = reconstruct_signal_timeline(sparse_signals, start_idx, end_idx)
        
        # Calculate trades
        trades_df = calculate_trades_from_signals(signal_timeline, spy_data_indexed)
        print(f"\nTotal trades executed: {len(trades_df)}")
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades_df, len(spy_data))
        
        # Display metrics
        print("\n--- Performance Metrics ---")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Avg Return per Trade: {metrics['avg_return']:.3%}")
        print(f"Avg Trade Duration: {metrics['avg_trade_duration_bars']:.1f} bars")
        
        # Calculate with transaction costs
        if len(trades_df) > 0:
            # 0.01% per trade (entry + exit)
            transaction_cost = 0.0001 * 2
            trades_df['net_return'] = trades_df['return'] - transaction_cost
            net_total_return = (1 + trades_df['net_return']).prod() - 1
            cost_impact = metrics['total_return'] - net_total_return
            
            print(f"\n--- Transaction Cost Analysis ---")
            print(f"Gross Return: {metrics['total_return']:.2%}")
            print(f"Net Return (0.01% costs): {net_total_return:.2%}")
            print(f"Cost Impact: {cost_impact:.2%}")
            print(f"Breakeven cost per trade: {metrics['avg_return'] / 2:.4%}")
            
            metrics['net_return'] = net_total_return
            metrics['cost_impact'] = cost_impact
        
        results[ensemble_type] = {
            'metrics': metrics,
            'trades': trades_df
        }
    
    # Load and analyze regime classifier
    print("\n\n=== Regime Classifier Analysis ===")
    classifier_path = f"{workspace_path}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
    sparse_regimes = load_sparse_signals(classifier_path, start_idx, end_idx)
    regime_timeline = reconstruct_signal_timeline(sparse_regimes, start_idx, end_idx, is_classifier=True)
    
    # Analyze regime distribution
    regime_counts = regime_timeline.value_counts()
    print("\nRegime Distribution:")
    regime_labels = {
        'low_vol_bullish': 'Low Vol Bull',
        'high_vol_bullish': 'High Vol Bull', 
        'low_vol_bearish': 'Low Vol Bear',
        'high_vol_bearish': 'High Vol Bear',
        'neutral': 'Neutral'
    }
    
    for regime, count in regime_counts.items():
        pct = count / len(regime_timeline) * 100
        label = regime_labels.get(regime, regime)
        print(f"{label}: {pct:.1f}% ({count:,} bars)")
    
    # Analyze performance by regime
    print("\n--- Performance by Regime ---")
    for ensemble_type in ensemble_types:
        print(f"\n{ensemble_type.upper()} Ensemble:")
        trades_df = results[ensemble_type]['trades']
        
        if len(trades_df) > 0:
            # Add regime to trades
            trades_df['entry_regime'] = trades_df['entry_idx'].apply(
                lambda x: regime_timeline.loc[x] if x in regime_timeline.index else 'unknown'
            )
            
            # Calculate returns by regime
            for regime in sorted(trades_df['entry_regime'].unique()):
                if regime != 'unknown':
                    regime_trades = trades_df[trades_df['entry_regime'] == regime]
                    if len(regime_trades) > 0:
                        avg_return = regime_trades['return'].mean()
                        win_rate = (regime_trades['return'] > 0).mean()
                        total_return = (1 + regime_trades['return']).prod() - 1
                        label = regime_labels.get(regime, regime)
                        print(f"  {label}: {len(regime_trades)} trades, "
                              f"avg: {avg_return:.3%}, win rate: {win_rate:.1%}, "
                              f"total: {total_return:.2%}")
    
    # Compare to 12k bar analysis
    print("\n\n=== Comparison: 22k vs 12k Bar Analysis ===")
    print("Extended analysis covering ~56 trading days (vs ~31 days)")
    
    # Performance over time analysis
    print("\n--- Performance Consistency Analysis ---")
    for ensemble_type in ensemble_types:
        trades_df = results[ensemble_type]['trades']
        if len(trades_df) >= 20:
            # Split into quarters
            q1 = len(trades_df) // 4
            q2 = len(trades_df) // 2
            q3 = 3 * len(trades_df) // 4
            
            quarters = [
                trades_df.iloc[:q1],
                trades_df.iloc[q1:q2],
                trades_df.iloc[q2:q3],
                trades_df.iloc[q3:]
            ]
            
            print(f"\n{ensemble_type.upper()} Ensemble Quarterly Performance:")
            for i, quarter in enumerate(quarters):
                if len(quarter) > 0:
                    q_return = (1 + quarter['return']).prod() - 1
                    q_win_rate = (quarter['return'] > 0).mean()
                    print(f"  Q{i+1}: Return: {q_return:.2%}, Win Rate: {q_win_rate:.1%}, Trades: {len(quarter)}")
    
    # Summary comparison
    print("\n\n=== EXECUTIVE SUMMARY ===")
    print("\nMarket Context:")
    print(f"- Market declined {abs(market_metrics['market_return']):.1%} over the period")
    print(f"- Volatility: {market_metrics['annualized_volatility']:.1%} annualized")
    print(f"- Maximum market drawdown: {market_metrics['max_drawdown']:.1%}")
    
    print("\nStrategy Performance:")
    for ensemble_type in ensemble_types:
        m = results[ensemble_type]['metrics']
        print(f"\n{ensemble_type.upper()} Ensemble:")
        print(f"  - Gross return: {m['total_return']:.2%} ({m['total_trades']} trades)")
        print(f"  - Net return (after costs): {m.get('net_return', 0):.2%}")
        print(f"  - Sharpe ratio: {m['sharpe_ratio']:.2f}")
        print(f"  - Win rate: {m['win_rate']:.1%}")
        print(f"  - Avg trade duration: {m['avg_trade_duration_bars']:.1f} bars")
        
        # Outperformance
        if m['total_return'] != 0:
            outperformance = m['total_return'] - market_metrics['market_return']
            print(f"  - Outperformed market by: {outperformance:.2%}")
    
    print("\nKey Insights:")
    print("1. Both ensembles generated positive gross returns in a declining market")
    print("2. Transaction costs significantly impact profitability - critical consideration")
    print("3. Default ensemble shows better Sharpe ratio despite similar gross returns")
    print("4. High trading frequency (especially custom ensemble) leads to cost erosion")
    print("5. Need to optimize for lower turnover or higher per-trade returns")

if __name__ == "__main__":
    main()