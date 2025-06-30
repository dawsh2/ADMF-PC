#!/usr/bin/env python3
"""Analyze Bollinger parameter sweep results with focus on filter effectiveness."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_strategy_traces(results_dir: Path) -> pd.DataFrame:
    """Load all strategy traces from the parameter sweep."""
    traces_dir = results_dir / "traces" / "bollinger_bands"
    all_traces = []
    
    for parquet_file in sorted(traces_dir.glob("*.parquet")):
        try:
            # Extract strategy number from filename
            strategy_num = int(parquet_file.stem.split('_')[-1])
            
            # Load trace data
            df = pd.read_parquet(parquet_file)
            df['strategy_id'] = strategy_num
            all_traces.append(df)
        except Exception as e:
            print(f"Error loading {parquet_file}: {e}")
    
    if all_traces:
        return pd.concat(all_traces, ignore_index=True)
    return pd.DataFrame()

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate key metrics for a set of trades."""
    if len(trades_df) == 0:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_return_bps': 0,
            'total_return_bps': 0,
            'sharpe_ratio': 0,
            'max_drawdown_bps': 0
        }
    
    # Calculate returns
    returns = trades_df['pnl_bps'].values
    
    # Calculate metrics
    metrics = {
        'num_trades': len(trades_df),
        'win_rate': (returns > 0).mean() * 100,
        'avg_return_bps': returns.mean(),
        'total_return_bps': returns.sum(),
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'max_drawdown_bps': returns.cumsum().expanding().max().sub(returns.cumsum()).max()
    }
    
    return metrics

def analyze_filter_impact(df: pd.DataFrame) -> Dict:
    """Analyze the impact of different filters on performance."""
    results = {}
    
    # Get trades (entry and exit pairs)
    entries = df[df['signal'] != 0].copy()
    if len(entries) == 0:
        return {}
    
    # Match entries with exits
    trades = []
    for idx, entry in entries.iterrows():
        # Find next opposite signal or end of data
        exit_mask = (df.index > idx)
        if entry['signal'] > 0:  # Long entry
            exit_mask &= (df['signal'] < 0)
        else:  # Short entry
            exit_mask &= (df['signal'] > 0)
        
        exits = df[exit_mask]
        if len(exits) > 0:
            exit_idx = exits.index[0]
            exit_row = df.loc[exit_idx]
        else:
            # Use last bar if no exit signal
            exit_idx = df.index[-1]
            exit_row = df.iloc[-1]
        
        # Calculate trade return
        if entry['signal'] > 0:  # Long
            pnl_bps = (exit_row['close'] - entry['close']) / entry['close'] * 10000
        else:  # Short
            pnl_bps = (entry['close'] - exit_row['close']) / entry['close'] * 10000
        
        trade = {
            'entry_time': entry.name,
            'exit_time': exit_row.name,
            'direction': 'long' if entry['signal'] > 0 else 'short',
            'entry_price': entry['close'],
            'exit_price': exit_row['close'],
            'pnl_bps': pnl_bps,
            'volume': entry.get('volume', np.nan),
            'volatility': entry.get('volatility', np.nan),
            'trend_strength': entry.get('trend_strength', np.nan),
            'vwap_ratio': entry['close'] / entry.get('vwap', entry['close']) if 'vwap' in entry else 1.0,
            'rsi': entry.get('rsi', 50),
            'bb_position': entry.get('bb_position', 0.5),
            'volume_ratio': entry.get('volume', 1) / entry.get('volume_ma', entry.get('volume', 1)) if 'volume_ma' in entry else 1.0
        }
        trades.append(trade)
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return {}
    
    # Baseline metrics (all trades)
    baseline = calculate_trade_metrics(trades_df)
    results['baseline'] = baseline
    
    # Volume filter analysis
    if 'volume_ratio' in trades_df.columns:
        for threshold in [0.8, 1.0, 1.2, 1.5, 2.0]:
            filtered = trades_df[trades_df['volume_ratio'] > threshold]
            metrics = calculate_trade_metrics(filtered)
            results[f'volume_ratio_>{threshold}'] = {
                **metrics,
                'filter_rate': (1 - len(filtered) / len(trades_df)) * 100
            }
    
    # Volatility filter analysis
    if 'volatility' in trades_df.columns and not trades_df['volatility'].isna().all():
        vol_percentiles = trades_df['volatility'].quantile([0.25, 0.5, 0.75])
        for pct, vol_threshold in zip([25, 50, 75], vol_percentiles):
            # High volatility trades
            filtered_high = trades_df[trades_df['volatility'] > vol_threshold]
            metrics_high = calculate_trade_metrics(filtered_high)
            results[f'volatility_top_{100-pct}%'] = {
                **metrics_high,
                'filter_rate': (1 - len(filtered_high) / len(trades_df)) * 100
            }
            
            # Low volatility trades
            filtered_low = trades_df[trades_df['volatility'] < vol_threshold]
            metrics_low = calculate_trade_metrics(filtered_low)
            results[f'volatility_bottom_{pct}%'] = {
                **metrics_low,
                'filter_rate': (1 - len(filtered_low) / len(trades_df)) * 100
            }
    
    # Trend filter analysis
    if 'trend_strength' in trades_df.columns and not trades_df['trend_strength'].isna().all():
        for threshold in [0.0, 0.1, 0.2, 0.3]:
            filtered = trades_df[trades_df['trend_strength'].abs() > threshold]
            metrics = calculate_trade_metrics(filtered)
            results[f'trend_strength_>{threshold}'] = {
                **metrics,
                'filter_rate': (1 - len(filtered) / len(trades_df)) * 100
            }
    
    # VWAP filter analysis
    if 'vwap_ratio' in trades_df.columns:
        # Long trades above VWAP
        long_above_vwap = trades_df[(trades_df['direction'] == 'long') & (trades_df['vwap_ratio'] > 1.0)]
        metrics = calculate_trade_metrics(long_above_vwap)
        results['long_above_vwap'] = {
            **metrics,
            'filter_rate': (1 - len(long_above_vwap) / len(trades_df[trades_df['direction'] == 'long'])) * 100
        }
        
        # Short trades below VWAP
        short_below_vwap = trades_df[(trades_df['direction'] == 'short') & (trades_df['vwap_ratio'] < 1.0)]
        metrics = calculate_trade_metrics(short_below_vwap)
        results['short_below_vwap'] = {
            **metrics,
            'filter_rate': (1 - len(short_below_vwap) / len(trades_df[trades_df['direction'] == 'short'])) * 100
        }
    
    # Long vs Short comparison
    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']
    
    results['long_only'] = calculate_trade_metrics(long_trades)
    results['short_only'] = calculate_trade_metrics(short_trades)
    
    # RSI extremes filter
    if 'rsi' in trades_df.columns and not trades_df['rsi'].isna().all():
        # Oversold longs
        oversold_longs = trades_df[(trades_df['direction'] == 'long') & (trades_df['rsi'] < 30)]
        results['oversold_longs'] = {
            **calculate_trade_metrics(oversold_longs),
            'filter_rate': (1 - len(oversold_longs) / len(long_trades)) * 100 if len(long_trades) > 0 else 100
        }
        
        # Overbought shorts
        overbought_shorts = trades_df[(trades_df['direction'] == 'short') & (trades_df['rsi'] > 70)]
        results['overbought_shorts'] = {
            **calculate_trade_metrics(overbought_shorts),
            'filter_rate': (1 - len(overbought_shorts) / len(short_trades)) * 100 if len(short_trades) > 0 else 100
        }
    
    # Bollinger Band position filter
    if 'bb_position' in trades_df.columns:
        # Trades near band extremes
        extreme_trades = trades_df[(trades_df['bb_position'] < 0.2) | (trades_df['bb_position'] > 0.8)]
        results['bb_extremes'] = {
            **calculate_trade_metrics(extreme_trades),
            'filter_rate': (1 - len(extreme_trades) / len(trades_df)) * 100
        }
    
    return results

def find_best_filter_combinations(df: pd.DataFrame) -> Dict:
    """Find filter combinations that achieve >1.5 bps return per trade."""
    trades = []
    entries = df[df['signal'] != 0].copy()
    
    # Build trades list with all features
    for idx, entry in entries.iterrows():
        exit_mask = (df.index > idx)
        if entry['signal'] > 0:
            exit_mask &= (df['signal'] < 0)
        else:
            exit_mask &= (df['signal'] > 0)
        
        exits = df[exit_mask]
        if len(exits) > 0:
            exit_idx = exits.index[0]
            exit_row = df.loc[exit_idx]
        else:
            exit_idx = df.index[-1]
            exit_row = df.iloc[-1]
        
        if entry['signal'] > 0:
            pnl_bps = (exit_row['close'] - entry['close']) / entry['close'] * 10000
        else:
            pnl_bps = (entry['close'] - exit_row['close']) / entry['close'] * 10000
        
        trade = {
            'pnl_bps': pnl_bps,
            'direction': 'long' if entry['signal'] > 0 else 'short',
            'volume_ratio': entry.get('volume', 1) / entry.get('volume_ma', entry.get('volume', 1)) if 'volume_ma' in entry else 1.0,
            'volatility_pct': entry.get('volatility', 50),
            'trend_strength': entry.get('trend_strength', 0),
            'vwap_above': entry['close'] > entry.get('vwap', entry['close']),
            'rsi': entry.get('rsi', 50),
            'bb_position': entry.get('bb_position', 0.5)
        }
        trades.append(trade)
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return {}
    
    # Test combinations of filters
    best_combinations = []
    
    # Define filter conditions
    filters = {
        'volume_high': lambda df: df['volume_ratio'] > 1.2,
        'volatility_high': lambda df: df['volatility_pct'] > df['volatility_pct'].quantile(0.7),
        'volatility_low': lambda df: df['volatility_pct'] < df['volatility_pct'].quantile(0.3),
        'trend_strong': lambda df: df['trend_strength'].abs() > 0.2,
        'vwap_aligned': lambda df: ((df['direction'] == 'long') & df['vwap_above']) | ((df['direction'] == 'short') & ~df['vwap_above']),
        'rsi_extreme': lambda df: ((df['direction'] == 'long') & (df['rsi'] < 30)) | ((df['direction'] == 'short') & (df['rsi'] > 70)),
        'bb_extreme': lambda df: (df['bb_position'] < 0.2) | (df['bb_position'] > 0.8)
    }
    
    # Test single filters
    for name, filter_func in filters.items():
        try:
            filtered = trades_df[filter_func(trades_df)]
            if len(filtered) > 10:  # Minimum trades for significance
                metrics = calculate_trade_metrics(filtered)
                if metrics['avg_return_bps'] > 1.5:
                    best_combinations.append({
                        'filters': [name],
                        'metrics': metrics,
                        'filter_rate': (1 - len(filtered) / len(trades_df)) * 100
                    })
        except:
            pass
    
    # Test two-filter combinations
    filter_names = list(filters.keys())
    for i in range(len(filter_names)):
        for j in range(i + 1, len(filter_names)):
            try:
                name1, name2 = filter_names[i], filter_names[j]
                mask = filters[name1](trades_df) & filters[name2](trades_df)
                filtered = trades_df[mask]
                
                if len(filtered) > 10:
                    metrics = calculate_trade_metrics(filtered)
                    if metrics['avg_return_bps'] > 1.5:
                        best_combinations.append({
                            'filters': [name1, name2],
                            'metrics': metrics,
                            'filter_rate': (1 - len(filtered) / len(trades_df)) * 100
                        })
            except:
                pass
    
    # Sort by average return
    best_combinations.sort(key=lambda x: x['metrics']['avg_return_bps'], reverse=True)
    
    return best_combinations[:10]  # Top 10 combinations

def main():
    """Main analysis function."""
    results_dir = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250623_062931")
    
    print("Loading Bollinger parameter sweep results...")
    df = load_strategy_traces(results_dir)
    
    if len(df) == 0:
        print("No data loaded!")
        return
    
    print(f"Loaded {len(df)} data points across {df['strategy_id'].nunique()} strategies")
    
    # Analyze filter impact for each strategy
    all_filter_results = {}
    best_strategies = []
    
    for strategy_id in df['strategy_id'].unique():
        strategy_df = df[df['strategy_id'] == strategy_id]
        
        # Analyze filters for this strategy
        filter_results = analyze_filter_impact(strategy_df)
        
        if filter_results and 'baseline' in filter_results:
            baseline_return = filter_results['baseline']['avg_return_bps']
            
            # Find best filter for this strategy
            best_filter_return = baseline_return
            best_filter_name = 'baseline'
            
            for filter_name, metrics in filter_results.items():
                if filter_name != 'baseline' and isinstance(metrics, dict):
                    if metrics.get('avg_return_bps', 0) > best_filter_return and metrics.get('num_trades', 0) > 10:
                        best_filter_return = metrics['avg_return_bps']
                        best_filter_name = filter_name
            
            all_filter_results[strategy_id] = filter_results
            
            if best_filter_return > 1.0:  # Strategies with >1 bps potential
                best_strategies.append({
                    'strategy_id': strategy_id,
                    'baseline_return': baseline_return,
                    'best_filter': best_filter_name,
                    'filtered_return': best_filter_return,
                    'improvement_bps': best_filter_return - baseline_return
                })
    
    # Sort by filtered return
    best_strategies.sort(key=lambda x: x['filtered_return'], reverse=True)
    
    # Generate comprehensive report
    report = []
    report.append("# Bollinger Band Parameter Sweep Filter Analysis")
    report.append(f"\nAnalyzed {len(all_filter_results)} strategies")
    report.append(f"Found {len(best_strategies)} strategies with >1 bps filtered return potential\n")
    
    # Top performing strategies with filters
    report.append("## Top 20 Strategies with Optimal Filters\n")
    report.append("| Strategy | Baseline Return | Best Filter | Filtered Return | Improvement | Filter Details |")
    report.append("|----------|----------------|-------------|-----------------|-------------|----------------|")
    
    for strategy in best_strategies[:20]:
        sid = strategy['strategy_id']
        filter_name = strategy['best_filter']
        filter_metrics = all_filter_results[sid][filter_name]
        
        report.append(f"| {sid} | {strategy['baseline_return']:.2f} bps | {filter_name} | "
                     f"{strategy['filtered_return']:.2f} bps | +{strategy['improvement_bps']:.2f} bps | "
                     f"{filter_metrics.get('num_trades', 0)} trades, "
                     f"{filter_metrics.get('win_rate', 0):.1f}% win rate |")
    
    # Overall filter effectiveness analysis
    report.append("\n## Filter Effectiveness Summary\n")
    
    # Aggregate filter performance across all strategies
    filter_performance = {}
    
    for strategy_id, filters in all_filter_results.items():
        baseline = filters.get('baseline', {})
        baseline_return = baseline.get('avg_return_bps', 0)
        
        for filter_name, metrics in filters.items():
            if filter_name != 'baseline' and isinstance(metrics, dict):
                improvement = metrics.get('avg_return_bps', 0) - baseline_return
                
                if filter_name not in filter_performance:
                    filter_performance[filter_name] = {
                        'improvements': [],
                        'win_rates': [],
                        'filter_rates': [],
                        'num_trades': []
                    }
                
                filter_performance[filter_name]['improvements'].append(improvement)
                filter_performance[filter_name]['win_rates'].append(metrics.get('win_rate', 0))
                filter_performance[filter_name]['filter_rates'].append(metrics.get('filter_rate', 0))
                filter_performance[filter_name]['num_trades'].append(metrics.get('num_trades', 0))
    
    # Calculate average performance for each filter
    report.append("### Average Filter Impact Across All Strategies\n")
    report.append("| Filter Type | Avg Improvement | Avg Win Rate | Avg Filter Rate | Strategies Improved |")
    report.append("|-------------|----------------|--------------|-----------------|-------------------|")
    
    filter_summary = []
    for filter_name, perf in filter_performance.items():
        improvements = [x for x in perf['improvements'] if x != 0]
        if improvements:
            avg_improvement = np.mean(improvements)
            avg_win_rate = np.mean([x for x in perf['win_rates'] if x > 0])
            avg_filter_rate = np.mean([x for x in perf['filter_rates'] if x > 0])
            strategies_improved = sum(1 for x in improvements if x > 0)
            
            filter_summary.append({
                'name': filter_name,
                'avg_improvement': avg_improvement,
                'avg_win_rate': avg_win_rate,
                'avg_filter_rate': avg_filter_rate,
                'strategies_improved': strategies_improved,
                'total_strategies': len(improvements)
            })
    
    # Sort by average improvement
    filter_summary.sort(key=lambda x: x['avg_improvement'], reverse=True)
    
    for fs in filter_summary[:20]:
        report.append(f"| {fs['name']} | {fs['avg_improvement']:.3f} bps | "
                     f"{fs['avg_win_rate']:.1f}% | {fs['avg_filter_rate']:.1f}% | "
                     f"{fs['strategies_improved']}/{fs['total_strategies']} |")
    
    # Find best filter combinations for high-return trades
    report.append("\n## Best Filter Combinations for >1.5 bps Returns\n")
    
    # Analyze top 10 strategies for combination opportunities
    for strategy in best_strategies[:10]:
        sid = strategy['strategy_id']
        strategy_df = df[df['strategy_id'] == sid]
        
        combinations = find_best_filter_combinations(strategy_df)
        
        if combinations:
            report.append(f"\n### Strategy {sid}")
            report.append("| Filter Combination | Avg Return | Win Rate | Num Trades | Filter Rate |")
            report.append("|-------------------|------------|----------|------------|-------------|")
            
            for combo in combinations[:5]:
                filters_str = " + ".join(combo['filters'])
                metrics = combo['metrics']
                report.append(f"| {filters_str} | {metrics['avg_return_bps']:.2f} bps | "
                             f"{metrics['win_rate']:.1f}% | {metrics['num_trades']} | "
                             f"{combo['filter_rate']:.1f}% |")
    
    # Specific filter type analysis
    report.append("\n## Detailed Filter Type Analysis\n")
    
    # Volume filter analysis
    report.append("### Volume Filters")
    report.append("Higher volume ratios generally improve performance:")
    volume_filters = [k for k in filter_performance.keys() if 'volume' in k]
    for vf in sorted(volume_filters):
        perf = filter_performance[vf]
        avg_imp = np.mean(perf['improvements'])
        report.append(f"- {vf}: {avg_imp:.3f} bps average improvement")
    
    # Volatility filter analysis
    report.append("\n### Volatility Filters")
    report.append("Performance varies by volatility regime:")
    vol_filters = [k for k in filter_performance.keys() if 'volatility' in k]
    for vf in sorted(vol_filters):
        perf = filter_performance[vf]
        avg_imp = np.mean(perf['improvements'])
        report.append(f"- {vf}: {avg_imp:.3f} bps average improvement")
    
    # Long vs Short analysis
    report.append("\n### Long vs Short Performance")
    long_perf = filter_performance.get('long_only', {})
    short_perf = filter_performance.get('short_only', {})
    
    if long_perf and short_perf:
        long_returns = [all_filter_results[sid]['long_only']['avg_return_bps'] 
                       for sid in all_filter_results 
                       if 'long_only' in all_filter_results[sid]]
        short_returns = [all_filter_results[sid]['short_only']['avg_return_bps'] 
                        for sid in all_filter_results 
                        if 'short_only' in all_filter_results[sid]]
        
        report.append(f"- Average long return: {np.mean(long_returns):.3f} bps")
        report.append(f"- Average short return: {np.mean(short_returns):.3f} bps")
    
    # Save report
    output_file = "/Users/daws/ADMF-PC/bollinger_filter_analysis_report.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {output_file}")
    
    # Save detailed results as CSV
    results_data = []
    for strategy in best_strategies:
        sid = strategy['strategy_id']
        for filter_name, metrics in all_filter_results[sid].items():
            if isinstance(metrics, dict):
                results_data.append({
                    'strategy_id': sid,
                    'filter': filter_name,
                    'num_trades': metrics.get('num_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'avg_return_bps': metrics.get('avg_return_bps', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'filter_rate': metrics.get('filter_rate', 0)
                })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("/Users/daws/ADMF-PC/bollinger_filter_detailed_results.csv", index=False)
    print("Detailed results saved to: bollinger_filter_detailed_results.csv")

if __name__ == "__main__":
    main()