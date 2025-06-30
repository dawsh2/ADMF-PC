# Test Set Validation
# Compare training vs test performance with optimal stops/targets

import pandas as pd
import numpy as np

def compare_train_test_performance(train_dir, test_dir, stop_pct=0.075, target_pct=0.10):
    """
    Compare performance between training and test sets
    """
    print("📊 Training vs Test Set Comparison")
    print("=" * 80)
    
    results = {}
    
    for dataset, run_dir in [('Training', train_dir), ('Test', test_dir)]:
        print(f"\n{dataset} Set Analysis:")
        
        # Load performance data
        perf_path = run_dir / 'performance_metrics.parquet'
        if perf_path.exists():
            perf_df = pd.read_parquet(perf_path)
            
            # Get high-frequency strategies only
            high_freq = perf_df[perf_df['num_trades'] >= 100]  # Adjust threshold as needed
            
            print(f"  Total strategies: {len(perf_df)}")
            print(f"  High-frequency strategies: {len(high_freq)}")
            
            if len(high_freq) > 0:
                # Show top performers
                top_5 = high_freq.nlargest(5, 'sharpe_ratio')
                
                print(f"\n  Top 5 by Sharpe (before stops):")
                for _, row in top_5.iterrows():
                    print(f"    {row['strategy_type']} - {row['strategy_hash'][:8]}")
                    print(f"      Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%")
                    print(f"      Period: {row.get('period', 'N/A')}, StdDev: {row.get('std_dev', 'N/A')}")
                
                # Store for comparison
                results[dataset] = {
                    'total_strategies': len(perf_df),
                    'high_freq_count': len(high_freq),
                    'avg_sharpe': high_freq['sharpe_ratio'].mean(),
                    'avg_return': high_freq['total_return'].mean(),
                    'best_sharpe': top_5.iloc[0]['sharpe_ratio'] if len(top_5) > 0 else 0,
                    'best_return': top_5.iloc[0]['total_return'] if len(top_5) > 0 else 0,
                    'best_params': {
                        'period': top_5.iloc[0].get('period', 'N/A'),
                        'std_dev': top_5.iloc[0].get('std_dev', 'N/A')
                    } if len(top_5) > 0 else {}
                }
        else:
            print(f"  ❌ No performance data found at {perf_path}")
    
    # Compare results
    if len(results) == 2:
        print("\n📈 Performance Comparison:")
        print("=" * 60)
        
        metrics = ['avg_sharpe', 'avg_return', 'best_sharpe', 'best_return']
        metric_names = ['Avg Sharpe', 'Avg Return', 'Best Sharpe', 'Best Return']
        
        for metric, name in zip(metrics, metric_names):
            train_val = results['Training'][metric]
            test_val = results['Test'][metric]
            
            if 'return' in metric:
                print(f"{name}: Train={train_val*100:.2f}%, Test={test_val*100:.2f}%")
            else:
                print(f"{name}: Train={train_val:.2f}, Test={test_val:.2f}")
            
            # Calculate degradation
            if train_val != 0:
                degradation = (test_val - train_val) / abs(train_val) * 100
                print(f"  Degradation: {degradation:+.1f}%")
        
        print("\n🎯 Overfitting Analysis:")
        train_sharpe = results['Training']['best_sharpe']
        test_sharpe = results['Test']['best_sharpe']
        
        if train_sharpe > 0:
            sharpe_degradation = (test_sharpe - train_sharpe) / train_sharpe * 100
            
            if abs(sharpe_degradation) < 20:
                print("✅ Minimal overfitting - strategy appears robust")
            elif abs(sharpe_degradation) < 50:
                print("⚠️ Moderate overfitting - strategy may still be viable")
            else:
                print("❌ Severe overfitting - strategy may not be reliable")
                
            print(f"   Sharpe degradation: {sharpe_degradation:+.1f}%")
    
    return results

# Example usage:
# train_dir = Path('/path/to/training/results')
# test_dir = Path('/path/to/test/results')
# compare_train_test_performance(train_dir, test_dir)