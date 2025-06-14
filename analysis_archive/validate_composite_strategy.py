"""
Validate composite strategy patterns and check for bar alignment issues.

This script:
1. Loads historical data and signals
2. Checks for look-ahead bias
3. Runs proper backtests with train/test/validation splits
4. Reports statistical significance of results
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

from src.analytics.backtesting_framework import (
    RobustBacktester, BacktestConfig, create_validation_report
)
from src.analytics.storage.hierarchical_parquet_storage import HierarchicalParquetStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_signal_data(workspace_path: str) -> pd.DataFrame:
    """Load signals and price data from hierarchical storage."""
    storage = HierarchicalParquetStorage(workspace_path)
    
    # Find signal data
    signal_files = list(Path(workspace_path).rglob("**/signals/*.parquet"))
    
    if not signal_files:
        logger.error("No signal files found")
        return pd.DataFrame()
        
    logger.info(f"Found {len(signal_files)} signal files")
    
    # Load and combine data
    all_data = []
    
    for file in signal_files:
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            
            # Extract metadata
            if table.schema.metadata:
                metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
                strategy_id = metadata.get('strategy_id', 'unknown')
                df['strategy_id'] = strategy_id
                
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        
    return combined_df


def load_price_data(workspace_path: str, symbol: str = None) -> pd.DataFrame:
    """Load price data from storage."""
    # Look for data files
    data_files = list(Path(workspace_path).rglob("**/data/*.parquet"))
    
    if not data_files:
        logger.error("No price data files found")
        return pd.DataFrame()
        
    price_data = []
    
    for file in data_files:
        try:
            df = pd.read_parquet(file)
            if symbol and 'symbol' in df.columns:
                df = df[df['symbol'] == symbol]
            price_data.append(df)
        except Exception as e:
            logger.error(f"Error loading price data from {file}: {e}")
            
    if not price_data:
        return pd.DataFrame()
        
    combined = pd.concat(price_data, ignore_index=True)
    
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values('timestamp')
        
    return combined


def merge_signals_and_prices(signals_df: pd.DataFrame, 
                           prices_df: pd.DataFrame) -> pd.DataFrame:
    """Merge signals with price data for backtesting."""
    
    # Ensure we have required columns
    if 'timestamp' not in signals_df.columns or 'timestamp' not in prices_df.columns:
        logger.error("Missing timestamp columns")
        return pd.DataFrame()
        
    # Set timestamp as index
    signals_df = signals_df.set_index('timestamp')
    prices_df = prices_df.set_index('timestamp')
    
    # Get signal value column
    signal_col = 'signal_value' if 'signal_value' in signals_df.columns else 'signal'
    
    # Merge on timestamp
    merged = prices_df.merge(
        signals_df[[signal_col, 'strategy_id']], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    # Forward fill signals (signals persist until changed)
    merged[signal_col] = merged[signal_col].fillna(0)
    
    return merged


def check_feature_calculation_timing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if features could have look-ahead bias.
    
    Examines how features are calculated relative to price data.
    """
    issues = []
    
    # Check if any feature columns exist
    feature_cols = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi_', 'macd_'))]
    
    if not feature_cols:
        logger.warning("No feature columns found in data")
        return {'issues': issues}
        
    # Check each feature
    for feature in feature_cols:
        if feature not in df.columns:
            continue
            
        # Check if feature changes before price
        feature_changes = df[feature].diff().abs() > 0
        price_changes = df['close'].diff().abs() > 0
        
        # Count how often feature changes without price change
        feature_only_changes = (feature_changes & ~price_changes).sum()
        
        if feature_only_changes > 0:
            issues.append({
                'feature': feature,
                'issue': 'feature_changes_without_price',
                'count': int(feature_only_changes),
                'severity': 'high' if feature_only_changes > 10 else 'medium'
            })
            
        # Check correlation with future returns
        future_returns = df['close'].pct_change().shift(-1)
        current_feature = df[feature]
        
        # Only check where both are not NaN
        mask = ~(current_feature.isna() | future_returns.isna())
        if mask.sum() > 30:  # Need enough data points
            corr = current_feature[mask].corr(future_returns[mask])
            
            if abs(corr) > 0.5:  # Suspiciously high correlation
                issues.append({
                    'feature': feature,
                    'issue': 'high_future_correlation',
                    'correlation': float(corr),
                    'severity': 'critical' if abs(corr) > 0.7 else 'high'
                })
                
    return {
        'issues': issues,
        'num_issues': len(issues),
        'critical_issues': len([i for i in issues if i.get('severity') == 'critical'])
    }


def validate_composite_strategy(workspace_path: str, 
                              strategy_id: str = 'trend_momentum_composite',
                              output_dir: str = 'validation_results'):
    """Main validation function."""
    
    logger.info(f"Validating strategy: {strategy_id}")
    logger.info(f"Workspace: {workspace_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading signal data...")
    signals_df = load_signal_data(workspace_path)
    
    if signals_df.empty:
        logger.error("No signal data found")
        return
        
    # Filter for specific strategy if provided
    if strategy_id and 'strategy_id' in signals_df.columns:
        signals_df = signals_df[signals_df['strategy_id'] == strategy_id]
        logger.info(f"Filtered to {len(signals_df)} signals for {strategy_id}")
        
    # Load price data
    logger.info("Loading price data...")
    prices_df = load_price_data(workspace_path)
    
    if prices_df.empty:
        logger.error("No price data found")
        return
        
    # Merge data
    logger.info("Merging signals and prices...")
    backtest_data = merge_signals_and_prices(signals_df, prices_df)
    
    if backtest_data.empty:
        logger.error("No data after merging")
        return
        
    logger.info(f"Merged data shape: {backtest_data.shape}")
    
    # Check for alignment issues
    logger.info("Checking for feature calculation timing issues...")
    timing_issues = check_feature_calculation_timing(backtest_data)
    
    if timing_issues['critical_issues'] > 0:
        logger.warning(f"Found {timing_issues['critical_issues']} critical timing issues!")
        
    # Extract feature columns
    feature_cols = [col for col in backtest_data.columns 
                   if col.startswith(('sma_', 'ema_', 'rsi_', 'macd_', 'bollinger_'))]
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    # Configure backtester
    config = BacktestConfig(
        train_ratio=0.6,
        test_ratio=0.2,
        validation_ratio=0.2,
        min_bars_warmup=200,
        min_trades_significance=30,
        feature_lag=1,  # Lag features by 1 bar
        signal_delay=1,  # Delay signal execution by 1 bar
        commission_bps=10,  # 10 bps commission
        slippage_bps=5  # 5 bps slippage
    )
    
    # Run backtest
    logger.info("Running robust backtest with proper data splits...")
    backtester = RobustBacktester(config)
    
    # Get signal column name
    signal_col = 'signal_value' if 'signal_value' in backtest_data.columns else 'signal'
    
    # Run validation
    results = backtester.validate_out_of_sample(
        backtest_data,
        signal_col,
        feature_cols
    )
    
    # Create reports
    logger.info("Creating validation reports...")
    
    # Main validation report
    create_validation_report(
        results,
        str(output_path / f'{strategy_id}_validation_report.json')
    )
    
    # Timing issues report
    timing_report_path = output_path / f'{strategy_id}_timing_issues.json'
    with open(timing_report_path, 'w') as f:
        import json
        json.dump(timing_issues, f, indent=2)
        
    # Summary report
    summary = {
        'strategy_id': strategy_id,
        'data_points': len(backtest_data),
        'date_range': {
            'start': str(backtest_data.index[0]),
            'end': str(backtest_data.index[-1])
        },
        'performance_summary': {
            'train': {
                'return': f"{results['train'].total_return:.2%}",
                'sharpe': f"{results['train'].sharpe_ratio:.2f}",
                'trades': len(results['train'].trades),
                'p_value': results['train'].p_value
            },
            'test': {
                'return': f"{results['test'].total_return:.2%}",
                'sharpe': f"{results['test'].sharpe_ratio:.2f}",
                'trades': len(results['test'].trades),
                'p_value': results['test'].p_value
            },
            'validation': {
                'return': f"{results['validation'].total_return:.2%}",
                'sharpe': f"{results['validation'].sharpe_ratio:.2f}",
                'trades': len(results['validation'].trades),
                'p_value': results['validation'].p_value
            }
        },
        'alignment_check': {
            'timing_issues_found': timing_issues['num_issues'],
            'critical_issues': timing_issues['critical_issues'],
            'max_feature_leakage': max(
                results['test'].feature_future_leakage.values()
            ) if results['test'].feature_future_leakage else 0
        },
        'statistical_significance': {
            'test_significant': results['test'].p_value < 0.05,
            'validation_significant': results['validation'].p_value < 0.05,
            'consistent_performance': abs(
                results['test'].avg_return_per_trade - 
                results['validation'].avg_return_per_trade
            ) < 0.001  # Less than 0.1% difference
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Strategy: {strategy_id}")
    print(f"Data points: {summary['data_points']:,}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print("\nPerformance Results:")
    print(f"  Train:      {summary['performance_summary']['train']['return']} return, "
          f"{summary['performance_summary']['train']['sharpe']} Sharpe, "
          f"{summary['performance_summary']['train']['trades']} trades")
    print(f"  Test:       {summary['performance_summary']['test']['return']} return, "
          f"{summary['performance_summary']['test']['sharpe']} Sharpe, "
          f"{summary['performance_summary']['test']['trades']} trades")
    print(f"  Validation: {summary['performance_summary']['validation']['return']} return, "
          f"{summary['performance_summary']['validation']['sharpe']} Sharpe, "
          f"{summary['performance_summary']['validation']['trades']} trades")
    print("\nStatistical Significance:")
    print(f"  Test p-value: {results['test'].p_value:.4f} "
          f"({'SIGNIFICANT' if results['test'].p_value < 0.05 else 'not significant'})")
    print(f"  Validation p-value: {results['validation'].p_value:.4f} "
          f"({'SIGNIFICANT' if results['validation'].p_value < 0.05 else 'not significant'})")
    print("\nAlignment Issues:")
    print(f"  Timing issues found: {timing_issues['num_issues']}")
    print(f"  Critical issues: {timing_issues['critical_issues']}")
    print(f"  Max feature leakage: {summary['alignment_check']['max_feature_leakage']:.3f}")
    
    if timing_issues['critical_issues'] > 0:
        print("\n⚠️  WARNING: Critical timing issues detected!")
        print("The high returns may be due to look-ahead bias.")
    elif results['test'].p_value > 0.05 or results['validation'].p_value > 0.05:
        print("\n⚠️  WARNING: Results not statistically significant!")
        print("The returns may be due to random chance.")
    else:
        print("\n✅ Results appear statistically significant with no critical issues.")
        
    print("="*60)
    
    # Save summary
    summary_path = output_path / f'{strategy_id}_summary.json'
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
        
    logger.info(f"All reports saved to {output_path}")
    
    return results, timing_issues


if __name__ == "__main__":
    import sys
    
    # Get workspace path from command line or use default
    workspace_path = sys.argv[1] if len(sys.argv) > 1 else "workspaces/expansive_grid_search_bc73ecec"
    strategy_id = sys.argv[2] if len(sys.argv) > 2 else "trend_momentum_composite"
    
    # Run validation
    results, issues = validate_composite_strategy(workspace_path, strategy_id)