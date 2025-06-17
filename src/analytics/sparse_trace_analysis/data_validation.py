"""
Data validation module for sparse trace analysis.

Provides validation functions to ensure data quality and consistency
for classifier and strategy analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def validate_signals_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate strategy signals DataFrame for common issues.
    
    Args:
        df: Signals DataFrame with columns ['bar_idx', 'signal_value', 'price']
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    # Check required columns
    required_cols = ['bar_idx', 'signal_value', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation['errors'].append(f"Missing required columns: {missing_cols}")
        validation['is_valid'] = False
        return validation
    
    # Basic statistics
    validation['stats'] = {
        'total_records': len(df),
        'unique_bar_indices': df['bar_idx'].nunique(),
        'signal_value_range': (df['signal_value'].min(), df['signal_value'].max()),
        'price_range': (df['price'].min(), df['price'].max()),
        'null_counts': df.isnull().sum().to_dict()
    }
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation['warnings'].append(f"Null values found: {null_counts.to_dict()}")
    
    # Check bar_idx ordering
    if not df['bar_idx'].is_monotonic_increasing:
        validation['errors'].append("bar_idx is not in ascending order")
        validation['is_valid'] = False
    
    # Check for duplicate bar indices
    duplicates = df['bar_idx'].duplicated().sum()
    if duplicates > 0:
        validation['warnings'].append(f"Found {duplicates} duplicate bar indices")
    
    # Check price validity
    invalid_prices = (df['price'] <= 0).sum()
    if invalid_prices > 0:
        validation['errors'].append(f"Found {invalid_prices} invalid prices (<=0)")
        validation['is_valid'] = False
    
    # Check signal value range
    signal_values = df['signal_value'].unique()
    expected_signals = {-1, 0, 1}
    unexpected_signals = set(signal_values) - expected_signals
    
    if unexpected_signals:
        validation['warnings'].append(
            f"Unexpected signal values found: {sorted(unexpected_signals)}"
        )
    
    # Check for reasonable price movements
    if len(df) > 1:
        price_changes = df['price'].pct_change().dropna()
        extreme_changes = (abs(price_changes) > 0.2).sum()  # >20% moves
        
        if extreme_changes > len(df) * 0.01:  # More than 1% of data
            validation['warnings'].append(
                f"Many extreme price movements detected: {extreme_changes}"
            )
    
    return validation


def validate_classifier_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate classifier DataFrame for common issues.
    
    Args:
        df: Classifier DataFrame with columns ['bar_idx', 'state']
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    # Check required columns
    required_cols = ['bar_idx', 'state']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation['errors'].append(f"Missing required columns: {missing_cols}")
        validation['is_valid'] = False
        return validation
    
    # Basic statistics
    validation['stats'] = {
        'total_changes': len(df),
        'unique_states': df['state'].nunique(),
        'state_list': sorted(df['state'].unique().tolist()),
        'bar_idx_range': (df['bar_idx'].min(), df['bar_idx'].max()),
        'null_counts': df.isnull().sum().to_dict()
    }
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation['errors'].append(f"Null values found: {null_counts.to_dict()}")
        validation['is_valid'] = False
    
    # Check bar_idx ordering
    if not df['bar_idx'].is_monotonic_increasing:
        validation['errors'].append("bar_idx is not in ascending order")
        validation['is_valid'] = False
    
    # Check for duplicate bar indices (should never happen for state changes)
    duplicates = df['bar_idx'].duplicated().sum()
    if duplicates > 0:
        validation['errors'].append(f"Found {duplicates} duplicate bar indices")
        validation['is_valid'] = False
    
    # Check state consistency (no immediate reversions)
    if len(df) > 2:
        for i in range(len(df) - 2):
            if df.iloc[i]['state'] == df.iloc[i + 2]['state']:
                # State A -> State B -> State A pattern
                validation['warnings'].append(
                    f"Quick state reversion detected at bars "
                    f"{df.iloc[i]['bar_idx']}-{df.iloc[i+2]['bar_idx']}"
                )
                break  # Only report first occurrence
    
    # Check for very short state durations
    if len(df) > 1:
        durations = df['bar_idx'].diff().dropna()
        short_durations = (durations < 5).sum()  # Less than 5 bars
        
        if short_durations > len(durations) * 0.5:  # More than 50%
            validation['warnings'].append(
                f"Many very short state durations: {short_durations}/{len(durations)}"
            )
    
    return validation


def validate_trade_data(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate trade data for consistency and correctness.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    if not trades:
        validation['stats'] = {'total_trades': 0}
        return validation
    
    # Check required fields
    required_fields = ['entry_bar', 'exit_bar', 'entry_price', 'exit_price', 'signal']
    
    missing_fields = set()
    for trade in trades:
        trade_missing = [field for field in required_fields if field not in trade]
        missing_fields.update(trade_missing)
    
    if missing_fields:
        validation['errors'].append(f"Trades missing required fields: {sorted(missing_fields)}")
        validation['is_valid'] = False
    
    # Basic statistics
    entry_bars = [trade.get('entry_bar', 0) for trade in trades]
    exit_bars = [trade.get('exit_bar', 0) for trade in trades]
    durations = [trade.get('bars_held', 0) for trade in trades]
    returns = [trade.get('net_log_return', 0) for trade in trades]
    
    validation['stats'] = {
        'total_trades': len(trades),
        'avg_duration': np.mean(durations) if durations else 0,
        'avg_return': np.mean(returns) if returns else 0,
        'bar_range': (min(entry_bars), max(exit_bars)) if entry_bars and exit_bars else (0, 0),
        'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0
    }
    
    # Validate trade logic
    for i, trade in enumerate(trades):
        entry_bar = trade.get('entry_bar', 0)
        exit_bar = trade.get('exit_bar', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        # Check bar ordering
        if exit_bar <= entry_bar:
            validation['errors'].append(f"Trade {i}: exit_bar <= entry_bar")
            validation['is_valid'] = False
        
        # Check price validity
        if entry_price <= 0 or exit_price <= 0:
            validation['errors'].append(f"Trade {i}: invalid prices")
            validation['is_valid'] = False
        
        # Check for extreme returns
        if 'net_log_return' in trade:
            log_return = trade['net_log_return']
            if abs(log_return) > 2:  # >100% single trade return
                validation['warnings'].append(f"Trade {i}: extreme return {log_return:.4f}")
    
    # Check for overlapping trades (should not happen in single strategy)
    sorted_trades = sorted(trades, key=lambda x: x.get('entry_bar', 0))
    
    for i in range(len(sorted_trades) - 1):
        current_exit = sorted_trades[i].get('exit_bar', 0)
        next_entry = sorted_trades[i + 1].get('entry_bar', 0)
        
        if current_exit > next_entry:
            validation['warnings'].append(f"Overlapping trades detected at trade {i}")
    
    return validation


def validate_workspace_structure(workspace_path: Path) -> Dict[str, Any]:
    """
    Validate workspace directory structure for completeness.
    
    Args:
        workspace_path: Path to workspace directory
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    if not workspace_path.exists():
        validation['errors'].append(f"Workspace path does not exist: {workspace_path}")
        validation['is_valid'] = False
        return validation
    
    # Check expected directory structure
    expected_dirs = [
        'traces',
        'traces/SPY_1m',
        'traces/SPY_1m/signals',
        'traces/SPY_1m/classifiers'
    ]
    
    missing_dirs = []
    for expected_dir in expected_dirs:
        dir_path = workspace_path / expected_dir
        if not dir_path.exists():
            missing_dirs.append(expected_dir)
    
    if missing_dirs:
        validation['errors'].append(f"Missing expected directories: {missing_dirs}")
        validation['is_valid'] = False
    
    # Count files in each directory
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    classifiers_dir = workspace_path / "traces" / "SPY_1m" / "classifiers"
    
    validation['stats'] = {
        'signals_files': len(list(signals_dir.rglob("*.parquet"))) if signals_dir.exists() else 0,
        'classifier_files': len(list(classifiers_dir.rglob("*.parquet"))) if classifiers_dir.exists() else 0,
        'total_size_mb': sum(f.stat().st_size for f in workspace_path.rglob("*") if f.is_file()) / (1024 * 1024)
    }
    
    # Check for metadata files
    metadata_file = workspace_path / "metadata.json"
    if not metadata_file.exists():
        validation['warnings'].append("No metadata.json file found")
    
    # Warn if very few files
    if validation['stats']['signals_files'] < 10:
        validation['warnings'].append(f"Very few signal files: {validation['stats']['signals_files']}")
    
    if validation['stats']['classifier_files'] < 5:
        validation['warnings'].append(f"Very few classifier files: {validation['stats']['classifier_files']}")
    
    return validation


def generate_validation_report(
    workspace_path: Path,
    sample_files: Optional[int] = 5
) -> str:
    """
    Generate comprehensive validation report for workspace.
    
    Args:
        workspace_path: Path to workspace to validate
        sample_files: Number of sample files to validate in detail
        
    Returns:
        Formatted validation report
    """
    report_lines = [
        "="*80,
        "WORKSPACE VALIDATION REPORT",
        "="*80,
        f"Workspace: {workspace_path}",
        ""
    ]
    
    # Validate workspace structure
    workspace_validation = validate_workspace_structure(workspace_path)
    
    report_lines.extend([
        "WORKSPACE STRUCTURE:",
        f"  Valid: {'✓' if workspace_validation['is_valid'] else '✗'}",
        f"  Signal files: {workspace_validation['stats'].get('signals_files', 0)}",
        f"  Classifier files: {workspace_validation['stats'].get('classifier_files', 0)}",
        f"  Total size: {workspace_validation['stats'].get('total_size_mb', 0):.1f} MB",
        ""
    ])
    
    if workspace_validation['errors']:
        report_lines.extend(["  Errors:"] + [f"    - {error}" for error in workspace_validation['errors']])
        report_lines.append("")
    
    if workspace_validation['warnings']:
        report_lines.extend(["  Warnings:"] + [f"    - {warning}" for warning in workspace_validation['warnings']])
        report_lines.append("")
    
    # Sample file validation
    if workspace_validation['is_valid'] and sample_files > 0:
        # Sample signals files
        signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
        if signals_dir.exists():
            signal_files = list(signals_dir.rglob("*.parquet"))[:sample_files]
            
            report_lines.extend([
                f"SAMPLE SIGNAL FILES VALIDATION ({len(signal_files)} files):",
                ""
            ])
            
            for file_path in signal_files:
                try:
                    df = pd.read_parquet(file_path)
                    df = df.rename(columns={'idx': 'bar_idx', 'val': 'signal_value', 'px': 'price'})
                    validation = validate_signals_dataframe(df)
                    
                    status = "✓" if validation['is_valid'] else "✗"
                    report_lines.append(f"  {file_path.name}: {status}")
                    
                    if validation['errors']:
                        for error in validation['errors']:
                            report_lines.append(f"    Error: {error}")
                    
                except Exception as e:
                    report_lines.append(f"  {file_path.name}: ✗ (Error: {e})")
            
            report_lines.append("")
        
        # Sample classifier files  
        classifiers_dir = workspace_path / "traces" / "SPY_1m" / "classifiers"
        if classifiers_dir.exists():
            classifier_files = list(classifiers_dir.rglob("*.parquet"))[:sample_files]
            
            report_lines.extend([
                f"SAMPLE CLASSIFIER FILES VALIDATION ({len(classifier_files)} files):",
                ""
            ])
            
            for file_path in classifier_files:
                try:
                    df = pd.read_parquet(file_path)
                    df = df.rename(columns={'idx': 'bar_idx', 'val': 'state'})
                    validation = validate_classifier_dataframe(df)
                    
                    status = "✓" if validation['is_valid'] else "✗"
                    report_lines.append(f"  {file_path.name}: {status}")
                    
                    if validation['errors']:
                        for error in validation['errors']:
                            report_lines.append(f"    Error: {error}")
                    
                except Exception as e:
                    report_lines.append(f"  {file_path.name}: ✗ (Error: {e})")
    
    return "\n".join(report_lines)