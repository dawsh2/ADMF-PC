#!/usr/bin/env python3
"""
Quick data structure check for ensemble analysis
"""

import sys
import os

# Try to use whatever Python environment has the packages
try:
    import pandas as pd
    import numpy as np
    print("✅ Successfully imported pandas and numpy")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# File paths
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet" 

def check_data_structure():
    """Check the basic structure of our data files"""
    print("🔍 Checking data structure...")
    
    # Check if files exist
    if not os.path.exists(DATA_PATH):
        print(f"❌ SPY data not found at {DATA_PATH}")
        return False
        
    # Check signal files
    signal_files = [
        f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet",
        f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_custom.parquet",
        f"{WORKSPACE_PATH}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
    ]
    
    for file_path in signal_files:
        if not os.path.exists(file_path):
            print(f"❌ Signal file not found: {file_path}")
            return False
        else:
            print(f"✅ Found: {os.path.basename(file_path)}")
    
    return True

def quick_data_sample():
    """Load and examine a small sample of each dataset"""
    print("\n📊 Loading data samples...")
    
    try:
        # Load just the first few rows to check structure
        print("Loading SPY data sample...")
        spy_data = pd.read_parquet(DATA_PATH)
        spy_sample = spy_data.head(5)
        print(f"SPY columns: {list(spy_sample.columns)}")
        print(f"SPY shape sample: {spy_sample.shape}")
        
        # Load signal samples
        default_path = f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet"
        custom_path = f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_custom.parquet"
        classifier_path = f"{WORKSPACE_PATH}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
        
        print("\nLoading signal samples...")
        default_data = pd.read_parquet(default_path)
        custom_data = pd.read_parquet(custom_path)
        classifier_data = pd.read_parquet(classifier_path)
        
        default_sample = default_data.head(5)
        custom_sample = custom_data.head(5)
        classifier_sample = classifier_data.head(5)
        
        print(f"Default ensemble columns: {list(default_sample.columns)}")
        print(f"Custom ensemble columns: {list(custom_sample.columns)}")
        print(f"Classifier columns: {list(classifier_sample.columns)}")
        
        print("\nFirst few rows of default ensemble:")
        print(default_sample)
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def get_data_info():
    """Get full dataset information"""
    print("\n📈 Getting full dataset info...")
    
    try:
        # Get full SPY dataset info
        spy_data = pd.read_parquet(DATA_PATH)
        print(f"SPY total rows: {len(spy_data):,}")
        
        if 'datetime' in spy_data.columns:
            spy_data['datetime'] = pd.to_datetime(spy_data['datetime'])
            print(f"Date range: {spy_data['datetime'].min()} to {spy_data['datetime'].max()}")
            last_12k_start = len(spy_data) - 12000
            print(f"Last 12k bars start at index: {last_12k_start:,}")
            print(f"Last 12k bars start date: {spy_data.iloc[last_12k_start]['datetime']}")
        
        # Get signal dataset info
        default_signals = pd.read_parquet(f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet")
        custom_signals = pd.read_parquet(f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_custom.parquet")
        classifier_signals = pd.read_parquet(f"{WORKSPACE_PATH}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet")
        
        print(f"\nSignal dataset sizes:")
        print(f"Default ensemble: {len(default_signals):,} signals")
        print(f"Custom ensemble: {len(custom_signals):,} signals")
        print(f"Classifier: {len(classifier_signals):,} signals")
        
        # Check signal ranges for last 12k bars
        analysis_start_idx = len(spy_data) - 12000
        
        default_filtered = default_signals[default_signals['bar_index'] >= analysis_start_idx]
        custom_filtered = custom_signals[custom_signals['bar_index'] >= analysis_start_idx]
        classifier_filtered = classifier_signals[classifier_signals['bar_index'] >= analysis_start_idx]
        
        print(f"\nSignals in last 12k bars:")
        print(f"Default ensemble: {len(default_filtered):,}")
        print(f"Custom ensemble: {len(custom_filtered):,}")
        print(f"Classifier: {len(classifier_filtered):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting data info: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Data Structure Check")
    print("=" * 50)
    
    if check_data_structure():
        if quick_data_sample():
            get_data_info()
            print("\n✅ Data structure check complete!")
        else:
            print("\n❌ Failed to load data samples")
    else:
        print("\n❌ Data structure check failed")