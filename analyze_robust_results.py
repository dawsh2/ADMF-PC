#!/usr/bin/env python3
"""Analyze robust parameter results on test set"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

# Find the most recent results
robust_dir = Path("config/keltner/robust_config/results")
result_dirs = sorted([d for d in robust_dir.iterdir() if d.is_dir()])

if len(result_dirs) < 2:
    print("Need both train and test results")
    exit(1)

# Assuming last two are test and train
train_dir = result_dirs[-2]  
test_dir = result_dirs[-1]

print("ROBUST PARAMETERS (P=10, M=3.0) PERFORMANCE")
print("="*80)

def analyze_trace(trace_file, metadata_file, dataset_name):
    """Analyze a single trace file"""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    df = pd.read_parquet(trace_file)
    
    # Count trades and calculate returns
    trades = 0
    in_trade = False
    trade_returns = []
    
    for i in range(len(df)):
        signal = df.iloc[i]['val']
        price = df.iloc[i]['px']
        
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = price
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            if 'entry_price' in locals():
                if trade_direction > 0:  # Long
                    ret = (price - entry_price) / entry_price
                else:  # Short
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    # Calculate metrics
    total_bars = metadata['total_bars']
    bars_per_day = 78  # 6.5 hours * 12 (5-min bars)
    trading_days = total_bars / bars_per_day
    
    if trade_returns:
        avg_return = np.mean(trade_returns) * 100
        total_return = sum(trade_returns) * 100
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        
        # Compound return
        cumulative = 1.0
        for r in trade_returns:
            cumulative *= (1 + r)
        compound_return = (cumulative - 1) * 100
        
        # Annualized
        annual_simple = (total_return / trading_days) * 252
        years = trading_days / 252
        annual_compound = ((cumulative ** (1/years)) - 1) * 100 if years > 0 else 0
    else:
        avg_return = total_return = win_rate = compound_return = annual_simple = annual_compound = 0
    
    # Get signal frequency
    comp_name = trace_file.stem
    signal_freq = metadata['components'][comp_name]['signal_frequency'] * 100
    
    return {
        'dataset': dataset_name,
        'total_bars': total_bars,
        'trading_days': trading_days,
        'trades': trades,
        'trades_per_day': trades / trading_days if trading_days > 0 else 0,
        'signal_freq': signal_freq,
        'avg_return': avg_return,
        'total_return': total_return,
        'compound_return': compound_return,
        'win_rate': win_rate,
        'annual_simple': annual_simple,
        'annual_compound': annual_compound
    }

# Analyze both datasets
results = []

# Train result
train_trace = train_dir / "traces" / "mean_reversion" / "SPY_5m_kb_robust_p10_m3.parquet"
train_metadata = train_dir / "metadata.json"
if train_trace.exists():
    train_result = analyze_trace(train_trace, train_metadata, "TRAIN")
    results.append(train_result)

# Test result  
test_trace = test_dir / "traces" / "mean_reversion" / "SPY_5m_kb_robust_p10_m3.parquet"
test_metadata = test_dir / "metadata.json"
if test_trace.exists():
    test_result = analyze_trace(test_trace, test_metadata, "TEST")
    results.append(test_result)

# Display comparison
if len(results) == 2:
    print(f"{'Metric':<25} {'TRAIN':>15} {'TEST':>15} {'Consistency':>15}")
    print("-"*80)
    
    train = results[0]
    test = results[1]
    
    metrics = [
        ('Trading days', 'trading_days', '.1f', False),
        ('Total trades', 'trades', 'd', False),
        ('Trades per day', 'trades_per_day', '.2f', True),
        ('Signal frequency %', 'signal_freq', '.2f', True),
        ('Win rate %', 'win_rate', '.1f', True),
        ('Avg return per trade %', 'avg_return', '.4f', True),
        ('Total return %', 'total_return', '.2f', True),
        ('Compound return %', 'compound_return', '.2f', True),
        ('Annual return % (simple)', 'annual_simple', '.2f', True),
        ('Annual return % (compound)', 'annual_compound', '.2f', True)
    ]
    
    for label, key, fmt, check_consistency in metrics:
        train_val = train[key]
        test_val = test[key]
        
        if check_consistency and train_val != 0:
            consistency = (test_val / train_val * 100) if train_val != 0 else 0
            consistency_str = f"{consistency:.0f}%"
        else:
            consistency_str = "N/A"
            
        print(f"{label:<25} {train_val:>15{fmt}} {test_val:>15{fmt}} {consistency_str:>15}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    
    # Performance consistency
    return_consistency = test['annual_compound'] / train['annual_compound'] * 100 if train['annual_compound'] != 0 else 0
    
    if test['annual_compound'] > 0 and train['annual_compound'] > 0:
        print(f"✅ Strategy profitable in both periods")
        print(f"✅ Return consistency: {return_consistency:.0f}%")
        if return_consistency > 80:
            print(f"✅ EXCELLENT consistency - robust parameters confirmed!")
        elif return_consistency > 50:
            print(f"✅ GOOD consistency - parameters are reasonably robust")
        else:
            print(f"⚠️  MODERATE consistency - some degradation but still profitable")
    elif test['annual_compound'] > 0:
        print(f"✅ Strategy profitable on test set")
        print(f"⚠️  But was not profitable on training set - unusual!")
    else:
        print(f"❌ Strategy not profitable on test set")
        print(f"❌ Parameters may not be as robust as expected")
    
    # Compare to previous results
    print(f"\nCOMPARISON TO PREVIOUS ATTEMPTS:")
    print(f"Period 50, M=1.0 (no filter): Train=4.60% → Test=0.03%")
    print(f"Period 30, M=1.0 (no filter): Train=10.71% → Test=-6.31%") 
    print(f"Period 10, M=3.0 (robust):    Train={train['annual_compound']:.2f}% → Test={test['annual_compound']:.2f}%")
    
else:
    print("ERROR: Could not find both train and test results")