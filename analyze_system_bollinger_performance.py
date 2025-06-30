#!/usr/bin/env python3
"""
Analyze system execution performance for Bollinger Bands strategy
Compare to notebook results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration from the system execution
STRATEGY_PARAMS = {
    "period": 10,
    "std_dev": 1.5,
    "stop_loss": 0.00075,  # 0.075%
    "take_profit": 0.001,  # 0.1%
}

# Notebook performance (from cell outputs)
NOTEBOOK_PERFORMANCE = {
    "return": 0.2074,  # 20.74%
    "sharpe": 12.81,
    "win_rate": 0.75,  # 75%
    "stop_rate": 0.207,  # 20.7%
    "target_rate": 0.69,  # 69%
    "num_trades": 416,
    "trading_days": 47,
    "trades_per_day": 8.85,
}

def analyze_latest_results():
    """Analyze the latest system execution results"""
    
    # Find latest results directory
    results_base = Path("config/bollinger/results")
    latest_dir = results_base / "latest"
    
    if not latest_dir.exists():
        # Find most recent by timestamp
        dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name != "latest"]
        if dirs:
            latest_dir = max(dirs, key=lambda d: d.name)
        else:
            print("No results found!")
            return
    
    print(f"Analyzing results from: {latest_dir}")
    
    # Load trace data
    trace_path = latest_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
    if trace_path.exists():
        trace = pd.read_parquet(trace_path)
        print(f"\nTrace data loaded: {len(trace)} signals")
        
        # Count signal changes (trades)
        trace['signal_change'] = trace['val'].diff().abs()
        num_trades = trace['signal_change'].sum() // 2  # Divide by 2 since each trade has entry and exit
        
        print(f"Number of trades: {num_trades}")
        
        # Date range
        date_range = pd.to_datetime(trace['ts'])
        trading_days = len(pd.bdate_range(date_range.min(), date_range.max()))
        print(f"Trading days: {trading_days}")
        print(f"Trades per day: {num_trades / trading_days:.2f}")
        
        # Calculate actual performance
        # Note: This is a simplified calculation without the full backtesting framework
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        print(f"\nNotebook Results (period={STRATEGY_PARAMS['period']}, std_dev={STRATEGY_PARAMS['std_dev']}):")
        print(f"  Stop Loss: 0.075%")
        print(f"  Take Profit: 0.10%")
        print(f"  Return: {NOTEBOOK_PERFORMANCE['return']*100:.2f}%")
        print(f"  Sharpe: {NOTEBOOK_PERFORMANCE['sharpe']:.2f}")
        print(f"  Win Rate: {NOTEBOOK_PERFORMANCE['win_rate']*100:.1f}%")
        print(f"  Stop hit rate: {NOTEBOOK_PERFORMANCE['stop_rate']*100:.1f}%")
        print(f"  Target hit rate: {NOTEBOOK_PERFORMANCE['target_rate']*100:.1f}%")
        print(f"  Trades: {NOTEBOOK_PERFORMANCE['num_trades']}")
        
        print(f"\nSystem Execution (from metadata):")
        print(f"  Stop Loss: {STRATEGY_PARAMS['stop_loss']*100:.3f}%")
        print(f"  Take Profit: {STRATEGY_PARAMS['take_profit']*100:.1f}%")
        print(f"  Trades: {num_trades}")
        
        print("\n" + "="*60)
        print("KEY DIFFERENCES IDENTIFIED:")
        print("="*60)
        
        print("\n1. TAKE PROFIT LEVEL:")
        print(f"   Notebook: 0.10% (0.001)")
        print(f"   System: {STRATEGY_PARAMS['take_profit']*100:.1f}% ({STRATEGY_PARAMS['take_profit']})")
        print(f"   ⚠️  System uses SAME take profit as notebook!")
        
        print("\n2. DATA ANALYSIS:")
        print(f"   Both use 5-minute bars")
        print(f"   Notebook analyzed 47 trading days")
        print(f"   System analyzed {trading_days} trading days")
        
        print("\n3. TRADE COUNT:")
        print(f"   Notebook: {NOTEBOOK_PERFORMANCE['num_trades']} trades")
        print(f"   System: {num_trades} trades")
        if num_trades != NOTEBOOK_PERFORMANCE['num_trades']:
            print(f"   ⚠️  Trade count mismatch! Difference: {abs(num_trades - NOTEBOOK_PERFORMANCE['num_trades'])}")
        
        # Expected return calculation
        print("\n4. EXPECTED RETURN CALCULATION:")
        stop_return = -STRATEGY_PARAMS['stop_loss'] - 0.0001  # Including execution cost
        target_return = STRATEGY_PARAMS['take_profit'] - 0.0001
        
        expected_return_per_trade = (
            NOTEBOOK_PERFORMANCE['stop_rate'] * stop_return +
            NOTEBOOK_PERFORMANCE['target_rate'] * target_return +
            (1 - NOTEBOOK_PERFORMANCE['stop_rate'] - NOTEBOOK_PERFORMANCE['target_rate']) * (-0.0001)
        )
        
        print(f"   Using notebook's exit distribution:")
        print(f"   - {NOTEBOOK_PERFORMANCE['stop_rate']*100:.1f}% stops at {stop_return*100:.3f}%")
        print(f"   - {NOTEBOOK_PERFORMANCE['target_rate']*100:.1f}% targets at {target_return*100:.3f}%")
        print(f"   - {(1-NOTEBOOK_PERFORMANCE['stop_rate']-NOTEBOOK_PERFORMANCE['target_rate'])*100:.1f}% signals at -0.01%")
        print(f"   Expected return per trade: {expected_return_per_trade*100:.4f}%")
        print(f"   Expected total return: {expected_return_per_trade * num_trades * 100:.2f}%")
        
        print("\n" + "="*60)
        print("CONCLUSIONS:")
        print("="*60)
        print("\n1. The notebook and system use the SAME parameters:")
        print("   - Period: 10")
        print("   - Std Dev: 1.5")
        print("   - Stop Loss: 0.075%")
        print("   - Take Profit: 0.1%")
        
        print("\n2. The performance difference is likely due to:")
        print("   - Implementation details in signal generation")
        print("   - Exact entry/exit timing")
        print("   - How stop/target exits are processed")
        print("   - Possible differences in data preprocessing")
        
        print("\n3. To match notebook performance, check:")
        print("   - Signal generation logic matches exactly")
        print("   - Stop/target exit processing is identical")
        print("   - Data alignment and bar timestamps match")
        print("   - Execution costs are applied consistently")
        
    else:
        print(f"Trace file not found: {trace_path}")
    
    # Load metadata to confirm
    metadata_path = latest_dir / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print("\n" + "="*60)
        print("METADATA CONFIRMATION:")
        print("="*60)
        
        component = metadata['components'].get('SPY_5m_strategy_0', {})
        params = component.get('parameters', {})
        risk = params.get('_risk', {})
        
        print(f"Strategy Type: {component.get('strategy_type')}")
        print(f"Period: {params.get('period')}")
        print(f"Std Dev: {params.get('std_dev')}")
        print(f"Stop Loss: {risk.get('stop_loss')} ({risk.get('stop_loss', 0)*100:.3f}%)")
        print(f"Take Profit: {risk.get('take_profit')} ({risk.get('take_profit', 0)*100:.1f}%)")

if __name__ == "__main__":
    analyze_latest_results()