# Detailed comparison of analysis vs execution trades
import pandas as pd
from pathlib import Path

# Load both trade lists
analysis_csv = 'analysis_trades_5edc4365.csv'
execution_csv = 'execution_trades_bollinger_fixed.csv'

if Path(analysis_csv).exists() and Path(execution_csv).exists():
    analysis_trades = pd.read_csv(analysis_csv)
    execution_trades = pd.read_csv(execution_csv)
    
    # Convert timestamps
    for df in [analysis_trades, execution_trades]:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    print("="*120)
    print("DETAILED TRADE-BY-TRADE COMPARISON")
    print("="*120)
    
    # Find where trades diverge
    print("\n1. TRADE COUNT:")
    print(f"   Analysis:  {len(analysis_trades)} trades")
    print(f"   Execution: {len(execution_trades)} trades")
    
    # Check entry times
    print("\n2. ENTRY TIME COMPARISON (first 20 trades):")
    print("-"*80)
    print(f"{'#':>3} {'Analysis Entry':>20} {'Execution Entry':>20} {'Match':>8} {'Time Diff':>15}")
    print("-"*80)
    
    for i in range(min(20, len(analysis_trades), len(execution_trades))):
        a_entry = analysis_trades.iloc[i]['entry_time']
        e_entry = execution_trades.iloc[i]['entry_time']
        
        time_diff = abs((a_entry - e_entry).total_seconds())
        match = "✓" if time_diff < 300 else "✗"  # Within 5 minutes
        
        print(f"{i+1:>3} {a_entry.strftime('%Y-%m-%d %H:%M'):>20} {e_entry.strftime('%Y-%m-%d %H:%M'):>20} "
              f"{match:>8} {time_diff/60:>14.1f}m")
    
    # Find first divergence
    first_mismatch = None
    for i in range(min(len(analysis_trades), len(execution_trades))):
        a_entry = analysis_trades.iloc[i]['entry_time']
        e_entry = execution_trades.iloc[i]['entry_time']
        if abs((a_entry - e_entry).total_seconds()) >= 300:
            first_mismatch = i
            break
    
    if first_mismatch is not None:
        print(f"\n⚠️  FIRST DIVERGENCE at trade #{first_mismatch + 1}")
        print(f"   Analysis:  {analysis_trades.iloc[first_mismatch]['entry_time']} {analysis_trades.iloc[first_mismatch]['dir']}")
        print(f"   Execution: {execution_trades.iloc[first_mismatch]['entry_time']} {execution_trades.iloc[first_mismatch]['dir']}")
    
    # Compare exit types
    print("\n3. EXIT TYPE COMPARISON (first 20 trades):")
    print("-"*80)
    print(f"{'#':>3} {'Entry Time':>16} {'Analysis Exit':>15} {'Execution Exit':>15} {'Match':>8}")
    print("-"*80)
    
    for i in range(min(20, len(analysis_trades), len(execution_trades))):
        a = analysis_trades.iloc[i]
        e = execution_trades.iloc[i]
        
        # Only compare if entry times are close
        time_diff = abs((a['entry_time'] - e['entry_time']).total_seconds())
        if time_diff < 300:  # Same trade
            exit_match = "✓" if a['exit_type'] == e['exit_type'] else "✗"
            print(f"{i+1:>3} {a['entry_time'].strftime('%m-%d %H:%M'):>16} "
                  f"{a['exit_type']:>15} {e['exit_type']:>15} {exit_match:>8}")
    
    # Performance comparison
    print("\n4. PERFORMANCE METRICS:")
    print("-"*50)
    
    # Calculate metrics for matching trades only
    matching_trades = 0
    a_returns = []
    e_returns = []
    
    for i in range(min(len(analysis_trades), len(execution_trades))):
        a = analysis_trades.iloc[i]
        e = execution_trades.iloc[i]
        
        time_diff = abs((a['entry_time'] - e['entry_time']).total_seconds())
        if time_diff < 300:  # Same trade
            matching_trades += 1
            a_returns.append(a['return'])
            e_returns.append(e['return'])
    
    print(f"Matching trades: {matching_trades}")
    if matching_trades > 0:
        a_returns = pd.Series(a_returns)
        e_returns = pd.Series(e_returns)
        
        print(f"\nFor matching trades:")
        print(f"  Analysis avg return:   {a_returns.mean()*100:>6.3f}%")
        print(f"  Execution avg return:  {e_returns.mean()*100:>6.3f}%")
        print(f"  Analysis win rate:     {(a_returns > 0).mean()*100:>6.1f}%")
        print(f"  Execution win rate:    {(e_returns > 0).mean()*100:>6.1f}%")
        print(f"  Analysis total return: {((1 + a_returns).prod() - 1)*100:>6.2f}%")
        print(f"  Execution total return:{((1 + e_returns).prod() - 1)*100:>6.2f}%")
    
    # Check for extra trades
    print("\n5. CHECKING FOR EXTRA TRADES:")
    
    # Find execution trades that don't match any analysis trade
    unmatched_execution = []
    for i, e_trade in execution_trades.iterrows():
        e_entry = e_trade['entry_time']
        matched = False
        for j, a_trade in analysis_trades.iterrows():
            if abs((e_entry - a_trade['entry_time']).total_seconds()) < 300:
                matched = True
                break
        if not matched:
            unmatched_execution.append((i+1, e_trade))
    
    if unmatched_execution:
        print(f"\n⚠️  Execution has {len(unmatched_execution)} extra trades not in analysis:")
        for trade_num, trade in unmatched_execution[:10]:  # Show first 10
            print(f"   Trade #{trade_num}: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} "
                  f"{trade['dir']} entry={trade['entry_price']:.2f}")
    
    # Find analysis trades that don't match any execution trade
    unmatched_analysis = []
    for i, a_trade in analysis_trades.iterrows():
        a_entry = a_trade['entry_time']
        matched = False
        for j, e_trade in execution_trades.iterrows():
            if abs((a_entry - e_trade['entry_time']).total_seconds()) < 300:
                matched = True
                break
        if not matched:
            unmatched_analysis.append((i+1, a_trade))
    
    if unmatched_analysis:
        print(f"\n⚠️  Analysis has {len(unmatched_analysis)} trades not in execution:")
        for trade_num, trade in unmatched_analysis[:10]:  # Show first 10
            print(f"   Trade #{trade_num}: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} "
                  f"{trade['dir']} entry={trade['entry_price']:.2f}")

else:
    print("Please ensure both analysis_trades_5edc4365.csv and execution_trades_bollinger_fixed.csv exist")