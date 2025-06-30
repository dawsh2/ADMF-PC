"""
Analyze the reported difference between universal analysis (523 profit targets) 
and execution engine (150 profit targets).
"""

import pandas as pd
import numpy as np

def analyze_reported_metrics():
    """Analyze the reported metrics to understand the discrepancy."""
    
    print("ANALYZING REPORTED PROFIT TARGET DIFFERENCES")
    print("=" * 80)
    
    print("\nReported Results:")
    print("- Universal Analysis: 523 profit targets hit")
    print("- Execution Engine: 150 profit targets hit")
    print("- Difference: 373 trades (71% fewer in execution)")
    
    print("\nPossible Explanations:")
    print("\n1. DIFFERENT SIGNAL HANDLING:")
    print("   - Universal might continue holding through neutral signals")
    print("   - Execution might exit on any non-confirming signal")
    
    print("\n2. STOP/TARGET CHECK ORDER:")
    print("   - Universal: Signal → Stop/Target (on exit path)")
    print("   - Execution: Stop/Target → Signal (every bar)")
    
    print("\n3. PRICE DATA PRECISION:")
    print("   - Universal might use exact high/low for stop/target")
    print("   - Execution might use close prices or different logic")
    
    print("\n4. RE-ENTRY LOGIC:")
    print("   - Both claim to prevent re-entry, but implementation differs")
    print("   - Execution might have stricter re-entry prevention")
    
    # Let's create a specific test case
    print("\n" + "="*80)
    print("TEST CASE: Stop/Target Check Order Impact")
    print("="*80)
    
    # Simulate a scenario
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01 09:30', periods=10, freq='1min'),
        'open': [100, 100.5, 101.8, 102.1, 101.5, 100.8, 99.5, 99.8, 100.2, 100.5],
        'high': [100.5, 101.2, 102.2, 102.5, 101.8, 101.0, 99.8, 100.1, 100.5, 100.8],
        'low': [99.8, 100.3, 101.5, 101.8, 100.5, 99.5, 99.2, 99.5, 99.8, 100.2],
        'close': [100.2, 101.0, 102.0, 102.0, 101.0, 99.8, 99.5, 100.0, 100.3, 100.5],
        'signal': [1, 0, 0, 0, -1, 0, 0, 1, 0, 0]  # Long, hold, hold, hold, short, hold, hold, long
    })
    
    print("\nTest scenario:")
    print("- Entry: $100 (long)")
    print("- Stop: $99 (-1%)")
    print("- Target: $102 (+2%)")
    print("- Bar 3: High reaches $102.2 (target hit)")
    print("- Bar 5: Signal changes to SHORT")
    
    # Universal approach
    print("\n--- Universal Approach ---")
    print("1. Enter long at bar 1 ($100)")
    print("2. Hold through bars 2-4")
    print("3. See exit signal at bar 5")
    print("4. Look back at bars 2-4, see target was hit at bar 3")
    print("5. Exit at target price ($102) with timestamp of bar 3")
    print("Result: PROFIT TARGET HIT ✓")
    
    # Execution approach
    print("\n--- Execution Approach ---")
    print("1. Enter long at bar 1 ($100)")
    print("2. Bar 2: Check stop/target - not hit")
    print("3. Bar 3: Check stop/target - TARGET HIT!")
    print("4. Exit immediately at $102")
    print("5. No position when SHORT signal arrives at bar 5")
    print("Result: PROFIT TARGET HIT ✓")
    
    print("\nIn this case, BOTH hit the profit target.")
    
    print("\n" + "="*80)
    print("TEST CASE 2: Signal Before Target")
    print("="*80)
    
    test_data2 = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01 10:00', periods=10, freq='1min'),
        'open': [100, 100.5, 101.0, 101.2, 101.5, 100.8, 99.5, 99.8, 100.2, 100.5],
        'high': [100.5, 100.8, 101.3, 101.5, 101.8, 101.0, 99.8, 100.1, 100.5, 100.8],
        'low': [99.8, 100.3, 100.8, 101.0, 101.2, 100.5, 99.2, 99.5, 99.8, 100.2],
        'close': [100.2, 100.6, 101.2, 101.3, 101.5, 100.7, 99.5, 100.0, 100.3, 100.5],
        'signal': [1, 0, -1, 0, 0, 0, 0, 1, 0, 0]  # Long, hold, SHORT (early), hold...
    })
    
    print("\nTest scenario 2:")
    print("- Entry: $100 (long)")
    print("- Stop: $99 (-1%)")
    print("- Target: $102 (+2%)")
    print("- Bar 3: Signal changes to SHORT (high only $101.3)")
    print("- Target never reached")
    
    print("\n--- Universal Approach ---")
    print("1. Enter long at bar 1 ($100)")
    print("2. See exit signal at bar 3")
    print("3. Look back at bars 2-3, target NOT hit (max $101.5)")
    print("4. Exit at signal price ($101.2)")
    print("Result: SIGNAL EXIT")
    
    print("\n--- Execution Approach ---")
    print("1. Enter long at bar 1 ($100)")
    print("2. Bar 2: Check stop/target - not hit")
    print("3. Bar 3: Check stop/target - not hit")
    print("4. Bar 3: Check signal - SHORT signal!")
    print("5. Exit at signal price ($101.2)")
    print("Result: SIGNAL EXIT")
    
    print("\nBoth get SIGNAL EXIT - no difference here either.")
    
    print("\n" + "="*80)
    print("HYPOTHESIS: The Real Difference")
    print("="*80)
    
    print("\nThe 373 trade difference might be due to:")
    print("\n1. NEUTRAL SIGNALS:")
    print("   - Universal might ignore NEUTRAL signals")
    print("   - Execution might exit on NEUTRAL (if prevent_reentry=True)")
    print("   - This creates MORE trades that can hit targets in universal")
    
    print("\n2. IMPLEMENTATION BUG:")
    print("   - One system might have a bug in target calculation")
    print("   - Rounding errors (1.02 * price vs price * 1.02)")
    print("   - Off-by-one errors in bar checking")
    
    print("\n3. DATA SYNCHRONIZATION:")
    print("   - Signal timestamps might not align with price data")
    print("   - One system might be using shifted/lagged data")
    
    print("\nRECOMMENDATION: ")
    print("1. Check how NEUTRAL signals are handled in each system")
    print("2. Verify the exact target price calculation")
    print("3. Log the first 10 trades where universal hits target but execution doesn't")
    print("4. For those trades, print every bar's OHLC to see where they diverge")

def create_diagnostic_script():
    """Create a script to diagnose the exact differences."""
    
    diagnostic_code = '''
# Diagnostic script to find exact differences

def diagnose_target_differences(universal_trades, execution_trades, price_data):
    """Find trades where universal hits target but execution doesn't."""
    
    # Find universal trades that hit targets
    universal_targets = universal_trades[universal_trades['exit_type'] == 'target']
    
    differences = []
    
    for _, u_trade in universal_targets.iterrows():
        # Find matching execution trade
        e_matches = execution_trades[
            abs(execution_trades['entry_time'] - u_trade['entry_time']) < pd.Timedelta(seconds=60)
        ]
        
        if len(e_matches) > 0:
            e_trade = e_matches.iloc[0]
            
            if e_trade['exit_type'] != 'target':
                # Found a difference!
                entry_time = u_trade['entry_time']
                exit_time = u_trade['exit_time']
                
                # Get all price bars during trade
                trade_bars = price_data[
                    (price_data['timestamp'] >= entry_time) & 
                    (price_data['timestamp'] <= exit_time)
                ]
                
                differences.append({
                    'entry_time': entry_time,
                    'entry_price': u_trade['entry_price'],
                    'universal_exit': u_trade['exit_type'],
                    'execution_exit': e_trade['exit_type'],
                    'target_price': u_trade['entry_price'] * 1.02,
                    'bars_during_trade': len(trade_bars),
                    'max_high': trade_bars['high'].max(),
                    'trade_bars': trade_bars
                })
    
    return differences

# Use this to find the specific cases where they differ
'''
    
    print("\n" + "="*80)
    print("DIAGNOSTIC CODE TO RUN:")
    print("="*80)
    print(diagnostic_code)

if __name__ == "__main__":
    analyze_reported_metrics()
    create_diagnostic_script()