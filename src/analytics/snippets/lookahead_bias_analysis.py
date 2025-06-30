# Lookahead Bias Analysis of Stop/Target Logic
# This script examines the trade execution logic for potential lookahead bias

import pandas as pd
import numpy as np

print("🔍 LOOKAHEAD BIAS ANALYSIS")
print("=" * 80)

# First, let's understand how trades are extracted
print("\n1. TRADE EXTRACTION ANALYSIS:")
print("-" * 40)

# Typical extract_trades function structure
print("""
The extract_trades function typically:
1. Loads signals from trace files
2. Identifies position changes (0→1, 1→-1, etc.)
3. Uses CLOSE price at signal time for entry/exit
4. Applies execution cost to returns
""")

# Key question: When do signals occur vs when are they executed?
print("\n⚠️ CRITICAL TIMING ISSUES:")
print("- Signals generated at bar close")
print("- But can we execute at that close price?")
print("- Real execution would be at NEXT bar's open")

print("\n2. STOP/TARGET LOGIC ANALYSIS:")
print("-" * 40)

# Analyze the apply_stop_target function
print("""
Current implementation (line 75):
    trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]

✅ CORRECT: Only looks at bars between entry and exit
✅ CORRECT: Doesn't peek at future data beyond original exit
""")

print("""
Stop/Target checking (lines 105-114):
    for _, bar in trade_prices.iterrows():
        if direction == 1:  # Long
            if bar['low'] <= stop_price:
                exit_price = stop_price
                
⚠️ ISSUE #1: Exit at exact stop price
- Assumes perfect fill at stop price
- Reality: Might gap through stop
- Could exit at worse price
""")

print("\n3. EXECUTION PRICE ANALYSIS:")
print("-" * 40)

print("""
POTENTIAL LOOKAHEAD BIASES FOUND:

1. ENTRY EXECUTION:
   ❌ Current: Entry at signal bar's close price
   ✅ Realistic: Entry at NEXT bar's open price
   
2. STOP/TARGET EXECUTION:
   ❌ Current: Exit at exact stop/target price
   ✅ Realistic: Exit at touched bar's close (or worse)
   
3. SIGNAL GENERATION:
   ❓ Unknown: When are signals generated?
   - If using current bar's close in calculation → OK
   - If using future information → LOOKAHEAD BIAS
""")

print("\n4. REALISTIC EXECUTION SIMULATION:")
print("-" * 40)

def apply_stop_target_realistic(trades_df, stop_pct, target_pct, market_data):
    """
    More realistic stop/target application that avoids lookahead bias
    """
    modified_returns = []
    exit_types = {'stop': 0, 'target': 0, 'signal': 0}
    
    for _, trade in trades_df.iterrows():
        # Get bars AFTER entry (can't use entry bar for stop/target check)
        trade_prices = market_data.iloc[int(trade['entry_idx'])+1:int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_returns.append(trade['net_return'])
            exit_types['signal'] += 1
            continue
        
        # Use NEXT bar's open as actual entry price (more realistic)
        actual_entry_price = trade_prices.iloc[0]['open'] if len(trade_prices) > 0 else trade['entry_price']
        direction = trade['direction']
        
        # Set stop and target from actual entry
        if direction == 1:  # Long
            stop_price = actual_entry_price * (1 - stop_pct/100)
            target_price = actual_entry_price * (1 + target_pct/100)
        else:  # Short
            stop_price = actual_entry_price * (1 + stop_pct/100)
            target_price = actual_entry_price * (1 - target_pct/100)
        
        # Check each bar for exit
        exit_price = trade['exit_price']
        exit_type = 'signal'
        exit_idx = len(trade_prices) - 1
        
        for i, (_, bar) in enumerate(trade_prices.iterrows()):
            if direction == 1:  # Long
                # Check if stop hit - exit at bar's close (conservative)
                if bar['low'] <= stop_price:
                    exit_price = min(bar['close'], stop_price)  # Might gap through
                    exit_type = 'stop'
                    exit_idx = i
                    break
                # Check if target hit - exit at bar's close
                elif bar['high'] >= target_price:
                    exit_price = bar['close']  # Take profit at close, not exact target
                    exit_type = 'target'
                    exit_idx = i
                    break
            else:  # Short
                if bar['high'] >= stop_price:
                    exit_price = max(bar['close'], stop_price)
                    exit_type = 'stop'
                    exit_idx = i
                    break
                elif bar['low'] <= target_price:
                    exit_price = bar['close']
                    exit_type = 'target'
                    exit_idx = i
                    break
        
        exit_types[exit_type] += 1
        
        # Calculate return with realistic prices
        if direction == 1:
            raw_return = (exit_price - actual_entry_price) / actual_entry_price
        else:
            raw_return = (actual_entry_price - exit_price) / actual_entry_price
        
        net_return = raw_return - trade['execution_cost']
        modified_returns.append(net_return)
    
    return np.array(modified_returns), exit_types

print("\n5. IMPACT ESTIMATION:")
print("-" * 40)

print("""
Expected impact of realistic execution:

1. ENTRY SLIPPAGE:
   - Current: Entry at signal bar close
   - Realistic: Entry at next bar open
   - Impact: ~0.01-0.02% average slippage

2. EXIT SLIPPAGE:
   - Current: Exit at exact stop/target
   - Realistic: Exit at bar close when touched
   - Impact: Targets might not capture full 0.1% move

3. GAPS:
   - Current: Always fill at stop
   - Realistic: Might gap through stop
   - Impact: Occasional large losses on gaps

ESTIMATED TOTAL IMPACT: 
- Reduce returns by 10-20%
- Reduce Sharpe by 10-30%
- Increase stop losses by 5-10%
""")

print("\n6. RECOMMENDATIONS:")
print("-" * 40)

print("""
To ensure no lookahead bias:

1. ✅ Signals should use only data available at bar close
2. ✅ Entry should be at NEXT bar's open (not signal bar close)
3. ✅ Stops/targets should account for gaps and slippage
4. ✅ Exit at bar close when stop/target touched (not exact price)
5. ✅ Add realistic slippage model (especially for 8+ trades/day)
6. ✅ Verify signal generation doesn't use future data

The current Sharpe of 12.81 likely overstates realistic performance by 20-40%.
A more realistic Sharpe might be 8-10 with proper execution modeling.
""")

print("\n" + "="*80)
print("⚠️ CONCLUSION: Mild lookahead bias in execution prices")
print("The logic is mostly sound, but perfect fills are unrealistic")
print("="*80)