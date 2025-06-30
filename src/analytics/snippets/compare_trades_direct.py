# Direct comparison using the trade data from notebooks
import pandas as pd
import numpy as np

print("="*120)
print("DETAILED TRADE-BY-TRADE COMPARISON")
print("="*120)

# Analysis trades from your output
analysis_data = [
    {'num': 1, 'entry_time': '2024-03-26 15:45', 'dir': 'LONG', 'entry': 521.11, 'stop': 520.72, 'target': 521.63, 'exit_type': 'signal', 'exit': 521.40, 'return': 0.00045, 'bars': 3},
    {'num': 2, 'entry_time': '2024-03-26 19:25', 'dir': 'LONG', 'entry': 520.21, 'stop': 519.82, 'target': 520.73, 'exit_type': 'stop', 'exit': 519.82, 'return': -0.00085, 'bars': 2},
    {'num': 3, 'entry_time': '2024-03-26 19:40', 'dir': 'LONG', 'entry': 519.19, 'stop': 518.80, 'target': 519.71, 'exit_type': 'target', 'exit': 519.71, 'return': 0.00090, 'bars': 1},
    {'num': 4, 'entry_time': '2024-03-27 13:30', 'dir': 'SHORT', 'entry': 521.45, 'stop': 521.84, 'target': 520.93, 'exit_type': 'signal', 'exit': 521.49, 'return': -0.00018, 'bars': 2},
    {'num': 5, 'entry_time': '2024-03-27 16:30', 'dir': 'SHORT', 'entry': 520.29, 'stop': 520.68, 'target': 519.77, 'exit_type': 'signal', 'exit': 520.20, 'return': 0.00008, 'bars': 2},
]

# Execution trades from your output
execution_data = [
    {'num': 1, 'entry_time': '2024-03-26 15:45', 'dir': 'LONG', 'entry': 521.11, 'stop': 520.72, 'target': 521.63, 'exit_type': 'signal', 'exit': 521.40, 'return': 0.00046, 'bars': 1},
    {'num': 2, 'entry_time': '2024-03-26 15:50', 'dir': 'SHORT', 'entry': 521.40, 'stop': 521.79, 'target': 520.88, 'exit_type': 'signal', 'exit': 521.40, 'return': -0.00009, 'bars': 1},
    {'num': 3, 'entry_time': '2024-03-26 19:25', 'dir': 'LONG', 'entry': 520.21, 'stop': 519.82, 'target': 520.73, 'exit_type': 'stop', 'exit': 519.82, 'return': -0.00085, 'bars': 1},
    {'num': 4, 'entry_time': '2024-03-26 19:40', 'dir': 'LONG', 'entry': 519.19, 'stop': 518.80, 'target': 519.71, 'exit_type': 'signal', 'exit': 519.11, 'return': -0.00025, 'bars': 1},
    {'num': 5, 'entry_time': '2024-03-27 13:30', 'dir': 'SHORT', 'entry': 521.45, 'stop': 521.84, 'target': 520.93, 'exit_type': 'signal', 'exit': 521.49, 'return': -0.00018, 'bars': 1},
]

analysis_df = pd.DataFrame(analysis_data)
execution_df = pd.DataFrame(execution_data)

print("\n🔍 KEY FINDINGS:\n")

print("1. EXTRA TRADE IN EXECUTION:")
print("   - Execution has trade #2 at 15:50 (SHORT) that doesn't exist in analysis")
print("   - This shifts all subsequent trade numbers\n")

print("2. DIFFERENT EXIT TYPES:")
print("   Analysis trade #3 (19:40 LONG):")
print("   - Analysis: exits at TARGET (519.71) with +0.090% return")
print("   - Execution: exits at SIGNAL (519.11) with -0.025% return")
print("   - This is a MAJOR difference - profitable vs losing trade!\n")

print("3. TRADE DURATION DIFFERENCES:")
print("   - Analysis trades last 1-3 bars (realistic)")
print("   - Execution trades mostly last 1 bar (too quick)")
print("   - Suggests positions are closing too early in execution\n")

print("4. PERFORMANCE IMPACT:")
analysis_return = 0.0475  # 4.75% from 100 trades
execution_return = 0.0036  # 0.36% from 100 trades
print(f"   - Analysis: 77% win rate, 4.75% return")
print(f"   - Execution: 52% win rate, 0.36% return")
print(f"   - Difference: {(analysis_return - execution_return)*100:.2f}% ({analysis_return/execution_return:.1f}x better)\n")

print("🐛 ROOT CAUSES TO INVESTIGATE:\n")
print("1. Why is execution creating an extra SHORT trade at 15:50?")
print("   - This could be a signal flipping issue")
print("   - Or a position management bug\n")

print("2. Why are profitable target exits becoming losing signal exits?")
print("   - Trade #3 should hit target at 519.71 but exits at 519.11")
print("   - The intrabar exit logic might not be working correctly\n")

print("3. Why are trades closing after just 1 bar?")
print("   - Positions might be closing on every new signal")
print("   - Exit memory might not be working properly")

print("\n📊 HYPOTHESIS:")
print("The execution engine is:")
print("1. Generating extra trades (signal flipping?)")
print("2. Not properly detecting intrabar stop/target hits")
print("3. Closing positions too quickly (1 bar vs multi-bar holds)")
print("\nThis combination turns a profitable strategy (4.75%) into a losing one (0.36%)")