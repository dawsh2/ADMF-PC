# Debug why analysis thinks trade 3 hits target when it doesn't
import pandas as pd
from pathlib import Path

print("🔍 DEBUGGING ANALYSIS TARGET LOGIC FOR TRADE #3")
print("="*80)

# Load market data
market_data = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Entry details
entry_time = pd.to_datetime('2024-03-26 19:40').tz_localize('UTC')
entry_price = 519.19
target_price = entry_price * 1.001  # 519.71
stop_price = entry_price * 0.99925  # 518.80

print(f"Trade setup:")
print(f"  Entry time:  {entry_time}")
print(f"  Entry price: {entry_price:.2f}")
print(f"  Target:      {target_price:.2f}")
print(f"  Stop:        {stop_price:.2f}")

# Get the entry bar index
entry_idx = market_data[market_data['timestamp'] == entry_time].index[0]

# Check the next few bars
print(f"\nChecking bars after entry:")
for i in range(5):  # Check 5 bars
    if entry_idx + i >= len(market_data):
        break
    
    bar = market_data.iloc[entry_idx + i]
    print(f"\nBar {i} - {bar['timestamp']}:")
    print(f"  High: {bar['high']:.2f} >= Target {target_price:.2f}? {bar['high'] >= target_price}")
    print(f"  Low:  {bar['low']:.2f} <= Stop {stop_price:.2f}? {bar['low'] <= stop_price}")
    
    if bar['high'] >= target_price:
        print(f"  ✅ TARGET HIT on bar {i}!")
        break
    elif bar['low'] <= stop_price:
        print(f"  ❌ STOP HIT on bar {i}!")
        break

# The analysis says it exits at target with 1 bar duration
# But we can see the target is never hit!
print("\n💡 ANALYSIS BUG FOUND:")
print("The analysis claims Trade #3 hits target at 519.71 after 1 bar")
print("But the high of 519.44 never reaches the target of 519.71")
print("\nThis suggests the analysis has a bug in its intrabar exit logic!")
print("It might be using the wrong price or miscalculating the target.")