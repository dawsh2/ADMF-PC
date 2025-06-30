# Debug exactly when entries and exits happen in the analysis
import pandas as pd
from pathlib import Path

print("🔍 DEBUGGING ENTRY/EXIT TIMING IN ANALYSIS")
print("="*80)

# Load market data
market_data = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Load signals
run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250627_185448')
strategy_index = pd.read_parquet(run_dir / 'strategy_index.parquet')
strategy_info = strategy_index[strategy_index['strategy_hash'] == '5edc43651004'].iloc[0]
signals = pd.read_parquet(run_dir / strategy_info['trace_path'])
signals['ts'] = pd.to_datetime(signals['ts'])

# Focus on the 19:40 trade
entry_time = pd.to_datetime('2024-03-26 19:40').tz_localize('UTC')

# Get the signal and market data
signal_at_1940 = signals[signals['ts'] == entry_time].iloc[0]
bar_at_1940 = market_data[market_data['timestamp'] == entry_time].iloc[0]

print(f"At 19:40:")
print(f"  Signal value: {signal_at_1940['val']}")
print(f"  Signal price: {signal_at_1940['px']:.2f}")
print(f"  Bar OHLC: O={bar_at_1940['open']:.2f}, H={bar_at_1940['high']:.2f}, L={bar_at_1940['low']:.2f}, C={bar_at_1940['close']:.2f}")
print(f"  Volume: {bar_at_1940['volume']}")

print("\n💡 KEY QUESTION:")
print("When does the signal get generated?")
print("- If signal is generated DURING the bar: Entry could be anywhere from open to close")
print("- If signal is generated AT BAR CLOSE: Entry must be at close price or next bar's open")

print("\n🔍 CHECKING SIGNAL TIMING:")
# Check if signal price matches any OHLC price
if abs(signal_at_1940['px'] - bar_at_1940['open']) < 0.01:
    print(f"  Signal price {signal_at_1940['px']:.2f} = OPEN price → Signal at bar open")
elif abs(signal_at_1940['px'] - bar_at_1940['high']) < 0.01:
    print(f"  Signal price {signal_at_1940['px']:.2f} = HIGH price → Signal at bar high")
elif abs(signal_at_1940['px'] - bar_at_1940['low']) < 0.01:
    print(f"  Signal price {signal_at_1940['px']:.2f} = LOW price → Signal at bar low")
elif abs(signal_at_1940['px'] - bar_at_1940['close']) < 0.01:
    print(f"  Signal price {signal_at_1940['px']:.2f} = CLOSE price → Signal at bar close ✅")
else:
    print(f"  Signal price {signal_at_1940['px']:.2f} doesn't match OHLC → Unknown timing")

print("\n📊 CORRECT BEHAVIOR:")
print("1. Signal generated at 19:40 bar CLOSE (519.19)")
print("2. Entry happens at 19:40 CLOSE or 19:45 OPEN")
print("3. Exit checks should start from 19:45 bar, NOT 19:40 bar")
print("4. Cannot use 19:40 high (519.97) for exit - that's lookahead!")

# Check what the analysis code actually does
print("\n🐛 ANALYSIS CODE BEHAVIOR:")
print("The analysis checks bars from entry_idx to current idx:")
print("  trade_bars = df.iloc[current_trade['entry_idx']:idx+1]")
print("This INCLUDES the entry bar in exit checks - LOOKAHEAD BIAS!")
print("\nIt should be:")
print("  trade_bars = df.iloc[current_trade['entry_idx']+1:idx+1]")
print("To only check bars AFTER entry.")