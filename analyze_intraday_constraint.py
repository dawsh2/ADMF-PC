import pandas as pd
import pytz

# Load trace
trace_path = "config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
trace = pd.read_parquet(trace_path)
trace["timestamp"] = pd.to_datetime(trace["ts"])
trace["timestamp_et"] = trace["timestamp"].dt.tz_convert(pytz.timezone("America/New_York"))
trace["hour"] = trace["timestamp_et"].dt.hour
trace["minute"] = trace["timestamp_et"].dt.minute
trace["date"] = trace["timestamp_et"].dt.date

print("Analyzing intraday constraint enforcement...")
print("=" * 60)

# Find signals at 15:55 (5 minutes before market close)
eod_mask = (trace["hour"] == 15) & (trace["minute"] == 55)
eod_signals = trace[eod_mask]

print(f"\nTotal signals at 15:55 ET: {len(eod_signals)}")
print("\nBreakdown by signal value:")
val_counts = eod_signals["val"].value_counts().sort_index()
for val, count in val_counts.items():
    print(f"  val={val}: {count} signals")

# Calculate percentage of exit signals
zero_count = len(eod_signals[eod_signals["val"] == 0])
exit_percentage = zero_count / len(eod_signals) * 100 if len(eod_signals) > 0 else 0
print(f"\nPercentage of 15:55 signals that are exits (val=0): {exit_percentage:.1f}%")

# Check if ALL days have an exit signal at 15:55
unique_dates = trace["date"].unique()
dates_with_1555_signal = eod_signals["date"].unique()

print(f"\nTotal trading days: {len(unique_dates)}")
print(f"Days with 15:55 signal: {len(dates_with_1555_signal)}")
print(f"Coverage: {len(dates_with_1555_signal) / len(unique_dates) * 100:.1f}%")

# Sample some days to see the pattern
print("\nSample end-of-day patterns (last 10 days with 15:55 signals):")
for date in eod_signals["date"].unique()[-10:]:
    day_signals = trace[trace["date"] == date]
    # Get signals from 15:50 onwards
    eod_window = day_signals[(day_signals["hour"] == 15) & (day_signals["minute"] >= 50)]
    
    print(f"\n{date}:")
    for _, sig in eod_window.iterrows():
        time_str = sig["timestamp_et"].strftime("%H:%M")
        print(f"  {time_str} -> val={sig['val']}")

# Check for any positions held overnight
print("\n\nChecking for overnight positions...")
overnight_positions = 0
for i in range(len(unique_dates) - 1):
    date1 = unique_dates[i]
    date2 = unique_dates[i + 1]
    
    # Last signal of day 1
    day1_signals = trace[trace["date"] == date1]
    if len(day1_signals) > 0:
        last_signal = day1_signals.iloc[-1]
        
        # First signal of day 2
        day2_signals = trace[trace["date"] == date2]
        if len(day2_signals) > 0:
            first_signal = day2_signals.iloc[0]
            
            # Check if position carried overnight (non-zero at end of day1 and same value at start of day2)
            if last_signal["val"] != 0:
                if first_signal["val"] == last_signal["val"]:
                    overnight_positions += 1
                    if overnight_positions <= 5:  # Show first 5 examples
                        print(f"  {date1} -> {date2}: Position {last_signal['val']} held overnight!")

print(f"\nTotal overnight positions found: {overnight_positions}")

# Final verdict
print("\n" + "=" * 60)
print("INTRADAY CONSTRAINT VERIFICATION:")
if exit_percentage > 95:
    print("✅ WORKING: Positions are systematically closed at 15:55 ET")
elif exit_percentage > 80:
    print("⚠️  PARTIALLY WORKING: Most positions closed at 15:55 ET")  
else:
    print("❌ NOT WORKING: Many positions remain open at market close")
print("=" * 60)