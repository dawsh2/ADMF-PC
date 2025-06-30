import pandas as pd

# Load positions
positions_df = pd.read_parquet("config/bollinger/results/20250627_171211/traces/portfolio/positions_close/positions_close.parquet")

# Load signals
signals_df = pd.read_parquet("config/bollinger/results/20250627_171211/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")

# Get positions where price went down
positions_df["price_change"] = positions_df["exit_price"] - positions_df["entry_price"]
down_moves = positions_df[positions_df["price_change"] < 0].head(10)

print("Positions where price went DOWN (should be profitable for SHORTS):")
print("="*60)
for idx, pos in down_moves.iterrows():
    entry_price = pos["entry_price"]
    exit_price = pos["exit_price"]
    price_change = pos["price_change"]
    current_return = (exit_price - entry_price) / entry_price * 100
    short_return = (entry_price - exit_price) / entry_price * 100
    
    print(f"\nPosition {idx}:")
    print(f"  Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
    print(f"  Price moved: ${price_change:.2f} ({price_change/entry_price*100:.3f}%)")
    print(f"  Current return (as LONG): {current_return:.3f}%")
    print(f"  Should be (if SHORT): {short_return:.3f}%")
    print(f"  Exit type: {pos.get('exit_type', 'unknown')}")

# Summary stats
neg_moves = (positions_df["price_change"] < 0).sum()
pos_moves = (positions_df["price_change"] > 0).sum()
total = len(positions_df)

print(f"\n\nPrice Movement Summary:")
print(f"Positions with negative price moves: {neg_moves}/{total} ({neg_moves/total*100:.1f}%)")
print(f"Positions with positive price moves: {pos_moves}/{total} ({pos_moves/total*100:.1f}%)")

print(f"\nSignal distribution:")
print(f"Long signals: {(signals_df["val"] > 0).sum()}")
print(f"Short signals: {(signals_df["val"] < 0).sum()}")
print(f"Flat signals: {(signals_df["val"] == 0).sum()}")

# Calculate what win rate SHOULD be if we handled shorts properly
# Assuming roughly equal distribution of longs/shorts
# Longs win when price goes up, shorts win when price goes down
estimated_long_wins = pos_moves / 2  # Half of positive moves
estimated_short_wins = neg_moves / 2  # Half of negative moves
estimated_total_wins = estimated_long_wins + estimated_short_wins
estimated_win_rate = estimated_total_wins / total * 100

print(f"\nEstimated CORRECT win rate (if shorts handled properly): ~{estimated_win_rate:.1f}%")
print(f"Current win rate (all treated as longs): 50.2%")
