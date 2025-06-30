"""Final comprehensive strategy comparison"""
import pandas as pd
import numpy as np

# Manually compile results from our analyses
strategies = [
    {
        'Strategy': 'Bollinger RSI Simple Signals',
        'Trades/Year': 699,
        'Avg Return': 0.0078,
        'Win Rate': 65.7,
        'Annual (0bp)': 8.3,
        'Annual (0.5bp)': -2.2,
        'Annual (1bp)': -11.6,
        'Annual (2bp)': -27.8,
        'Sharpe': 1.20,
        'Best Feature': 'Higher win rate, larger edge per trade'
    },
    {
        'Strategy': 'Bollinger RSI Simple (Filtered)',
        'Trades/Year': 131,
        'Avg Return': 0.0430,
        'Win Rate': 70.6,
        'Annual (0bp)': 5.8,
        'Annual (0.5bp)': 4.4,
        'Annual (1bp)': 3.1,
        'Annual (2bp)': 0.4,
        'Sharpe': 0.5,  # Estimated
        'Best Feature': 'Volume > 1.2x + Momentum aligned filter'
    },
    {
        'Strategy': 'RSI Bands',
        'Trades/Year': 1845,
        'Avg Return': 0.0005,
        'Win Rate': 68.4,
        'Annual (0bp)': 0.9,
        'Annual (0.5bp)': -16.1,
        'Annual (1bp)': -30.3,
        'Annual (2bp)': -51.8,
        'Sharpe': 0.17,
        'Best Feature': 'High win rate but tiny edge'
    },
    {
        'Strategy': 'Bollinger RSI Confirmed',
        'Trades/Year': 1481,
        'Avg Return': 0.0047,
        'Win Rate': 57.5,
        'Annual (0bp)': 7.3,
        'Annual (0.5bp)': -6.0,  # Estimated
        'Annual (1bp)': -20.2,
        'Annual (2bp)': -40.0,  # Estimated
        'Sharpe': 0.55,
        'Best Feature': 'More trades than Simple Signals'
    },
    {
        'Strategy': 'Bollinger Bands',
        'Trades/Year': 5552,
        'Avg Return': -0.0008,
        'Win Rate': 59.8,
        'Annual (0bp)': -4.5,
        'Annual (0.5bp)': -40.0,  # Estimated
        'Annual (1bp)': -68.5,
        'Annual (2bp)': -90.0,  # Estimated
        'Sharpe': -0.17,
        'Best Feature': 'None - loses money even without costs'
    }
]

df = pd.DataFrame(strategies)

print("=== COMPREHENSIVE STRATEGY COMPARISON ===\n")

# Sort by 1bp annual return
df = df.sort_values('Annual (1bp)', ascending=False)

print(f"{'Strategy':<35} {'T/Year':<8} {'Avg%':<8} {'WR%':<6} {'0bp':<8} {'0.5bp':<8} {'1bp':<8} {'2bp':<8} {'Sharpe':<8}")
print("-" * 110)

for _, row in df.iterrows():
    print(f"{row['Strategy']:<35} {row['Trades/Year']:<8} {row['Avg Return']:>6.3f}  {row['Win Rate']:>5.1f}  "
          f"{row['Annual (0bp)']:>6.1f}%  {row['Annual (0.5bp)']:>6.1f}%  {row['Annual (1bp)']:>6.1f}%  "
          f"{row['Annual (2bp)']:>6.1f}%  {row['Sharpe']:>6.2f}")

print("\n=== KEY FINDINGS ===\n")

print("1. ONLY ONE STRATEGY IS PROFITABLE at realistic costs:")
print(f"   - Bollinger RSI Simple (Filtered): 3.1% annual at 1bp cost")
print(f"   - Requires Volume > 1.2x and momentum alignment")
print(f"   - Only 131 trades per year (0.52 per day)\n")

print("2. Edge per trade is CRITICAL:")
print("   - Need >0.02% avg return per trade to overcome 1bp costs")
print("   - Original Bollinger RSI: 0.0078% is too small")
print("   - Filtered version: 0.043% is sufficient\n")

print("3. More trades ≠ Better returns:")
print("   - RSI Bands: 1,845 trades/year but only 0.0005% edge = disaster")
print("   - Quality > Quantity for mean reversion\n")

print("4. Without filtering, NO strategy beats costs:")
print("   - Best unfiltered (Bollinger RSI Simple): -11.6% at 1bp")
print("   - Even 0.5bp costs make most strategies unprofitable\n")

print("=== REQUIREMENTS FOR PROFITABLE INTRADAY MEAN REVERSION ===\n")
print("1. Minimum edge: 0.02% per trade (after filtering)")
print("2. Execution costs: <1bp per side (institutional level)")
print("3. Filters: Volume and momentum alignment critical")
print("4. Alternative: Use 10x leverage (futures) to magnify small edges")
print("5. Or: Switch to swing trading (hold 2-5 days) to amortize costs")

# Calculate minimum trades needed
print("\n=== MINIMUM VIABLE STRATEGY ===")
for cost in [0.5, 1.0, 2.0]:
    min_edge = cost * 2 / 100  # Round trip cost in decimal
    min_edge_pct = min_edge * 100
    print(f"\nAt {cost}bp cost:")
    print(f"  Need >{min_edge_pct:.3f}% average return per trade")
    print(f"  At 70% win rate, winners must average >{min_edge_pct/0.7*2:.3f}%")