#!/usr/bin/env python3
"""Check if Keltner 3.0 multiplier is too conservative"""

# Keltner Bands formula:
# Upper = SMA + (multiplier * ATR)
# Lower = SMA - (multiplier * ATR)

# With multiplier = 3.0, the bands are 3 ATRs away from the SMA
# This is VERY wide - price rarely moves 3 ATRs from its average

# Typical multipliers:
# - 1.5 to 2.0: Normal trading (generates reasonable signals)
# - 2.5: Conservative (fewer signals)
# - 3.0: Very conservative (rare signals)
# - 4.0+: Extreme moves only

print("Keltner Bands Multiplier Analysis")
print("=" * 50)
print("\nYour ensemble uses:")
print("- Keltner: period=26, multiplier=3.0 (VERY conservative)")
print("- Bollinger: period=11, std_dev=2.0 (standard)")
print("\nLikely issue: Keltner 3.0 multiplier is too wide!")
print("The bands are so far from price that signals are rare.")
print("\nRecommendation: Try multiplier=1.5 or 2.0 for Keltner")
print("\nTo verify, run: python test_individual_strategies.py")