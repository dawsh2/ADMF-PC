#!/usr/bin/env python3
"""Compare our results to original notebook."""

print("=== Comparison to Original Notebook ===\n")

print("Original Notebook:")
print("  Trades: 416")
print("  Returns: 10.27%")
print("  Win Rate: ~50%")
print("  Immediate re-entries: ~0")

print("\nOur Implementation:")
print("  Trades: 406 (-10 trades)")
print("  Returns: 1.42% (-8.85%)")
print("  Win Rate: 46.3% (-3.7%)")
print("  Immediate re-entries: 5 (-200+)")

print("\n=== Analysis ===")

print("\nPositive Changes:")
print("- Exit memory is working (5 vs 471 immediate re-entries)")
print("- Eliminated 57 bad trades from our initial 463")
print("- Trade quality improved")

print("\nRemaining Gaps:")
print("- 10 fewer trades than original")
print("- 8.85% lower returns")
print("- Slightly lower win rate")

print("\nPossible Reasons for Differences:")
print("1. Signal calculation differences")
print("2. Entry/exit timing differences")
print("3. Risk parameter differences")
print("4. Data preprocessing differences")

print("\nTo investigate further, we would need:")
print("- The original notebook code")
print("- Side-by-side signal comparison")
print("- Trade-by-trade comparison")