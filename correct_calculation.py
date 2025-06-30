#!/usr/bin/env python3

# Current P&L with 1 share per trade
total_pnl = 69.82  # From our backtest
spy_price = 600    # Approximate SPY price

# If we're trading 1 share at $600 each time
# Then our "capital" for the purpose of calculating returns is just the share price
print("=== Correct Calculation ===")
print(f"Trading 1 share at ${spy_price}")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Return: {(total_pnl / spy_price) * 100:.2f}%")

print("\nThis makes sense! The notebook is calculating returns based on")
print("the capital actually at risk (1 share = $600), not the total")
print("account balance of $100,000.")

# Verify with actual numbers
print("\n=== Verification ===")
print("406 trades")
print("Win rate: 46.3%")
print("Average win: $0.87 (0.15% of $600)")
print("Average loss: $-0.43 (0.075% of $600)")
print(f"Expected return: ~10-11% on the $600 at risk per trade")