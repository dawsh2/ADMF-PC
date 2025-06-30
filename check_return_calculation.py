#!/usr/bin/env python3
"""Check how returns should be calculated for short positions."""

print("=== Return Calculation for Short Positions ===")

# Example: SHORT position
entry_price = 100.00
stop_loss_price = 100.075  # 0.075% higher (stop loss for short)

# Simple calculation (WRONG for shorts)
simple_return = (stop_loss_price - entry_price) / entry_price
print(f"\nSimple return calculation:")
print(f"  (Exit - Entry) / Entry = ({stop_loss_price} - {entry_price}) / {entry_price}")
print(f"  = {simple_return:.5f} = {simple_return * 100:.3f}%")
print("  ❌ This shows +0.075% but it's actually a LOSS!")

# Correct calculation for SHORT positions
# For shorts: profit when price goes DOWN, loss when price goes UP
correct_return = (entry_price - stop_loss_price) / entry_price
print(f"\nCorrect return for SHORT position:")
print(f"  (Entry - Exit) / Entry = ({entry_price} - {stop_loss_price}) / {entry_price}")
print(f"  = {correct_return:.5f} = {correct_return * 100:.3f}%")
print("  ✓ This correctly shows -0.075% loss")

# General formula
print("\n\n=== Correct Return Formula ===")
print("For LONG positions:")
print("  Return = (Exit Price - Entry Price) / Entry Price")
print("\nFor SHORT positions:")
print("  Return = (Entry Price - Exit Price) / Entry Price")
print("  OR")
print("  Return = -1 * (Exit Price - Entry Price) / Entry Price")

print("\n\n=== The Fix ===")
print("When calculating returns, we need to check position direction:")
print("if quantity > 0:  # Long position")
print("    return_pct = (exit_price - entry_price) / entry_price * 100")
print("else:  # Short position")
print("    return_pct = (entry_price - exit_price) / entry_price * 100")