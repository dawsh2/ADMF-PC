#!/usr/bin/env python3
"""
Identify exact parameters for strategy 1029 based on the config grid
"""

# Base parameters
periods = [10, 15, 20, 30, 50]  # 5 options
multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  # 5 options

# Count filter combinations from config
# This is complex due to nested parameter sweeps
print("Strategy 1029 parameter identification")
print("=" * 50)

# With 2750 total strategies and base of 5x5 = 25 combinations
# That means ~110 filter variations per base combination

strategies_per_base = 2750 // 25  # = 110
print(f"Total strategies: 2750")
print(f"Base combinations (period × multiplier): {len(periods)} × {len(multipliers)} = 25")
print(f"Filter variations per base: ~{strategies_per_base}")

# Strategy 1029 breakdown
strategy_id = 1029
base_combo_idx = strategy_id // strategies_per_base  # Which of the 25 base combos
filter_idx = strategy_id % strategies_per_base  # Which filter variant

# Decode base parameters
period_idx = base_combo_idx // len(multipliers)
multiplier_idx = base_combo_idx % len(multipliers)

period = periods[period_idx]
multiplier = multipliers[multiplier_idx]

print(f"\nStrategy 1029 likely parameters:")
print(f"- Base combination index: {base_combo_idx} (of 25)")
print(f"- Period: {period}")
print(f"- Multiplier: {multiplier}")
print(f"- Filter variant: {filter_idx} (of ~{strategies_per_base})")

print("\nNote: The exact filter combination depends on how parameter sweeps were expanded.")
print("Filter variant 29 would be somewhere in the middle of the filter list,")
print("possibly one of the combined filters or regime-specific filters.")

# Create estimated config
print("\nEstimated configuration for strategy 1029:")
print(f"""
strategy:
  - type: keltner_bands
    period: {period}
    multiplier: {multiplier}
    atr_period: 20  # Default if not specified
    filter: <filter variant {filter_idx}>
    
# This corresponds to strategy_{strategy_id} in the compiled run
""")