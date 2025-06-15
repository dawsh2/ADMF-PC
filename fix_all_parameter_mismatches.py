#!/usr/bin/env python3
"""
Fix all parameter naming mismatches between strategies and features
"""

# Map of feature parameter corrections needed
# Format: (file, old_param_name, new_param_name)
fixes_needed = [
    # fibonacci_retracement expects 'period' not 'fib_period'
    ('src/strategy/strategies/indicators/structure.py', "'fib_period'", "'period'"),
    
    # swing_points expects 'period' not 'swing_period'  
    ('src/strategy/strategies/indicators/structure.py', "'swing_period'", "'period'"),
    
    # support_resistance expects 'period' not 'sr_period'
    ('src/strategy/strategies/indicators/structure.py', "'sr_period'", "'period'"),
    
    # aroon expects 'period' not 'aroon_period'
    ('src/strategy/strategies/indicators/trend.py', "'aroon_period'", "'period'"),
    
    # ultimate_oscillator expects period1/2/3 not uo_period1/2/3
    ('src/strategy/strategies/indicators/oscillators.py', "'uo_period1'", "'period1'"),
    ('src/strategy/strategies/indicators/oscillators.py', "'uo_period2'", "'period2'"), 
    ('src/strategy/strategies/indicators/oscillators.py', "'uo_period3'", "'period3'"),
    
    # Check other potential issues
    ('src/strategy/strategies/indicators/trend.py', "'psar_af'", "'af'"),
    ('src/strategy/strategies/indicators/trend.py', "'psar_max_af'", "'max_af'"),
    ('src/strategy/strategies/indicators/trend.py', "'lr_period'", "'period'"),
    ('src/strategy/strategies/indicators/trend.py', "'supertrend_period'", "'period'"),
    ('src/strategy/strategies/indicators/trend.py', "'supertrend_multiplier'", "'multiplier'"),
]

print("Parameter fixes to apply:")
for file, old, new in fixes_needed:
    print(f"  {file}: {old} -> {new}")

print("\nThese fixes will ensure strategies pass the correct parameter names to features.")