#!/usr/bin/env python3
"""Debug range parsing"""

from src.core.coordinator.config.clean_syntax_parser import CleanSyntaxParser

parser = CleanSyntaxParser()

# Test the multiplier range specifically
multiplier_range = parser._parse_range('range(0.5, 4.0, 0.2)')

print(f"Multiplier values: {multiplier_range}")
print(f"Count: {len(multiplier_range)}")
print(f"Last value: {multiplier_range[-1]}")

# Check what's happening
current = 0.5
step = 0.2
stop = 4.0
values = []
while current <= stop + step/2:
    print(f"Current: {current}, Stop + step/2: {stop + step/2}, Compare: {current <= stop + step/2}")
    values.append(round(current, 6))
    current += step
    if len(values) > 20:  # Safety break
        break
        
print(f"\nManual calculation gives: {len(values)} values")
print(f"Last few: {values[-3:]}")