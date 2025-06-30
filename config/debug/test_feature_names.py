#!/usr/bin/env python3
"""
Test feature naming conventions to debug RSI filter.
"""

print("🔍 Feature Naming Investigation:\n")

print("Current RSI filter template: 'rsi({period}) < {threshold}'")
print("This generates: 'rsi(14) < 50'\n")

print("Possible issues:")
print("1. Feature system expects 'rsi_14' not 'rsi(14)'")
print("2. Filter evaluator might not parse function-style notation")
print("3. Feature might not be auto-created from filter expression\n")

# Check how features are named in the codebase
from pathlib import Path
import re

# Search for RSI feature patterns
feature_files = [
    "src/strategy/components/features/indicators/oscillators.py",
    "src/strategy/strategies/indicators/oscillators.py",
]

print("📊 Checking RSI implementations:\n")

for file_path in feature_files:
    if Path(file_path).exists():
        with open(file_path) as f:
            content = f.read()
            
        # Look for RSI class or function definitions
        rsi_patterns = [
            r'class.*RSI.*:',
            r'def.*rsi.*\(',
            r'feature_name.*=.*["\']rsi',
            r'name.*=.*["\']rsi',
        ]
        
        print(f"In {file_path}:")
        for pattern in rsi_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches[:3]:  # Show first 3 matches
                    print(f"  Found: {match}")
        
        # Look for how RSI is registered
        if 'register_component' in content:
            # Find the registration
            reg_pattern = r'@register_component.*\n.*class\s+(\w+)'
            matches = re.findall(reg_pattern, content, re.MULTILINE)
            for match in matches:
                if 'rsi' in match.lower():
                    print(f"  Registered as: {match}")

print("\n💡 Solution Options:")
print("1. Change filter template to use 'rsi_14' instead of 'rsi(14)'")
print("2. Add explicit feature mapping in config")
print("3. Fix feature name resolution in the framework")