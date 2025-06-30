#!/usr/bin/env python3
"""Check what's in the component registry"""

from src.core.components.discovery import get_component_registry

registry = get_component_registry()

print("=== REGISTERED STRATEGIES ===")
strategies = []
for name, info in registry._components.items():
    if info.component_type == 'strategy':
        strategies.append(name)

strategies.sort()
for s in strategies:
    print(f"  {s}")

print(f"\nTotal strategies: {len(strategies)}")

# Check if bollinger_bands is there
if 'bollinger_bands' in strategies:
    print("\n✅ bollinger_bands is registered")
else:
    print("\n❌ bollinger_bands is NOT registered")
    # Check similar names
    bollinger_like = [s for s in strategies if 'bollinger' in s]
    if bollinger_like:
        print(f"Found similar: {bollinger_like}")