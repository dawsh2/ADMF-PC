#\!/usr/bin/env python3
"""Check what strategies are registered"""

from src.core.components.discovery import get_component_registry

registry = get_component_registry()

print("Registered strategies:")
print("="*60)

# Get all registered components
for name, info in registry._components.items():
    if info.component_type == 'strategy':
        print(f"  - {name}")
        if 'keltner' in name.lower():
            print(f"    ^ Found Keltner strategy: {name}")

# Check specific names
test_names = ['keltner_bands', 'keltner_band', 'keltner', 'KeltnerBands']
print(f"\nChecking specific names:")
for name in test_names:
    info = registry.get_component(name)
    if info:
        print(f"  ✓ {name} is registered")
    else:
        print(f"  ✗ {name} is NOT registered")