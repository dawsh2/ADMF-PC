"""Temporarily fix the monitoring import issue for testing."""

import os

# Read the file
file_path = "src/core/infrastructure/capabilities.py"
with open(file_path, 'r') as f:
    content = f.read()

# Comment out the monitoring import
new_content = content.replace(
    "from .monitoring import MetricsCollector, PerformanceTracker, ComponentHealthCheck",
    "# from .monitoring import MetricsCollector, PerformanceTracker, ComponentHealthCheck\n"
    "# Temporarily disabled for testing without numpy\n"
    "MetricsCollector = None\n"
    "PerformanceTracker = None\n"
    "ComponentHealthCheck = None"
)

# Write back
with open(file_path, 'w') as f:
    f.write(new_content)

print(f"Fixed {file_path}")
print("You can now run: python3 test_direct_imports.py")
print("\nTo restore, run: git checkout src/core/infrastructure/capabilities.py")