import sys
sys.path.insert(0, '.')
from src.core.coordinator.config.clean_syntax_parser import CleanSyntaxParser
import json

parser = CleanSyntaxParser()

# Test simple RSI filter with multiple thresholds
test_filter = {'rsi_below': {'threshold': [40, 50, 60, 70]}}
strategies = parser._expand_strategy('test', {'period': 20, 'filter': [test_filter]})

print(f"Generated {len(strategies)} strategies:")
print(json.dumps(strategies, indent=2))