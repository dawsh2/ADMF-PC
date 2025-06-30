# De-Nightmare-ifying Strategy Development

## The Problem We Solved

The nightmare was: **"Why is this always a nightmare? I don't want to be hardcoding stuff in state.py or elsewhere for each strategy."**

The core issues were:
1. Legacy feature system incompatible with new architecture
2. Metadata loss during strategy compilation
3. No clear migration path from old to new system
4. Feature discovery failing silently

## The Solution: Clear Patterns & Automation

### 1. Strategy Declaration Pattern

**Use the `@strategy` decorator with `feature_discovery`:**

```python
@strategy(
    name='rsi_bands',
    feature_discovery=lambda params: [
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'overbought': {'type': 'float', 'range': (60, 90), 'default': 70},
        'oversold': {'type': 'float', 'range': (10, 40), 'default': 30},
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14}
    }
)
def rsi_bands_signal(data: Dict[str, Any], params: Dict[str, Any]) -> Signal:
    # Strategy logic here
    pass
```

### 2. Feature Specification

**Always use FeatureSpec for clarity:**

```python
# Good - explicit feature requirements
feature_discovery=lambda params: [
    FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
    FeatureSpec('sma', {'period': params.get('sma_period', 20)})
]

# Bad - legacy format
feature_config={
    'rsi': {'period': 14}
}
```

### 3. Automated Migration Tool

Create a tool to automatically migrate strategies:

```python
#!/usr/bin/env python3
"""Migrate legacy strategies to new feature_discovery format."""

import ast
import re
from pathlib import Path

def migrate_strategy_file(filepath: Path):
    """Migrate a single strategy file from legacy to new format."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find @strategy decorators with feature_config
    pattern = r'@strategy\((.*?)feature_config\s*=\s*({.*?})(.*?)\)'
    
    def replace_feature_config(match):
        before = match.group(1)
        config_str = match.group(2)
        after = match.group(3)
        
        # Parse the feature config
        try:
            config = ast.literal_eval(config_str)
            
            # Convert to feature_discovery
            specs = []
            for feature, params in config.items():
                param_str = ', '.join(f"'{k}': params.get('{k}', {v})" 
                                    for k, v in params.items())
                specs.append(f"FeatureSpec('{feature}', {{{param_str}}})")
            
            discovery = f"feature_discovery=lambda params: [{', '.join(specs)}]"
            
            return f"@strategy({before}{discovery}{after})"
        except:
            return match.group(0)  # Return unchanged if parsing fails
    
    # Replace all occurrences
    new_content = re.sub(pattern, replace_feature_config, content, flags=re.DOTALL)
    
    # Add import if needed
    if 'FeatureSpec' not in new_content and 'feature_discovery' in new_content:
        new_content = "from src.strategy.types import FeatureSpec\n" + new_content
    
    # Save migrated file
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Migrated {filepath}")

# Run migration
if __name__ == "__main__":
    for file in Path("src/strategy/strategies/indicators").glob("*.py"):
        if "feature_config" in file.read_text():
            migrate_strategy_file(file)
```

### 4. Strategy Validation Tool

Create a validation tool to catch issues early:

```python
#!/usr/bin/env python3
"""Validate strategy definitions and feature discovery."""

from src.core.components.discovery import discover_components
from src.strategy.types import FeatureSpec

def validate_strategies():
    """Validate all discovered strategies."""
    strategies = discover_components('strategy')
    
    issues = []
    
    for name, info in strategies.items():
        # Check for feature_discovery
        if 'feature_discovery' not in info:
            issues.append(f"❌ {name}: Missing feature_discovery")
            continue
        
        # Test feature discovery with default params
        try:
            params = {}
            for param, config in info.get('parameter_space', {}).items():
                params[param] = config.get('default', 0)
            
            features = info['feature_discovery'](params)
            
            # Validate each feature spec
            for spec in features:
                if not isinstance(spec, FeatureSpec):
                    issues.append(f"❌ {name}: feature_discovery must return FeatureSpec objects")
                elif not hasattr(spec, 'canonical_name'):
                    issues.append(f"❌ {name}: FeatureSpec missing canonical_name")
        except Exception as e:
            issues.append(f"❌ {name}: feature_discovery failed: {e}")
    
    if issues:
        print("Strategy validation issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ All strategies validated successfully!")
        return True

if __name__ == "__main__":
    validate_strategies()
```

### 5. Development Workflow

**Step-by-step process for adding new strategies:**

1. **Copy a working example:**
   ```bash
   cp src/strategy/strategies/indicators/oscillators.py my_new_indicators.py
   ```

2. **Use the template:**
   ```python
   @strategy(
       name='my_indicator',
       feature_discovery=lambda params: [
           FeatureSpec('indicator_name', {'param': params.get('param', default)})
       ],
       parameter_space={
           'param': {'type': 'int', 'range': (5, 50), 'default': 20}
       }
   )
   def my_indicator_signal(data: Dict[str, Any], params: Dict[str, Any]) -> Signal:
       # Your logic here
       pass
   ```

3. **Test immediately:**
   ```bash
   # Create test config
   cat > config/test_my_indicator.yaml << EOF
   data:
     symbols: ["SPY"]
     start_date: "2024-03-01"
     end_date: "2024-03-31"
   
   strategies:
     my_indicator:
       params:
         param: 25
   EOF
   
   # Run signal generation
   python main.py --config config/test_my_indicator.yaml --signal-generation --bars 100
   ```

4. **Validate:**
   ```bash
   python validate_strategies.py
   ```

### 6. Common Patterns Reference

**Pattern 1: Single Indicator**
```python
feature_discovery=lambda params: [
    FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
]
```

**Pattern 2: Multiple Indicators**
```python
feature_discovery=lambda params: [
    FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
    FeatureSpec('sma', {'period': params.get('sma_period', 20)})
]
```

**Pattern 3: Conditional Features**
```python
feature_discovery=lambda params: [
    FeatureSpec('macd', {
        'fast_period': params.get('fast_period', 12),
        'slow_period': params.get('slow_period', 26),
        'signal_period': params.get('signal_period', 9)
    })
] + ([FeatureSpec('volume_sma', {'period': 20})] if params.get('use_volume', False) else [])
```

### 7. Debugging Checklist

When something goes wrong:

1. **Check feature discovery:**
   ```python
   # In your strategy file, add debug print
   print(f"Features requested: {feature_discovery(params)}")
   ```

2. **Check metadata preservation:**
   ```python
   # In state.py, add debug print
   print(f"Strategy metadata: {strategy_func._component_info}")
   ```

3. **Check signal generation:**
   ```bash
   # Use DEBUG log level
   python main.py --config your_config.yaml --signal-generation --bars 10 --log-level DEBUG
   ```

4. **Check feature availability:**
   ```python
   # List all available features
   from src.strategy.components.features.hub import FeatureHub
   print(FeatureHub.available_features())
   ```

## The Result

With these patterns and tools:
- ✅ No more hardcoding in state.py
- ✅ Clear migration path from legacy format
- ✅ Automated validation catches issues early
- ✅ Consistent patterns make development predictable
- ✅ 📡 emoji appears when signals are published!

The "nightmare" is replaced with a clear, repeatable process.