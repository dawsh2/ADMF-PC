# Pandas/Numpy Dependency Analysis

## Why The Configs Can't Run Without Pandas

The indicator strategy configs themselves are **correct** and use the right parameter names. The issue is that the main.py script and the ADMF-PC system require pandas/numpy for several core components:

### 1. Import Chain Issues

When running `python main.py --config config/indicators/...`, the following import chain occurs:

```
main.py
└── src/core/coordinator/coordinator.py
    └── src/core/__init__.py
        └── src/core/components/__init__.py
            └── src/core/components/protocols.py
                └── src/core/events/__init__.py
                    └── src/core/events/tracing/__init__.py
                        └── src/core/events/storage/hierarchical.py
                            └── import pandas as pd  # REQUIRED HERE
```

### 2. Core Dependencies on Pandas/Numpy

Several core modules actually use pandas/numpy:

1. **src/data/loaders.py** - Uses pandas for CSV data loading:
   ```python
   class SimpleCSVLoader:
       def load_bars(self, file_path: str) -> pd.DataFrame:
   ```

2. **src/core/events/storage/hierarchical.py** - Uses pandas for event storage:
   ```python
   def to_dataframe(self) -> pd.DataFrame:
       """Convert events to pandas DataFrame for analysis."""
   ```

3. **src/strategy/protocols.py** - Imports pandas but only uses it for type hints:
   ```python
   def extract_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
   ```

### 3. Why Strategies Still Work

The individual strategy functions themselves **don't need pandas**. They work with dictionaries:

```python
def sma_crossover(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # No pandas usage - just dictionaries
    fast_sma = features.get(f'sma_{fast_period}')
    slow_sma = features.get(f'sma_{slow_period}')
```

### 4. The Real Problem

The configs are correct, but the system architecture requires pandas for:
- Data loading (CSV → DataFrame)
- Event storage and analysis
- Feature calculation in some components
- Type hints in protocols

## Solutions

### Option 1: Install Dependencies (Recommended)
```bash
pip install pandas numpy
python main.py --config config/indicators/crossover/test_sma_crossover.yaml --signal-generation --bars 100
```

### Option 2: Create Pandas-Free Runner
Create a minimal runner that:
1. Loads data without pandas (using dictionaries)
2. Skips event storage
3. Runs strategies directly
4. Outputs results without DataFrame conversion

### Option 3: Mock Dependencies
As shown in the tests, we can mock pandas/numpy:
```python
sys.modules['pandas'] = MockPandas()
sys.modules['numpy'] = MockNumpy()
```

But this would require extensive mocking throughout the system.

## Summary

- **Configs are correct** ✓
- **Strategies work without pandas** ✓
- **System architecture requires pandas** ✗

The indicator configs will work perfectly once pandas/numpy are installed. The configs themselves use the correct parameter names and strategy references.