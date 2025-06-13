# Strategy Module Review Summary

## Current State

The `src/strategy/strategies/` directory is in a transitional state with 17 strategy files showing:

### ✅ What's Working Well
1. **New @strategy decorator pattern** is clean and enables automatic discovery
2. **Binary signals (-1, 0, 1)** implemented in newer strategies
3. **Good separation** between different strategy types (crossovers, oscillators, channels)
4. **Feature configuration** in decorator metadata

### ⚠️ Issues Identified

1. **Duplicate Implementations**
   - `momentum.py` and `simple_momentum.py` have overlapping functionality
   - `mean_reversion.py` is just a wrapper around `mean_reversion_simple.py`
   - `ma_crossover.py` duplicates functionality in `crossovers.py`

2. **Mixed Paradigms**
   - Old: Class-based strategies (ArbitrageStrategy, MarketMakingStrategy)
   - New: Function-based with @strategy decorator
   - Some files contain both patterns

3. **Inconsistent Naming**
   - Some files use `_strategy` suffix, others don't
   - `simple_` prefix used inconsistently
   - No clear convention

4. **Complex __init__.py**
   - Lazy loading for backward compatibility
   - Mix of old and new imports
   - Some imports commented out due to issues

## Recommendations

### Immediate Actions
1. **Consolidate duplicates**: Merge momentum variants into single file
2. **Remove wrappers**: Delete `mean_reversion.py`, rename `mean_reversion_simple.py`
3. **Standardize naming**: Remove `_strategy` suffix, avoid `simple_` prefix

### Future Organization
Consider reorganizing into subdirectories:
- `core/` - fundamental strategies (momentum, mean reversion, trend)
- `indicators/` - indicator-based strategies (MA, RSI, MACD)
- `rules/` - the 16 trading rules
- `advanced/` - complex strategies (arbitrage, market making)

### Migration Path
1. Complete transition to @strategy decorated functions
2. Move class-based strategies to legacy directory
3. Simplify __init__.py to only export function strategies
4. Update dependent code to use new patterns

The module would benefit from this cleanup to make it more maintainable and consistent with the project's architecture patterns.