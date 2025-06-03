# Multi-Strategy Signal Collection Fix Summary

## Issues Fixed

### 1. **Event Subscription Mechanism**
- **Problem**: The original implementation used a complex event subscription/unsubscription mechanism with callbacks that was prone to timing issues and signal duplication.
- **Solution**: Replaced with direct signal collection by calling sub-container strategies directly.

### 2. **Indicator Configuration Mismatch**
- **Problem**: Bollinger Bands indicator configuration used `num_std` parameter but IndicatorHub expected `std_dev`.
- **Solution**: Updated the parameter name in `IndicatorContainer._initialize_self()` to use `std_dev`.

### 3. **Strategy Initialization in Multi-Strategy Mode**
- **Problem**: Sub-containers weren't being properly initialized, causing strategies to be None.
- **Solution**: Added explicit `await sub_container._initialize_self()` call and ensured proper configuration format for sub-containers.

### 4. **Signal Collection and Aggregation**
- **Problem**: Signals were being lost due to asynchronous event handling and timing issues.
- **Solution**: Direct synchronous signal collection from sub-containers with proper data sharing.

## Code Changes

### File: `/Users/daws/ADMF-PC/src/execution/containers_pipeline.py`

#### 1. Fixed Bollinger Bands Parameter (Line ~294)
```python
# Changed from:
parameters={'period': period, 'num_std': 2}
# To:
parameters={'period': period, 'std_dev': 2}
```

#### 2. Rewrote `_process_signals` Method (Lines 531-612)
- Removed event subscription/unsubscription mechanism
- Added direct sub-container processing
- Ensured indicator and market data are shared with sub-containers
- Direct signal collection without async delays

#### 3. Updated `_initialize_multi_strategy` Method (Lines 489-532)
- Ensured sub-container config is in single-strategy format
- Added explicit initialization call for sub-containers
- Fixed sub-container naming

## Benefits

1. **Reliability**: Direct signal collection eliminates timing issues and race conditions.
2. **Simplicity**: Removed complex event subscription logic in favor of direct method calls.
3. **Performance**: No async delays needed for signal collection.
4. **Debugging**: Easier to trace signal flow through direct calls.

## Verification

The fix was verified with a test script that:
1. Created a multi-strategy container with momentum and mean reversion strategies
2. Verified both strategies were properly initialized
3. Confirmed each strategy received its required indicators (SMA_20 for momentum, BB_20 for mean reversion)
4. Tested signal generation with sample data

## Additional Notes

- The momentum strategy requires `SMA_{lookback_period}` (default SMA_20) but it currently calculates momentum internally. Consider moving momentum calculation to IndicatorHub for consistency.
- The signal aggregation functionality is stubbed out - implement proper weighted voting when needed.
- Consider adding unit tests to prevent regression of these fixes.