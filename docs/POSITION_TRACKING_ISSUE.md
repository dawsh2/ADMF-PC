# Position Tracking Issue Analysis

## Problem
Only 1 position open/close event is being recorded despite 545+ orders being created.

## Root Cause
The issue is in how ORDER events are being traced by MultiStrategyTracer:

1. **Sparse Storage Misuse**: The tracer uses `TemporalSparseStorage` which is designed to store only signal CHANGES (e.g., when a strategy goes from long to short).

2. **Single Strategy ID**: All orders are stored with strategy_id='portfolio_orders', so the sparse storage treats them as coming from a single source.

3. **Change Detection**: The storage only records an order if it's different from the previous order. Since most orders have the same "direction" (buy/sell), they're considered redundant and not stored.

## Evidence
- 547 orders created but most have None metadata
- Only 4 BUY and 2 SELL orders have metadata (these were the "changes")
- Order indices match signal change points, confirming orders ARE being created

## Example Flow
```
Bar 72: Strategy signal = 1 (long)
  → Order created: BUY (stored - first order)
  → Fill executed
  → Position opened ✓

Bar 79: Strategy signal = -1 (short)  
  → Order created: SELL (direction changed, stored)
  → Fill executed
  → Position closed ✓

Bar 246: Strategy signal = 1 (long)
  → Order created: BUY (NOT stored - same direction as last non-flat)
  → Fill executed  
  → Position opened (but no trace!)

Bar 337: Strategy signal = -1 (short)
  → Order created: SELL (NOT stored - same as bar 79)
  → Fill executed
  → Position closed (but no trace!)
```

## Solution Options

### Option 1: Don't Use Sparse Storage for Orders
Orders and fills should use regular storage that records every event, not just changes.

### Option 2: Use Unique Strategy IDs
Instead of 'portfolio_orders', use unique IDs like 'order_{order_id}' to force storage of each order.

### Option 3: Add Force Storage Flag
Add a parameter to sparse storage to disable change detection for certain event types.

## Impact
This explains why:
- Position events are missing (they depend on order metadata)
- Risk management can't track positions properly  
- Trade reconstruction shows only 1 trade instead of ~270+
- Performance metrics are completely wrong

## Immediate Fix
The quickest fix is to modify MultiStrategyTracer to use unique IDs for orders:

```python
# Instead of:
strategy_id='portfolio_orders'

# Use:
strategy_id=f"order_{payload.get('order_id', 'unknown')}"
```

This will force storage of every order with its full metadata.