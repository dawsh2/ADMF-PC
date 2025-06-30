# Signal to Order Mismatch Analysis

## Problem Summary
- **Expected**: 623,088 signal changes across 1800 strategies
- **Actual**: Only 150 orders generated
- **Loss Rate**: 99.98% of signals are not converting to orders

## Root Causes Identified

### 1. Single Order Per Symbol Constraint
The portfolio's `can_create_order` method (lines 122-145 in `src/portfolio/state.py`) prevents creating new orders if there's already a pending order for the same symbol:

```python
def can_create_order(self, symbol: str) -> bool:
    """Check if we should create a new order for this symbol.
    
    This prevents race conditions by ensuring we don't create multiple
    orders for the same symbol while one is pending.
    """
    # ... code that checks for pending orders
    return len(pending_for_symbol) == 0
```

This means:
- Only ONE order can be pending per symbol at any time
- With 1800 strategies all trading the same symbol (SPY), only the first signal gets converted
- All subsequent signals are ignored until the order fills

### 2. Signal Filtering Conditions
The portfolio also filters signals based on:
- **FLAT signals** are ignored (line 560)
- **Exit memory** prevents re-entry after risk exits (lines 542-557)
- **Signal strength** requirements

### 3. No Multi-Strategy Aggregation
The system appears to process each strategy's signals independently without aggregation:
- Each strategy sends its own signals
- No component aggregates or nets out opposing signals
- First signal wins, others are discarded

## Impact

With 1800 strategies generating signals:
- Strategy 0 sends BUY signal → Order created
- Strategies 1-1799 send signals → All ignored (pending order exists)
- Order fills
- Next signal from any strategy → New order
- Repeat

This explains why we see ~150 orders instead of 600,000+. The orders roughly correspond to the number of times the position fully exits and re-enters.

## Recommendations

### Option 1: Implement Signal Aggregation
Add a component that:
- Collects signals from all strategies
- Nets out opposing signals
- Sends one aggregated signal to portfolio

### Option 2: Multi-Position Support
Allow multiple positions per symbol:
- Track positions by (symbol, strategy_id)
- Allow each strategy to manage its own position
- Aggregate for risk limits

### Option 3: Order Queue Management
Instead of blocking new orders:
- Queue signals while order is pending
- Process queue when order fills
- Net out opposing signals in queue

### Option 4: Reduce Strategy Count
For single-symbol, single-position trading:
- Use fewer strategies
- Or use ensemble strategies that internally aggregate

## Current Workaround

The system is working as designed for:
- Preventing race conditions
- Maintaining single position per symbol
- Sequential order processing

But this design doesn't scale to 1800 concurrent strategies on the same symbol.