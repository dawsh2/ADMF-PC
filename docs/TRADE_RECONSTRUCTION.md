# Trade Reconstruction Guide

## Overview

The universal topology generates four types of trace events that together allow complete trade reconstruction:

1. **Strategy Signals** - The trading decisions
2. **Portfolio Orders** - Order creation events
3. **Execution Fills** - Order execution confirmations
4. **Position Events** - Position lifecycle tracking

## Event Metadata Structure

### 1. Strategy Signals
```python
{
    "idx": 100,           # Bar index
    "ts": 1234567890,     # Timestamp
    "sym": "SPY",         # Symbol
    "val": 1,             # Signal value (1=long, -1=short, 0=flat)
    "strat": "strategy_0", # Strategy ID
    "px": 450.50,         # Price at signal
    "metadata": {
        "parameters": {...},  # Strategy parameters
        "indicators": {...}   # Indicator values at signal
    }
}
```

### 2. Portfolio Orders
```python
{
    "idx": 101,           # Bar index when order created
    "ts": 1234567891,     # Timestamp
    "sym": "SPY",         # Symbol
    "val": 1,             # Direction
    "metadata": {
        "order_id": "ord_123",
        "side": "BUY",
        "quantity": 100,
        "price": 450.55,
        "order_type": "MARKET",
        "strategy_id": "strategy_0",
        "signal_bar": 100,    # Bar that generated the signal
        "reason": "signal"    # or "stop_loss", "take_profit", etc.
    }
}
```

### 3. Execution Fills
```python
{
    "idx": 101,           # Bar index when filled
    "ts": 1234567892,     # Timestamp
    "sym": "SPY",         # Symbol
    "metadata": {
        "fill_id": "fill_456",
        "order_id": "ord_123",
        "price": 450.60,
        "quantity": 100,
        "commission": 1.00,
        "side": "BUY",
        "timestamp": 1234567892
    }
}
```

### 4. Position Events

#### Position Open
```python
{
    "idx": 101,           # Bar index when position opened
    "ts": 1234567892,     # Timestamp
    "sym": "SPY",         # Symbol
    "metadata": {
        "position_id": "pos_789",
        "strategy_id": "strategy_0",
        "entry_price": 450.60,
        "quantity": 100,
        "side": "LONG",
        "entry_bar": 101,
        "entry_order_id": "ord_123",
        "entry_fill_id": "fill_456"
    }
}
```

#### Position Close
```python
{
    "idx": 150,           # Bar index when position closed
    "ts": 1234568000,     # Timestamp
    "sym": "SPY",         # Symbol
    "metadata": {
        "position_id": "pos_789",
        "strategy_id": "strategy_0",
        "exit_price": 451.20,
        "quantity": 100,
        "realized_pnl": 60.00,
        "exit_bar": 150,
        "exit_type": "stop_loss",    # or "signal", "take_profit", "eod"
        "exit_reason": "Stop loss triggered at 451.20",
        "exit_order_id": "ord_124",
        "exit_fill_id": "fill_457",
        "bars_held": 49,
        "max_profit": 120.00,
        "max_loss": -30.00
    }
}
```

## Reconstruction Process

### Method 1: Using Position Events (Simplest)

Position events contain all necessary information for basic trade analysis:

```python
import pandas as pd
import json

# Load position events
opens = pd.read_parquet('traces/portfolio/positions_open/position_open.parquet')
closes = pd.read_parquet('traces/portfolio/positions_close/position_close.parquet')

# Parse metadata
opens['meta'] = opens['metadata'].apply(json.loads)
closes['meta'] = closes['metadata'].apply(json.loads)

# Extract position IDs
opens['position_id'] = opens['meta'].apply(lambda x: x.get('position_id'))
closes['position_id'] = closes['meta'].apply(lambda x: x.get('position_id'))

# Join opens and closes
trades = pd.merge(
    opens[['position_id', 'idx', 'meta']].rename(columns={'idx': 'entry_bar', 'meta': 'entry_meta'}),
    closes[['position_id', 'idx', 'meta']].rename(columns={'idx': 'exit_bar', 'meta': 'exit_meta'}),
    on='position_id',
    how='inner'
)

# Calculate trade metrics
trades['entry_price'] = trades['entry_meta'].apply(lambda x: x['entry_price'])
trades['exit_price'] = trades['exit_meta'].apply(lambda x: x['exit_price'])
trades['pnl'] = trades['exit_meta'].apply(lambda x: x['realized_pnl'])
trades['bars_held'] = trades['exit_bar'] - trades['entry_bar']
trades['exit_type'] = trades['exit_meta'].apply(lambda x: x.get('exit_type'))
```

### Method 2: Order-Fill Matching (More Detailed)

For more detailed analysis including partial fills and order timing:

```python
# Load all events
orders = pd.read_parquet('traces/portfolio/orders/portfolio_orders.parquet')
fills = pd.read_parquet('traces/execution/fills/execution_fills.parquet')

# Parse metadata
orders['order_data'] = orders['metadata'].apply(json.loads)
fills['fill_data'] = fills['metadata'].apply(json.loads)

# Extract order IDs
orders['order_id'] = orders['order_data'].apply(lambda x: x['order_id'])
fills['order_id'] = fills['fill_data'].apply(lambda x: x['order_id'])

# Match orders to fills
order_fills = pd.merge(
    orders[['order_id', 'idx', 'order_data']].rename(columns={'idx': 'order_bar'}),
    fills[['order_id', 'idx', 'fill_data']].rename(columns={'idx': 'fill_bar'}),
    on='order_id'
)

# Group by position lifecycle
# Entry orders have reason='signal', exit orders have reason='stop_loss' etc
entry_fills = order_fills[order_fills['order_data'].apply(lambda x: x.get('reason') == 'signal')]
exit_fills = order_fills[order_fills['order_data'].apply(lambda x: x.get('reason') != 'signal')]
```

### Method 3: Full Event Chain (Most Complete)

Link signals → orders → fills → positions for complete trade lifecycle:

```sql
-- Using DuckDB for complex joins
WITH signal_orders AS (
    SELECT 
        s.idx as signal_bar,
        s.val as signal_value,
        o.idx as order_bar,
        json_extract_string(o.metadata, '$.order_id') as order_id,
        json_extract_string(o.metadata, '$.strategy_id') as strategy_id
    FROM signals s
    JOIN orders o ON s.strat = json_extract_string(o.metadata, '$.strategy_id')
        AND o.idx >= s.idx 
        AND o.idx <= s.idx + 2  -- Orders created within 2 bars of signal
),
order_fills AS (
    SELECT 
        so.*,
        f.idx as fill_bar,
        json_extract_string(f.metadata, '$.price') as fill_price,
        json_extract_string(f.metadata, '$.fill_id') as fill_id
    FROM signal_orders so
    JOIN fills f ON so.order_id = json_extract_string(f.metadata, '$.order_id')
),
complete_trades AS (
    SELECT 
        of.*,
        po.idx as position_open_bar,
        pc.idx as position_close_bar,
        json_extract_string(pc.metadata, '$.realized_pnl') as pnl,
        json_extract_string(pc.metadata, '$.exit_type') as exit_type
    FROM order_fills of
    LEFT JOIN position_opens po ON of.fill_id = json_extract_string(po.metadata, '$.entry_fill_id')
    LEFT JOIN position_closes pc ON json_extract_string(po.metadata, '$.position_id') = 
                                    json_extract_string(pc.metadata, '$.position_id')
)
SELECT * FROM complete_trades;
```

## Common Issues in Trade Reconstruction

### 1. Missing Position Events
Early in the backtest, position tracking may not be fully initialized. Look for:
- Orders/fills without corresponding position events
- Position events starting after bar 0

### 2. Incomplete Trades
Positions opened near the end of the backtest may not have closing events.

### 3. Multiple Partial Fills
A single order may have multiple fills. Aggregate by order_id first.

### 4. Risk Management Exits
These create internal orders that may not have originating signals. Check the 'reason' field in order metadata.

## Best Practices

1. **Always validate data completeness**:
   ```python
   print(f"Orders: {len(orders)}")
   print(f"Fills: {len(fills)}")  
   print(f"Position Opens: {len(opens)}")
   print(f"Position Closes: {len(closes)}")
   ```

2. **Check for orphaned events**:
   ```python
   unmatched_orders = orders[~orders['order_id'].isin(fills['order_id'])]
   unclosed_positions = opens[~opens['position_id'].isin(closes['position_id'])]
   ```

3. **Use position events for simple analysis**, order-fill matching for detailed analysis

4. **Consider time windows** when matching events - they may not occur on the same bar

5. **Handle edge cases** like EOD closes, partial fills, and risk exits explicitly