# Trace Structure Documentation

## Overview

The universal topology generates trace files that capture the complete flow of events through the trading system. These traces are stored in Parquet format for efficient storage and querying.

## Trace Types and Event Flow

### 1. Strategy Signals (`traces/signals/`)
- **Location**: `traces/signals/{strategy_type}/{component_id}.parquet`
- **Source**: Strategy components
- **Content**: Signal state changes (long/short/flat)
- **Key Fields**:
  - `idx`: Bar index
  - `ts`: Timestamp
  - `sym`: Symbol
  - `val`: Signal value (1=long, -1=short, 0=flat)
  - `strat`: Strategy ID
  - `px`: Price at signal
  - `metadata`: Strategy parameters and configuration

### 2. Portfolio Orders (`traces/portfolio/orders/`)
- **Location**: `traces/portfolio/orders/portfolio_orders.parquet`
- **Source**: Portfolio state when processing signals
- **Content**: Order creation events
- **Key Fields**:
  - `idx`: Bar index when order was created
  - `ts`: Timestamp
  - `sym`: Symbol
  - `val`: Direction (long/short)
  - `metadata`: Contains order details (side, quantity, price, order_id)

### 3. Execution Fills (`traces/execution/fills/`)
- **Location**: `traces/execution/fills/execution_fills.parquet`
- **Source**: Execution engine when orders are filled
- **Content**: Order execution confirmations
- **Key Fields**:
  - `idx`: Bar index when fill occurred
  - `ts`: Timestamp
  - `sym`: Symbol
  - `metadata`: Contains fill details (price, quantity, commission, fill_id)

### 4. Position Events (`traces/portfolio/positions_*/`)
- **Open**: `traces/portfolio/positions_open/position_open.parquet`
- **Close**: `traces/portfolio/positions_close/position_close.parquet`
- **Source**: Portfolio state when positions change
- **Content**: Position lifecycle events
- **Key Fields**:
  - `idx`: Bar index of position event
  - `ts`: Timestamp
  - `sym`: Symbol
  - `metadata`: Contains position details:
    - For opens: entry_price, quantity, strategy_id
    - For closes: exit_price, realized_pnl, exit_type, exit_reason

## Event Flow Sequence

```
1. Strategy generates signal (SIGNAL event)
   ↓
2. Portfolio receives signal and creates order (ORDER event)
   ↓
3. Execution engine receives order and creates fill (FILL event)
   ↓
4. Portfolio receives fill and updates position
   ↓
5. Portfolio emits POSITION_OPEN (first fill) or updates existing position
   ↓
6. When position closes, Portfolio emits POSITION_CLOSE
```

## Risk Management Exit Flow

When risk management triggers an exit (stop loss, take profit, trailing stop):

1. Portfolio's BAR event handler checks exit conditions
2. If exit triggered, generates internal FLAT signal with exit metadata
3. Creates closing order
4. On fill, emits POSITION_CLOSE with exit_type and exit_reason

## Important Notes

### Re-entry After Risk Exits

**Current Behavior**: The system has no cooldown mechanism. If a strategy maintains a long signal after a stop loss exit, the portfolio will immediately create a new order on the next bar.

**Example Scenario**:
1. Bar 100: Strategy signal = 1 (long), position opened
2. Bar 150: Stop loss triggered, position closed (exit_type = "stop_loss")
3. Bar 151: Strategy signal still = 1, NEW position opened immediately

**Potential Issues**:
- Can lead to "whipsawing" in volatile markets
- May repeatedly hit stop losses if market conditions haven't changed
- Increases transaction costs

### Trace Discrepancies

You may notice discrepancies between:
- Number of orders/fills (e.g., 724)
- Number of position events (e.g., 1 open, 1 close)

**Reasons**:
1. **Incomplete trades**: Positions opened but not closed by end of backtest
2. **Position tracking initialization**: Early versions may not have tracked all positions
3. **Flat positions**: Orders that result in closing positions don't generate new POSITION_OPEN events

## Using the Traces

### With DuckDB
```sql
-- Load all traces
CREATE VIEW signals AS SELECT * FROM read_parquet('traces/signals/*/*.parquet');
CREATE VIEW orders AS SELECT * FROM read_parquet('traces/portfolio/orders/*.parquet');
CREATE VIEW fills AS SELECT * FROM read_parquet('traces/execution/fills/*.parquet');
CREATE VIEW position_opens AS SELECT * FROM read_parquet('traces/portfolio/positions_open/*.parquet');
CREATE VIEW position_closes AS SELECT * FROM read_parquet('traces/portfolio/positions_close/*.parquet');

-- Analyze re-entry patterns
WITH risk_exits AS (
    SELECT idx as exit_bar, json_extract_string(metadata, '$.symbol') as symbol
    FROM position_closes
    WHERE json_extract_string(metadata, '$.exit_type') IN ('stop_loss', 'trailing_stop')
)
SELECT 
    o.idx - re.exit_bar as bars_to_reentry,
    COUNT(*) as occurrences
FROM orders o
JOIN risk_exits re ON json_extract_string(o.metadata, '$.symbol') = re.symbol
WHERE o.idx > re.exit_bar AND o.idx <= re.exit_bar + 10
GROUP BY 1
ORDER BY 1;
```

### With Pandas
```python
import pandas as pd
import json

# Load traces
signals = pd.read_parquet('traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')
orders = pd.read_parquet('traces/portfolio/orders/portfolio_orders.parquet')

# Parse metadata
orders['order_data'] = orders['metadata'].apply(json.loads)
orders['side'] = orders['order_data'].apply(lambda x: x.get('side'))
orders['quantity'] = orders['order_data'].apply(lambda x: x.get('quantity'))
```

## Future Improvements

1. **Cooldown Period**: Implement a configurable cooldown after risk exits
2. **Signal Override**: Allow risk manager to override strategy signals temporarily
3. **Position Tracking**: Ensure all positions generate open/close events
4. **Exit Analytics**: Enhanced metadata for analyzing exit effectiveness