# Trading System Issues Summary

## Issues Found

### 1. **PnL Calculation Broken**
- `realized_pnl` is always 0 for all trades
- Should be: (exit_price - entry_price) * quantity
- Example: Entry $521.11, Exit $521.40, Quantity 100 → Should be $29.00 PnL

### 2. **Position Sizing Mismatch**
- Code shows: `return 1` (position size should be 1)
- Trades show: `quantity = 100` for all trades
- Metadata shows: correct signed quantities (1 for LONG, -1 for SHORT)
- Something is multiplying position size by 100

### 3. **Return Percentage Scaling**
- Return percentages appear to be already multiplied by 100
- `return_pct = 0.055650` means 0.055650%, not 5.565%
- This could cause confusion in analysis

### 4. **Stop/Target Logic Inverted**
- Found trades hitting `take_profit` at -0.1% LOSS
- Found trades hitting `stop_loss` at +0.075% GAIN
- This indicates the stop loss and take profit logic is inverted for one or both directions

### 5. **Metadata Structure**
- Metadata contains correct signed quantities (1, -1)
- But trades DataFrame shows unsigned quantity (100)
- Direction information is lost in the trades DataFrame

## Root Cause Analysis

The combination of these issues suggests:

1. **Data Processing Pipeline Issue**: Somewhere between signal generation and trade recording:
   - Signed quantities (1, -1) are being converted to unsigned (100)
   - PnL calculation is failing (returning 0)
   - Return percentages are being pre-multiplied by 100

2. **Direction Logic Issue**: For at least one direction (LONG or SHORT):
   - Stop loss and take profit levels are inverted
   - Positions exit at take profit when losing money
   - Positions exit at stop loss when making money

## Next Steps

1. Run `analyze_by_signal_type(df, trades_df)` to identify which direction has inverted logic
2. Fix PnL calculation to properly compute (exit_price - entry_price) * quantity
3. Trace where position size is being multiplied by 100
4. Fix the inverted stop/target logic for the affected direction
5. Ensure signed quantities are preserved through the pipeline