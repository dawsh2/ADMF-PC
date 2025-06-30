# Run All Diagnostics

After running your backtest with 463 trades, please run these diagnostic scripts in order:

## 1. Check Implementation
```bash
python check_ohlc_implementation.py
```
This verifies that the OHLC fix and risk manager changes are actually in the code.

## 2. Debug Trade Count
```bash
python debug_463_trades.py
```
This will show:
- Exit type breakdown
- Stop losses that exit with gains (shouldn't happen!)
- Immediate re-entries count
- Whether signals have OHLC data

## 3. Analyze Bar Ranges
```bash
python analyze_bar_ranges.py
```
This checks how many bars have ranges wide enough to trigger both SL and TP.

## 4. Compare with Notebook
```bash
python compare_with_notebook_exactly.py
```
This compares your results with the expected notebook results.

## Possible Issues

### If stop losses are exiting with GAINS:
The exit price calculation might be inverted or using the wrong price.

### If immediate re-entries are high:
Exit memory is still not working (strategy_id not propagating).

### If OHLC data is missing:
The strategy changes didn't take effect.

### If trade count INCREASED from 453 to 463:
The OHLC logic might be too aggressive, or we're checking exits multiple times per bar.

## Next Steps

Based on the diagnostic output, we'll need to:
1. Verify the correct files are being used
2. Check if there's a logic error in the exit checking
3. Possibly add logging to trace exactly what's happening

Please run these diagnostics and share the output!