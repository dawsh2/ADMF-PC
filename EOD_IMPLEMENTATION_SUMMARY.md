# End-of-Day (EOD) Position Closing Implementation

## Overview
The `--close-eod` flag prevents overnight holding by:
1. Blocking new entries after 3:30 PM ET
2. Forcing all positions to close at 3:50 PM ET

## Implementation Details

### Files Modified
1. **`src/core/cli/parser.py`** - Added `--close-eod` CLI argument
2. **`main.py`** - Passes the flag to config as `config['execution']['close_eod']`
3. **`src/strategy/state.py`** - Applies EOD filters when loading strategies
4. **`src/strategy/components/eod_timeframe_helper.py`** - Helper functions for timeframe detection

### How It Works
When `--close-eod` is used:
```
CLI flag → main.py → config → strategy state → filter applied to all strategies
```

The system automatically detects your data's timeframe and calculates the correct bar numbers:

| Timeframe | Entry Cutoff | Force Exit | Filter Expression |
|-----------|--------------|------------|-------------------|
| 1-min | bar < 360 | bar >= 380 | `(bar_of_day < 360 or (bar_of_day >= 380 and signal == 0))` |
| 5-min | bar < 72 | bar >= 76 | `(bar_of_day < 72 or (bar_of_day >= 76 and signal == 0))` |
| 15-min | bar < 24 | bar >= 25 | `(bar_of_day < 24 or (bar_of_day >= 25 and signal == 0))` |

### Important Notes on Your Data

1. **UTC Timestamps**: Your data files use UTC timestamps
   - 13:30 UTC = 9:30 AM ET (market open)
   - 19:30 UTC = 3:30 PM ET (entry cutoff)
   - 19:50 UTC = 3:50 PM ET (force exit)
   - 20:00 UTC = 4:00 PM ET (market close)

2. **Data Coverage**:
   - `SPY_1m.csv`: 390 bars per day (complete)
   - `SPY_5m.csv`: 78 bars per day (complete)
   - `SPY_15m.csv`: 26 bars per day (complete)

3. **Pre/After Market**: Some files include extended hours trading, but `bar_of_day` is calculated from regular market open (9:30 AM ET).

## Usage

```bash
# Without EOD closing (allows overnight positions)
python main.py --config config.yaml --backtest

# With EOD closing (prevents overnight holding)
python main.py --config config.yaml --backtest --close-eod
```

## Example Filter Applied

For a 5-minute strategy with existing filter:
```yaml
# Original filter
filter: "signal != 0 and volume > 1000000"

# With --close-eod, becomes:
filter: "(signal != 0 and volume > 1000000) and (bar_of_day < 72 or (bar_of_day >= 76 and signal == 0))"
```

## Why These Times?

Based on the analysis from earlier sessions:
- Overnight holding was causing significant losses in test data
- A 3:30 PM entry cutoff improved 5/10 strategies to profitability
- 3:50 PM gives 10 minutes to exit positions before market close
- This prevents gap risk from overnight news and events