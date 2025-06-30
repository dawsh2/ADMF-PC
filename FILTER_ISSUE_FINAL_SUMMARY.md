# Filter Issue Summary - Signal Generation Mode

## The Problem
Filters defined in the config are NOT being applied during signal generation mode (`--signal-generation`).

## Evidence
1. **Single filter config**: Always produces 726 signals regardless of threshold (0.8, 1.1, etc.)
2. **Multiple filter config**: Correctly expands to 3 strategies but each produces identical 726 signals
3. **No enhanced_metadata.json**: Filter metadata tracking is not implemented
4. **Identical signal files**: All strategies produce exact same output (22898 bytes)

## Root Cause
The signal generation mode does not apply filters to the strategy signals. The filters are:
- Parsed correctly by the config parser
- Expanded correctly by the parameter expander
- But NOT applied during strategy execution

## Why It Worked in Training
The original analysis likely used a different mode (possibly `--backtest` or a custom analysis script) that does apply filters post-signal generation.

## Current State
- **Raw signals**: 726 on test data (vs 3,481 on training data)
- **Expected filtered**: ~590 signals with 0.8 threshold (estimated)
- **Performance impact**: -2.06 bps/trade without filter vs expected 0.68 bps/trade with filter

## Solutions
1. **Use different execution mode**: Try `--backtest` instead of `--signal-generation`
2. **Post-process signals**: Apply filters after signal generation
3. **Custom analysis**: Run the same analysis pipeline that was used originally
4. **Fix implementation**: Add filter application to signal generation mode

## Key Insight
The 74% reduction in signals (3,481 → 726) between training and test data indicates a fundamentally different market regime. The test period appears to have:
- Lower volatility
- Fewer Keltner band crosses
- Different market dynamics

Without the volatility filter working, you're trading all signals including low-volatility periods where mean reversion performs poorly.