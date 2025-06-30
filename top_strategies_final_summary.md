# Top Keltner Strategies - Final Summary

## Key Discovery
The "2826-signal group" isn't an ensemble - it's **11 identical strategies** that converged to the same signals despite different Keltner parameters (different periods/multipliers produced same trades).

## Top 10 Strategies (with Win Rates)

| Signals | RPT (bps) | Win% | Trades/Day | Annual% | Filter Type | Recommendation |
|---------|-----------|------|------------|---------|-------------|----------------|
| **47** | 4.09 | 60.9% | 0.1 | 0.9% | Master regime (Vol+VWAP+Time) | Too few trades |
| **303** | 0.76 | 59.6% | 0.6 | 1.2% | RSI/Volume combination | Low frequency |
| **2826** | 0.68 | **73.7%** | 5.7 | 9.7% | Volatility filter (ATR-based) | **BEST CHOICE** |
| 3261 | 0.55 | 70.6% | 6.5 | 9.0% | Minimal filtering | Good alternative |
| 3262 | 0.45 | 67.9% | 6.5 | 7.4% | Baseline (no filter) | Acceptable |
| 2305 | 0.42 | **77.0%** | 4.7 | 4.9% | Light volume filter | High win rate |
| 1500 | 0.41 | 63.2% | 3.0 | 3.1% | Long-only variant | Meets 2-3/day target |
| 2326 | 0.32 | 72.0% | 4.6 | 3.7% | Mixed filters | Marginal |
| 2073 | 0.30 | 75.9% | 4.1 | 3.1% | Unknown filter | Marginal |
| 3481 | 0.23 | 68.5% | 7.0 | 4.1% | No filter | Baseline |

## The Winner: 2826-Signal Strategy

### What it is:
- **NOT an ensemble** - just one strategy (pick any of the 11)
- Volatility filter: Trades when ATR > threshold
- Multiple Keltner parameters converged to same signals

### Performance:
- **0.68 bps/trade** (0.18 bps after 0.5 bps costs)
- **73.7% win rate** - excellent consistency
- **5.7 trades/day** - good frequency
- **2.6% annual net** (9.7% gross)

### Why it works:
- High win rate (73.7%) provides consistency
- Volatility filter targets better trading conditions
- Good frequency balances edge vs opportunities
- Light filtering (18.8% reduction) avoids overfitting

## Implementation

You would run **ONE strategy** from the 2826 group:
```yaml
strategy: keltner_bands
period: [any from 10-50 that produces 2826 signals]
multiplier: [any from 1.0-3.0 that produces 2826 signals]
filter: volatility_above: {threshold: ~1.1-1.2}
```

**NOT** an ensemble of all 11 (they're identical anyway).

## Alternative if you need exactly 2-3 trades/day:
- **1500 signals**: 0.41 bps, 63.2% win rate, 3.0 trades/day
- But annual return drops to 3.1% (vs 9.7% for 2826)

The 2826 strategy remains the clear winner for practical trading.