# Analytics Implementation Plan

**Ready to execute the full workflow with proper regime data**

*Based on docs/analytics/analytics-workflow.md and lessons learned*

---

## Current Status

✅ **Completed:**
- 1,157 strategies analyzed (need Sharpe correction)
- 28 profitable strategies identified (pending validation)
- SQL templates and tooling created
- Critical pitfalls documented

🔄 **Waiting for:**
- New classifier data with 20-30m averages (less sensitive regimes)
- Proper regime transitions vs current 5-6 minute chaos

---

## Phase 1: Classifier Validation (Ready to Execute)

### Step 1.1: State Distribution Analysis
**Target:** Balanced regime distribution (~20-33% per state)

```sql
-- Use templates/correct_sharpe_calculation.sql framework
-- Check regime balance across new classifiers
SELECT 
    classifier_id, 
    regime, 
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) as pct_time,
    CASE 
        WHEN COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) < 5 THEN 'RARE_STATE'
        WHEN COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY classifier_id) > 60 THEN 'DOMINANT_STATE'
        ELSE 'BALANCED'
    END as distribution_flag
FROM classifier_states 
GROUP BY classifier_id, regime;
```

### Step 1.2: Regime Persistence Analysis  
**Target:** Average regime duration >15-20 minutes (vs current 5-6 min)

```sql
-- Validate regime stability with new 20-30m averaged data
-- Should show much more stable regimes than current
```

### Step 1.3: Market Differentiation Test
**Key:** Do regimes actually predict different market conditions?

---

## Phase 2: Classifier Ranking (Automated Pipeline)

### Quality Scorecard
**Weighted scoring system:**
- Distribution balance: 25%
- Regime stability: 30% 
- Predictive power: 30%
- Sample adequacy: 15%

**Selection criteria:**
- Composite score >0.6
- Grade A or B classification
- **Target:** Top 10-15 classifiers from ~100 combinations

---

## Phase 3: Strategy-Regime Performance Matrix

### Critical Updates Needed

#### 1. **Fix Sharpe Ratio Calculation**
```sql
-- WRONG (current): Per-trade scaling
SQRT(252 * assumed_trades_per_day)  -- Inflates low-frequency strategies

-- CORRECT: Wall-clock time scaling  
-- Calculate daily returns, then annualize
WITH daily_returns AS (
    SELECT strategy_id, DATE(entry_time) as date, SUM(net_return_bps) as daily_bps
    FROM trades GROUP BY strategy_id, DATE(entry_time)
)
SELECT AVG(daily_bps) / STDDEV(daily_bps) * SQRT(252) as correct_sharpe
```

#### 2. **Regime-Strategy Performance Analysis**
```sql
-- Analyze 28 profitable strategies across regimes
-- Identify which strategies outperform in specific market conditions
-- Build regime-conditional alpha matrix
```

#### 3. **Duration Limit Integration**
```sql
-- Apply 30-minute duration limits where beneficial
-- RSI strategies showed 9.93 Sharpe with duration limits vs 0.34 without
```

---

## Phase 4: Strategy Quality Filtering

### Updated Criteria (Post-Analysis)

#### 1. **Realistic Performance Thresholds**
```sql
-- UPDATED: Based on actual analysis results
WHERE net_sharpe > 0.3           -- Lowered from 0.5 (few strategies meet 0.5)
  AND net_return_bps > 0         -- Must be profitable after 0.5 bps costs
  AND trades_per_day >= 1.0      -- Minimum frequency  
  AND total_trades >= 100        -- Statistical significance
```

#### 2. **Cost Robustness Validation**
```sql
-- 0.5 bps transaction costs eliminated 97% of strategies
-- Validate remaining 28 strategies maintain profitability
```

#### 3. **Duration Optimization**
```sql
-- Test duration limits on all profitable strategies
-- Some may benefit from 30-60 minute caps like RSI strategies
```

---

## Phase 5: Ensemble Construction

### Allocation Matrix Framework
```sql
-- For each (classifier, regime) pair:
-- 1. Rank profitable strategies by regime-specific performance
-- 2. Weight by corrected Sharpe ratios  
-- 3. Apply diversification constraints (max 40% per strategy)
-- 4. Ensure minimum 5% allocation threshold
```

### Expected Outcomes
- **~10-15 qualified classifiers** (from 100+ combinations)
- **~15-25 qualified strategies** (from 28 profitable, post-Sharpe correction)
- **Allocation matrix:** Each regime gets 3-8 strategies with optimized weights

---

## Implementation Sequence

### Immediate (Once New Data Arrives)
1. **Run Phase 1 classifier validation** - identify stable classifiers
2. **Execute Sharpe correction script** on all 1,157 strategies  
3. **Re-validate the 28 profitable strategies** with correct metrics
4. **Apply duration limits optimization** to remaining profitable strategies

### Week 1
1. **Complete classifier ranking** - select top 10-15
2. **Strategy-regime performance analysis** - build alpha matrix
3. **Final strategy quality filtering** - robust candidates only

### Week 2  
1. **Build ensemble allocation matrix** - optimized weights per regime
2. **Walk-forward validation setup** - out-of-sample testing
3. **Paper trading preparation** - real-time regime detection

---

## Key Metrics to Track

### Classifier KPIs
- **Regime duration:** Target >15-20 minutes (vs current 5-6)
- **State balance:** Std deviation <10%
- **Predictive power:** >20 bps differentiation between regimes

### Strategy KPIs  
- **Corrected Sharpe:** >0.3 after wall-clock time calculation
- **Cost survival:** Profitable after 0.5 bps transaction costs
- **Duration optimization:** Performance boost from time limits

### Ensemble KPIs
- **Coverage:** >80% of time in identifiable regimes
- **Diversification:** <0.5 average strategy correlation  
- **Regime alpha:** Clear outperformance in assigned regimes

---

## Risk Management

### Known Pitfalls (Learned)
1. **Sharpe inflation** - Use wall-clock time only
2. **Overnight gap risk** - Filter out overnight trades
3. **Transaction cost reality** - 0.5 bps eliminates most edges
4. **Sample size requirements** - Minimum 100 trades, 20 days
5. **Duration risk** - Longer trades often toxic (30-60+ min)

### Monitoring Setup
```sql
-- Real-time alerts for:
-- 1. Regime transitions (classifier state changes)
-- 2. Strategy performance deviation (>2 sigma from baseline)  
-- 3. Correlation spikes (strategies becoming too similar)
-- 4. Drawdown breaches (approaching risk limits)
```

---

## Expected Timeline

**Data Arrival → Paper Trading Ready:** 2-3 weeks

- **Week 1:** Classifier validation, Sharpe correction, strategy re-ranking
- **Week 2:** Regime-strategy analysis, ensemble construction  
- **Week 3:** Validation, monitoring setup, paper trading preparation

**Success Criteria:**
- 10-15 validated classifiers with stable 15-20 minute regimes
- 15-25 strategies with corrected Sharpe >0.3 and robust performance
- Allocation matrix providing clear regime-conditional strategy selection
- Real-time regime detection and portfolio rebalancing system

---

*This plan leverages all lessons learned while executing the original analytics workflow design with proper regime data.*