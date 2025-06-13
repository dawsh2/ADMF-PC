# Strategy Filtering Hierarchy for Ensemble Optimization

## Problem Statement

Starting with ~15,000 unique parameter combinations from strategy backtesting, need to efficiently filter down to a manageable subset (~50 per regime) for genetic algorithm optimization while preserving signal quality and computational efficiency.

## Filtering Hierarchy (In Order)

### 1. Activity/Trade Count Filtering
**Purpose:** Eliminate statistical noise from low-sample strategies  
**Cost:** Trivial (simple counting)  
**Criteria:**
- Minimum total trades across all periods (e.g., >50)
- Minimum trades per regime of interest (e.g., >10)
- Sufficient activity to generate meaningful statistics

**Rationale:** A strategy with Sharpe 3.0 but only 3 trades is statistical noise, not a tradeable signal.

---

### 2. Sensitivity Analysis 
**Purpose:** Filter out parameter-sensitive/fragile strategies  
**Cost:** Moderate (parameter space analysis)  
**Method:** Performance clustering around parameter neighborhoods

**Keep strategies where:**
- Performance forms stable plateau around optimal parameters
- Small parameter changes don't destroy performance
- Strategy demonstrates robustness across parameter variations

**Reject strategies where:**
- Performance shows cliff edges with minor parameter tweaks
- Optimal parameters are isolated peaks in noise
- No consistent performance neighborhood exists

---

### 3. Sharpe Ratio Filtering
**Purpose:** Keep only globally high-performing strategies  
**Cost:** Trivial (simple sorting)  
**Criteria:**
- Global Sharpe ratio above threshold (e.g., >1.0)
- Focus on strategies with demonstrated edge across all market conditions

**Rationale:** If strategy can't perform globally, regime-specific optimization won't help.

---

### 4. Regime-Specific Sorting
**Purpose:** Rank strategies by regime performance  
**Cost:** Low (regime classification + sorting)  
**Method:**
- Sort strategies by Sharpe ratio within each regime (Bull/Bear/Flat)
- Maintain separate rankings per regime
- Keep top N performers per regime for correlation analysis

---

### 5. Correlation Filtering (Per Regime)
**Purpose:** Eliminate redundant/highly correlated strategies  
**Cost:** Moderate (correlation matrix computation)  
**Method:**
- Calculate signal correlation matrix within each regime
- Use clustering or greedy selection to keep uncorrelated strategies
- Maintain diversity while preserving top performers

**Target:** ~50 unique, uncorrelated strategies per regime

## Expected Filtering Results

```
Initial Universe:     15,000 strategies
    ↓ Activity Filter (seconds)
Statistically Valid:   5,000 strategies (67% reduction)
    ↓ Sensitivity Analysis (minutes)
Robust Strategies:       500 strategies (90% reduction)
    ↓ Sharpe Filtering (seconds)  
High Performers:         200 strategies (60% reduction)
    ↓ Regime Sorting (seconds)
Regime Ranked:           200 strategies (organized)
    ↓ Correlation Filter (seconds)
Final Per Regime:     ~50 strategies per regime
```

## Key Advantages

### Computational Efficiency
- **Front-load cheap filters:** Activity and Sharpe filtering are nearly instant
- **Expensive operations on small sets:** Sensitivity and correlation analysis only on worthy candidates
- **Total time savings:** Hours → Minutes for full filtering pipeline

### Signal Quality Preservation
- **Statistical validity:** Ensures sufficient sample sizes
- **Robustness:** Filters out parameter-fitted noise
- **Performance focus:** Maintains only strategies with demonstrated edge
- **Diversity:** Prevents over-concentration in similar signals

### Logical Flow
- **Noise elimination first:** Remove statistical artifacts before analysis
- **Quality before quantity:** Establish performance threshold before optimization
- **Regime-specific preparation:** Organize for regime-adaptive ensemble building
- **Correlation last:** Final diversity filter on pre-qualified candidates

## Implementation Notes

### Activity Filtering Thresholds
- **Conservative:** 100+ total trades, 20+ per regime
- **Moderate:** 50+ total trades, 10+ per regime  
- **Aggressive:** 25+ total trades, 5+ per regime

### Sensitivity Analysis Approach
- **Parameter neighborhood analysis:** Test performance stability around optimal parameters
- **Performance clustering:** Look for normal distributions vs chaotic performance
- **Robustness scoring:** Quantify parameter sensitivity for filtering decisions

### Correlation Filtering Strategy
- **Hierarchical clustering:** Group similar strategies, keep best from each cluster
- **Greedy selection:** Iteratively select highest-performing uncorrelated strategies
- **Threshold-based:** Simple correlation cutoff (e.g., |correlation| < 0.6)

## Final Output

**Per-regime strategy sets** ready for genetic algorithm optimization:
- Bull regime: ~50 robust, uncorrelated, high-performing strategies
- Bear regime: ~50 robust, uncorrelated, high-performing strategies  
- Flat regime: ~50 robust, uncorrelated, high-performing strategies

Each set represents the highest-quality, most diverse subset of the original strategy universe, optimized for ensemble weight optimization via GA + WFV.
