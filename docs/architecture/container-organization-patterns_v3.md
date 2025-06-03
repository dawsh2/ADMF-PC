# Container Organization for Combinatorial Search

## Philosophy: Hierarchical Organization for Efficient Search

The ADMF-PC container hierarchy isn't about abstract "organizational patterns" - it's about efficiently exploring the combinatorial search space of trading system configurations. By organizing containers hierarchically, we can fix certain components while varying others, dramatically reducing computational overhead and making results easier to analyze.

## Core Insight: Minimize Container Count Through Smart Hierarchy

When you have:
- 2 classifiers
- 4 risk managers  
- 10 portfolios
- 40 strategies

That's 3,200 possible combinations. The key is organizing them efficiently:

```
Least variations → Outermost container
Most variations → Innermost container
```

This minimizes container creation and maximizes component reuse.

## Invertible Hierarchies for Different Search Questions

The powerful insight is that you can invert the container hierarchy based on what you're searching for:

### Search Question 1: "How does my strategy perform across different market conditions?"
**Fix the strategy (outer), vary classifiers and risk (inner)**

```yaml
# Strategy-outer organization
search_mode: "strategy_validation"
hierarchy:
  strategies:
    - name: "momentum_strategy"
      type: "momentum"
      parameters: {fast: 10, slow: 30}
      vary:
        classifiers: ["hmm", "pattern", "volatility"]
        risk_profiles: ["conservative", "balanced", "aggressive"]
```

Container structure:
```
┌─────────────────────┐
│ Momentum Strategy   │ (1 container - strategy logic computed once)
│  ├─ HMM Classifier  │ 
│  │  ├─ Conservative │ (test all risk levels)
│  │  ├─ Balanced     │
│  │  └─ Aggressive   │
│  ├─ Pattern Class   │
│  │  └─ ...          │
│  └─ Volatility Class│
│     └─ ...          │
└─────────────────────┘
```

### Search Question 2: "Which strategies work best in bull markets?"
**Fix the classifier (outer), vary strategies and risk (inner)**

```yaml
# Classifier-outer organization  
search_mode: "regime_optimization"
hierarchy:
  classifiers:
    - name: "hmm_bull_detector"
      type: "hmm"
      parameters: {states: 3}
      vary:
        strategies: 
          - {type: "momentum", params: [...]}
          - {type: "mean_reversion", params: [...]}
          - {type: "breakout", params: [...]}
        risk_profiles: ["conservative", "balanced", "aggressive"]
```

Container structure:
```
┌─────────────────────┐
│ HMM Classifier      │ (1 container - regime detection computed once)
│  ├─ Conservative    │
│  │  ├─ Momentum     │ (test all strategies)
│  │  ├─ MeanReversion│
│  │  └─ Breakout     │
│  ├─ Balanced        │
│  │  └─ ...          │
└─────────────────────┘
```

### Search Question 3: "What's the optimal risk configuration?"
**Fix risk parameters (outer), vary strategies and classifiers (inner)**

```yaml
# Risk-outer organization
search_mode: "risk_optimization"
hierarchy:
  risk_profiles:
    - name: "conservative"
      max_position_pct: 2.0
      stop_loss: 1.5
      vary:
        strategies: [momentum, mean_reversion, breakout]
        classifiers: ["hmm", "pattern"]
```

## Practical Examples

### Example 1: Strategy Parameter Optimization
**Question**: "What are the best parameters for my momentum strategy?"

```yaml
# Fix strategy type, search parameter space
search:
  fixed: 
    component: "momentum_strategy"
  vary:
    fast_period: [5, 10, 15, 20]
    slow_period: [20, 30, 40, 50]
    classifiers: ["hmm", "pattern"]  
    risk_levels: ["conservative", "balanced"]

# Creates efficient hierarchy:
# MomentumStrategy(5,20) → HMM → Conservative
#                       → HMM → Balanced  
#                       → Pattern → Conservative
#                       → Pattern → Balanced
# MomentumStrategy(5,30) → ...
```

### Example 2: Regime Detector Comparison
**Question**: "Which classifier best identifies profitable regimes?"

```yaml
# Fix strategies and risk, compare classifiers
search:
  fixed:
    strategies: ["momentum", "mean_reversion"]
    risk_profile: "balanced"
  vary:
    classifiers: 
      - {type: "hmm", states: [2, 3, 4]}
      - {type: "pattern", lookback: [20, 50]}
      - {type: "ml_based", model: ["rf", "xgboost"]}
```

### Example 3: Multi-Phase Combinatorial Search

```yaml
# Phase 1: Find best strategies for each regime
phase1:
  fixed: {classifier: "hmm"}
  vary: {strategies: all}
  output: "best_strategies_per_regime"

# Phase 2: Optimize risk parameters for winners
phase2:
  fixed: {strategies: "phase1.best_strategies_per_regime"}
  vary: {risk_parameters: grid_search}
  output: "optimal_risk_params"

# Phase 3: Validate across different classifiers
phase3:
  fixed: 
    strategies: "phase1.best_strategies_per_regime"
    risk: "phase2.optimal_risk_params"
  vary: {classifiers: ["hmm", "pattern", "ml"]}
```

## Implementation Benefits

### 1. Computational Efficiency
```python
# Inefficient: Recreate everything for each combination
for combo in all_3200_combinations:
    container = create_everything_from_scratch(combo)  # Wasteful!

# Efficient: Reuse outer containers
classifier = create_once(HMMClassifier())
for risk in risk_profiles:
    risk_container = create_once(risk, parent=classifier)
    for strategy in strategies:
        # Only create what changes
        strategy_container = Container(strategy, parent=risk_container)
```

### 2. Clear Search Structure
- **What's fixed**: Outer containers (computed once)
- **What varies**: Inner containers (search space)
- **What's measured**: Performance metrics at each combination

### 3. Natural Parallelization
```python
# Each outer container can be processed independently
parallel_jobs = []
for outer_container in fixed_components:
    job = JobSpec(
        fixed=outer_container,
        search_space=inner_variations
    )
    parallel_jobs.append(job)

# Distribute across workers
ray.get([process_search.remote(job) for job in parallel_jobs])
```

## Configuration Examples

### Simple Two-Level Search
```yaml
# Question: How do my strategies perform across risk levels?
hierarchy: "strategy_first"
containers:
  - type: "strategy"
    name: "momentum"
    parameters: {fast: 10, slow: 30}
    children:
      - type: "risk"
        profiles: ["conservative", "balanced", "aggressive"]
```

### Complex Multi-Level Search
```yaml
# Question: Full combinatorial exploration
hierarchy: "custom"
search_space:
  level1: 
    type: "classifier"
    options: ["hmm", "pattern"]
  level2:
    type: "risk" 
    options: ["conservative", "balanced", "aggressive"]
  level3:
    type: "portfolio"
    options: ["equal_weight", "risk_parity", "kelly"]
  level4:
    type: "strategy"
    options: 
      - {type: "momentum", params: [...]}
      - {type: "mean_reversion", params: [...]}

# Automatically creates optimal hierarchy based on variation count
```

## Best Practices

### 1. Order by Variation Frequency
```yaml
# If you change strategies often but rarely change classifiers:
hierarchy:
  - classifiers    # Rarely change (outer)
  - risk_profiles  # Sometimes change
  - portfolios     # Change more often  
  - strategies     # Change frequently (inner)
```

### 2. Order by Computational Cost
```yaml
# If classifier is expensive but strategies are cheap:
hierarchy:
  - classifiers    # Expensive (compute once - outer)
  - strategies     # Cheap (can recompute - inner)
```

### 3. Order by Search Question
```yaml
# "How does strategy X perform?" → Strategy outer
# "What works in regime Y?" → Classifier outer  
# "Optimal risk for portfolio Z?" → Risk outer
```

## Migration from Old Concepts

### Old Documentation (Abstract)
"Compare organizational patterns for research into market microstructure differences between institutional hierarchies..."

### New Documentation (Practical)
"Organize containers hierarchically to efficiently search the combinatorial space of trading system configurations. Fix what you're testing, vary what you're optimizing."

### Old Mental Model
- Different "organizational philosophies"
- Abstract "research paradigms"
- Vague "institutional patterns"

### New Mental Model  
- Combinatorial search optimization
- Hierarchical container efficiency
- Invertible based on search question

## Summary

The container organization in ADMF-PC is about:

1. **Minimizing container count** through smart hierarchies
2. **Enabling efficient combinatorial search** by fixing outer components
3. **Inverting hierarchies** based on what question you're asking
4. **Reusing expensive computations** in outer containers

It's not about abstract "organizational patterns" - it's about practical computational efficiency and clear search structure for finding optimal trading system configurations.
