# CLI Piping Benefits & Use Cases

## Why Piping for Trading System Workflows?

### 1. **Natural Workflow Expression**
```bash
# Reads like English: generate signals, optimize ensemble, validate
python main.py --signal-generation grid.yaml | \
python main.py --signal-replay --optimize-ensemble | \
python main.py --backtest --validate
```

### 2. **Automatic Parameter Inheritance**
When you pipe phases together, later phases automatically inherit:
- Strategy configurations
- Feature definitions
- Data parameters
- Artifact paths (signals, weights, etc.)

No need to repeatedly specify the same parameters!

### 3. **Composability**
Mix and match phases as needed:
```bash
# Try different ensemble methods on same signals
cat saved_signals.json | python main.py --signal-replay --optimize-ensemble --method mean-variance
cat saved_signals.json | python main.py --signal-replay --optimize-ensemble --method max-sharpe
cat saved_signals.json | python main.py --signal-replay --optimize-ensemble --method risk-parity
```

### 4. **Parallel Processing**
Leverage Unix tools for natural parallelization:
```bash
# Run 10 different parameter sets in parallel
parallel -j 10 'python main.py --signal-generation {} --output-format pipe' \
  ::: params/*.yaml | \
python main.py --signal-replay --from-pipe --merge --optimize-ensemble
```

### 5. **Debugging & Inspection**
Save intermediate results for debugging:
```bash
# Save each phase output
python main.py --signal-generation grid.yaml | tee phase1.json | \
python main.py --signal-replay --from-pipe | tee phase2.json | \
python main.py --backtest --from-pipe > final.json

# Inspect what went wrong
jq '.results' phase1.json
jq '.artifacts.signals' phase1.json
```

## Real-World Use Cases

### Use Case 1: Regime-Adaptive System Development
```bash
#!/bin/bash
# Build a complete regime-adaptive trading system

# 1. Generate signals across parameter grid
python main.py --signal-generation regime_grid.yaml --output-format pipe | \

# 2. Find optimal ensemble weights per market regime  
python main.py --signal-replay --from-pipe --optimize-ensemble --by-regime | \

# 3. Validate on out-of-sample data
python main.py --backtest --from-pipe --validate --start-date 2024-01-01
```

### Use Case 2: Strategy Research Pipeline
```bash
# Research pipeline with conditional execution

# Generate signals and check quality
python main.py --signal-generation research.yaml --output-format pipe | \
tee signals.json | \

# Only optimize if we have enough high-quality signals
jq -e '.results.sharpe_ratio > 1.0 and .results.total_signals > 1000' && \
python main.py --signal-replay --from-pipe --optimize-ensemble || \
echo "Signal quality too low, aborting optimization"
```

### Use Case 3: Incremental Development
```bash
# Develop iteratively, reusing expensive computations

# Step 1: Generate signals (expensive, ~1 hour)
python main.py --signal-generation full_backtest.yaml \
  --output-format json > signals_v1.json

# Step 2: Try different ensemble methods (cheap, ~1 minute each)
for method in mean_variance max_sharpe risk_parity; do
  cat signals_v1.json | \
  python main.py --signal-replay - --optimize-ensemble --method $method \
    --output-format json > ensemble_${method}.json
done

# Step 3: Compare results
python compare_ensembles.py ensemble_*.json
```

### Use Case 4: Production Pipeline
```bash
# Production workflow with monitoring and error handling

set -eo pipefail  # Exit on error

# Run with monitoring
python main.py --signal-generation prod_config.yaml --output-format pipe | \
pv -l -N "Signal Generation" | \  # Monitor progress
python main.py --signal-replay --from-pipe --optimize-ensemble | \
pv -l -N "Optimization" | \
python main.py --backtest --from-pipe --validate || {
  # Error handling
  echo "Pipeline failed!" | mail -s "Trading System Error" ops@company.com
  exit 1
}

# Success notification
echo "Pipeline completed successfully" | \
mail -s "Trading System Update" team@company.com
```

### Use Case 5: A/B Testing Strategies
```bash
# Compare two strategy configurations

# Version A
python main.py --signal-generation strategy_v1.yaml | \
python main.py --signal-replay --from-pipe --optimize | \
python main.py --backtest --from-pipe > results_v1.json

# Version B  
python main.py --signal-generation strategy_v2.yaml | \
python main.py --signal-replay --from-pipe --optimize | \
python main.py --backtest --from-pipe > results_v2.json

# Compare
python compare_strategies.py results_v1.json results_v2.json
```

## Integration with Unix Tools

### 1. **jq - JSON Processing**
```bash
# Extract specific metrics
python main.py --signal-generation config.yaml --output-format json | \
jq '.results.total_signals'

# Filter based on results
python main.py --backtest config.yaml --output-format json | \
jq 'select(.results.sharpe_ratio > 1.5)'
```

### 2. **GNU Parallel**
```bash
# Process multiple configs in parallel
parallel --tag 'python main.py --signal-generation {} --output-format json' \
  ::: configs/*.yaml | \
jq -s 'max_by(.results.sharpe_ratio)'
```

### 3. **tee - Save Intermediate Results**
```bash
# Save while piping
python main.py --signal-generation config.yaml | \
tee >(gzip > signals.json.gz) | \
python main.py --signal-replay --from-pipe
```

### 4. **pv - Progress Monitoring**
```bash
# Monitor data flow
python main.py --signal-generation large_config.yaml | \
pv -l -s 1000000 | \  # Expect ~1M lines
python main.py --signal-replay --from-pipe
```

## Best Practices

### 1. **Use Meaningful Phase Names**
```bash
# Good: Clear what each phase does
--signal-generation
--signal-replay --optimize-ensemble
--backtest --validate

# Bad: Generic names
--phase1
--phase2  
--phase3
```

### 2. **Save Intermediate Results**
```bash
# Always tee important intermediate results
python main.py --signal-generation expensive.yaml | \
tee signals_$(date +%Y%m%d).json | \
python main.py --signal-replay --from-pipe
```

### 3. **Add Progress Indicators**
```bash
# For long-running phases
python main.py --signal-generation grid.yaml --progress | \
python main.py --signal-replay --from-pipe --progress
```

### 4. **Handle Errors Gracefully**
```bash
# Use pipefail and error checking
set -eo pipefail

python main.py --signal-generation config.yaml || {
  echo "Signal generation failed"
  exit 1
}
```

## Summary

CLI piping brings several key advantages:

1. **Intuitive** - Workflows read naturally
2. **Composable** - Mix and match phases easily  
3. **Efficient** - Reuse expensive computations
4. **Debuggable** - Inspect intermediate results
5. **Parallelizable** - Natural parallel processing
6. **Integrable** - Works with Unix ecosystem

This makes complex trading system development more accessible and efficient!