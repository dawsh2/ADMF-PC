# Declarative vs Code: What Can and Can't Be Done

## What We CAN Do Declaratively

### 1. Complex Multi-Phase Workflows ✅
```yaml
phases:
  - name: optimization
    depends_on: exploration
    conditions:
      - metric_threshold:
          phase: exploration
          metric: best_sharpe
          operator: ">"
          threshold: 1.0
```

### 2. Dynamic Value Resolution ✅
```yaml
config:
  parameters: "{optimization.outputs.best_params}"
  start_date: "{phase1.end_date + 1}"
  weights: "{calculate: params / sum(params)}"
```

### 3. Conditional Execution ✅
```yaml
conditions:
  - type: expression
    expression: "results['phase1']['sharpe_ratio'] > 1.5 and results['phase1']['max_drawdown'] < 0.2"
```

### 4. Input/Output Data Flow ✅
```yaml
inputs:
  signals: "{phase1.outputs.signal_files}"
  config: "./config/production.yaml"
outputs:
  report: "./results/final_report.pdf"
  metrics: "./results/metrics.json"
```

### 5. Parameter Sweeps and Iterations ✅
```yaml
iterations:
  type: parameter_grid
  parameters:
    threshold: [0.01, 0.02, 0.03]
    window: [10, 20, 30]
```

### 6. Aggregation and Statistics ✅
```yaml
aggregation:
  type: statistical
  operations: [mean, std, percentiles]
  group_by: regime
```

### 7. Pattern Composition ✅
```yaml
topology: backtest       # Topology pattern
sequence: walk_forward   # Sequence pattern
workflow: research       # Workflow pattern
```

### 8. File I/O and Persistence ✅
```yaml
outputs:
  trades: "./results/trades.csv"
  metrics: "./results/metrics.json"
  plots: "./results/charts/*.png"
```

### 9. Sub-phase Dependencies ✅
```yaml
sub_phases:
  - name: train
  - name: test
    depends_on: train
    config:
      params: "{train.optimal_parameters}"
```

### 10. Custom Behaviors via Plugins ✅
```yaml
behaviors:
  - type: custom
    handler: my_custom_analyzer
    config:
      specific_params: value
```

## What We CAN'T Do Declaratively (Without Extensions)

### 1. Complex Mathematical Operations ❌
While we can do simple calculations, complex math requires code:
```python
# Can't express this declaratively:
def kelly_criterion(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.var(excess_returns)
```

**Solution**: Register custom functions that can be called from YAML
```yaml
config:
  position_size:
    type: function
    name: kelly_criterion
    params: [returns, risk_free_rate]
```

### 2. Custom Strategy Logic ❌
Trading strategies still need code:
```python
# Can't express strategy logic in YAML:
class MyComplexStrategy:
    def generate_signal(self, data):
        # Complex logic with multiple conditions
        if self.detect_pattern(data) and self.confirm_trend(data):
            return Signal(...)
```

**Solution**: Strategies remain as plugins, YAML configures them

### 3. Real-time Data Processing ❌
Streaming/real-time processing needs code:
```python
# Can't handle streaming declaratively:
async def process_tick(tick):
    update_orderbook(tick)
    check_signals()
    manage_positions()
```

**Solution**: Create streaming topology pattern with handlers

### 4. Complex Event Processing ❌
Advanced event correlation needs code:
```python
# Complex event patterns:
if (event_a and event_b within 5 seconds) or event_c:
    trigger_action()
```

**Solution**: Event processing patterns with rule engine

### 5. Custom Visualizations ❌
While we can specify what to plot, rendering needs code:
```python
# Can't create custom plots declaratively:
fig = go.Figure()
fig.add_trace(go.Candlestick(...))
fig.add_trace(go.Scatter(...))
```

**Solution**: Visualization templates referenced from YAML

## The 80/20 Rule

The declarative approach handles ~80% of use cases:
- ✅ Standard backtesting workflows
- ✅ Parameter optimization
- ✅ Walk-forward analysis
- ✅ Monte Carlo simulations
- ✅ Multi-phase workflows
- ✅ Signal generation and replay
- ✅ Risk analysis
- ✅ Report generation

The remaining 20% requiring code:
- ❌ Novel strategy development
- ❌ Custom indicators
- ❌ Advanced mathematics
- ❌ Real-time trading
- ❌ Custom visualizations

## Best Practices

### 1. Use Declarative When Possible
```yaml
# Good: Declarative workflow
workflow: adaptive_ensemble
phases:
  - topology: signal_generation
    sequence: walk_forward
```

### 2. Extend with Plugins When Needed
```python
# Custom function registered for use in YAML
@register_function("advanced_sharpe")
def advanced_sharpe_ratio(returns, risk_free_rate):
    # Complex calculation
    return result
```

### 3. Keep Business Logic Separate
```yaml
# YAML defines WHAT and WHEN
phases:
  - name: optimization
    when: market_is_stable
    
# Code defines HOW
def market_is_stable():
    # Implementation
```

## Conclusion

The declarative approach can handle the vast majority of trading system workflows. The key limitations are around:

1. **Custom algorithms** - Still need code
2. **Real-time processing** - Requires event loops
3. **Complex math** - Beyond simple expressions
4. **Novel visualizations** - Need rendering code

However, these can be addressed through:
- Plugin system for custom functions
- Template system for visualizations  
- Handler registration for real-time
- Expression engine for complex conditions

The result is a system where:
- **Users** work entirely in YAML
- **Developers** extend capabilities via plugins
- **Researchers** share patterns not code
- **Systems** are reproducible and versionable