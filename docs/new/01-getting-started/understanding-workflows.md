# Understanding Workflows

Workflows are the heart of ADMF-PC's power. They allow you to compose complex trading system operations from simple building blocks - all through configuration.

## 🧩 What Are Workflows?

A workflow is a sequence of operations that transform market data into trading decisions and performance metrics. Think of workflows as recipes that combine ingredients (components) in specific ways to achieve your goals.

### Simple Workflow: Backtest
```
Load Data → Run Strategy → Execute Trades → Report Results
```

### Complex Workflow: Walk-Forward Optimization
```
Split Data → Optimize Parameters → Validate Performance → Move Window → Repeat
```

## 🏗️ Workflow Building Blocks

ADMF-PC provides four fundamental building blocks:

### 1. **Backtest**
- Simulates trading with historical data
- Generates performance metrics
- Can capture signals for later analysis

### 2. **Optimization**  
- Searches for best parameters
- Supports grid, random, and Bayesian search
- Can optimize across multiple objectives

### 3. **Analysis**
- Statistical validation
- Performance attribution  
- Regime analysis

### 4. **Validation**
- Out-of-sample testing
- Walk-forward analysis
- Monte Carlo simulation

## 🔄 How Workflows Execute

### The Coordinator's Role

The Coordinator acts as the workflow conductor:

1. **Parses Configuration**: Understands what you want to achieve
2. **Creates Containers**: Spins up isolated environments for each phase
3. **Manages Data Flow**: Passes results between phases
4. **Ensures Reproducibility**: Controls random seeds and execution order

### Execution Flow Example

Let's trace through a simple optimization workflow:

```yaml
workflow:
  type: "optimization"
  phases:
    - name: "parameter_search"
      type: "optimization"
      # ... configuration ...
      
    - name: "validation"  
      type: "backtest"
      # ... configuration ...
```

The Coordinator:
1. Creates an OptimizationContainer for phase 1
2. Runs parameter search, collecting results
3. Stores results in PhaseDataStore
4. Creates a BacktestContainer for phase 2  
5. Passes best parameters to validation
6. Runs validation backtest
7. Aggregates final results

## 📊 Common Workflow Patterns

### 1. Simple Backtest
The most basic workflow - just run a strategy:

```yaml
workflow:
  type: "backtest"
  
# Rest of configuration...
```

### 2. Parameter Optimization
Find the best parameters for your strategy:

```yaml
workflow:
  type: "optimization"
  
optimization:
  method: "grid"
  parameters:
    fast_period: [5, 10, 20]
    slow_period: [20, 50, 100]
  objective: "sharpe_ratio"
  
# Rest of configuration...
```

### 3. Multi-Phase Workflow
Complex workflow with multiple steps:

```yaml
workflow:
  type: "multi_phase"
  phases:
    - name: "initial_optimization"
      type: "optimization"
      config:
        method: "random"
        n_trials: 1000
        
    - name: "regime_analysis"
      type: "analysis"
      config:
        analyzer: "regime_detection"
        
    - name: "regime_specific_optimization"
      type: "optimization"
      config:
        method: "grid"
        group_by: "regime"
        
    - name: "final_validation"
      type: "backtest"
      config:
        use_best_params: true
```

### 4. Walk-Forward Analysis
Rolling window optimization and validation:

```yaml
workflow:
  type: "walk_forward"
  
walk_forward:
  train_period_days: 252
  test_period_days: 63
  step_days: 63
  
# Rest of configuration...
```

## 🚀 Workflow Advantages

### 1. **Composability**
Build complex workflows from simple pieces:
- No custom code needed
- Reusable components
- Clear, declarative configuration

### 2. **Reproducibility**
Every workflow execution is identical:
- Controlled random seeds
- Deterministic execution order
- Versioned configurations

### 3. **Scalability**
Workflows scale efficiently:
- Parallel execution of independent phases
- Signal replay for 10-100x speedup
- Container pooling for resource efficiency

### 4. **Flexibility**
Mix and match components:
- Any strategy type
- Any optimization method
- Any analysis technique

## 📝 Workflow Configuration

### Basic Structure

Every workflow configuration has:

```yaml
workflow:
  type: "workflow_type"        # backtest, optimization, multi_phase, etc.
  name: "descriptive_name"     # For identification
  description: "explanation"   # What this workflow does
  
# Workflow-specific configuration
# ... 
```

### Phase Management

For multi-phase workflows:

```yaml
phases:
  - name: "phase_1"
    type: "optimization"
    config:
      # Phase-specific config
    outputs:
      - best_params
      - performance_metrics
      
  - name: "phase_2"  
    type: "backtest"
    inputs:
      parameters: "${phase_1.best_params}"
```

### Data Flow

Results flow between phases automatically:
- Phase outputs are stored in PhaseDataStore
- Later phases can reference earlier results
- Use template syntax: `${phase_name.output_name}`

## 🎯 Choosing the Right Workflow

| Goal | Workflow Type | Key Features |
|------|--------------|--------------|
| Test a strategy | `backtest` | Simple, fast, straightforward |
| Find best parameters | `optimization` | Grid/random/Bayesian search |
| Validate robustness | `walk_forward` | Rolling window validation |
| Complex analysis | `multi_phase` | Flexible composition |
| Regime adaptation | `multi_phase` | Regime detection + optimization |
| Production prep | `multi_phase` | Optimize → validate → analyze |

## 🔧 Workflow Best Practices

### 1. Start Simple
Begin with basic backtest, then add complexity:
```
Backtest → Optimization → Walk-Forward → Multi-Phase
```

### 2. Use Signal Capture
For optimization workflows, capture signals first:
```yaml
capture_signals: true
signal_replay: true  # 10-100x faster optimization
```

### 3. Validate Everything
Always include out-of-sample validation:
```yaml
phases:
  - name: "optimize"
    data_split: "train"  # 80%
  - name: "validate"  
    data_split: "test"   # 20%
```

### 4. Monitor Resources
Set resource limits for large workflows:
```yaml
infrastructure:
  max_workers: 8
  memory_limit_gb: 16
  enable_monitoring: true
```

## 🎓 Learning Exercises

### Exercise 1: Modify Simple Backtest
Take the simple backtest and:
1. Change it to use 2 years of data
2. Add a second strategy
3. Modify risk parameters

### Exercise 2: Create Optimization Workflow
Convert the simple backtest to optimization:
1. Define parameter ranges
2. Set optimization objective
3. Add validation phase

### Exercise 3: Build Multi-Phase Workflow
Create a workflow that:
1. Optimizes parameters
2. Analyzes regime performance
3. Validates results

## 🤔 Common Questions

**Q: Can I create custom workflow types?**  
A: Yes! See [Custom Workflows](../08-advanced-topics/custom-workflows.md) for advanced patterns.

**Q: How do I debug workflows?**  
A: Enable verbose logging and use the workflow visualization tools. See [Debugging Guide](troubleshooting.md).

**Q: What's the performance impact of multi-phase?**  
A: Minimal! The Coordinator efficiently manages resources, and signal replay dramatically speeds up subsequent phases.

**Q: Can workflows be resumed if interrupted?**  
A: Yes! The checkpoint system allows resuming from any completed phase.

## 🎯 Key Takeaways

1. **Workflows = Composable Operations**: Build complex from simple
2. **Coordinator = Conductor**: Orchestrates all workflow execution
3. **Phases = Building Blocks**: Each phase is an isolated operation
4. **Configuration = Complete Specification**: No code needed
5. **Results Flow Automatically**: Phase outputs become next phase inputs

---

Ready to dive deeper? Continue to [Core Concepts](../02-core-concepts/README.md) →