# Workflow Composition

Workflow composition is ADMF-PC's approach to building sophisticated trading operations from simple, reusable building blocks. This enables creating complex institutional-grade workflows through configuration alone, without any custom programming.

## 🧩 The Building Block Philosophy

Instead of writing custom code for each workflow, ADMF-PC provides four fundamental building blocks that can be composed infinitely:

### Core Building Blocks

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  BACKTEST   │  │OPTIMIZATION │  │  ANALYSIS   │  │ VALIDATION  │
│             │  │             │  │             │  │             │
│ • Strategy  │  │ • Parameter │  │ • Statistical│  │ • Out-of-   │
│   execution │  │   search    │  │   analysis  │  │   sample    │
│ • Risk mgmt │  │ • Objective │  │ • Regime    │  │   testing   │
│ • Portfolio │  │   optimization│  │   detection │  │ • Walk-     │
│   tracking  │  │ • Constraint│  │ • Performance│  │   forward   │
│ • Reporting │  │   handling  │  │   attribution│  │ • Monte     │
│             │  │             │  │             │  │   Carlo     │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

**Key Insight**: Complex workflows emerge from composing these simple blocks in different patterns - no new code required!

## 🔄 Composition Patterns

### 1. Sequential Composition (Pipeline)

Operations execute one after another, with outputs feeding inputs:

```yaml
# Research workflow: Optimize → Analyze → Validate
workflow:
  type: "sequential"
  phases:
    - name: "parameter_optimization"
      type: "optimization"
      config:
        method: "grid"
        parameters:
          fast_period: [5, 10, 15, 20]
          slow_period: [20, 30, 40, 50]
        objective: "sharpe_ratio"
        
    - name: "performance_analysis"
      type: "analysis"
      inputs: ["parameter_optimization.best_params"]
      config:
        analyzers: ["drawdown", "regime_performance"]
        
    - name: "out_of_sample_validation"
      type: "backtest"
      inputs: ["parameter_optimization.best_params"]
      config:
        data_split: "test"  # Uses different data than optimization
```

**Execution Flow**:
```
Optimize Parameters → Analyze Results → Validate Performance
      ↓                    ↓                    ↓
  Best Parameters    Performance Metrics   Final Validation
```

### 2. Parallel Composition (Branching)

Operations execute simultaneously on different data or parameters:

```yaml
# Multi-strategy development: Test different approaches in parallel
workflow:
  type: "parallel"
  branches:
    momentum_branch:
      type: "optimization"
      config:
        strategy_type: "momentum"
        parameters:
          fast_period: [5, 10, 20]
          slow_period: [20, 50, 100]
          
    mean_reversion_branch:
      type: "optimization"
      config:
        strategy_type: "mean_reversion"
        parameters:
          lookback: [10, 20, 30]
          threshold: [1.5, 2.0, 2.5]
          
    ml_branch:
      type: "optimization"
      config:
        strategy_type: "sklearn_model"
        parameters:
          model_type: ["random_forest", "svm", "gradient_boosting"]
          
  aggregation:
    type: "ensemble_optimization"
    inputs: ["momentum_branch.results", "mean_reversion_branch.results", "ml_branch.results"]
```

**Execution Flow**:
```
                    ┌─ Momentum Optimization ─┐
Data Preparation ──├─ Mean Reversion Opt ────┼─→ Ensemble Optimization
                    └─ ML Model Optimization ─┘
```

### 3. Conditional Composition (Adaptive)

Operations execute based on conditions or intermediate results:

```yaml
# Adaptive workflow: Different paths based on market conditions
workflow:
  type: "conditional"
  
  initial_phase:
    name: "market_analysis"
    type: "analysis"
    config:
      analyzers: ["regime_detection", "volatility_analysis"]
      
  conditions:
    - condition: "market_analysis.regime == 'trending'"
      workflow:
        type: "optimization"
        config:
          strategy_types: ["momentum", "breakout"]
          
    - condition: "market_analysis.regime == 'sideways'"
      workflow:
        type: "optimization"
        config:
          strategy_types: ["mean_reversion", "pairs_trading"]
          
    - condition: "market_analysis.volatility > 0.3"
      workflow:
        type: "analysis"
        config:
          analyzers: ["risk_assessment"]
        next:
          type: "optimization"
          config:
            risk_constraints:
              max_volatility: 0.2
```

### 4. Iterative Composition (Refinement)

Operations repeat with refinement based on previous results:

```yaml
# Iterative improvement: Refine strategy through multiple iterations
workflow:
  type: "iterative"
  max_iterations: 5
  convergence_threshold: 0.01
  
  iteration_workflow:
    - name: "parameter_optimization"
      type: "optimization"
      config:
        method: "bayesian"  # More efficient for iterative refinement
        n_trials: 100
        
    - name: "performance_evaluation"
      type: "analysis"
      inputs: ["parameter_optimization.best_params"]
      
    - name: "strategy_refinement"
      type: "optimization"
      inputs: ["performance_evaluation.bottlenecks"]
      config:
        focus_areas: "${performance_evaluation.improvement_areas}"
        
  convergence_check:
    metric: "sharpe_ratio"
    improvement_threshold: 0.01
```

## 🏗️ Advanced Composition Examples

### Institutional Research Workflow

```yaml
# Complete research workflow for strategy development
workflow:
  type: "multi_phase"
  name: "institutional_strategy_research"
  
  phases:
    # Phase 1: Initial Discovery
    - name: "broad_parameter_search"
      type: "optimization"
      config:
        method: "random"
        n_trials: 10000
        parameters:
          # Broad parameter ranges
          fast_period: [3, 50]
          slow_period: [10, 200]
          signal_threshold: [0.001, 0.1]
      container_count: 1000  # Massive parallel search
      
    # Phase 2: Regime Analysis
    - name: "regime_detection"
      type: "analysis"
      inputs: ["broad_parameter_search.top_100_results"]
      config:
        analyzers: ["hmm_regime", "volatility_regime", "trend_regime"]
        
    # Phase 3: Regime-Specific Optimization
    - name: "regime_specific_optimization"
      type: "optimization"
      inputs: ["regime_detection.regimes"]
      config:
        method: "grid"
        group_by_regime: true
        parameters:
          # Refined parameter ranges based on initial search
          fast_period: ["${broad_parameter_search.best_fast_range}"]
          slow_period: ["${broad_parameter_search.best_slow_range}"]
          
    # Phase 4: Ensemble Weight Optimization
    - name: "ensemble_optimization"
      type: "optimization"
      method: "signal_replay"  # 10x faster for ensemble weights
      inputs: ["regime_specific_optimization.regime_strategies"]
      config:
        optimization_target: "ensemble_weights"
        constraints:
          min_weight: 0.1
          max_weight: 0.4
          sum_weights: 1.0
          
    # Phase 5: Walk-Forward Validation
    - name: "walk_forward_validation"
      type: "validation"
      inputs: ["ensemble_optimization.final_ensemble"]
      config:
        method: "walk_forward"
        train_period_days: 252
        test_period_days: 63
        step_days: 21
        
    # Phase 6: Risk Analysis
    - name: "risk_analysis"
      type: "analysis"
      inputs: ["walk_forward_validation.results"]
      config:
        analyzers: ["drawdown_analysis", "var_analysis", "stress_testing"]
        
    # Phase 7: Final Strategy Construction
    - name: "strategy_construction"
      type: "backtest"
      inputs: [
        "ensemble_optimization.final_ensemble",
        "risk_analysis.risk_limits"
      ]
      config:
        include_transaction_costs: true
        include_slippage: true
        generate_tearsheet: true
```

### Production Deployment Workflow

```yaml
# Workflow for deploying strategy to production
workflow:
  type: "production_deployment"
  
  phases:
    # Pre-deployment validation
    - name: "final_validation"
      type: "backtest"
      config:
        data_source: "latest_data"  # Most recent data
        include_recent_regimes: true
        stress_test_scenarios: ["covid_crash", "rate_hikes", "tech_selloff"]
        
    # Shadow trading phase
    - name: "shadow_trading"
      type: "live_simulation"
      inputs: ["final_validation.validated_strategy"]
      config:
        duration_days: 30
        paper_trading: true
        real_time_data: true
        alert_thresholds:
          max_drawdown: 0.05
          min_sharpe: 1.0
          
    # Gradual capital allocation
    - name: "gradual_deployment"
      type: "live_trading"
      inputs: ["shadow_trading.performance_metrics"]
      config:
        initial_capital_pct: 0.1  # Start with 10%
        scale_up_schedule:
          - weeks: 2, capital_pct: 0.25
          - weeks: 4, capital_pct: 0.5
          - weeks: 8, capital_pct: 1.0
        auto_scale_conditions:
          min_sharpe: 1.5
          max_drawdown: 0.03
```

## 📊 Phase Data Management

### Automatic Data Flow

The Coordinator automatically manages data flow between phases:

```python
class PhaseDataManager:
    """Manages data flow between workflow phases"""
    
    def __init__(self):
        self.phase_outputs = {}
        self.data_transformers = {}
        
    def register_phase_output(self, phase_name: str, output_data: Dict):
        """Register output data from completed phase"""
        self.phase_outputs[phase_name] = {
            'data': output_data,
            'timestamp': datetime.now(),
            'schema_version': '1.0'
        }
        
    def get_phase_input(self, phase_name: str, input_spec: str) -> Any:
        """Get input data for phase using reference syntax"""
        
        # Parse input specification: "phase_name.output_key"
        if '.' in input_spec:
            source_phase, output_key = input_spec.split('.', 1)
        else:
            source_phase = input_spec
            output_key = 'default'
            
        # Retrieve data
        if source_phase not in self.phase_outputs:
            raise ValueError(f"Phase {source_phase} output not available")
            
        phase_data = self.phase_outputs[source_phase]['data']
        
        # Navigate nested keys
        for key in output_key.split('.'):
            if isinstance(phase_data, dict) and key in phase_data:
                phase_data = phase_data[key]
            else:
                raise ValueError(f"Output key {output_key} not found in {source_phase}")
                
        return phase_data
        
    def transform_data(self, data: Any, transformer_name: str) -> Any:
        """Apply data transformation between phases"""
        if transformer_name in self.data_transformers:
            transformer = self.data_transformers[transformer_name]
            return transformer(data)
        return data
```

### Data Serialization and Storage

```python
class PhaseDataStorage:
    """Efficient storage and retrieval of phase data"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_phase_data(self, phase_name: str, data: Any, format: str = "pickle"):
        """Save phase data to disk"""
        
        file_path = self.storage_path / f"{phase_name}.{format}"
        
        if format == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "parquet":
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path)
            else:
                raise ValueError("Parquet format requires DataFrame")
                
    def load_phase_data(self, phase_name: str, format: str = "pickle") -> Any:
        """Load phase data from disk"""
        
        file_path = self.storage_path / f"{phase_name}.{format}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Phase data file not found: {file_path}")
            
        if format == "pickle":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif format == "json":
            with open(file_path, 'r') as f:
                return json.load(f)
        elif format == "parquet":
            return pd.read_parquet(file_path)
```

## 🎛️ Workflow Configuration Patterns

### Template-Based Configuration

```yaml
# Workflow template with parameterization
workflow_template: &optimization_template
  type: "optimization"
  config:
    method: "grid"
    objective: "sharpe_ratio"
    constraints:
      max_drawdown: 0.15
      min_trades: 50

# Use template with different parameters
workflows:
  momentum_optimization:
    <<: *optimization_template
    config:
      parameters:
        fast_period: [5, 10, 20]
        slow_period: [20, 50, 100]
        
  mean_reversion_optimization:
    <<: *optimization_template
    config:
      parameters:
        lookback: [10, 20, 30]
        threshold: [1.5, 2.0, 2.5]
```

### Environment-Specific Workflows

```yaml
# Different workflows for different environments
environments:
  research: &research_env
    max_containers: 1000
    timeout_hours: 24
    checkpointing: true
    
  production: &prod_env
    max_containers: 10
    timeout_hours: 1
    checkpointing: false
    
workflows:
  strategy_research:
    environment: *research_env
    phases:
      - name: "broad_search"
        container_count: 1000
        
  strategy_validation:
    environment: *prod_env
    phases:
      - name: "final_check"
        container_count: 1
```

## 🚀 Performance Optimization

### Signal Replay for Speed

```yaml
# Use signal replay for 10-100x speedup in optimization phases
workflow:
  type: "multi_phase"
  
  phases:
    # Phase 1: Generate signals (slower but necessary)
    - name: "signal_generation"
      type: "signal_generation"
      config:
        strategy_types: ["momentum", "mean_reversion"]
        capture_signals: true
        output_path: "signals/strategy_signals.pkl"
        
    # Phase 2: Fast parameter optimization using captured signals
    - name: "parameter_optimization"
      type: "optimization"
      method: "signal_replay"  # 10-100x faster
      inputs: ["signal_generation.signals"]
      config:
        parameters:
          signal_threshold: [0.01, 0.02, 0.05, 0.1]
          position_size: [0.01, 0.02, 0.05]
          stop_loss: [0.01, 0.02, 0.03]
```

### Parallel Execution

```yaml
# Maximize parallelization for independent operations
workflow:
  type: "parallel"
  
  infrastructure:
    max_workers: 32
    memory_per_worker: "2GB"
    load_balancing: "dynamic"
    
  branches:
    # Each branch runs independently
    short_term_strategies:
      container_count: 10
      timeframe: "1m"
      
    medium_term_strategies:
      container_count: 10
      timeframe: "5m"
      
    long_term_strategies:
      container_count: 10
      timeframe: "1h"
```

## 🔍 Workflow Monitoring

### Real-Time Progress Tracking

```python
class WorkflowMonitor:
    """Monitors workflow execution progress"""
    
    def __init__(self):
        self.phase_progress = {}
        self.estimated_completion = {}
        
    def update_phase_progress(self, phase_name: str, progress_pct: float):
        """Update progress for specific phase"""
        self.phase_progress[phase_name] = {
            'progress': progress_pct,
            'timestamp': datetime.now(),
            'estimated_completion': self.estimate_completion_time(phase_name, progress_pct)
        }
        
    def get_overall_progress(self) -> Dict:
        """Get overall workflow progress"""
        total_phases = len(self.phase_progress)
        completed_phases = sum(1 for p in self.phase_progress.values() if p['progress'] >= 100)
        
        overall_progress = (completed_phases / total_phases * 100) if total_phases > 0 else 0
        
        return {
            'overall_progress': overall_progress,
            'completed_phases': completed_phases,
            'total_phases': total_phases,
            'phase_details': self.phase_progress
        }
```

## 🎯 Benefits of Workflow Composition

### 1. **Zero-Code Complexity**
Build sophisticated workflows without programming:
- Research pipelines
- Production deployment
- Risk management validation
- Performance optimization

### 2. **Infinite Flexibility**
Compose building blocks in unlimited ways:
- Sequential workflows
- Parallel processing
- Conditional logic
- Iterative refinement

### 3. **Automatic Optimization**
Built-in performance optimization:
- Signal replay for speed
- Parallel execution
- Resource management
- Progress tracking

### 4. **Production Ready**
Enterprise-grade workflow management:
- Fault tolerance
- Checkpointing and resume
- Monitoring and alerting
- Audit trails

## 🤔 Common Questions

**Q: Can I create custom workflow types?**
A: Yes! While the four core building blocks handle most cases, you can define custom workflow patterns through configuration and composition.

**Q: How do I debug complex workflows?**
A: Use phase-by-phase execution, detailed logging, and the workflow visualization tools to understand data flow and identify issues.

**Q: What's the performance impact of multi-phase workflows?**
A: Minimal! The Coordinator efficiently manages resources, and signal replay dramatically speeds up subsequent phases.

**Q: Can workflows be modified during execution?**
A: Not during execution, but you can checkpoint workflows and resume with modified configurations.

## 🎯 Key Takeaways

1. **Building Blocks > Custom Code**: Four blocks compose infinitely
2. **Configuration > Programming**: Complex workflows through YAML
3. **Automatic Data Flow**: Phase outputs become next phase inputs
4. **Performance Optimized**: Signal replay and parallel execution
5. **Production Ready**: Enterprise-grade workflow management

Workflow composition enables building institutional-grade trading operations through simple configuration, making complex research and deployment workflows accessible without programming.

---

Next: [Isolation Benefits](isolation-benefits.md) - Why isolated event buses are revolutionary →