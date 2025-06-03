# Multi-Phase Workflows

Multi-phase workflows enable building sophisticated research and deployment pipelines by composing simple operations into complex, automated sequences. This is where ADMF-PC's true power emerges.

## 🧩 Understanding Multi-Phase Workflows

Multi-phase workflows break complex operations into sequential phases:

```
Phase 1: Data Preparation → Phase 2: Parameter Search → Phase 3: Regime Analysis → Phase 4: Ensemble Building → Phase 5: Validation
```

Each phase:
- Operates on outputs from previous phases
- Runs in isolated containers for reliability
- Can use different optimization methods and resources
- Automatically manages data flow between phases

## 🚀 Basic Multi-Phase Workflow

### Simple Research Pipeline

```yaml
# config/basic_multi_phase.yaml
workflow:
  type: "multi_phase"
  name: "Basic Research Pipeline"
  description: "Optimize → Analyze → Validate workflow"
  
data:
  symbols: ["SPY"]
  start_date: "2022-01-01"
  end_date: "2023-12-31"
  
# Phase definitions
phases:
  # Phase 1: Parameter Optimization
  - name: "parameter_optimization"
    type: "optimization"
    config:
      strategies:
        - type: "momentum"
          optimization_target: true
          
      optimization:
        method: "grid"
        parameters:
          fast_period: [5, 10, 15, 20]
          slow_period: [20, 30, 40, 50]
          signal_threshold: [0.01, 0.02, 0.05]
        objective: "sharpe_ratio"
        
    # Phase outputs
    outputs:
      - best_parameters
      - top_10_parameter_sets
      - optimization_history
      
  # Phase 2: Performance Analysis
  - name: "performance_analysis"
    type: "analysis"
    inputs: ["parameter_optimization.best_parameters"]
    config:
      analyzers:
        - "drawdown_analysis"
        - "regime_performance"
        - "correlation_analysis"
        - "monte_carlo_simulation"
        
    outputs:
      - performance_metrics
      - regime_breakdown
      - risk_assessment
      
  # Phase 3: Out-of-Sample Validation
  - name: "validation"
    type: "backtest"
    inputs: ["parameter_optimization.best_parameters"]
    config:
      data_split: "test"              # Use reserved test data
      include_transaction_costs: true
      generate_detailed_report: true
      
    outputs:
      - validation_results
      - final_performance_report

# Resource management across phases
infrastructure:
  phase_resources:
    parameter_optimization:
      max_workers: 16
      memory_limit_gb: 32
      
    performance_analysis:
      max_workers: 4
      memory_limit_gb: 8
      
    validation:
      max_workers: 1
      memory_limit_gb: 4

reporting:
  output_path: "reports/multi_phase_research/"
  include_phase_reports: true
  generate_summary_report: true
```

### Run Multi-Phase Workflow

```bash
# Run complete workflow
python main.py config/basic_multi_phase.yaml

# Run specific phase only
python main.py config/basic_multi_phase.yaml --phase parameter_optimization

# Resume from specific phase
python main.py config/basic_multi_phase.yaml --resume-from performance_analysis
```

## 📊 Advanced Multi-Phase Patterns

### Institutional Research Pipeline

```yaml
# config/institutional_research.yaml
workflow:
  type: "institutional_research"
  name: "Comprehensive Strategy Development"
  description: "Full institutional-grade research pipeline"
  
data:
  symbols: ["SPY", "QQQ", "IWM", "TLT", "GLD", "VXX"]
  start_date: "2018-01-01"
  end_date: "2023-12-31"
  
phases:
  # Phase 1: Broad Parameter Discovery
  - name: "broad_parameter_search"
    type: "optimization"
    config:
      strategies:
        - type: "momentum"
        - type: "mean_reversion"
        - type: "breakout"
        - type: "pairs_trading"
        
      optimization:
        method: "random"
        n_trials: 10000              # Large search space
        parameters:
          # Broad parameter ranges
          fast_period: [3, 50]
          slow_period: [10, 200]
          lookback_period: [5, 100]
          signal_threshold: [0.001, 0.1]
          
    container_count: 1000            # Massive parallelization
    timeout_hours: 6
    
    outputs:
      - top_100_strategies
      - parameter_importance
      - performance_distribution
      
  # Phase 2: Regime Detection and Analysis
  - name: "regime_analysis"
    type: "analysis"
    inputs: ["broad_parameter_search.top_100_strategies"]
    config:
      regime_detection:
        methods: ["hmm", "k_means", "change_point"]
        n_regimes: [2, 3, 4]
        features: ["volatility", "trend", "volume", "correlation"]
        
      regime_performance_analysis:
        strategy_performance_by_regime: true
        regime_transition_analysis: true
        regime_persistence_analysis: true
        
    outputs:
      - regime_classifications
      - regime_specific_performance
      - regime_transition_probabilities
      
  # Phase 3: Regime-Specific Optimization
  - name: "regime_specific_optimization"
    type: "optimization"
    inputs: [
      "broad_parameter_search.top_100_strategies",
      "regime_analysis.regime_classifications"
    ]
    config:
      optimization:
        method: "bayesian"
        group_by_regime: true        # Optimize per regime
        n_trials_per_regime: 200
        
      ensemble_optimization:
        enabled: true
        ensemble_methods: ["voting", "weighted_average", "meta_learning"]
        
    outputs:
      - regime_optimized_strategies
      - ensemble_weights
      - regime_switching_rules
      
  # Phase 4: Signal Capture for Fast Iteration
  - name: "signal_capture"
    type: "signal_generation"
    inputs: ["regime_specific_optimization.regime_optimized_strategies"]
    config:
      capture_all_signals: true
      include_regime_context: true
      signal_compression: true       # Compress for storage efficiency
      
    outputs:
      - compressed_signals
      - signal_metadata
      - regime_signal_distribution
      
  # Phase 5: Ensemble Weight Optimization (Super Fast)
  - name: "ensemble_optimization"
    type: "optimization"
    method: "signal_replay"          # 100x faster using captured signals
    inputs: ["signal_capture.compressed_signals"]
    config:
      optimization:
        method: "differential_evolution"
        n_generations: 100
        population_size: 50
        
        parameters:
          strategy_weights: [0.0, 1.0]  # Weight per strategy
          regime_confidence_threshold: [0.5, 0.95]
          rebalance_frequency: [5, 30]  # Days
          
      constraints:
        sum_weights: 1.0
        min_weight: 0.05
        max_weight: 0.4
        max_correlation: 0.8
        
    container_count: 100
    
    outputs:
      - optimal_ensemble_weights
      - ensemble_performance_metrics
      - weight_stability_analysis
      
  # Phase 6: Walk-Forward Validation
  - name: "walk_forward_validation"
    type: "walk_forward"
    inputs: ["ensemble_optimization.optimal_ensemble_weights"]
    config:
      walk_forward:
        train_period_days: 504       # 2 years
        test_period_days: 126        # 6 months
        step_days: 63                # Quarterly
        
      validation_metrics:
        - "performance_consistency"
        - "parameter_stability"
        - "regime_adaptation_effectiveness"
        - "ensemble_weight_stability"
        
    outputs:
      - walk_forward_results
      - parameter_evolution
      - regime_performance_validation
      
  # Phase 7: Risk Analysis and Stress Testing
  - name: "risk_analysis"
    type: "analysis"
    inputs: ["walk_forward_validation.walk_forward_results"]
    config:
      risk_analysis:
        var_analysis: true
        stress_testing: true
        scenario_analysis: true
        
      stress_scenarios:
        - name: "covid_crash"
          date_range: ["2020-02-01", "2020-04-01"]
        - name: "rate_hike_cycle"
          date_range: ["2022-03-01", "2023-06-01"]
        - name: "tech_selloff"
          date_range: ["2022-01-01", "2022-10-01"]
          
      tail_risk_analysis:
        confidence_levels: [0.95, 0.99, 0.999]
        extreme_scenario_modeling: true
        
    outputs:
      - risk_metrics
      - stress_test_results
      - tail_risk_assessment
      - risk_recommendations
      
  # Phase 8: Final Strategy Construction
  - name: "strategy_construction"
    type: "backtest"
    inputs: [
      "ensemble_optimization.optimal_ensemble_weights",
      "risk_analysis.risk_recommendations"
    ]
    config:
      include_transaction_costs: true
      include_market_impact: true
      realistic_execution_delays: true
      
      # Apply risk recommendations
      dynamic_risk_management: true
      regime_aware_position_sizing: true
      
      # Generate production-ready configuration
      generate_deployment_config: true
      
    outputs:
      - final_strategy_performance
      - production_configuration
      - implementation_guide
      - monitoring_recommendations
```

## 🔄 Conditional and Branching Workflows

### Adaptive Research Workflow

```yaml
workflow:
  type: "adaptive_multi_phase"
  
phases:
  # Phase 1: Initial Assessment
  - name: "initial_assessment"
    type: "analysis"
    config:
      market_condition_analysis:
        - "volatility_regime"
        - "trend_strength"
        - "correlation_structure"
        
    outputs:
      - market_conditions
      - recommended_strategy_types
      
  # Conditional branching based on market conditions
  - name: "strategy_selection"
    type: "conditional"
    condition: "${initial_assessment.market_conditions.volatility} > 0.25"
    
    # High volatility branch
    if_true:
      type: "optimization"
      config:
        strategies:
          - type: "defensive"
          - type: "volatility_trading"
        optimization:
          method: "robust_optimization"
          risk_penalty: 0.5
          
    # Low volatility branch  
    if_false:
      type: "optimization"
      config:
        strategies:
          - type: "momentum"
          - type: "mean_reversion"
        optimization:
          method: "return_maximization"
          
    outputs:
      - selected_strategies
      - optimization_results
      
  # Convergence phase
  - name: "ensemble_construction"
    type: "ensemble_optimization"
    inputs: ["strategy_selection.selected_strategies"]
    config:
      adaptive_weighting: true
      market_condition_awareness: true
```

### Parallel Research Branches

```yaml
workflow:
  type: "parallel_research"
  
  # Parallel research branches
  parallel_phases:
    # Branch A: Technical Analysis Focus
    technical_branch:
      phases:
        - name: "technical_optimization"
          type: "optimization"
          config:
            strategies: ["momentum", "mean_reversion", "breakout"]
            
        - name: "technical_validation"
          type: "walk_forward"
          inputs: ["technical_optimization.best_strategies"]
          
    # Branch B: Machine Learning Focus
    ml_branch:
      phases:
        - name: "feature_engineering"
          type: "feature_engineering"
          config:
            feature_types: ["technical", "fundamental", "sentiment"]
            
        - name: "ml_optimization"
          type: "optimization"
          inputs: ["feature_engineering.features"]
          config:
            strategies: ["sklearn_model", "tensorflow_model"]
            
        - name: "ml_validation"
          type: "walk_forward"
          inputs: ["ml_optimization.best_models"]
          
    # Branch C: Alternative Data Focus
    alternative_data_branch:
      phases:
        - name: "alt_data_integration"
          type: "data_integration"
          config:
            data_sources: ["sentiment", "satellite", "economic"]
            
        - name: "alt_data_optimization"
          type: "optimization"
          inputs: ["alt_data_integration.processed_data"]
          
  # Convergence phase - combine all branches
  convergence_phase:
    name: "multi_branch_ensemble"
    type: "ensemble_optimization"
    inputs: [
      "technical_branch.technical_validation.results",
      "ml_branch.ml_validation.results", 
      "alternative_data_branch.alt_data_optimization.results"
    ]
    config:
      ensemble_method: "hierarchical_voting"
      cross_validation: true
```

## 📊 Data Flow Management

### Advanced Data Flow Patterns

```yaml
# Complex data dependencies
phases:
  - name: "data_preprocessing"
    type: "data_processing"
    outputs:
      - cleaned_data
      - feature_engineered_data
      - regime_labels
      
  - name: "strategy_optimization_a"
    type: "optimization"
    inputs: ["data_preprocessing.cleaned_data"]
    outputs:
      - strategy_a_results
      
  - name: "strategy_optimization_b"
    type: "optimization" 
    inputs: [
      "data_preprocessing.cleaned_data",
      "data_preprocessing.feature_engineered_data"
    ]
    outputs:
      - strategy_b_results
      
  - name: "ensemble_optimization"
    type: "optimization"
    inputs: [
      "strategy_optimization_a.strategy_a_results",
      "strategy_optimization_b.strategy_b_results",
      "data_preprocessing.regime_labels"
    ]
    config:
      # Use all inputs in ensemble optimization
      input_processing:
        strategy_a_weight: 0.4
        strategy_b_weight: 0.6
        regime_conditioning: true
```

### Data Transformation Between Phases

```yaml
# Automatic data transformation
phases:
  - name: "signal_generation"
    type: "signal_generation"
    outputs:
      - raw_signals
      
  - name: "signal_processing"
    type: "transformation"
    inputs: ["signal_generation.raw_signals"]
    config:
      transformations:
        - type: "signal_smoothing"
          method: "exponential_smoothing"
          alpha: 0.3
          
        - type: "signal_filtering"
          method: "strength_threshold"
          threshold: 0.5
          
        - type: "signal_aggregation"
          method: "time_weighted_average"
          window: 5
          
    outputs:
      - processed_signals
      
  - name: "portfolio_optimization"
    type: "optimization"
    inputs: ["signal_processing.processed_signals"]
    # Automatically receives processed signals
```

## 🛠️ Workflow Management

### Checkpointing and Resume

```yaml
workflow:
  checkpointing:
    enabled: true
    checkpoint_frequency: "per_phase"   # or "hourly", "daily"
    checkpoint_location: "checkpoints/workflow_id/"
    
    # Auto-save configuration
    auto_save:
      phase_completion: true
      significant_results: true
      error_conditions: true
      
    # Resume configuration
    resume_policy:
      auto_resume_on_failure: true
      max_resume_attempts: 3
      resume_from_last_checkpoint: true
```

### Error Handling and Recovery

```yaml
workflow:
  error_handling:
    # Phase-level error handling
    phase_error_policies:
      parameter_optimization:
        on_failure: "retry"
        max_retries: 2
        fallback_method: "reduced_parameter_space"
        
      walk_forward_validation:
        on_failure: "continue_with_warning"
        partial_results: "acceptable"
        
      final_validation:
        on_failure: "abort_workflow"
        critical: true
        
    # Global error handling
    global_error_policy:
      timeout_action: "save_partial_results"
      resource_exhaustion: "graceful_degradation"
      unexpected_error: "checkpoint_and_abort"
```

### Resource Management

```yaml
workflow:
  resource_management:
    # Dynamic resource allocation
    dynamic_allocation: true
    
    # Phase-specific resources
    phase_resources:
      broad_search:
        priority: "low"
        max_containers: 1000
        memory_per_container: "512MB"
        cpu_per_container: 0.1
        
      ensemble_optimization:
        priority: "medium"
        max_containers: 100
        memory_per_container: "2GB"
        cpu_per_container: 0.5
        
      final_validation:
        priority: "high"
        max_containers: 1
        memory_per_container: "8GB"
        cpu_per_container: 4.0
        
    # Resource constraints
    global_limits:
      max_total_memory_gb: 128
      max_total_cpu_cores: 32
      max_concurrent_containers: 1000
```

## 📈 Monitoring and Progress Tracking

### Real-Time Workflow Monitoring

```yaml
workflow:
  monitoring:
    enabled: true
    
    # Progress tracking
    progress_tracking:
      phase_progress: true
      container_progress: true
      estimated_completion: true
      
    # Performance monitoring
    performance_monitoring:
      resource_utilization: true
      bottleneck_detection: true
      cost_tracking: true
      
    # Alerts
    alerts:
      phase_completion: true
      performance_degradation: true
      resource_exhaustion: true
      estimated_delay: true
      
    # Visualization
    real_time_dashboard: true
    progress_charts: true
    resource_usage_charts: true
```

### Workflow Analytics

```yaml
reporting:
  workflow_analytics:
    # Execution analysis
    execution_analysis:
      - "phase_timing_analysis"
      - "resource_efficiency_analysis"
      - "bottleneck_identification"
      - "cost_benefit_analysis"
      
    # Results analysis
    results_analysis:
      - "cross_phase_performance_tracking"
      - "data_flow_impact_analysis"
      - "ensemble_contribution_analysis"
      - "validation_consistency_analysis"
      
    # Optimization recommendations
    optimization_recommendations:
      - "workflow_structure_optimization"
      - "resource_allocation_optimization"
      - "phase_ordering_optimization"
      - "parallel_execution_opportunities"
```

## 🎯 Production Deployment Workflows

### Staged Deployment Pipeline

```yaml
workflow:
  type: "production_deployment"
  
phases:
  # Stage 1: Final Validation
  - name: "pre_deployment_validation"
    type: "comprehensive_validation"
    config:
      validation_tests:
        - "out_of_sample_performance"
        - "stress_testing"
        - "parameter_stability"
        - "risk_assessment"
        - "capacity_analysis"
        
      acceptance_criteria:
        min_sharpe_ratio: 1.0
        max_drawdown: 0.15
        min_win_rate: 0.45
        
  # Stage 2: Paper Trading
  - name: "paper_trading"
    type: "live_simulation"
    inputs: ["pre_deployment_validation.validated_strategy"]
    config:
      duration_days: 30
      real_time_data: true
      paper_trading_only: true
      
      monitoring:
        real_time_alerts: true
        performance_tracking: true
        deviation_alerts: true
        
  # Stage 3: Shadow Trading
  - name: "shadow_trading"
    type: "live_simulation"
    inputs: ["paper_trading.performance_validation"]
    config:
      duration_days: 60
      shadow_mode: true           # Run alongside existing strategy
      comparison_baseline: "current_production_strategy"
      
  # Stage 4: Gradual Rollout
  - name: "gradual_rollout"
    type: "live_trading"
    inputs: ["shadow_trading.comparative_results"]
    config:
      rollout_schedule:
        week_1: 0.1               # 10% allocation
        week_2: 0.25              # 25% allocation
        week_4: 0.5               # 50% allocation
        week_8: 1.0               # Full allocation
        
      safety_checks:
        performance_monitoring: true
        automatic_rollback: true
        risk_limit_enforcement: true
```

## 🤔 Common Questions

**Q: How do I debug failed workflows?**
A: Use phase-by-phase execution, check logs for specific phases, and examine checkpoint data. Enable detailed monitoring for complex workflows.

**Q: Can I modify workflows during execution?**
A: No during execution, but you can checkpoint and resume with modified configurations.

**Q: How do I handle data dependencies between phases?**
A: Use the `inputs` specification in phase configuration. Data flows automatically between phases.

**Q: What's the maximum number of phases?**
A: No hard limit, but 5-10 phases is typical. Very long workflows become harder to debug and manage.

## 📝 Multi-Phase Workflow Checklist

- [ ] Plan data flow between phases
- [ ] Set appropriate resource limits per phase
- [ ] Enable checkpointing for long workflows
- [ ] Configure error handling policies
- [ ] Set up progress monitoring
- [ ] Plan for workflow resumption
- [ ] Validate intermediate results
- [ ] Document workflow purpose and structure

## 📈 Next Steps

- **Deploy to Production**: [Live Trading Guide](live-trading.md)
- **Learn Advanced Patterns**: [Patterns Documentation](../06-patterns/README.md)
- **Build Custom Components**: [Advanced Topics](../08-advanced-topics/README.md)

---

Continue to [Live Trading](live-trading.md) to deploy your workflows to production →