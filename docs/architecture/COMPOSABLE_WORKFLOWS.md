# Composable Workflows: Turning ADMF-PC into a Strategy Research Automaton

## Motivation

Traditional quantitative research is often a manual, iterative process:
1. Run a backtest
2. Analyze results
3. Adjust parameters
4. Repeat until satisfactory

**Composable Workflows** transform this into an automated research engine that can:
- Intelligently search parameter spaces
- Adapt to market conditions
- Build statistical confidence
- Optimize ensemble strategies
- All without human intervention

## Core Concept

Instead of adding another layer, we built composability directly into the workflow system:

```
Coordinator
    ↓
WorkflowExecutor (detects and handles composability)
    ↓
Workflow (with optional composable methods)
    ↓
Sequencer (phase orchestration)
    ↓
Topology (container arrangement)
```

Any workflow can become composable by implementing one or more optional methods:
- `should_continue()` - Iterate until conditions are met
- `get_branches()` - Branch to other workflows based on results
- `modify_config_for_next()` - Evolve configuration between iterations

## Implementation

### The Protocol

```python
@runtime_checkable
class WorkflowProtocol(Protocol):
    """All workflows implement this."""
    
    defaults: Dict[str, Any]
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """Required: Define workflow phases."""
        ...
    
    # Optional composable methods
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """Should workflow continue iterating?"""
        return False  # Default: execute once
    
    def get_branches(self, result: Dict[str, Any]) -> Optional[List[WorkflowBranch]]:
        """Get conditional branches based on results."""
        return None  # Default: no branching
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Modify config for next iteration."""
        return config  # Default: unchanged
```

### Simple Workflow (Non-Composable)

```python
class SimpleBacktestWorkflow:
    """Just runs once - no composable methods implemented."""
    
    defaults = {'trace_level': 'normal'}
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        return {
            "backtest": PhaseConfig(
                name="backtest",
                sequence="single_pass",
                topology="backtest",
                description="Run single backtest",
                config=config
            )
        }
```

### Composable Workflow Example

```python
class AdaptiveStrategySearchWorkflow:
    """Searches until finding profitable strategies."""
    
    defaults = {
        'target_sharpe': 1.5,
        'target_win_rate': 0.55,
        'max_iterations': 10
    }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        return {
            "parameter_search": PhaseConfig(
                name="parameter_search",
                sequence="train_test",
                topology="backtest",
                description="Search parameter space",
                config=config
            ),
            "strategy_analysis": PhaseConfig(
                name="strategy_analysis",
                sequence="single_pass",
                topology="analysis",
                description="Analyze found strategies",
                config={'metrics': ['sharpe_ratio', 'win_rate']},
                depends_on=['parameter_search']
            )
        }
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """Continue if haven't found good strategies yet."""
        if result.get('aggregated_results', {}).get('best_sharpe', 0) > 1.5:
            return False  # Found good strategy!
        
        return iteration < self.defaults['max_iterations']
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Expand parameter space based on results."""
        # Intelligently expand search space
        if 'parameter_space' in config:
            # Add more parameter values around promising areas
            best_params = result.get('best_parameters', {})
            # ... expansion logic ...
        
        return config
    
    def get_branches(self, result: Dict[str, Any]) -> Optional[List[WorkflowBranch]]:
        """Branch to different approaches if stuck."""
        best_sharpe = result.get('aggregated_results', {}).get('best_sharpe', 0)
        
        if best_sharpe < 0.5:
            # Not finding anything - try regime-based approach
            return [WorkflowBranch(
                condition=lambda r: True,
                workflow='regime_adaptive_ensemble',
                config_modifier=lambda c, r: {**c, 'note': 'Switched to regime-based'}
            )]
        
        return None
```

### The WorkflowExecutor

The executor automatically detects and handles composability:

```python
class WorkflowExecutor:
    def execute(self, workflow_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        workflow = self.workflows[workflow_name]
        
        # Detect if composable
        is_composable = (
            hasattr(workflow, 'should_continue') or 
            hasattr(workflow, 'get_branches') or
            hasattr(workflow, 'modify_config_for_next')
        )
        
        if is_composable:
            return self._execute_composable(workflow, workflow_name, config)
        else:
            return self._execute_simple(workflow, workflow_name, config)
```

## Real-World Use Cases

### 1. Adaptive Market Regime Search

**Motivation**: Markets change. Strategies need to adapt automatically.

**Implementation**:

```python
class RegimeAdaptiveWorkflow:
    """Detects regime changes and reoptimizes."""
    
    def __init__(self):
        self.current_regime = None
        self.regime_history = []
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        phases = {}
        
        if self.current_regime is None:
            # Initial regime detection
            phases["regime_detection"] = PhaseConfig(
                name="regime_detection",
                sequence="single_pass",
                topology="analysis",
                description="Detect current market regime",
                config={'method': 'hmm', 'n_regimes': 3}
            )
        
        # Always optimize for current/detected regime
        phases["optimization"] = PhaseConfig(
            name="optimization",
            sequence="train_test",
            topology="backtest",
            description=f"Optimize for regime: {self.current_regime}",
            config={
                **config,
                'regime_filter': self.current_regime
            }
        )
        
        return phases
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """Continue monitoring for regime changes."""
        # In production, would have more sophisticated stopping criteria
        return iteration < 100
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Update regime if changed."""
        detected_regime = result.get('phase_results', {}).get(
            'regime_detection', {}
        ).get('dominant_regime')
        
        if detected_regime and detected_regime != self.current_regime:
            self.current_regime = detected_regime
            self.regime_history.append({
                'iteration': iteration,
                'regime': detected_regime
            })
        
        return config
```

### 2. Statistical Confidence Builder

**Motivation**: Run simulations until statistical significance is achieved.

**Implementation**:

```python
class MonteCarloConfidenceWorkflow:
    """Runs until confidence intervals are tight enough."""
    
    defaults = {
        'initial_iterations': 100,
        'max_iterations': 10000,
        'target_ci_width': 0.1,
        'confidence_level': 0.95
    }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        return {
            "monte_carlo": PhaseConfig(
                name="monte_carlo",
                sequence="monte_carlo",
                topology="backtest",
                description=f"Run {config.get('mc_iterations', 100)} Monte Carlo simulations",
                config=config
            )
        }
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """Continue if confidence interval too wide."""
        ci_width = self._get_ci_width(result)
        return ci_width > self.defaults['target_ci_width']
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Double iterations each time."""
        current = config.get('mc_iterations', self.defaults['initial_iterations'])
        config['mc_iterations'] = min(current * 2, self.defaults['max_iterations'])
        return config
    
    def _get_ci_width(self, result: Dict[str, Any]) -> float:
        """Extract confidence interval width from results."""
        metrics = result.get('aggregated_results', {})
        upper = metrics.get('sharpe_ci_upper', 0)
        lower = metrics.get('sharpe_ci_lower', 0)
        return upper - lower
```

### 3. Multi-Strategy Ensemble Builder

**Motivation**: Automatically discover and combine complementary strategies.

**Implementation**:

```python
class EnsembleBuilderWorkflow:
    """Builds optimal ensemble of strategies."""
    
    def __init__(self):
        self.discovered_strategies = []
        self.ensemble_performance = None
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        phases = {}
        
        if not self.discovered_strategies:
            # Discovery phase
            phases["discovery"] = PhaseConfig(
                name="discovery",
                sequence="train_test",
                topology="backtest",
                description="Discover strategy candidates",
                config={
                    **config,
                    'parameter_space': self._get_search_space()
                }
            )
        else:
            # Ensemble optimization phase
            phases["ensemble"] = PhaseConfig(
                name="ensemble",
                sequence="single_pass",
                topology="ensemble_optimization",
                description="Optimize strategy weights",
                config={
                    'strategies': self.discovered_strategies,
                    'method': 'mean_variance_optimization'
                }
            )
        
        return phases
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """Continue until we have enough good strategies."""
        if len(self.discovered_strategies) >= 5:
            # Enough strategies - check ensemble performance
            if self.ensemble_performance and self.ensemble_performance['sharpe'] > 2.0:
                return False
        
        return iteration < 20
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """Add discovered strategies and search new areas."""
        # Extract good strategies from results
        good_strategies = self._extract_good_strategies(result)
        self.discovered_strategies.extend(good_strategies)
        
        # Update ensemble performance if available
        if 'ensemble' in result.get('phase_results', {}):
            self.ensemble_performance = result['phase_results']['ensemble']
        
        return config
```

## User Experience

Simple YAML configuration for complex research:

```yaml
# Run adaptive strategy search
workflow: adaptive_strategy_search
target_sharpe: 2.0
max_iterations: 20

data:
  symbols: [SPY, QQQ, IWM]
  start: '2020-01-01'
  end: '2023-12-31'

parameter_space:
  strategies:
    momentum:
      lookback: [10, 20, 50]
    mean_reversion:
      period: [14, 21, 28]
```

The system automatically:
1. Searches the parameter space
2. Analyzes results
3. Expands search in promising directions
4. Branches to different approaches if needed
5. Stops when targets are met

## Advanced Patterns

### Composition Through Configuration

Workflows can be parameterized for different behaviors:

```yaml
# Conservative search
workflow: adaptive_strategy_search
target_sharpe: 1.2
target_win_rate: 0.60
expansion_factor: 1.2  # Expand slowly

# Aggressive search  
workflow: adaptive_strategy_search
target_sharpe: 3.0
target_win_rate: 0.65
expansion_factor: 2.0  # Expand aggressively
max_iterations: 50
```

### Nested Workflows

Workflows can execute other workflows through branching:

```python
def get_branches(self, result: Dict[str, Any]) -> Optional[List[WorkflowBranch]]:
    """Branch to specialized workflows based on initial results."""
    
    if self._looks_like_trending_market(result):
        return [WorkflowBranch(
            condition=lambda r: True,
            workflow='trend_following_optimization'
        )]
    elif self._looks_like_mean_reverting_market(result):
        return [WorkflowBranch(
            condition=lambda r: True,
            workflow='mean_reversion_optimization'
        )]
    
    # Continue with generic search
    return None
```

## Benefits

1. **Native Integration**: Composability is part of the workflow system, not bolted on
2. **Flexibility**: Any workflow can be made composable by adding methods
3. **Simplicity**: Simple workflows stay simple - no forced complexity
4. **Power**: Complex research automation with minimal code
5. **Extensibility**: Easy to add new composable patterns

## Examples in the Codebase

- `adaptive_strategy_search.py` - Searches until finding profitable strategies
- `regime_adaptive_ensemble.py` - Multi-phase workflow with complex dependencies
- `train_test_optimization.py` - Simple non-composable optimization workflow

## Conclusion

By making workflows composable by default, ADMF-PC becomes a true research automaton. Users can:
- Define high-level research goals
- Let workflows intelligently explore solution spaces
- Automatically adapt to results and market conditions
- Build sophisticated research pipelines with simple configuration

This isn't just automation - it's intelligent automation that learns and adapts, turning quantitative research from a manual process into a guided exploration powered by your domain expertise and the system's execution capabilities.