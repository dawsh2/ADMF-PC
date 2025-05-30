# ADMF-PC Workflow Composition Architecture

## Overview

The Workflow Composition Architecture extends the ADMF-PC Protocol + Composition design to enable complex, multi-phase workflows through declarative composition of simple building blocks. Instead of creating specialized managers for every workflow variant, we compose existing workflow types (Backtest, Optimization, Analysis, Validation) into sophisticated execution patterns.

## Core Concept: Workflows as Building Blocks

```
Simple Workflows (Building Blocks):
├── BACKTEST      - Single strategy evaluation
├── OPTIMIZATION  - Parameter search
├── ANALYSIS      - Performance analysis
├── VALIDATION    - Out-of-sample testing
└── LIVE_TRADING  - Real-time execution

Composite Workflows (Composed from above):
├── REGIME_ADAPTIVE_GRID_SEARCH    - Multi-phase optimization
├── WALK_FORWARD_VALIDATION        - Rolling window testing
├── ENSEMBLE_OPTIMIZATION          - Strategy combination
└── CUSTOM_WORKFLOWS              - User-defined compositions
```

## Architecture Benefits

1. **No New Code for New Workflows**: Define new patterns through configuration
2. **Reuses Existing Infrastructure**: Each phase uses proven managers
3. **Clean Separation of Concerns**: Coordinator sequences, managers execute
4. **Infinite Composability**: Mix and match phases as needed
5. **Maintains Protocol + Composition**: No inheritance, just orchestration

## Implementation

### 1. Workflow Template Definition

```python
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class WorkflowPhaseDefinition:
    """Defines a single phase in a composite workflow"""
    name: str
    workflow_type: WorkflowType
    config_generator: Callable[..., Dict[str, Any]]
    depends_on: List[str] = field(default_factory=list)
    continue_on_failure: bool = False
    timeout_seconds: Optional[int] = None

@dataclass
class CompositeWorkflowTemplate:
    """Template for a multi-phase workflow"""
    name: str
    description: str
    phases: List[WorkflowPhaseDefinition]
    aggregation_strategy: Callable[[Dict[str, Any]], Any]
    
class WorkflowTemplateRegistry:
    """Registry of workflow templates"""
    def __init__(self):
        self._templates: Dict[str, CompositeWorkflowTemplate] = {}
        self._register_standard_templates()
    
    def register(self, template: CompositeWorkflowTemplate):
        """Register a new workflow template"""
        self._templates[template.name] = template
    
    def get(self, name: str) -> CompositeWorkflowTemplate:
        """Retrieve a workflow template"""
        if name not in self._templates:
            raise ValueError(f"Unknown workflow template: {name}")
        return self._templates[name]
    
    def _register_standard_templates(self):
        """Register built-in workflow templates"""
        
        # Regime Adaptive Grid Search Ensemble
        self.register(CompositeWorkflowTemplate(
            name="regime_adaptive_grid_search_ensemble",
            description="Multi-phase optimization with regime-specific parameters",
            phases=[
                WorkflowPhaseDefinition(
                    name="parameter_discovery",
                    workflow_type=WorkflowType.OPTIMIZATION,
                    config_generator=self._generate_param_discovery_config
                ),
                WorkflowPhaseDefinition(
                    name="regime_analysis",
                    workflow_type=WorkflowType.ANALYSIS,
                    config_generator=self._generate_regime_analysis_config,
                    depends_on=["parameter_discovery"]
                ),
                WorkflowPhaseDefinition(
                    name="ensemble_optimization",
                    workflow_type=WorkflowType.OPTIMIZATION,
                    config_generator=self._generate_ensemble_config,
                    depends_on=["parameter_discovery", "regime_analysis"]
                ),
                WorkflowPhaseDefinition(
                    name="validation",
                    workflow_type=WorkflowType.VALIDATION,
                    config_generator=self._generate_validation_config,
                    depends_on=["ensemble_optimization"]
                )
            ],
            aggregation_strategy=self._aggregate_optimization_results
        ))
```

### 2. Enhanced Coordinator with Composition Support

```python
class Coordinator:
    """Enhanced Coordinator with workflow composition support"""
    
    def __init__(self, shared_services: Dict[str, Any] = None):
        # Existing initialization...
        self.workflow_registry = WorkflowTemplateRegistry()
        self.composite_executor = CompositeWorkflowExecutor(self)
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute a workflow - simple or composite"""
        
        # Check if this is a registered composite workflow
        if self.workflow_registry.is_composite(config.workflow_type):
            return await self.composite_executor.execute(config)
        
        # Otherwise, execute as simple workflow
        return await self._execute_simple_workflow(config)
    
    async def _execute_simple_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute a single workflow (existing implementation)"""
        # Create container
        container_id = await self._create_workflow_container(
            config.workflow_id, config, context
        )
        
        # Get appropriate manager
        manager = self.manager_factory.create_manager(
            config.workflow_type, container_id
        )
        
        # Execute
        return await manager.execute(config, context)
```

### 3. Composite Workflow Executor

```python
class CompositeWorkflowExecutor:
    """Executes multi-phase composite workflows"""
    
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.phase_data_store = PhaseDataStore()
    
    async def execute(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute a composite workflow"""
        template = self.coordinator.workflow_registry.get(config.workflow_type)
        
        # Initialize composite result
        composite_result = CompositeWorkflowResult(
            workflow_id=config.workflow_id,
            workflow_type=config.workflow_type,
            template_name=template.name
        )
        
        # Execute phases in order
        for phase_def in template.phases:
            # Check dependencies
            if not self._dependencies_met(phase_def, composite_result):
                if not phase_def.continue_on_failure:
                    composite_result.mark_failed(f"Dependencies not met for {phase_def.name}")
                    break
                continue
            
            # Generate phase configuration
            phase_config = self._generate_phase_config(
                phase_def, config, composite_result
            )
            
            # Execute phase
            try:
                phase_result = await self._execute_phase(
                    phase_def, phase_config
                )
                composite_result.add_phase_result(phase_def.name, phase_result)
                
                # Store intermediate data
                self.phase_data_store.store(
                    config.workflow_id, 
                    phase_def.name, 
                    phase_result
                )
                
            except Exception as e:
                composite_result.add_phase_error(phase_def.name, str(e))
                if not phase_def.continue_on_failure:
                    break
        
        # Aggregate results
        final_results = template.aggregation_strategy(
            composite_result.get_all_phase_results()
        )
        composite_result.set_final_results(final_results)
        
        return composite_result
    
    async def _execute_phase(
        self, 
        phase_def: WorkflowPhaseDefinition,
        phase_config: WorkflowConfig
    ) -> WorkflowResult:
        """Execute a single phase"""
        # Add timeout if specified
        if phase_def.timeout_seconds:
            return await asyncio.wait_for(
                self.coordinator._execute_simple_workflow(phase_config),
                timeout=phase_def.timeout_seconds
            )
        else:
            return await self.coordinator._execute_simple_workflow(phase_config)
```

### 4. Configuration Generators

```python
class ConfigGenerators:
    """Configuration generators for workflow phases"""
    
    @staticmethod
    def parameter_discovery_config(
        base_config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Generate config for parameter discovery phase"""
        return {
            'optimization_config': {
                'algorithm': 'grid',
                'parameter_space': base_config.parameters['parameter_space'],
                'objective': 'sharpe_ratio',
                'regime_aware': True,
                'capture_signals': True  # Important for signal replay
            },
            'backtest_config': {
                'start_date': base_config.parameters['train_start'],
                'end_date': base_config.parameters['train_end'],
                'initial_capital': 100000
            },
            'data_config': base_config.data_config
        }
    
    @staticmethod
    def regime_analysis_config(
        base_config: WorkflowConfig,
        param_discovery_results: WorkflowResult
    ) -> Dict[str, Any]:
        """Generate config for regime analysis phase"""
        return {
            'analysis_config': {
                'analysis_type': 'regime_performance',
                'trades_data': param_discovery_results.final_results['all_trades'],
                'regime_classifiers': base_config.parameters['regime_classifiers'],
                'metrics': ['sharpe', 'returns', 'drawdown', 'win_rate']
            }
        }
    
    @staticmethod
    def ensemble_optimization_config(
        base_config: WorkflowConfig,
        param_results: WorkflowResult,
        regime_results: WorkflowResult
    ) -> Dict[str, Any]:
        """Generate config for ensemble optimization phase"""
        return {
            'optimization_config': {
                'algorithm': 'genetic',
                'objective': 'risk_adjusted_returns',
                'signal_replay': True,  # Use captured signals
                'signal_logs': param_results.final_results['signal_logs'],
                'regime_optimal_params': regime_results.final_results['optimal_params_by_regime'],
                'weight_constraints': {
                    'min_weight': 0.0,
                    'max_weight': 1.0,
                    'sum_to_one': True
                }
            }
        }
```

### 5. Example: Creating Custom Workflows

```python
# Define a custom workflow template
custom_template = CompositeWorkflowTemplate(
    name="adaptive_regime_with_risk_sweep",
    description="Regime optimization followed by risk parameter tuning",
    phases=[
        # Phase 1: Standard parameter discovery
        WorkflowPhaseDefinition(
            name="parameter_discovery",
            workflow_type=WorkflowType.OPTIMIZATION,
            config_generator=ConfigGenerators.parameter_discovery_config
        ),
        
        # Phase 2: Regime analysis
        WorkflowPhaseDefinition(
            name="regime_analysis",
            workflow_type=WorkflowType.ANALYSIS,
            config_generator=ConfigGenerators.regime_analysis_config,
            depends_on=["parameter_discovery"]
        ),
        
        # Phase 3: Risk parameter sweep (NEW!)
        WorkflowPhaseDefinition(
            name="risk_parameter_sweep",
            workflow_type=WorkflowType.OPTIMIZATION,
            config_generator=lambda base_config, regime_results: {
                'optimization_config': {
                    'algorithm': 'grid',
                    'parameter_space': {
                        'max_position_size': [0.01, 0.02, 0.03, 0.05],
                        'max_drawdown': [0.05, 0.10, 0.15, 0.20],
                        'position_sizing_method': ['fixed', 'volatility', 'kelly']
                    },
                    'signal_replay': True,
                    'objective': 'risk_adjusted_sharpe',
                    'fixed_strategy_params': regime_results['optimal_params']
                }
            },
            depends_on=["regime_analysis"]
        ),
        
        # Phase 4: Ensemble with risk-aware weights
        WorkflowPhaseDefinition(
            name="risk_aware_ensemble",
            workflow_type=WorkflowType.OPTIMIZATION,
            config_generator=lambda base_config, risk_results: {
                'optimization_config': {
                    'algorithm': 'bayesian',
                    'objective': 'min_drawdown_sharpe',
                    'signal_replay': True,
                    'risk_parameters': risk_results['optimal_risk_params']
                }
            },
            depends_on=["risk_parameter_sweep"]
        ),
        
        # Phase 5: Walk-forward validation
        WorkflowPhaseDefinition(
            name="walk_forward_validation",
            workflow_type=WorkflowType.VALIDATION,
            config_generator=ConfigGenerators.walk_forward_config,
            depends_on=["risk_aware_ensemble"]
        )
    ],
    aggregation_strategy=lambda results: {
        'optimal_strategy_params': results['regime_analysis']['optimal_params_by_regime'],
        'optimal_risk_params': results['risk_parameter_sweep']['best_params'],
        'ensemble_weights': results['risk_aware_ensemble']['weights'],
        'validation_metrics': results['walk_forward_validation']['metrics']
    }
)

# Register the custom workflow
coordinator.workflow_registry.register(custom_template)

# Use it like any other workflow
config = WorkflowConfig(
    workflow_type="adaptive_regime_with_risk_sweep",
    parameters={
        'parameter_space': {...},
        'regime_classifiers': ['hmm', 'pattern'],
        'train_start': '2020-01-01',
        'train_end': '2022-12-31',
        'test_start': '2023-01-01',
        'test_end': '2023-12-31'
    }
)

result = await coordinator.execute_workflow(config)
```

### 6. Workflow Composition Patterns

#### Pattern 1: Sequential Pipeline
```python
# Each phase depends on the previous
phases = [
    phase_1,
    phase_2 (depends on phase_1),
    phase_3 (depends on phase_2),
    phase_4 (depends on phase_3)
]
```

#### Pattern 2: Parallel Branches
```python
# Multiple phases can run in parallel
phases = [
    phase_1,
    phase_2a (depends on phase_1),
    phase_2b (depends on phase_1),  # Parallel with 2a
    phase_3 (depends on phase_2a, phase_2b)
]
```

#### Pattern 3: Conditional Execution
```python
# Phases execute based on conditions
WorkflowPhaseDefinition(
    name="advanced_optimization",
    workflow_type=WorkflowType.OPTIMIZATION,
    config_generator=lambda base_config, prev_results: {
        # Only run if basic optimization found good params
        'enabled': prev_results['sharpe'] > 1.0,
        'optimization_config': {...}
    },
    continue_on_failure=True  # Skip if not enabled
)
```

#### Pattern 4: Iterative Refinement
```python
# Loop phases until convergence
def create_iterative_workflow(max_iterations=5):
    phases = [initial_phase]
    
    for i in range(max_iterations):
        phases.extend([
            WorkflowPhaseDefinition(
                name=f"refinement_{i}",
                workflow_type=WorkflowType.OPTIMIZATION,
                config_generator=lambda cfg, prev: {
                    'optimization_config': {
                        'algorithm': 'gradient',
                        'starting_point': prev[f'refinement_{i-1}']['best_params'] if i > 0 else None
                    }
                }
            ),
            WorkflowPhaseDefinition(
                name=f"convergence_check_{i}",
                workflow_type=WorkflowType.ANALYSIS,
                config_generator=lambda cfg, prev: {
                    'analysis_config': {
                        'check_convergence': True,
                        'tolerance': 0.001
                    }
                }
            )
        ])
    
    return CompositeWorkflowTemplate("iterative_refinement", phases=phases)
```

### 7. Phase Data Management

```python
class PhaseDataStore:
    """Manages data flow between phases"""
    
    def __init__(self, storage_path: Path = Path("./workflow_data")):
        self.storage_path = storage_path
        self.in_memory_cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def store(self, workflow_id: str, phase_name: str, result: WorkflowResult):
        """Store phase results"""
        # Extract key data for next phases
        phase_data = self._extract_phase_data(phase_name, result)
        
        # Store large data to disk
        if self._is_large_data(phase_data):
            file_path = self.storage_path / f"{workflow_id}_{phase_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(phase_data, f)
            self.metadata[f"{workflow_id}_{phase_name}"] = {
                'location': 'disk',
                'path': str(file_path),
                'size': file_path.stat().st_size
            }
        else:
            # Keep small data in memory
            self.in_memory_cache[f"{workflow_id}_{phase_name}"] = phase_data
            self.metadata[f"{workflow_id}_{phase_name}"] = {
                'location': 'memory',
                'size': sys.getsizeof(phase_data)
            }
    
    def retrieve(self, workflow_id: str, phase_name: str) -> Any:
        """Retrieve phase results"""
        key = f"{workflow_id}_{phase_name}"
        
        if key in self.in_memory_cache:
            return self.in_memory_cache[key]
        
        metadata = self.metadata.get(key)
        if metadata and metadata['location'] == 'disk':
            with open(metadata['path'], 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def _extract_phase_data(self, phase_name: str, result: WorkflowResult) -> Any:
        """Extract relevant data based on phase type"""
        if phase_name == "parameter_discovery":
            return {
                'signal_logs': result.final_results.get('signal_logs'),
                'parameter_performance': result.final_results.get('parameter_performance'),
                'regime_classifications': result.final_results.get('regime_classifications')
            }
        elif phase_name == "regime_analysis":
            return {
                'optimal_params_by_regime': result.final_results.get('optimal_params_by_regime'),
                'regime_performance': result.final_results.get('regime_performance')
            }
        # ... etc for other phases
        
        return result.final_results
```

### 8. Workflow Monitoring and Visualization

```python
class WorkflowMonitor:
    """Monitor composite workflow execution"""
    
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.active_workflows: Dict[str, WorkflowStatus] = {}
        
        # Subscribe to workflow events
        coordinator.event_bus.subscribe('workflow.phase.started', self.on_phase_start)
        coordinator.event_bus.subscribe('workflow.phase.completed', self.on_phase_complete)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow status"""
        status = self.active_workflows.get(workflow_id)
        if not status:
            return {'status': 'not_found'}
        
        return {
            'workflow_id': workflow_id,
            'template': status.template_name,
            'current_phase': status.current_phase,
            'completed_phases': status.completed_phases,
            'phase_timings': status.phase_timings,
            'progress': len(status.completed_phases) / len(status.total_phases),
            'estimated_completion': status.estimate_completion()
        }
    
    def generate_workflow_diagram(self, template_name: str) -> str:
        """Generate Mermaid diagram of workflow"""
        template = self.coordinator.workflow_registry.get(template_name)
        
        mermaid = "graph TD\n"
        for i, phase in enumerate(template.phases):
            # Add node
            mermaid += f"    {phase.name}[{phase.name}<br/>{phase.workflow_type}]\n"
            
            # Add dependencies
            for dep in phase.depends_on:
                mermaid += f"    {dep} --> {phase.name}\n"
        
        return mermaid
```

## Benefits of This Architecture

### 1. **Infinite Flexibility**
- Create any workflow by composing existing building blocks
- No code changes needed for new patterns
- Easy experimentation with different phase orders

### 2. **Clean Separation of Concerns**
- Coordinator: Orchestrates phase execution
- Managers: Execute specific workflow types
- Templates: Define workflow patterns
- Generators: Create phase configurations

### 3. **Reusability**
- Each phase uses proven, tested managers
- Common patterns become templates
- Share phase definitions across workflows

### 4. **Maintainability**
- New workflows don't require new code
- Changes to managers automatically benefit all workflows
- Clear, declarative workflow definitions

### 5. **Debugging and Monitoring**
- Each phase is a standard workflow with full logging
- Clear phase boundaries for debugging
- Progress tracking and visualization

### 6. **Protocol + Composition Victory**
- No inheritance needed
- Pure composition of existing components
- Protocols ensure compatibility
- Maximum flexibility with minimal complexity

## Example Use Cases

### 1. Research Workflow
```python
research_workflow = CompositeWorkflowTemplate(
    name="signal_quality_research",
    phases=[
        # Generate signals without execution
        signal_generation_phase,
        # Analyze MAE/MFE
        mae_mfe_analysis_phase,
        # Test different classifiers
        classifier_comparison_phase,
        # Generate research report
        report_generation_phase
    ]
)
```

### 2. Production Deployment Workflow
```python
production_workflow = CompositeWorkflowTemplate(
    name="strategy_production_deployment",
    phases=[
        # Run full backtest
        backtest_phase,
        # Validate on recent data
        validation_phase,
        # Risk assessment
        risk_analysis_phase,
        # Deploy to paper trading
        paper_trading_phase,
        # Monitor for 1 week
        monitoring_phase,
        # Deploy to live
        live_deployment_phase
    ]
)
```

### 3. Continuous Improvement Workflow
```python
continuous_improvement = CompositeWorkflowTemplate(
    name="weekly_strategy_tuning",
    phases=[
        # Analyze recent performance
        performance_analysis_phase,
        # Identify underperforming regimes
        regime_diagnosis_phase,
        # Retune parameters for weak regimes
        targeted_optimization_phase,
        # Validate improvements
        improvement_validation_phase,
        # Update production parameters
        parameter_update_phase
    ]
)
```

## Conclusion

The Workflow Composition Architecture transforms the Coordinator from a simple orchestrator into a powerful workflow engine, all while maintaining the clean Protocol + Composition design. By treating workflows as composable building blocks, we achieve unlimited flexibility without adding complexity.

This is the power of Protocol + Composition: simple components that combine into sophisticated systems, with no inheritance, no tight coupling, and maximum reusability.
