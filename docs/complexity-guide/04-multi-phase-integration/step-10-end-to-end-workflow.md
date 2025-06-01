# Step 10: End-to-End Workflow

**Status**: Multi-Phase Integration Step
**Complexity**: Very High
**Prerequisites**: [Step 9: Parameter Expansion](step-09-parameter-expansion.md) completed
**Architecture Ref**: [WORKFLOW_COMPOSITION.md](../../legacy/WORKFLOW_COMPOSITION.MD)

## 🎯 Objective

Implement complete multi-phase optimization workflow:
- Orchestrate complex optimization pipelines
- Coordinate multiple phases with state persistence
- Enable workflow composition and reuse
- Support checkpoint/resume functionality
- Aggregate and analyze results across phases

## 📋 Required Reading

Before starting:
1. [MULTIPHASE_OPTIMIZATION.md](../../legacy/MULTIPHASE_OPTIMIZATION.md)
2. [WORKFLOW_COMPOSITION.md](../../legacy/WORKFLOW_COMPOSITION.MD)
3. [Coordinator Architecture](../../core/coordinator/README.md)

## 🏗️ Implementation Tasks

### 1. Workflow Definition and Configuration

```python
# src/core/coordinator/workflow_definition.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import yaml
import json

class PhaseType(Enum):
    """Types of workflow phases"""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    PARAMETER_SEARCH = "parameter_search"
    SIGNAL_CAPTURE = "signal_capture"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    ANALYSIS = "analysis"

@dataclass
class PhaseDefinition:
    """Defines a single phase in the workflow"""
    name: str
    phase_type: PhaseType
    config: Dict[str, Any]
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution control
    enabled: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    timeout_minutes: Optional[int] = None
    
    # Resource requirements
    memory_gb: float = 4.0
    cpu_cores: int = 1
    gpu_required: bool = False
    
    # Validation
    validation_func: Optional[str] = None
    success_criteria: Optional[Dict[str, Any]] = None
    
    def validate_config(self) -> bool:
        """Validate phase configuration"""
        required_keys = {
            PhaseType.DATA_PREPARATION: ['start_date', 'end_date', 'symbols'],
            PhaseType.PARAMETER_SEARCH: ['method', 'param_definitions', 'n_iterations'],
            PhaseType.OPTIMIZATION: ['objective', 'constraints'],
            PhaseType.VALIDATION: ['validation_method', 'metrics']
        }
        
        if self.phase_type in required_keys:
            for key in required_keys[self.phase_type]:
                if key not in self.config:
                    raise ValueError(f"Missing required key '{key}' for {self.phase_type}")
        
        return True

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    name: str
    version: str
    description: str
    
    # Phases
    phases: List[PhaseDefinition]
    
    # Global configuration
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    parallel_phases: bool = True
    checkpoint_enabled: bool = True
    checkpoint_frequency: int = 1  # After every N phases
    
    # Resource limits
    max_total_memory_gb: float = 32.0
    max_total_cores: int = 16
    
    # Output settings
    output_directory: str = "./workflow_outputs"
    save_intermediate_results: bool = True
    compression: str = "gzip"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'WorkflowDefinition':
        """Load workflow definition from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse phases
        phases = []
        for phase_data in data.get('phases', []):
            phase = PhaseDefinition(
                name=phase_data['name'],
                phase_type=PhaseType(phase_data['type']),
                config=phase_data.get('config', {}),
                depends_on=phase_data.get('depends_on', []),
                **{k: v for k, v in phase_data.items() 
                   if k not in ['name', 'type', 'config', 'depends_on']}
            )
            phases.append(phase)
        
        return cls(
            name=data['name'],
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            phases=phases,
            **{k: v for k, v in data.items() 
               if k not in ['name', 'version', 'description', 'phases']}
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save workflow definition to YAML file"""
        data = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'phases': [
                {
                    'name': phase.name,
                    'type': phase.phase_type.value,
                    'config': phase.config,
                    'depends_on': phase.depends_on,
                    'enabled': phase.enabled,
                    'memory_gb': phase.memory_gb,
                    'cpu_cores': phase.cpu_cores
                }
                for phase in self.phases
            ],
            'global_config': self.global_config,
            'parallel_phases': self.parallel_phases,
            'checkpoint_enabled': self.checkpoint_enabled,
            'output_directory': self.output_directory
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate entire workflow definition"""
        # Check phase dependencies
        phase_names = {phase.name for phase in self.phases}
        for phase in self.phases:
            for dep in phase.depends_on:
                if dep not in phase_names:
                    raise ValueError(f"Phase '{phase.name}' depends on unknown phase '{dep}'")
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise ValueError("Circular dependencies detected in workflow")
        
        # Validate individual phases
        for phase in self.phases:
            phase.validate_config()
        
        # Check resource limits
        total_memory = sum(p.memory_gb for p in self.phases if p.enabled)
        total_cores = sum(p.cpu_cores for p in self.phases if p.enabled)
        
        if not self.parallel_phases:
            # Sequential execution - check max per phase
            max_memory = max(p.memory_gb for p in self.phases if p.enabled)
            max_cores = max(p.cpu_cores for p in self.phases if p.enabled)
            if max_memory > self.max_total_memory_gb:
                raise ValueError(f"Phase memory requirement {max_memory}GB exceeds limit")
            if max_cores > self.max_total_cores:
                raise ValueError(f"Phase CPU requirement {max_cores} exceeds limit")
        else:
            # Parallel execution - check total
            if total_memory > self.max_total_memory_gb:
                raise ValueError(f"Total memory requirement {total_memory}GB exceeds limit")
            if total_cores > self.max_total_cores:
                raise ValueError(f"Total CPU requirement {total_cores} exceeds limit")
        
        return True
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""
        # Build adjacency list
        graph = {phase.name: phase.depends_on for phase in self.phases}
        
        # Track visit states
        white = set(graph.keys())  # Not visited
        gray = set()  # Currently visiting
        black = set()  # Visited
        
        def visit(node):
            if node in black:
                return False
            if node in gray:
                return True  # Cycle detected
            
            gray.add(node)
            white.discard(node)
            
            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True
            
            gray.remove(node)
            black.add(node)
            return False
        
        while white:
            if visit(white.pop()):
                return True
        
        return False
```

### 2. Workflow Executor

```python
# src/core/coordinator/workflow_executor.py
class WorkflowExecutor:
    """
    Executes multi-phase workflows with state management.
    Handles orchestration, checkpointing, and error recovery.
    """
    
    def __init__(self, workflow: WorkflowDefinition):
        self.workflow = workflow
        self.phase_executors = self._create_phase_executors()
        self.state_manager = WorkflowStateManager(workflow.output_directory)
        self.resource_manager = ResourceManager(
            max_memory_gb=workflow.max_total_memory_gb,
            max_cores=workflow.max_total_cores
        )
        self.logger = ComponentLogger("WorkflowExecutor", workflow.name)
        
        # Execution tracking
        self.execution_id = self._generate_execution_id()
        self.start_time = None
        self.phase_results: Dict[str, PhaseResult] = {}
        self.execution_graph = self._build_execution_graph()
    
    def _create_phase_executors(self) -> Dict[str, PhaseExecutor]:
        """Create executor for each phase type"""
        executors = {}
        
        for phase in self.workflow.phases:
            if phase.phase_type == PhaseType.DATA_PREPARATION:
                executor = DataPreparationExecutor(phase)
            elif phase.phase_type == PhaseType.PARAMETER_SEARCH:
                executor = ParameterSearchExecutor(phase)
            elif phase.phase_type == PhaseType.OPTIMIZATION:
                executor = OptimizationExecutor(phase)
            elif phase.phase_type == PhaseType.VALIDATION:
                executor = ValidationExecutor(phase)
            else:
                executor = GenericPhaseExecutor(phase)
            
            executors[phase.name] = executor
        
        return executors
    
    def execute(self, resume_from_checkpoint: bool = False) -> WorkflowResult:
        """Execute the complete workflow"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting workflow execution: {self.execution_id}")
        
        # Load checkpoint if resuming
        if resume_from_checkpoint:
            checkpoint = self.state_manager.load_latest_checkpoint()
            if checkpoint:
                self.phase_results = checkpoint.completed_phases
                self.logger.info(f"Resuming from checkpoint with {len(self.phase_results)} completed phases")
        
        # Validate workflow
        self.workflow.validate()
        
        try:
            # Execute phases according to dependencies
            if self.workflow.parallel_phases:
                result = self._execute_parallel()
            else:
                result = self._execute_sequential()
            
            # Final aggregation
            final_result = self._aggregate_results()
            
            # Save final results
            self.state_manager.save_final_results(self.execution_id, final_result)
            
            self.logger.info(f"Workflow completed successfully in {self._get_elapsed_time()}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            self._handle_failure(e)
            raise
    
    def _execute_sequential(self) -> WorkflowResult:
        """Execute phases sequentially"""
        execution_order = self._topological_sort()
        
        for phase_name in execution_order:
            if phase_name in self.phase_results:
                self.logger.info(f"Skipping completed phase: {phase_name}")
                continue
            
            phase = next(p for p in self.workflow.phases if p.name == phase_name)
            if not phase.enabled:
                self.logger.info(f"Skipping disabled phase: {phase_name}")
                continue
            
            self.logger.info(f"Executing phase: {phase_name}")
            
            # Check dependencies
            for dep in phase.depends_on:
                if dep not in self.phase_results:
                    raise RuntimeError(f"Dependency '{dep}' not satisfied for phase '{phase_name}'")
            
            # Allocate resources
            with self.resource_manager.allocate(phase.memory_gb, phase.cpu_cores):
                # Execute phase
                executor = self.phase_executors[phase_name]
                
                # Pass results from dependencies
                dependency_results = {
                    dep: self.phase_results[dep] 
                    for dep in phase.depends_on
                }
                
                result = self._execute_phase_with_retry(
                    executor, dependency_results, phase
                )
                
                self.phase_results[phase_name] = result
                
                # Checkpoint if enabled
                if self.workflow.checkpoint_enabled and \
                   len(self.phase_results) % self.workflow.checkpoint_frequency == 0:
                    self._save_checkpoint()
        
        return self._create_workflow_result()
    
    def _execute_parallel(self) -> WorkflowResult:
        """Execute phases in parallel where possible"""
        execution_graph = self._build_execution_graph()
        completed = set(self.phase_results.keys())
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            
            while len(completed) < len([p for p in self.workflow.phases if p.enabled]):
                # Find phases ready to execute
                ready_phases = []
                for phase in self.workflow.phases:
                    if phase.name in completed or not phase.enabled:
                        continue
                    
                    # Check if dependencies are satisfied
                    if all(dep in completed for dep in phase.depends_on):
                        ready_phases.append(phase)
                
                # Submit ready phases
                for phase in ready_phases:
                    if phase.name not in futures:
                        future = executor.submit(
                            self._execute_phase_wrapper, phase
                        )
                        futures[future] = phase.name
                
                # Wait for any phase to complete
                if futures:
                    done, pending = concurrent.futures.wait(
                        futures.keys(), 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        phase_name = futures.pop(future)
                        try:
                            result = future.result()
                            self.phase_results[phase_name] = result
                            completed.add(phase_name)
                            
                            self.logger.info(f"Phase '{phase_name}' completed")
                            
                            # Checkpoint if needed
                            if self.workflow.checkpoint_enabled:
                                self._save_checkpoint()
                                
                        except Exception as e:
                            self.logger.error(f"Phase '{phase_name}' failed: {e}")
                            raise
                
                # Avoid busy waiting
                if not futures and len(completed) < len([p for p in self.workflow.phases if p.enabled]):
                    time.sleep(0.1)
        
        return self._create_workflow_result()
    
    def _execute_phase_wrapper(self, phase: PhaseDefinition) -> PhaseResult:
        """Wrapper for parallel phase execution"""
        # Allocate resources
        self.resource_manager.wait_for_resources(phase.memory_gb, phase.cpu_cores)
        
        with self.resource_manager.allocate(phase.memory_gb, phase.cpu_cores):
            executor = self.phase_executors[phase.name]
            
            dependency_results = {
                dep: self.phase_results[dep] 
                for dep in phase.depends_on
            }
            
            return self._execute_phase_with_retry(
                executor, dependency_results, phase
            )
    
    def _execute_phase_with_retry(self, executor: PhaseExecutor,
                                 dependency_results: Dict[str, PhaseResult],
                                 phase: PhaseDefinition) -> PhaseResult:
        """Execute phase with retry logic"""
        last_error = None
        
        for attempt in range(phase.max_retries):
            try:
                self.logger.info(f"Executing phase '{phase.name}' (attempt {attempt + 1})")
                
                # Set timeout if specified
                if phase.timeout_minutes:
                    signal.alarm(phase.timeout_minutes * 60)
                
                # Execute
                result = executor.execute(
                    dependency_results,
                    self.workflow.global_config
                )
                
                # Validate result
                if phase.validation_func:
                    validator = self._get_validator(phase.validation_func)
                    if not validator(result):
                        raise ValueError("Phase validation failed")
                
                # Check success criteria
                if phase.success_criteria:
                    if not self._check_success_criteria(result, phase.success_criteria):
                        raise ValueError("Success criteria not met")
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Phase '{phase.name}' failed on attempt {attempt + 1}: {e}"
                )
                
                if attempt < phase.max_retries - 1 and phase.retry_on_failure:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    break
        
        raise RuntimeError(f"Phase '{phase.name}' failed after {phase.max_retries} attempts: {last_error}")
```

### 3. State Management

```python
# src/core/coordinator/workflow_state.py
class WorkflowStateManager:
    """
    Manages workflow state, checkpoints, and results.
    Enables resume functionality and result persistence.
    """
    
    def __init__(self, output_directory: str):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger = ComponentLogger("WorkflowStateManager", "state")
    
    def save_checkpoint(self, execution_id: str, 
                       completed_phases: Dict[str, PhaseResult],
                       metadata: Dict[str, Any]) -> str:
        """Save workflow checkpoint"""
        checkpoint = WorkflowCheckpoint(
            execution_id=execution_id,
            timestamp=datetime.now(),
            completed_phases=completed_phases,
            metadata=metadata
        )
        
        # Generate checkpoint filename
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{execution_id}_{checkpoint.timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as JSON for readability
        json_file = checkpoint_file.with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_file}")
        
        return str(checkpoint_file)
    
    def load_latest_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        """Load most recent checkpoint"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        self.logger.info(f"Loading checkpoint: {latest_file}")
        
        with open(latest_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    def save_phase_result(self, execution_id: str, phase_name: str, 
                         result: PhaseResult) -> None:
        """Save individual phase result"""
        phase_dir = self.results_dir / execution_id / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_file = phase_dir / "result.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Save artifacts
        if hasattr(result, 'artifacts'):
            artifacts_dir = phase_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            
            for name, artifact in result.artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_parquet(artifacts_dir / f"{name}.parquet")
                elif isinstance(artifact, dict):
                    with open(artifacts_dir / f"{name}.json", 'w') as f:
                        json.dump(artifact, f, indent=2, default=str)
                elif isinstance(artifact, np.ndarray):
                    np.save(artifacts_dir / f"{name}.npy", artifact)
        
        # Save summary
        if hasattr(result, 'summary'):
            with open(phase_dir / "summary.json", 'w') as f:
                json.dump(result.summary(), f, indent=2, default=str)
    
    def save_final_results(self, execution_id: str, 
                          workflow_result: WorkflowResult) -> None:
        """Save final workflow results"""
        results_file = self.results_dir / execution_id / "workflow_result.pkl"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'wb') as f:
            pickle.dump(workflow_result, f)
        
        # Generate report
        report = self._generate_workflow_report(workflow_result)
        report_file = self.results_dir / execution_id / "workflow_report.html"
        report_file.write_text(report)
        
        self.logger.info(f"Saved final results: {results_file}")
```

### 4. Phase Executors

```python
# src/core/coordinator/phase_executors.py
class PhaseExecutor(ABC):
    """Base class for phase executors"""
    
    def __init__(self, phase_definition: PhaseDefinition):
        self.phase_def = phase_definition
        self.logger = ComponentLogger(f"PhaseExecutor_{phase_definition.name}", 
                                    phase_definition.name)
    
    @abstractmethod
    def execute(self, dependency_results: Dict[str, PhaseResult],
               global_config: Dict[str, Any]) -> PhaseResult:
        """Execute the phase"""
        pass

class ParameterSearchExecutor(PhaseExecutor):
    """Executor for parameter search phases"""
    
    def execute(self, dependency_results: Dict[str, PhaseResult],
               global_config: Dict[str, Any]) -> PhaseResult:
        """Execute parameter search"""
        config = self.phase_def.config
        
        # Get signals from previous phase if available
        signals = None
        for dep_name, dep_result in dependency_results.items():
            if hasattr(dep_result, 'captured_signals'):
                signals = dep_result.captured_signals
                break
        
        if signals is None:
            raise ValueError("No captured signals found in dependencies")
        
        # Create replay engine
        replay_engine = SignalReplayEngine(MemoryStorageBackend())
        replay_engine.loaded_signals = signals
        
        # Setup parameter definitions
        param_defs = self._create_param_definitions(config['param_definitions'])
        
        # Create optimizer
        optimizer = ReplayParameterOptimizer(replay_engine, param_defs)
        
        # Run optimization
        start_time = time.time()
        
        result = optimizer.optimize_with_method(
            method=config['method'],
            n_iterations=config['n_iterations'],
            objective=config.get('objective', 'sharpe_ratio'),
            constraints=config.get('constraints')
        )
        
        elapsed_time = time.time() - start_time
        
        # Create phase result
        return ParameterSearchResult(
            phase_name=self.phase_def.name,
            best_params=result.best_config,
            best_value=result.best_value,
            all_results=result.all_results,
            method=config['method'],
            n_iterations=config['n_iterations'],
            elapsed_time=elapsed_time,
            artifacts={
                'optimization_history': result.convergence_history,
                'parameter_importance': self._analyze_parameter_importance(result)
            }
        )
    
    def _create_param_definitions(self, config_defs: Dict) -> Dict[str, ParameterDefinition]:
        """Create parameter definitions from config"""
        param_defs = {}
        
        for name, def_config in config_defs.items():
            param_defs[name] = ParameterDefinition(
                name=name,
                param_type=def_config['type'],
                bounds=def_config['bounds'],
                default_value=def_config.get('default'),
                distribution=def_config.get('distribution', 'uniform')
            )
        
        return param_defs

class ValidationExecutor(PhaseExecutor):
    """Executor for validation phases"""
    
    def execute(self, dependency_results: Dict[str, PhaseResult],
               global_config: Dict[str, Any]) -> PhaseResult:
        """Execute validation phase"""
        config = self.phase_def.config
        
        # Get optimization results
        opt_results = None
        for dep_name, dep_result in dependency_results.items():
            if isinstance(dep_result, ParameterSearchResult):
                opt_results = dep_result
                break
        
        if opt_results is None:
            raise ValueError("No optimization results found in dependencies")
        
        # Get validation data
        validation_data = self._load_validation_data(config)
        
        # Run validation methods
        validation_results = {}
        
        if 'monte_carlo' in config['validation_method']:
            mc_validator = MonteCarloValidator(MonteCarloConfig(
                n_simulations=config.get('n_simulations', 1000)
            ))
            validation_results['monte_carlo'] = mc_validator.validate_strategy(
                opt_results.best_backtest_result
            )
        
        if 'bootstrap' in config['validation_method']:
            bootstrap_validator = BootstrapValidator(
                n_bootstrap=config.get('n_bootstrap', 10000)
            )
            validation_results['bootstrap'] = bootstrap_validator.validate_strategy_metrics(
                opt_results.best_backtest_result.trades
            )
        
        if 'walk_forward' in config['validation_method']:
            # Run walk-forward validation
            wf_results = self._run_walk_forward_validation(
                opt_results.best_params, validation_data, config
            )
            validation_results['walk_forward'] = wf_results
        
        # Aggregate validation results
        is_valid = self._assess_validation(validation_results, config.get('thresholds', {}))
        
        return ValidationResult(
            phase_name=self.phase_def.name,
            validation_methods=config['validation_method'],
            validation_results=validation_results,
            is_valid=is_valid,
            metrics=self._extract_validation_metrics(validation_results),
            artifacts={
                'validation_report': self._generate_validation_report(validation_results)
            }
        )
```

### 5. Result Aggregation and Analysis

```python
# src/core/coordinator/result_aggregation.py
class WorkflowResultAggregator:
    """
    Aggregates results across workflow phases.
    Provides comprehensive analysis and reporting.
    """
    
    def __init__(self, phase_results: Dict[str, PhaseResult]):
        self.phase_results = phase_results
        self.logger = ComponentLogger("WorkflowResultAggregator", "aggregation")
    
    def aggregate(self) -> WorkflowResult:
        """Aggregate all phase results into final workflow result"""
        # Extract key results
        optimization_results = self._extract_optimization_results()
        validation_results = self._extract_validation_results()
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics()
        
        # Generate insights
        insights = self._generate_insights()
        
        # Create comprehensive result
        return WorkflowResult(
            phase_results=self.phase_results,
            optimization_summary=optimization_results,
            validation_summary=validation_results,
            metrics=summary_metrics,
            insights=insights,
            execution_metadata=self._get_execution_metadata()
        )
    
    def _extract_optimization_results(self) -> Dict[str, Any]:
        """Extract and summarize optimization results"""
        opt_results = {}
        
        for phase_name, result in self.phase_results.items():
            if isinstance(result, ParameterSearchResult):
                opt_results[phase_name] = {
                    'best_params': result.best_params,
                    'best_value': result.best_value,
                    'improvement': self._calculate_improvement(result),
                    'convergence_speed': self._analyze_convergence(result),
                    'parameter_sensitivity': result.artifacts.get('parameter_importance')
                }
        
        return opt_results
    
    def _generate_insights(self) -> List[str]:
        """Generate actionable insights from results"""
        insights = []
        
        # Parameter insights
        param_importance = self._aggregate_parameter_importance()
        if param_importance:
            top_params = sorted(param_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:3]
            insights.append(
                f"Most important parameters: {', '.join([p[0] for p in top_params])}"
            )
        
        # Performance insights
        performance_summary = self._analyze_performance_across_phases()
        if performance_summary['improvement_trend'] > 0:
            insights.append(
                f"Performance improved {performance_summary['improvement_trend']:.1%} "
                f"through optimization phases"
            )
        
        # Validation insights
        validation_summary = self._extract_validation_results()
        if validation_summary:
            robustness_score = validation_summary.get('robustness_score', 0)
            if robustness_score > 0.8:
                insights.append("Strategy shows high robustness in validation")
            elif robustness_score < 0.5:
                insights.append("WARNING: Strategy shows low robustness - consider revision")
        
        # Resource usage insights
        resource_usage = self._analyze_resource_usage()
        if resource_usage['memory_efficiency'] < 0.5:
            insights.append(
                "Consider memory optimization - current efficiency is "
                f"{resource_usage['memory_efficiency']:.1%}"
            )
        
        return insights
    
    def generate_report(self) -> str:
        """Generate comprehensive HTML report"""
        from jinja2 import Template
        
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Workflow Execution Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .phase { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                .metric { display: inline-block; margin: 10px; padding: 5px; }
                .insight { background: #f0f0f0; padding: 10px; margin: 5px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            </style>
        </head>
        <body>
            <h1>Workflow Execution Report</h1>
            
            <h2>Executive Summary</h2>
            <div class="summary">
                <p>Total Phases: {{ phase_count }}</p>
                <p>Execution Time: {{ execution_time }}</p>
                <p>Final Performance: {{ final_performance }}</p>
            </div>
            
            <h2>Phase Results</h2>
            {% for phase_name, result in phases.items() %}
            <div class="phase">
                <h3>{{ phase_name }}</h3>
                <p>Type: {{ result.phase_type }}</p>
                <p>Status: {{ result.status }}</p>
                {% if result.metrics %}
                <h4>Metrics:</h4>
                <table>
                    {% for metric, value in result.metrics.items() %}
                    <tr><td>{{ metric }}</td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endfor %}
            
            <h2>Insights</h2>
            {% for insight in insights %}
            <div class="insight">{{ insight }}</div>
            {% endfor %}
            
            <h2>Resource Usage</h2>
            <canvas id="resourceChart"></canvas>
            
            <script>
                // Add charts using Chart.js or similar
            </script>
        </body>
        </html>
        """)
        
        # Prepare template data
        template_data = {
            'phase_count': len(self.phase_results),
            'execution_time': self._format_execution_time(),
            'final_performance': self._get_final_performance(),
            'phases': self._prepare_phase_data(),
            'insights': self._generate_insights()
        }
        
        return template.render(**template_data)
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_workflow.py`:

```python
class TestWorkflowDefinition:
    """Test workflow definition and validation"""
    
    def test_workflow_creation(self):
        """Test creating workflow definition"""
        phases = [
            PhaseDefinition(
                name="data_prep",
                phase_type=PhaseType.DATA_PREPARATION,
                config={'start_date': '2024-01-01', 'end_date': '2024-12-31'}
            ),
            PhaseDefinition(
                name="optimization",
                phase_type=PhaseType.OPTIMIZATION,
                config={'method': 'bayesian', 'n_iterations': 100},
                depends_on=['data_prep']
            )
        ]
        
        workflow = WorkflowDefinition(
            name="test_workflow",
            version="1.0",
            description="Test workflow",
            phases=phases
        )
        
        assert workflow.validate()
        assert len(workflow.phases) == 2
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies"""
        phases = [
            PhaseDefinition("A", PhaseType.OPTIMIZATION, {}, depends_on=["B"]),
            PhaseDefinition("B", PhaseType.OPTIMIZATION, {}, depends_on=["C"]),
            PhaseDefinition("C", PhaseType.OPTIMIZATION, {}, depends_on=["A"])
        ]
        
        workflow = WorkflowDefinition("circular", "1.0", "", phases)
        
        with pytest.raises(ValueError, match="Circular dependencies"):
            workflow.validate()

class TestWorkflowExecutor:
    """Test workflow execution"""
    
    def test_sequential_execution(self):
        """Test sequential phase execution"""
        workflow = create_test_workflow(parallel=False)
        executor = WorkflowExecutor(workflow)
        
        result = executor.execute()
        
        # All phases should complete
        assert len(result.phase_results) == len(workflow.phases)
        
        # Check execution order
        execution_times = [
            (name, result.phase_results[name].start_time)
            for name in result.phase_results
        ]
        
        # Should be in dependency order
        assert execution_times[0][0] == "data_prep"
        assert execution_times[1][0] == "param_search"
```

### Integration Tests

Create `tests/integration/test_step10_workflow_integration.py`:

```python
def test_complete_optimization_workflow():
    """Test full optimization workflow end-to-end"""
    # Create workflow definition
    workflow = WorkflowDefinition.from_yaml("test_configs/optimization_workflow.yaml")
    
    # Execute workflow
    executor = WorkflowExecutor(workflow)
    result = executor.execute()
    
    # Verify all phases completed
    assert all(phase.name in result.phase_results for phase in workflow.phases)
    
    # Check optimization improved performance
    initial_performance = result.phase_results['initial_backtest'].metrics['sharpe_ratio']
    final_performance = result.phase_results['final_validation'].metrics['sharpe_ratio']
    assert final_performance > initial_performance
    
    # Verify checkpointing worked
    checkpoint_files = list(Path(workflow.output_directory).glob("checkpoints/*.pkl"))
    assert len(checkpoint_files) > 0

def test_workflow_resume_from_checkpoint():
    """Test resuming workflow from checkpoint"""
    workflow = create_test_workflow()
    executor = WorkflowExecutor(workflow)
    
    # Simulate partial execution
    executor._execute_sequential()
    
    # Force failure after first phase
    if len(executor.phase_results) >= 1:
        raise Exception("Simulated failure")
    
    # Create new executor and resume
    new_executor = WorkflowExecutor(workflow)
    result = new_executor.execute(resume_from_checkpoint=True)
    
    # Should complete successfully
    assert len(result.phase_results) == len(workflow.phases)
    
    # First phase should not be re-executed
    # (Check by comparing timestamps or execution counts)
```

### System Tests

Create `tests/system/test_step10_production_workflow.py`:

```python
def test_production_scale_workflow():
    """Test workflow at production scale"""
    # Create large-scale workflow
    workflow_config = {
        'name': 'production_optimization',
        'phases': [
            {
                'name': 'data_prep',
                'type': 'data_preparation',
                'config': {
                    'symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],
                    'start_date': '2020-01-01',
                    'end_date': '2024-01-01'
                }
            },
            {
                'name': 'feature_engineering',
                'type': 'feature_engineering',
                'config': {
                    'features': ['momentum', 'volatility', 'volume', 'correlation'],
                    'lookback_periods': [5, 10, 20, 50, 100]
                },
                'depends_on': ['data_prep']
            },
            {
                'name': 'signal_capture',
                'type': 'signal_capture',
                'config': {
                    'strategies': ['momentum', 'mean_reversion', 'pairs'],
                    'capture_context': True
                },
                'depends_on': ['feature_engineering']
            },
            {
                'name': 'param_optimization',
                'type': 'parameter_search',
                'config': {
                    'method': 'bayesian',
                    'n_iterations': 500,
                    'param_definitions': {
                        'risk_level': {'type': 'continuous', 'bounds': [0.01, 0.05]},
                        'position_size': {'type': 'continuous', 'bounds': [0.05, 0.25]},
                        'stop_loss': {'type': 'continuous', 'bounds': [0.02, 0.10]}
                    }
                },
                'depends_on': ['signal_capture'],
                'memory_gb': 16,
                'cpu_cores': 8
            },
            {
                'name': 'validation',
                'type': 'validation',
                'config': {
                    'validation_method': ['monte_carlo', 'bootstrap', 'walk_forward'],
                    'n_simulations': 1000,
                    'confidence_level': 0.95
                },
                'depends_on': ['param_optimization']
            }
        ],
        'parallel_phases': True,
        'checkpoint_enabled': True,
        'max_total_memory_gb': 32,
        'max_total_cores': 16
    }
    
    workflow = WorkflowDefinition.from_dict(workflow_config)
    executor = WorkflowExecutor(workflow)
    
    # Monitor resource usage
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    
    # Execute workflow
    start_time = time.time()
    result = executor.execute()
    execution_time = time.time() - start_time
    
    # Stop monitoring
    resource_stats = resource_monitor.stop()
    
    # Verify completion
    assert result.is_complete
    assert all(phase_result.is_successful for phase_result in result.phase_results.values())
    
    # Check performance
    assert execution_time < 3600  # Should complete within 1 hour
    assert resource_stats['peak_memory_gb'] < 32
    assert resource_stats['avg_cpu_utilization'] > 0.5  # Good utilization
    
    # Verify results quality
    final_sharpe = result.optimization_summary['param_optimization']['best_value']
    assert final_sharpe > 1.0  # Reasonable performance
    
    # Check validation passed
    assert result.validation_summary['validation']['is_valid']
```

## ✅ Validation Checklist

### Workflow Definition
- [ ] YAML loading/saving works
- [ ] Dependency validation correct
- [ ] Circular dependency detection
- [ ] Resource limits enforced

### Execution Flow
- [ ] Sequential execution correct
- [ ] Parallel execution efficient
- [ ] Dependencies respected
- [ ] Error handling robust

### State Management
- [ ] Checkpointing works
- [ ] Resume functionality
- [ ] Results persisted
- [ ] Artifacts saved

### Integration
- [ ] All phase types supported
- [ ] Result aggregation accurate
- [ ] Report generation works
- [ ] Resource monitoring functional

## 📊 Performance Considerations

### Optimization Strategies
```python
class OptimizedWorkflowExecutor(WorkflowExecutor):
    """Optimized executor with advanced features"""
    
    def __init__(self, workflow):
        super().__init__(workflow)
        self.execution_cache = {}
        self.result_cache = TTLCache(maxsize=1000, ttl=3600)
    
    def _execute_with_caching(self, phase):
        cache_key = self._get_cache_key(phase)
        
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        result = super()._execute_phase(phase)
        self.result_cache[cache_key] = result
        
        return result
```

### Resource Optimization
- Dynamic resource allocation
- Memory pooling
- Lazy loading of data
- Result streaming

## 🐛 Common Issues

1. **Phase Dependencies**
   - Validate dependency graph
   - Handle missing dependencies
   - Detect circular references

2. **Resource Exhaustion**
   - Monitor memory usage
   - Implement resource limits
   - Use spillover to disk

3. **Long Running Workflows**
   - Implement timeouts
   - Add progress tracking
   - Support cancellation

## 🎯 Success Criteria

Step 10 is complete when:
1. ✅ Workflow definition flexible and validated
2. ✅ Execution engine handles all phase types
3. ✅ State management enables resume
4. ✅ Result aggregation comprehensive
5. ✅ Production-scale testing passes

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.8: Memory & Batch Processing](step-10.8-memory-batch.md)

## 📚 Additional Resources

- [Workflow Patterns](../references/workflow-patterns.md)
- [State Management Best Practices](../references/state-management.md)
- [Distributed Workflow Orchestration](../references/distributed-orchestration.md)