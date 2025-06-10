"""
Declarative Workflow Manager

Interprets workflow patterns defined in YAML rather than Python code.
Handles multi-phase workflows with dependencies, inputs/outputs, and conditional logic.
"""

from typing import Dict, Any, Optional, List, Set
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime
import re
import uuid

from .protocols import PhaseConfig, TopologyBuilderProtocol, WorkflowProtocol, SequenceProtocol
from .topology import TopologyBuilder
from .sequencer import Sequencer
from .config.pattern_loader import PatternLoader
from .config.resolver import ConfigResolver
from ..components.discovery import discover_components

# Import Pydantic validation (with fallback)
from .config import (
    PYDANTIC_AVAILABLE,
    WorkflowConfig,
    validate_workflow_dict,
    get_validation_errors
)

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Executes workflows based on declarative patterns.
    
    Workflows define multi-phase processes with:
    - Phase dependencies
    - Input/output data flow
    - Conditional execution
    - Dynamic configuration
    """
    
    def __init__(self, 
                 topology_builder: Optional[TopologyBuilderProtocol] = None,
                 sequencer: Optional[Any] = None,
                 shared_services: Optional[Dict[str, Any]] = None,
                 pattern_loader: Optional[PatternLoader] = None,
                 config_resolver: Optional[ConfigResolver] = None):
        """Initialize workflow manager."""
        self.coordinator_id = str(uuid.uuid4())
        self.shared_services = shared_services or {}
        
        # Shared components
        self.pattern_loader = pattern_loader or PatternLoader()
        self.config_resolver = config_resolver or ConfigResolver()
        
        # Component discovery
        self.discovered_workflows = self._discover_workflows()
        self.discovered_sequences = self._discover_sequences()
        
        # Core components  
        self.topology_builder = topology_builder or TopologyBuilder(self.pattern_loader, self.config_resolver)
        self.sequencer = sequencer or Sequencer(self.topology_builder, self.pattern_loader, self.config_resolver)
        
        # Pattern loading
        self.workflow_patterns = self.pattern_loader.load_patterns('workflows')
        self.phase_outputs = {}  # Store outputs from completed phases
        
        # Event tracing support
        self.event_tracer = None
        self._setup_event_tracing()
        
        logger.info(f"Coordinator {self.coordinator_id} initialized with "
                   f"{len(self.discovered_workflows)} discovered workflows, "
                   f"{len(self.discovered_sequences)} discovered sequences, "
                   f"and {len(self.workflow_patterns)} workflow patterns")
        
    
    def _discover_workflows(self) -> Dict[str, WorkflowProtocol]:
        """Discover all available workflows."""
        try:
            components = discover_components(
                "src.core.coordinator.workflows",
                protocol_type=WorkflowProtocol,
                component_type="workflow"
            )
            return {name: comp() for name, comp in components.items()}
        except Exception as e:
            logger.warning(f"Failed to discover workflows: {e}")
            return {}
    
    def _discover_sequences(self) -> Dict[str, SequenceProtocol]:
        """Discover all available sequences."""
        try:
            components = discover_components(
                "src.core.coordinator.sequences",
                protocol_type=SequenceProtocol,
                component_type="sequence"
            )
            return {name: comp() for name, comp in components.items()}
        except Exception as e:
            logger.warning(f"Failed to discover sequences: {e}")
            return {}
    
    def _setup_event_tracing(self):
        """Setup event tracing if enabled."""
        if self.shared_services.get('enable_event_tracing', False):
            try:
                from ..events.tracing import EventTracer
                self.event_tracer = EventTracer(
                    enabled=True,
                    trace_dir=self.shared_services.get('trace_dir', './traces')
                )
                logger.info("Event tracing initialized")
            except ImportError:
                logger.warning("Event tracing not available")
    
    def run_workflow(self, workflow_definition: Dict[str, Any],
                    workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a workflow from definition.
        
        Args:
            workflow_definition: Dict containing:
                - workflow: Name of workflow pattern or inline definition
                - config: User configuration
                - context: Additional context
            workflow_id: Optional workflow ID for tracking
                
        Returns:
            Workflow execution results
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        # Extract configuration
        if isinstance(workflow_definition, dict) and 'config' in workflow_definition:
            config = workflow_definition['config']
            workflow_spec = workflow_definition.get('workflow', 'simple_backtest')
        else:
            # Backward compatibility - treat entire dict as config
            config = workflow_definition
            workflow_spec = config.get('workflow', 'simple_backtest')
        
        # VALIDATE CONFIGURATION FIRST (if Pydantic available)
        if PYDANTIC_AVAILABLE and WorkflowConfig:
            try:
                validation_errors = get_validation_errors(config)
                if validation_errors:
                    logger.error(f"Configuration validation failed for workflow {workflow_id}")
                    for error in validation_errors:
                        logger.error(f"  - {error}")
                    
                    return {
                        'success': False,
                        'workflow_id': workflow_id,
                        'error': 'Configuration validation failed',
                        'validation_errors': validation_errors,
                        'summary': {
                            'workflow': workflow_spec,
                            'success': False,
                            'validation_failed': True,
                            'error_count': len(validation_errors)
                        }
                    }
                else:
                    logger.info(f"Configuration validation passed for workflow {workflow_id}")
            except Exception as e:
                logger.warning(f"Configuration validation failed with error: {e}")
                logger.debug("Proceeding without validation")
        else:
            logger.debug("Pydantic validation not available, skipping validation")
        
        # Apply trace level presets if specified
        config = self._apply_trace_level_config(config)
        
        # Get workflow pattern
        if isinstance(workflow_spec, str):
            pattern_name = workflow_spec
            
            # Check discovered workflows first (for composable support)
            if pattern_name in self.discovered_workflows:
                return self._execute_discovered_workflow(pattern_name, config, workflow_id)
            
            # Then check patterns
            pattern = self.workflow_patterns.get(pattern_name)
            if not pattern:
                raise ValueError(f"Unknown workflow: {pattern_name}")
        else:
            # Inline workflow definition
            pattern = workflow_spec
        
        workflow_name = pattern.get('name', pattern_name if isinstance(workflow_spec, str) else 'unnamed_workflow')
        logger.info(f"Starting workflow: {workflow_name} (ID: {workflow_id})")
        
        # Set instance variables for results path building
        self._current_workflow_id = workflow_name
        self._current_phase_name = None
        
        # Initialize workflow context
        workflow_context = {
            'workflow_name': workflow_name,
            'workflow_id': workflow_id,
            'start_time': start_time,
            'config': config,
            'pattern': pattern,
            'phase_outputs': {},
            'phase_results': {}
        }
        
        # Execute phases
        phases = pattern.get('phases', [])
        results = {
            'workflow': workflow_name,
            'phases': {},
            'success': True,
            'outputs': {}
        }
        
        for phase_def in phases:
            phase_name = phase_def.get('name')
            
            # Check dependencies
            if not self._check_dependencies(phase_def, workflow_context):
                logger.info(f"Skipping phase {phase_name} - dependencies not met")
                continue
            
            # Check conditions
            if not self._check_conditions(phase_def, workflow_context):
                logger.info(f"Skipping phase {phase_name} - conditions not met")
                continue
            
            # Execute phase
            logger.info(f"Executing phase: {phase_name}")
            self._current_phase_name = phase_name
            phase_result = self._execute_phase(phase_def, workflow_context)
            
            # Store results
            results['phases'][phase_name] = phase_result
            workflow_context['phase_results'][phase_name] = phase_result
            
            # Handle outputs
            if 'outputs' in phase_def:
                phase_outputs = self._process_outputs(
                    phase_def['outputs'], 
                    phase_result, 
                    workflow_context
                )
                workflow_context['phase_outputs'][phase_name] = phase_outputs
                results['outputs'][phase_name] = phase_outputs
            
            # Check if phase failed
            if not phase_result.get('success', True):
                results['success'] = False
                if phase_def.get('required', True):
                    logger.error(f"Required phase {phase_name} failed, stopping workflow")
                    break
        
        # Process workflow-level outputs
        if 'outputs' in pattern:
            workflow_outputs = self._process_outputs(
                pattern['outputs'],
                results,
                workflow_context
            )
            results['outputs']['workflow'] = workflow_outputs
        
        # Add summary
        results['summary'] = self._create_summary(results, workflow_context)
        
        return results
    
    def _check_dependencies(self, phase_def: Dict[str, Any], 
                           context: Dict[str, Any]) -> bool:
        """Check if phase dependencies are satisfied."""
        depends_on = phase_def.get('depends_on', [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        for dep in depends_on:
            if dep not in context['phase_results']:
                return False
            if not context['phase_results'][dep].get('success', True):
                return False
        
        return True
    
    def _check_conditions(self, phase_def: Dict[str, Any], 
                         context: Dict[str, Any]) -> bool:
        """Check if phase conditions are met."""
        conditions = phase_def.get('conditions', [])
        if not conditions:
            return True
        
        if isinstance(conditions, dict):
            conditions = [conditions]
        
        for condition in conditions:
            if not self._evaluate_condition(condition, context):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                           context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        cond_type = condition.get('type', 'expression')
        
        if cond_type == 'expression':
            expr = condition.get('expression')
            try:
                # Safe evaluation with limited context
                safe_context = {
                    'results': context.get('phase_results', {}),
                    'config': context.get('config', {}),
                    'outputs': context.get('phase_outputs', {})
                }
                return eval(expr, {"__builtins__": {}}, safe_context)
            except Exception as e:
                logger.error(f"Failed to evaluate condition: {e}")
                return False
                
        elif cond_type == 'metric_threshold':
            phase = condition.get('phase')
            metric = condition.get('metric')
            operator = condition.get('operator', '>')
            threshold = condition.get('threshold')
            
            value = self._extract_value(
                f"phase_results.{phase}.{metric}", 
                context
            )
            
            if value is None:
                return False
            
            if operator == '>':
                return value > threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<':
                return value < threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            
        return True
    
    def _execute_phase(self, phase_def: Dict[str, Any], 
                      workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single phase."""
        phase_name = phase_def.get('name')
        
        # Build phase configuration
        phase_config = self._build_phase_config(phase_def, workflow_context)
        
        # Create PhaseConfig object
        phase = PhaseConfig(
            name=phase_name,
            description=phase_def.get('description', f'Phase {phase_name}'),  # Add description
            topology=phase_def.get('topology', 'backtest'),
            sequence=phase_def.get('sequence', phase_def.get('pattern', 'single_pass')),
            config=phase_config,
            output=phase_def.get('output', {})
        )
        
        # Execute using sequencer
        try:
            result = self.sequencer.execute_sequence(phase, workflow_context)
            return result
        except Exception as e:
            logger.error(f"Phase {phase_name} failed: {e}")
            return {
                'success': False,
                'phase_name': phase_name,
                'error': str(e)
            }
    
    def _build_phase_config(self, phase_def: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Build configuration for a phase."""
        # Start with base config
        config = context.get('config', {}).copy()
        
        # Apply phase-specific config
        if 'config' in phase_def:
            config.update(self._resolve_config(phase_def['config'], context))
        
        # Process inputs
        if 'inputs' in phase_def:
            inputs = self._resolve_inputs(phase_def['inputs'], context)
            config['inputs'] = inputs
        
        return config
    
    def _resolve_config(self, config_spec: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve configuration values."""
        resolved = {}
        
        for key, value in config_spec.items():
            resolved[key] = self._resolve_value(value, context)
        
        return resolved
    
    def _resolve_inputs(self, inputs_spec: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve input specifications."""
        resolved = {}
        
        for key, spec in inputs_spec.items():
            value = self._resolve_value(spec, context)
            
            # Load file inputs if needed
            if isinstance(value, str) and Path(value).exists():
                try:
                    if value.endswith('.json'):
                        with open(value) as f:
                            value = json.load(f)
                    elif value.endswith('.yaml'):
                        with open(value) as f:
                            value = yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"Failed to load input file {value}: {e}")
            
            resolved[key] = value
        
        return resolved
    
    def _process_outputs(self, outputs_spec: Any, 
                        result: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Process and save outputs."""
        if isinstance(outputs_spec, list):
            # Convert list format to dict
            outputs_spec = {item: item for item in outputs_spec}
        
        processed = {}
        
        for key, spec in outputs_spec.items():
            if isinstance(spec, str):
                # Simple path specification
                path = self._resolve_value(spec, context)
                value = self._extract_value(key, result)
                
                if path.startswith('./') or path.startswith('/'):
                    # Save to file
                    self._save_output(path, value)
                    processed[key] = path
                else:
                    # Just store value
                    processed[key] = value
                    
            elif isinstance(spec, dict):
                # Complex output specification
                out_type = spec.get('type', 'extract')
                
                if out_type == 'extract':
                    source = spec.get('source', key)
                    value = self._extract_value(source, result)
                    path = spec.get('path')
                    
                    if path:
                        path = self._resolve_value(path, context)
                        self._save_output(path, value)
                        processed[key] = path
                    else:
                        processed[key] = value
                        
                elif out_type == 'aggregate':
                    # Aggregate multiple values
                    sources = spec.get('sources', [])
                    aggregated = {}
                    for source in sources:
                        value = self._extract_value(source, result)
                        aggregated[source] = value
                    processed[key] = aggregated
        
        return processed
    
    def _save_output(self, path: str, value: Any) -> None:
        """Save output to file with proper results directory structure."""
        # Build hierarchical results path
        path_obj = self._build_results_path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if path.endswith('.json'):
                with open(path_obj, 'w') as f:
                    json.dump(value, f, indent=2, default=str)
            elif path.endswith('.yaml'):
                with open(path_obj, 'w') as f:
                    yaml.dump(value, f)
            else:
                # Save as text
                with open(path_obj, 'w') as f:
                    f.write(str(value))
            
            logger.info(f"Saved output to {path_obj}")
        except Exception as e:
            logger.error(f"Failed to save output to {path_obj}: {e}")
    
    def _build_results_path(self, path: str) -> Path:
        """Build proper results path with workflow context."""
        # If path is absolute, use as-is
        if Path(path).is_absolute():
            return Path(path)
        
        # Build hierarchical path: results/{workflow_id}/{phase_name}/{filename}
        base_dir = Path(self.config_resolver.config.get('results_dir', './results'))
        workflow_id = getattr(self, '_current_workflow_id', 'default')
        phase_name = getattr(self, '_current_phase_name', 'default')
        
        # Create timestamped workflow directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workflow_dir = base_dir / f"{workflow_id}_{timestamp}"
        
        # Build full path
        if phase_name and phase_name != 'default':
            return workflow_dir / phase_name / path
        else:
            return workflow_dir / path
    
    def _resolve_value(self, spec: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value specification."""
        return self.config_resolver.resolve_value(spec, context)
    
    def _extract_value(self, path: str, data: Dict[str, Any]) -> Any:
        """Extract value from nested dict using dot notation."""
        return self.config_resolver.extract_value(path, data)
    
    def _create_summary(self, results: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow execution summary."""
        summary = {
            'workflow': context.get('workflow_name'),
            'success': results.get('success'),
            'phases_executed': len(results.get('phases', {})),
            'phases_succeeded': sum(
                1 for p in results.get('phases', {}).values() 
                if p.get('success', True)
            ),
            'duration': str(datetime.now() - context.get('start_time')),
            'outputs_generated': len(results.get('outputs', {}))
        }
        
        # Add key metrics if available
        metrics = {}
        for phase_name, phase_result in results.get('phases', {}).items():
            if 'aggregated' in phase_result:
                metrics[phase_name] = phase_result['aggregated']
        
        if metrics:
            summary['key_metrics'] = metrics
        
        return summary
    
    def _apply_trace_level_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply trace level presets if specified."""
        try:
            from ..events.tracing import (
                get_trace_level_from_config,
                apply_trace_level
            )
            
            trace_level = get_trace_level_from_config(config)
            if trace_level:
                config = apply_trace_level(config, trace_level)
                logger.info(f"Applied trace level preset: {trace_level}")
                
        except ImportError:
            logger.debug("Trace level presets not available")
            
        return config
    
    def _aggregate_phase_results(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all phases."""
        aggregated = {
            'total_phases': len(phase_results),
            'successful_phases': sum(1 for r in phase_results.values() if r.get('success', False)),
            'phase_metrics': {}
        }
        
        # Extract and aggregate metrics
        for phase_name, result in phase_results.items():
            if 'aggregate_metrics' in result:
                aggregated['phase_metrics'][phase_name] = result['aggregate_metrics']
            elif 'metrics' in result:
                aggregated['phase_metrics'][phase_name] = result['metrics']
        
        # Calculate primary metric if available
        primary_metric = None
        for phase_name, metrics in aggregated['phase_metrics'].items():
            if 'sharpe_ratio' in metrics:
                if primary_metric is None or metrics['sharpe_ratio'] > primary_metric:
                    primary_metric = metrics['sharpe_ratio']
                    aggregated['best_phase'] = phase_name
        
        if primary_metric is not None:
            aggregated['primary_metric'] = primary_metric
        
        return aggregated
    
    def _execute_discovered_workflow(self, workflow_name: str, 
                                   config: Dict[str, Any], 
                                   workflow_id: str) -> Dict[str, Any]:
        """Execute a discovered workflow (supports composable)."""
        workflow = self.discovered_workflows[workflow_name]
        
        # Check if workflow is composable
        is_composable = (
            hasattr(workflow, 'should_continue') or 
            hasattr(workflow, 'get_branches') or
            hasattr(workflow, 'modify_config_for_next')
        )
        
        if is_composable:
            return self._execute_composable_workflow(workflow, workflow_name, config, workflow_id)
        else:
            return self._execute_simple_workflow(workflow, workflow_name, config, workflow_id)
    
    def _execute_composable_workflow(self, workflow: WorkflowProtocol,
                                   workflow_name: str,
                                   config: Dict[str, Any],
                                   workflow_id: str) -> Dict[str, Any]:
        """Execute a composable workflow with iteration/branching."""
        logger.info(f"Executing composable workflow: {workflow_name}")
        
        # Initialize tracking
        iteration_results = []
        branch_results = {}
        iteration = 0
        max_iterations = config.get('max_iterations', 100)
        current_config = config.copy()
        
        # Iteration loop
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Workflow iteration {iteration}")
            
            # Check if we should continue
            if hasattr(workflow, 'should_continue') and iteration > 1:
                if not workflow.should_continue(current_config, iteration_results):
                    logger.info("Workflow stopping - should_continue returned False")
                    break
            
            # Modify config for next iteration
            if hasattr(workflow, 'modify_config_for_next') and iteration > 1:
                current_config = workflow.modify_config_for_next(
                    current_config, iteration_results
                )
            
            # Execute this iteration
            iteration_context = {
                'workflow_name': workflow_name,
                'workflow_id': workflow_id,
                'iteration': iteration,
                'config': current_config
            }
            
            result = self._execute_simple_workflow(
                workflow, workflow_name, current_config, 
                f"{workflow_id}_iter{iteration}"
            )
            
            iteration_results.append(result)
            
            # Check for branches
            if hasattr(workflow, 'get_branches'):
                branches = workflow.get_branches(current_config, result)
                if branches:
                    logger.info(f"Workflow branching into {len(branches)} paths")
                    for branch in branches:
                        branch_id = branch.branch_id
                        branch_config = branch.config
                        branch_result = self._execute_simple_workflow(
                            workflow, workflow_name, branch_config,
                            f"{workflow_id}_branch_{branch_id}"
                        )
                        branch_results[branch_id] = branch_result
        
        # Aggregate results
        success = all(r.get('success', False) for r in iteration_results)
        if branch_results:
            success = success and all(r.get('success', False) for r in branch_results.values())
        
        return {
            'workflow_id': workflow_id,
            'workflow_name': workflow_name,
            'success': success,
            'iterations': len(iteration_results),
            'iteration_results': iteration_results,
            'branch_results': branch_results,
            'aggregated_results': self._aggregate_iteration_results(iteration_results),
            'workflow_type': 'composable',
            'coordinator_id': self.coordinator_id
        }
    
    def _execute_simple_workflow(self, workflow: WorkflowProtocol,
                               workflow_name: str,
                               config: Dict[str, Any],
                               workflow_id: str) -> Dict[str, Any]:
        """Execute a simple (non-composable) workflow."""
        start_time = datetime.now()
        
        # Apply workflow defaults
        if hasattr(workflow, 'defaults'):
            config = self._apply_defaults(config, workflow.defaults)
        
        # Get phase definitions
        phases = workflow.get_phases(config)
        
        # Create execution context
        context = {
            'workflow_name': workflow_name,
            'workflow_id': workflow_id,
            'start_time': start_time,
            'config': config,
            'phase_data': {},
            'phase_results': {}
        }
        
        # Execute phases
        phase_results = {}
        for phase_name, phase_config in phases.items():
            logger.info(f"Executing phase: {phase_name}")
            
            try:
                # Create PhaseConfig
                phase = PhaseConfig(
                    name=phase_name,
                    topology=phase_config.get('topology', 'backtest'),
                    sequence=phase_config.get('sequence', 'single_pass'),
                    config=phase_config.get('config', config),
                    output=phase_config.get('output', {})
                )
                
                # Execute phase
                result = self.sequencer.execute_sequence(phase, context)
                phase_results[phase_name] = result
                
                # Store phase output
                if result.get('success', False):
                    output_data = result.get('output', {})
                    context['phase_data'][phase_name] = output_data
                    
            except Exception as e:
                logger.error(f"Phase {phase_name} failed: {e}")
                phase_results[phase_name] = {
                    'success': False,
                    'error': str(e)
                }
                break
        
        # Aggregate results
        success = all(r.get('success', False) for r in phase_results.values())
        
        return {
            'workflow_id': workflow_id,
            'workflow_name': workflow_name,
            'success': success,
            'phase_results': phase_results,
            'aggregated_results': self._aggregate_phase_results(phase_results),
            'workflow_type': 'simple',
            'coordinator_id': self.coordinator_id,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    def _apply_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge defaults with config."""
        result = defaults.copy()
        self._deep_merge(result, config)
        return result
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge update into base."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _aggregate_iteration_results(self, iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple iterations."""
        if not iteration_results:
            return {}
        
        aggregated = {
            'total_iterations': len(iteration_results),
            'successful_iterations': sum(1 for r in iteration_results if r.get('success', False)),
            'iteration_metrics': []
        }
        
        # Extract metrics from each iteration
        for i, result in enumerate(iteration_results):
            metrics = result.get('aggregated_results', {})
            if metrics:
                aggregated['iteration_metrics'].append({
                    'iteration': i + 1,
                    'metrics': metrics
                })
        
        return aggregated
    
    def _process_workflow_outputs(self, outputs_spec: Any, phase_results: Dict[str, Any], 
                                 context: Dict[str, Any]) -> None:
        """Process workflow-level outputs."""
        if outputs_spec:
            workflow_outputs = self._process_outputs(outputs_spec, phase_results, context)
            context['workflow_outputs'] = workflow_outputs