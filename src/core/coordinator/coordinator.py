"""
Coordinator implementation that manages workflow execution.

The Coordinator is the main entry point that:
1. Discovers and manages workflow patterns
2. Maintains oversight of phase completion
3. Handles result streaming
4. Manages distributed execution (future)
5. Always delegates to Sequencer for phase execution

All complexity is properly delegated to workflows and sequences.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..components.discovery import discover_components
from .protocols import (
    WorkflowProtocol, 
    SequenceProtocol,
    TopologyBuilderProtocol,
    PhaseConfig
)
from .topology import TopologyBuilder
from .sequencer import Sequencer

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinator that manages workflow execution.
    
    This is the main entry point for the system. It:
    1. Discovers available workflows and sequences
    2. Manages workflow execution with phase oversight
    3. Handles event tracing setup (optional)
    4. Delegates phase execution to Sequencer
    5. Aggregates results across phases
    6. Returns results to user
    """
    
    def __init__(self, shared_services: Optional[Dict[str, Any]] = None):
        """
        Initialize coordinator.
        
        Args:
            shared_services: Optional shared services like event tracer
        """
        self.coordinator_id = str(uuid.uuid4())
        self.shared_services = shared_services or {}
        
        # Discover available components
        self.workflows = self._discover_workflows()
        sequences = self._discover_sequences()
        
        # Create topology builder and sequencer
        self.topology_builder = TopologyBuilder()
        self.sequencer = Sequencer(sequences, self.topology_builder)
        
        # Optional event tracing
        self.event_tracer = None
        self._setup_event_tracing()
        
        logger.info(f"Coordinator {self.coordinator_id} initialized with "
                   f"{len(self.workflows)} workflows and {len(sequences)} sequences")
    
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
    
    def run_workflow(
        self,
        config: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow based on configuration.
        
        This is the main entry point for users.
        
        Args:
            config: User configuration including:
                - workflow: Name of workflow to run (default: simple_backtest)
                - data: Data configuration
                - strategies: Strategy configurations
                - Any workflow-specific parameters
            workflow_id: Optional workflow ID for tracking
            
        Returns:
            Dict containing:
                - workflow_id: Unique workflow execution ID
                - success: Whether workflow completed successfully
                - results: Workflow execution results
                - metadata: Execution metadata
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        # Apply trace level presets if specified
        config = self._apply_trace_level_config(config)
        
        # Get workflow name
        workflow_name = config.get('workflow', 'simple_backtest')
        
        logger.info(f"Starting workflow: {workflow_name} (ID: {workflow_id})")
        
        try:
            # Execute workflow
            result = self._execute_workflow(workflow_name, config)
            
            # Add event trace if available
            if self.event_tracer:
                trace_summary = self.event_tracer.get_summary()
                result['trace_summary'] = trace_summary
                logger.info(f"Event trace: {trace_summary.get('total_events', 0)} events")
            
            # Add coordinator metadata
            result['workflow_id'] = workflow_id
            result['coordinator_id'] = self.coordinator_id
            result['start_time'] = start_time.isoformat()
            result['end_time'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                'workflow_id': workflow_id,
                'success': False,
                'error': str(e),
                'workflow_name': workflow_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
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
    
    # Backward compatibility methods
    
    def execute_workflow(
        self,
        config: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Backward compatibility alias for run_workflow."""
        return self.run_workflow(config, workflow_id)
    
    def _execute_workflow(self, workflow_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a workflow with the given configuration.
        
        Handles both simple and composable workflows:
        - Simple workflows execute once
        - Composable workflows can iterate and branch
        
        Args:
            workflow_name: Name of the workflow to execute
            config: User configuration
            
        Returns:
            Dict containing:
                - success: Whether workflow completed successfully
                - phase_results: Results from each phase (or iterations)
                - aggregated_results: Aggregated metrics/results
                - metadata: Execution metadata
        """
        # Get workflow
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        
        # Check if workflow is composable
        is_composable = (
            hasattr(workflow, 'should_continue') or 
            hasattr(workflow, 'get_branches') or
            hasattr(workflow, 'modify_config_for_next')
        )
        
        if is_composable:
            return self._execute_composable(workflow, workflow_name, config)
        else:
            return self._execute_simple(workflow, workflow_name, config)
    
    def _execute_simple(self, workflow: WorkflowProtocol, workflow_name: str, 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a simple (non-composable) workflow once."""
        start_time = datetime.now()
        
        # Apply workflow defaults to config
        config = self._apply_defaults(config, workflow.defaults)
        
        # Get phase definitions
        phases = workflow.get_phases(config)
        
        # Create execution context for inter-phase data
        context = {
            'workflow_name': workflow_name,
            'start_time': start_time,
            'config': config,
            'phase_data': {}  # Stores output from each phase
        }
        
        # Execute phases in dependency order
        phase_results = {}
        for phase_name, phase_config in self._order_phases(phases).items():
            logger.info(f"Executing phase: {phase_name}")
            
            try:
                # Resolve inter-phase data references
                self._resolve_phase_inputs(phase_config, context)
                
                # Execute phase
                result = self._execute_phase(phase_config, context)
                phase_results[phase_name] = result
                
                # Store phase output for subsequent phases
                if result.get('success', False):
                    output_data = result.get('output', {})
                    
                    # Check if we should store full results or just output
                    if phase_config.config.get('store_full_results', False):
                        # Store complete results including metrics, trades, etc.
                        context['phase_data'][phase_name] = {
                            'output': output_data,
                            'metrics': result.get('aggregated_results', {}),
                            'summary': result.get('summary', {}),
                            'results_path': result.get('results_path')  # If saved to disk
                        }
                    else:
                        # Default: just store the output
                        context['phase_data'][phase_name] = output_data
                    
            except Exception as e:
                logger.error(f"Phase {phase_name} failed: {e}")
                phase_results[phase_name] = {
                    'success': False,
                    'error': str(e)
                }
                
                # Fail fast - don't continue if a phase fails
                break
        
        # Aggregate results
        success = all(r.get('success', False) for r in phase_results.values())
        
        return {
            'success': success,
            'workflow_name': workflow_name,
            'phase_results': phase_results,
            'aggregated_results': self._aggregate_results(phase_results),
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'phases_executed': len(phase_results),
                'phases_succeeded': sum(1 for r in phase_results.values() 
                                       if r.get('success', False))
            }
        }
    
    def _execute_composable(self, workflow: WorkflowProtocol, workflow_name: str,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a composable workflow with iteration and branching support."""
        start_time = datetime.now()
        iteration = 0
        max_iterations = config.get('max_iterations', 10)
        
        # Track all results across iterations
        all_results = []
        branch_results = []
        
        # Initial config
        current_config = self._apply_defaults(config, workflow.defaults)
        
        while iteration < max_iterations:
            logger.info(f"Executing {workflow_name} iteration {iteration + 1}")
            
            # Execute workflow
            result = self._execute_simple(workflow, workflow_name, current_config)
            result['iteration'] = iteration
            all_results.append(result)
            
            # Check if should continue
            if hasattr(workflow, 'should_continue'):
                if not workflow.should_continue(result, iteration, current_config):
                    logger.info(f"Workflow {workflow_name} stopping at iteration {iteration + 1}")
                    break
            
            # Check for branches
            if hasattr(workflow, 'get_branches'):
                branches = workflow.get_branches(result, iteration, current_config)
                for branch_name, branch_config in branches.items():
                    logger.info(f"Executing branch: {branch_name}")
                    branch_result = self._execute_simple(workflow, f"{workflow_name}_branch_{branch_name}", branch_config)
                    branch_result['branch_name'] = branch_name
                    branch_result['parent_iteration'] = iteration
                    branch_results.append(branch_result)
            
            # Modify config for next iteration
            if hasattr(workflow, 'modify_config_for_next'):
                current_config = workflow.modify_config_for_next(result, iteration, current_config)
            
            iteration += 1
        
        # Return aggregated results
        return {
            'success': any(r.get('success', False) for r in all_results),
            'workflow_name': workflow_name,
            'iterations': all_results,
            'branches': branch_results,
            'aggregated_results': self._aggregate_composable_results(all_results, branch_results),
            'metadata': {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'total_iterations': len(all_results),
                'branches_executed': len(branch_results)
            }
        }
    
    def _apply_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Apply workflow defaults to user config."""
        # Deep merge defaults with config
        result = defaults.copy()
        
        def deep_merge(base: dict, override: dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(result, config)
        return result
    
    def _order_phases(self, phases: Dict[str, PhaseConfig]) -> Dict[str, PhaseConfig]:
        """Order phases based on dependencies."""
        # Simple topological sort
        ordered = {}
        remaining = phases.copy()
        
        while remaining:
            # Find phases with no remaining dependencies
            ready = [
                name for name, phase in remaining.items()
                if all(dep in ordered for dep in phase.depends_on)
            ]
            
            if not ready:
                raise ValueError("Circular dependency in phases")
            
            # Add ready phases to ordered list
            for name in ready:
                ordered[name] = remaining.pop(name)
        
        return ordered
    
    def _resolve_phase_inputs(self, phase_config: PhaseConfig, context: Dict[str, Any]):
        """Resolve inter-phase data references in phase config."""
        # Replace references like {phase1.output.signals} with actual data
        for input_key, reference in phase_config.input.items():
            if reference.startswith('{') and reference.endswith('}'):
                # Parse reference
                path = reference[1:-1].split('.')
                if len(path) >= 2 and path[0] in context['phase_data']:
                    # Navigate to referenced data
                    data = context['phase_data'][path[0]]
                    for key in path[1:]:
                        if isinstance(data, dict) and key in data:
                            data = data[key]
                        else:
                            data = None
                            break
                    
                    # Update phase config with actual data
                    phase_config.config[input_key] = data
    
    def _execute_phase(self, phase_config: PhaseConfig, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single phase with proper context."""
        # Add results directory to context if specified
        if 'results_dir' in context.get('config', {}):
            context['results_dir'] = context['config']['results_dir']
        
        # Add workflow metadata
        context['workflow_name'] = context.get('workflow_name', 'unknown')
        
        # Delegate to sequencer
        result = self.sequencer.execute_sequence(phase_config, context)
        
        # Add topology info (sequencer already adds phase_name and sequence)
        result['topology'] = phase_config.topology
        
        return result
    
    def _aggregate_results(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across all phases."""
        # Simple aggregation - can be enhanced with ResultAggregator components
        aggregated = {
            'total_phases': len(phase_results),
            'successful_phases': sum(1 for r in phase_results.values() 
                                   if r.get('success', False)),
            'phase_metrics': {}
        }
        
        # Collect key metrics from each phase
        for phase_name, result in phase_results.items():
            if result.get('metrics'):
                aggregated['phase_metrics'][phase_name] = result['metrics']
        
        # Calculate summary metrics if available
        if aggregated['phase_metrics']:
            all_metrics = list(aggregated['phase_metrics'].values())
            if all_metrics and 'sharpe_ratio' in all_metrics[0]:
                # Example: average Sharpe across phases
                aggregated['avg_sharpe_ratio'] = sum(
                    m.get('sharpe_ratio', 0) for m in all_metrics
                ) / len(all_metrics)
        
        return aggregated
    
    def _aggregate_composable_results(self, iteration_results: List[Dict[str, Any]], 
                                    branch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from composable workflow execution."""
        # Find best iteration
        best_iteration = None
        best_metric = float('-inf')
        
        for i, result in enumerate(iteration_results):
            if result.get('success', False):
                # Extract primary metric (could be made configurable)
                metric = self._extract_primary_metric(result)
                if metric > best_metric:
                    best_metric = metric
                    best_iteration = i
        
        aggregated = {
            'best_iteration': best_iteration,
            'best_metric': best_metric,
            'total_iterations': len(iteration_results),
            'successful_iterations': sum(1 for r in iteration_results if r.get('success', False)),
            'branches_executed': len(branch_results),
            'iteration_metrics': []
        }
        
        # Collect metrics from each iteration
        for i, result in enumerate(iteration_results):
            if result.get('aggregated_results', {}).get('phase_metrics'):
                iteration_metric = {
                    'iteration': i,
                    'success': result.get('success', False),
                    'primary_metric': self._extract_primary_metric(result)
                }
                aggregated['iteration_metrics'].append(iteration_metric)
        
        # Add branch results summary
        if branch_results:
            aggregated['branch_summary'] = {
                'total_branches': len(branch_results),
                'successful_branches': sum(1 for r in branch_results if r.get('success', False))
            }
        
        return aggregated
    
    def _extract_primary_metric(self, result: Dict[str, Any]) -> float:
        """Extract the primary metric from a result for comparison."""
        # Try different paths where the primary metric might be
        paths = [
            ['aggregated_results', 'avg_sharpe_ratio'],
            ['aggregated_results', 'phase_metrics', 'backtest', 'sharpe_ratio'],
            ['phase_results', 'backtest', 'metrics', 'sharpe_ratio'],
            ['metrics', 'sharpe_ratio']
        ]
        
        for path in paths:
            value = result
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None and isinstance(value, (int, float)):
                return float(value)
        
        return 0.0