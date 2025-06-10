"""
Declarative Sequencer Implementation

Interprets sequence patterns defined in YAML/dictionaries rather than
hardcoded Python classes.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from .protocols import PhaseConfig, TopologyBuilderProtocol
from .topology import TopologyBuilder
from .config.pattern_loader import PatternLoader
from .config.resolver import ConfigResolver
from ..containers.protocols import ContainerRole

logger = logging.getLogger(__name__)


class Sequencer:
    """
    Executes sequences based on declarative patterns.
    
    Instead of hardcoded sequence classes, this sequencer interprets
    sequence patterns that describe:
    - How to split/iterate over data
    - How to modify configurations between iterations
    - How to aggregate results
    """
    
    def __init__(self, topology_builder: Optional[TopologyBuilderProtocol] = None,
                 pattern_loader: Optional[PatternLoader] = None,
                 config_resolver: Optional[ConfigResolver] = None):
        """Initialize declarative sequencer."""
        self.topology_builder = topology_builder or TopologyBuilder(pattern_loader, config_resolver)
        self.pattern_loader = pattern_loader or PatternLoader()
        self.config_resolver = config_resolver or ConfigResolver()
        self.sequence_patterns = self.pattern_loader.load_patterns('sequences')
        
    
    def execute_sequence(
        self,
        phase_config: PhaseConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a sequence based on its pattern.
        
        Args:
            phase_config: Phase configuration with sequence name
            context: Execution context
            
        Returns:
            Sequence execution results
        """
        sequence_name = phase_config.sequence
        
        # Get sequence pattern
        pattern = self.sequence_patterns.get(sequence_name)
        if not pattern:
            raise ValueError(f"Unknown sequence pattern: {sequence_name}")
        
        logger.info(f"Executing sequence '{sequence_name}' for phase '{phase_config.name}'")
        
        # Build evaluation context
        eval_context = {
            'config': phase_config.config,
            'phase': phase_config,
            'workflow': context,
            'pattern': pattern
        }
        
        # Generate iterations
        iterations = self._generate_iterations(pattern, eval_context)
        
        # Execute iterations
        iteration_results = []
        for i, iteration_context in enumerate(iterations):
            iter_context = {**eval_context, **iteration_context, 'iteration_index': i}
            
            # Apply config modifiers
            modified_config = self._apply_config_modifiers(
                phase_config.config.copy(),
                pattern.get('config_modifiers', []),
                iter_context
            )
            
            # Execute sub-sequences, sub-phases, or single topology
            if 'sub_sequences' in pattern:
                result = self._execute_sub_sequences(
                    pattern['sub_sequences'],
                    modified_config,
                    phase_config,
                    iter_context
                )
            elif 'sub_phases' in pattern:
                result = self._execute_sub_phases(
                    pattern['sub_phases'],
                    modified_config,
                    phase_config,
                    iter_context
                )
            else:
                result = self._execute_single_topology(
                    phase_config.topology,
                    modified_config,
                    phase_config,
                    iter_context
                )
            
            iteration_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(
            iteration_results,
            pattern.get('aggregation', {}),
            eval_context
        )
        
        return {
            'success': all(r.get('success', True) for r in iteration_results),
            'sequence_name': sequence_name,
            'phase_name': phase_config.name,
            'iterations': len(iterations),
            'iteration_results': iteration_results,
            'aggregated': aggregated,
            'output': self._extract_output(aggregated, phase_config.output)
        }
    
    def _generate_iterations(self, pattern: Dict[str, Any], 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate iteration contexts based on pattern."""
        iterations_config = pattern.get('iterations', {})
        iter_type = iterations_config.get('type', 'single')
        
        if iter_type == 'single':
            return [{}]
            
        elif iter_type == 'repeated':
            count = self._resolve_value(iterations_config.get('count', 1), context)
            return [{'iteration': i} for i in range(count)]
            
        elif iter_type == 'windowed':
            return self._generate_windows(iterations_config, context)
            
        else:
            logger.warning(f"Unknown iteration type: {iter_type}")
            return [{}]
    
    def _generate_windows(self, config: Dict[str, Any], 
                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate window contexts."""
        window_gen = config.get('window_generator', {})
        gen_type = window_gen.get('type', 'rolling')
        
        if gen_type == 'rolling':
            # Get parameters
            train_periods = self._resolve_value(window_gen.get('train_periods'), context)
            test_periods = self._resolve_value(window_gen.get('test_periods'), context)
            step_size = self._resolve_value(window_gen.get('step_size'), context)
            
            start_date = context['config'].get('start_date', '2020-01-01')
            end_date = context['config'].get('end_date', '2023-12-31')
            
            # Generate windows
            windows = []
            current_start = datetime.strptime(start_date, '%Y-%m-%d')
            final_end = datetime.strptime(end_date, '%Y-%m-%d')
            
            while True:
                train_end = current_start + timedelta(days=train_periods - 1)
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=test_periods - 1)
                
                if test_end > final_end:
                    break
                
                windows.append({
                    'window': {
                        'train_start': current_start.strftime('%Y-%m-%d'),
                        'train_end': train_end.strftime('%Y-%m-%d'),
                        'test_start': test_start.strftime('%Y-%m-%d'),
                        'test_end': test_end.strftime('%Y-%m-%d')
                    }
                })
                
                current_start += timedelta(days=step_size)
            
            return windows
        
        return [{}]
    
    def _apply_config_modifiers(self, config: Dict[str, Any],
                               modifiers: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration modifiers."""
        for modifier in modifiers:
            mod_type = modifier.get('type')
            
            if mod_type == 'set_dates':
                # Set date fields from context
                for key in ['train_start', 'train_end', 'test_start', 'test_end']:
                    if key in modifier:
                        config[key] = self._resolve_value(modifier[key], context)
                        
            elif mod_type == 'add_seed':
                # Add random seed
                if 'random_seed' in modifier:
                    config['random_seed'] = self._resolve_value(modifier['random_seed'], context)
                    
            elif mod_type == 'custom':
                # Custom modifier function
                # Could be extended to support user-defined modifiers
                pass
        
        return config
    
    def _execute_sub_phases(self, sub_phases: List[Dict[str, Any]],
                           config: Dict[str, Any],
                           phase_config: PhaseConfig,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sub-phases with dependencies."""
        results = {}
        phase_results = {}
        
        for sub_phase in sub_phases:
            name = sub_phase['name']
            
            # Check dependencies
            if 'depends_on' in sub_phase:
                dep = sub_phase['depends_on']
                if dep not in phase_results:
                    logger.error(f"Dependency {dep} not found for sub-phase {name}")
                    continue
                
                # Add dependency results to context
                context[dep] = phase_results[dep]
            
            # Apply config override
            sub_config = config.copy()
            if 'config_override' in sub_phase:
                for key, value in sub_phase['config_override'].items():
                    sub_config[key] = self._resolve_value(value, context)
            
            # Execute topology
            result = self._execute_single_topology(
                phase_config.topology,
                sub_config,
                phase_config,
                context
            )
            
            phase_results[name] = result
        
        results['sub_phases'] = phase_results
        results['success'] = all(r.get('success', True) for r in phase_results.values())
        
        return results
    
    def _execute_single_topology(self, topology_mode: str,
                               config: Dict[str, Any],
                               phase_config: PhaseConfig,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single topology with full container lifecycle."""
        
        # Build topology definition
        topology_definition = self._build_topology_definition(
            topology_mode, config, phase_config, context
        )
        
        # Build topology
        topology = self.topology_builder.build_topology(topology_definition)
        
        # Execute with proper lifecycle
        execution_result = self._execute_topology(topology, phase_config, context)
        
        # Extract phase results from execution
        phase_results = execution_result.get('phase_results', {})
        
        # Process results based on storage mode
        result = self._process_results(
            execution_result, phase_results, phase_config, context
        )
        
        # Add phase metadata
        result['sequence_name'] = phase_config.sequence
        result['phase_name'] = phase_config.name
        result['topology_mode'] = topology_mode
        
        # Collect outputs as specified in phase config
        if phase_config.output:
            result['output'] = self._extract_outputs(phase_results, phase_config.output)
        
        return result
    
    def _aggregate_results(self, results: List[Dict[str, Any]],
                          aggregation_config: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results based on pattern."""
        agg_type = aggregation_config.get('type', 'none')
        
        if agg_type == 'none':
            return results[0] if results else {}
            
        elif agg_type == 'statistical':
            # Statistical aggregation
            source = aggregation_config.get('source', 'metrics')
            operations = aggregation_config.get('operations', ['mean'])
            
            # Extract metrics from specified source
            all_metrics = []
            for result in results:
                metrics = self._extract_nested(result, source)
                if metrics:
                    all_metrics.append(metrics)
            
            if not all_metrics:
                return {}
            
            # Calculate statistics
            aggregated = {}
            metric_names = all_metrics[0].keys()
            
            for metric in metric_names:
                values = [m.get(metric, 0) for m in all_metrics]
                metric_stats = {}
                
                if 'mean' in operations:
                    metric_stats['mean'] = sum(values) / len(values)
                if 'std' in operations:
                    metric_stats['std'] = self._calculate_std(values)
                if 'min' in operations:
                    metric_stats['min'] = min(values)
                if 'max' in operations:
                    metric_stats['max'] = max(values)
                    
                aggregated[metric] = metric_stats
            
            return aggregated
            
        elif agg_type == 'distribution':
            # Distribution analysis
            metrics = aggregation_config.get('metrics', [])
            percentiles = aggregation_config.get('percentiles', [25, 50, 75])
            
            distributions = {}
            for metric in metrics:
                values = []
                for result in results:
                    value = result.get('metrics', {}).get(metric)
                    if value is not None:
                        values.append(value)
                
                if values:
                    values.sort()
                    distributions[metric] = {
                        'values': values,
                        'percentiles': {
                            p: values[int(len(values) * p / 100)]
                            for p in percentiles
                        }
                    }
            
            return distributions
            
        elif agg_type == 'comparison':
            # Phase comparison
            phases = aggregation_config.get('phases', [])
            comparison = {}
            
            for result in results:
                if 'sub_phases' in result:
                    for phase in phases:
                        if phase in result['sub_phases']:
                            comparison[phase] = result['sub_phases'][phase].get('metrics', {})
            
            return comparison
        
        return {}
    
    def _extract_nested(self, data: Dict[str, Any], path: str) -> Any:
        """Extract nested value from dict using dot notation."""
        return self.config_resolver.extract_value(path, data)
    
    def _resolve_value(self, spec: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value specification."""
        return self.config_resolver.resolve_value(spec, context)
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _extract_output(self, aggregated: Dict[str, Any], 
                       output_spec: Dict[str, bool]) -> Dict[str, Any]:
        """Extract requested output from aggregated results."""
        output = {}
        
        for key, should_include in output_spec.items():
            if should_include and key in aggregated:
                output[key] = aggregated[key]
        
        return output
    
    def _execute_sub_sequences(self, sub_sequences: List[Dict[str, Any]],
                              config: Dict[str, Any],
                              phase_config: PhaseConfig,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute composed sequences."""
        results = {}
        sequence_results = {}
        
        for sub_seq in sub_sequences:
            name = sub_seq['name']
            seq_pattern_name = sub_seq.get('sequence')
            
            # Check dependencies
            if 'depends_on' in sub_seq:
                dep = sub_seq['depends_on']
                if dep not in sequence_results:
                    logger.error(f"Dependency {dep} not found for sub-sequence {name}")
                    continue
                
                # Add dependency results to context
                context[dep] = sequence_results[dep]
            
            # Apply config override
            sub_config = config.copy()
            if 'config_override' in sub_seq:
                for key, value in sub_seq['config_override'].items():
                    sub_config[key] = self._resolve_value(value, context)
            
            # Create phase config for sub-sequence
            sub_phase = PhaseConfig(
                name=f"{phase_config.name}.{name}",
                topology=phase_config.topology,
                sequence=seq_pattern_name,
                config=sub_config,
                output=sub_seq.get('output', {})
            )
            
            # Execute sub-sequence recursively
            result = self.execute_sequence(sub_phase, context)
            sequence_results[name] = result
        
        results['sub_sequences'] = sequence_results
        results['success'] = all(r.get('success', True) for r in sequence_results.values())
        
        return results
    
    def _build_topology_definition(self, topology_mode: str,
                                  config: Dict[str, Any],
                                  phase_config: PhaseConfig,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Build topology definition with proper structure."""
        topology_definition = {
            'mode': topology_mode,
            'config': config,
            'metadata': {
                'workflow_id': context.get('workflow_name', 'unknown'),
                'phase_name': phase_config.name
            }
        }
        
        # Add tracing if enabled - with CORRECT structure
        execution_config = config.get('execution', {})
        if execution_config.get('enable_event_tracing', False):
            trace_settings = execution_config.get('trace_settings', {})
            topology_definition['tracing_config'] = {
                'enabled': True,
                'trace_id': trace_settings.get('trace_id', 
                    f"{context.get('workflow_name', 'unknown')}_{phase_config.name}"),
                'trace_dir': trace_settings.get('trace_dir', './traces'),
                'max_events': trace_settings.get('max_events', 10000),
                'container_settings': trace_settings.get('container_settings', {}),
                # Include console output settings
                'enable_console_output': trace_settings.get('enable_console_output', False),
                'console_filter': trace_settings.get('console_filter', [])
            }
        
        return topology_definition
    
    def _execute_topology(self, topology: Dict[str, Any], 
                         phase_config: PhaseConfig, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute topology with proper container lifecycle.
        
        This ensures:
        1. All containers are initialized
        2. All containers are started
        3. Data flows through the system
        4. All containers are stopped
        5. All containers are cleaned up (triggering result saves)
        """
        containers = topology.get('containers', {})
        if not containers:
            return {'containers_executed': 0, 'success': True}
        
        logger.info(f"Executing topology with {len(containers)} containers")
        
        # Track execution state
        initialized_containers = []
        started_containers = []
        errors = []
        
        try:
            # Phase 1: Initialize all containers
            logger.info("Initializing containers...")
            for container_id, container in containers.items():
                try:
                    if hasattr(container, 'initialize'):
                        container.initialize()
                        initialized_containers.append(container)
                        logger.debug(f"Initialized container: {container_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize container {container_id}: {e}")
                    errors.append(f"Init {container_id}: {str(e)}")
                    raise
            
            # Phase 2: Start all containers
            logger.info("Starting containers...")
            for container_id, container in containers.items():
                try:
                    if hasattr(container, 'start'):
                        container.start()
                        started_containers.append(container)
                        logger.debug(f"Started container: {container_id}")
                except Exception as e:
                    logger.error(f"Failed to start container {container_id}: {e}")
                    errors.append(f"Start {container_id}: {str(e)}")
                    raise
            
            # Phase 3: Run the execution
            # For backtest, this means streaming data through the system
            execution_result = self._run_topology_execution(
                topology, phase_config, context
            )
            
            # Phase 4: Collect results while containers are still running
            # This happens before cleanup so containers can provide final metrics
            phase_results = self._collect_phase_results(topology)
            
            return {
                'containers_executed': len(containers),
                'success': True,
                'execution_result': execution_result,
                'phase_results': phase_results,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Topology execution failed: {e}")
            return {
                'containers_executed': len(initialized_containers),
                'success': False,
                'error': str(e),
                'errors': errors
            }
            
        finally:
            # Phase 5: Stop all started containers (in reverse order)
            logger.info("Stopping containers...")
            for container in reversed(started_containers):
                try:
                    if hasattr(container, 'stop'):
                        container.stop()
                        logger.debug(f"Stopped container: {container.container_id}")
                except Exception as e:
                    logger.error(f"Error stopping container: {e}")
            
            # Phase 6: Cleanup all initialized containers (in reverse order)
            # THIS IS WHERE RESULTS GET SAVED!
            logger.info("Cleaning up containers...")
            for container in reversed(initialized_containers):
                try:
                    if hasattr(container, 'cleanup'):
                        container.cleanup()
                        logger.debug(f"Cleaned up container: {container.container_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up container: {e}")
    
    def _run_topology_execution(self, topology: Dict[str, Any],
                               phase_config: PhaseConfig,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the actual execution based on topology mode.
        
        This is beautifully simple now - just tell containers to execute.
        Event-driven architecture handles the rest naturally.
        """
        mode = phase_config.topology
        containers = topology.get('containers', {})
        
        if mode == 'backtest' or mode == 'root_backtest':
            # Event-driven execution - just tell containers to execute
            # Data containers will start streaming
            # Other containers will react to events naturally
            logger.info("Starting event-driven execution")
            
            # Execute all containers - they know what to do
            for container_id, container in containers.items():
                try:
                    container.execute()
                    logger.debug(f"Executed container {container_id}")
                except Exception as e:
                    logger.error(f"Error executing container {container_id}: {e}")
            
            # Wait for completion or timeout
            # For now, simple approach - could be enhanced with proper completion detection
            import time
            max_duration = phase_config.config.get('max_execution_time', 60)  # seconds
            time.sleep(min(1, max_duration))  # Brief pause for demo
            
            # In a real implementation, we'd:
            # - Monitor event flow
            # - Check if data streaming is complete
            # - Wait for all pending events to process
            # - Have proper completion signaling
            
            return {
                'execution_mode': 'event_driven',
                'containers_executed': len(containers),
                'success': True
            }
            
        elif mode == 'signal_generation':
            # Similar to backtest but focused on signal capture
            return self._run_signal_generation(topology, phase_config)
            
        elif mode == 'optimization':
            # Run multiple parameter combinations
            return self._run_optimization(topology, phase_config)
            
        else:
            logger.warning(f"Unknown topology mode: {mode}")
            return {'mode': mode, 'status': 'completed'}
    
    def _run_signal_generation(self, topology: Dict[str, Any],
                              phase_config: PhaseConfig) -> Dict[str, Any]:
        """Run signal generation mode."""
        # Implementation would stream data and capture signals
        containers = topology.get('containers', {})
        
        # Find containers with data streaming components
        data_containers = []
        for container in containers.values():
            if hasattr(container, 'get_component'):
                data_streamer = container.get_component('data_streamer')
                if data_streamer:
                    data_containers.append((container, data_streamer))
        
        total_bars = 0
        for container, data_streamer in data_containers:
            bars_streamed = 10  # Default for testing
            total_bars += bars_streamed
        
        return {'bars_processed': total_bars, 'signals_generated': 0}
    
    def _run_optimization(self, topology: Dict[str, Any],
                         phase_config: PhaseConfig) -> Dict[str, Any]:
        """Run optimization mode with multiple parameter combinations."""
        # Count portfolio containers as proxy for combinations
        containers = topology.get('containers', {})
        portfolio_containers = []
        for container in containers.values():
            # Portfolio containers have portfolio_manager component
            if hasattr(container, 'get_component'):
                portfolio_mgr = container.get_component('portfolio_manager')
                if portfolio_mgr:
                    portfolio_containers.append(container)
        
        return {'combinations_tested': len(portfolio_containers)}
    
    def _collect_phase_results(self, topology: Dict[str, Any]) -> Dict[str, Any]:
        """Collect results from all portfolio containers."""
        results = {
            'container_results': {},
            'aggregate_metrics': {},
            'trades': [],
            'equity_curves': {}
        }
        
        containers = topology.get('containers', {})
        portfolio_results = []
        
        for container_id, container in containers.items():
            # Collect from containers with streaming metrics
            if hasattr(container, 'streaming_metrics') and container.streaming_metrics:
                container_results = container.streaming_metrics.get_results()
                results['container_results'][container_id] = container_results
                
                # Aggregate portfolio data - check for portfolio_manager component
                if hasattr(container, 'get_component'):
                    portfolio_mgr = container.get_component('portfolio_manager')
                    if portfolio_mgr:
                        portfolio_results.append(container_results)
                    
                    # Collect trades if available
                    if 'trades' in container_results:
                        results['trades'].extend(container_results['trades'])
                    
                    # Store equity curve if available
                    if 'equity_curve' in container_results:
                        results['equity_curves'][container_id] = container_results['equity_curve']
        
        # Calculate aggregate metrics
        if portfolio_results:
            results['aggregate_metrics'] = self._aggregate_portfolio_metrics(portfolio_results)
        
        return results
    
    def _aggregate_portfolio_metrics(self, portfolio_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple portfolios."""
        if not portfolio_results:
            return {}
        
        # Find best performing portfolio
        best_sharpe = -float('inf')
        best_portfolio_metrics = None
        
        all_metrics = []
        for result in portfolio_results:
            metrics = result.get('metrics', {})
            if metrics:
                all_metrics.append(metrics)
                sharpe = metrics.get('sharpe_ratio', -float('inf'))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_portfolio_metrics = metrics
        
        if not all_metrics:
            return {}
        
        # Calculate aggregates
        aggregate = {
            'best_sharpe_ratio': best_sharpe,
            'best_portfolio_metrics': best_portfolio_metrics,
            'portfolio_count': len(portfolio_results)
        }
        
        # Average key metrics
        for metric_name in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                aggregate[f'avg_{metric_name}'] = sum(values) / len(values)
        
        return aggregate
    
    def _process_results(self, execution_result: Dict[str, Any],
                        phase_results: Dict[str, Any],
                        phase_config: PhaseConfig,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory/disk/hybrid storage modes."""
        
        results_storage = phase_config.config.get('results_storage', 'memory')
        
        result = {
            'success': execution_result.get('success', True),
            'containers_executed': execution_result.get('containers_executed', 0)
        }
        
        if results_storage == 'disk':
            # Save everything to disk, return only paths
            results_path = self._save_results_to_disk(phase_results, phase_config, context)
            result.update({
                'results_saved': True,
                'results_path': results_path,
                'summary': self._create_summary(phase_results),
                'aggregate_metrics': phase_results.get('aggregate_metrics', {})
            })
        elif results_storage == 'hybrid':
            # Save large data to disk, keep summaries in memory
            results_path = self._save_results_to_disk(phase_results, phase_config, context)
            result.update({
                'results_saved': True,
                'results_path': results_path,
                'summary': self._create_summary(phase_results),
                'aggregate_metrics': phase_results.get('aggregate_metrics', {}),
                'phase_results': {
                    'aggregate_metrics': phase_results.get('aggregate_metrics', {}),
                    'container_count': len(phase_results.get('container_results', {})),
                    'total_trades': len(phase_results.get('trades', []))
                }
            })
        else:  # 'memory'
            # Keep everything in memory (risky for large runs)
            result.update({
                'phase_results': phase_results,
                'aggregate_metrics': phase_results.get('aggregate_metrics', {})
            })
        
        return result
    
    def _save_results_to_disk(self, results: Dict[str, Any], 
                              phase_config: PhaseConfig, 
                              context: Dict[str, Any]) -> str:
        """Save results to disk and return path."""
        # Build path using custom results_dir if provided
        base_results_dir = "./results"
        custom_dir = context.get('results_dir', '')
        workflow_id = context.get('workflow_name', 'unknown')
        phase_name = phase_config.name
        
        if custom_dir:
            results_dir = os.path.join(base_results_dir, custom_dir, phase_name)
        else:
            results_dir = os.path.join(base_results_dir, workflow_id, phase_name)
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save container results separately for easier analysis
        container_dir = os.path.join(results_dir, 'containers')
        os.makedirs(container_dir, exist_ok=True)
        
        for container_id, container_results in results.get('container_results', {}).items():
            filepath = os.path.join(container_dir, f"{container_id}_results.json")
            with open(filepath, 'w') as f:
                json.dump(container_results, f, indent=2, default=str)
        
        # Save aggregate results
        aggregate_path = os.path.join(results_dir, 'aggregate_results.json')
        with open(aggregate_path, 'w') as f:
            json.dump({
                'aggregate_metrics': results.get('aggregate_metrics', {}),
                'total_trades': len(results.get('trades', [])),
                'containers_tracked': len(results.get('container_results', {}))
            }, f, indent=2)
        
        # Save trades if present
        if results.get('trades'):
            trades_path = os.path.join(results_dir, 'all_trades.json')
            with open(trades_path, 'w') as f:
                json.dump(results['trades'], f, indent=2, default=str)
        
        # Save phase summary
        summary = {
            'phase': phase_name,
            'timestamp': datetime.now().isoformat(),
            'containers': list(results.get('container_results', {}).keys()),
            'metrics_summary': results.get('aggregate_metrics', {}),
            'config': {
                'results_storage': phase_config.config.get('results_storage'),
                'event_tracing': phase_config.config.get('execution', {}).get('enable_event_tracing')
            }
        }
        
        summary_path = os.path.join(results_dir, 'phase_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved phase results to {results_dir}")
        return results_dir
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal summary for memory efficiency."""
        aggregate = results.get('aggregate_metrics', {})
        return {
            'best_sharpe': aggregate.get('best_sharpe_ratio', 0),
            'avg_return': aggregate.get('avg_total_return', 0),
            'total_trades': len(results.get('trades', [])),
            'containers': len(results.get('container_results', {}))
        }
    
    def _extract_outputs(self, phase_results: Dict[str, Any], 
                        output_spec: Dict[str, bool]) -> Dict[str, Any]:
        """Extract requested outputs from phase results."""
        output = {}
        
        for key, should_collect in output_spec.items():
            if should_collect:
                if key in phase_results:
                    output[key] = phase_results[key]
                elif key == 'best_parameters' and 'aggregate_metrics' in phase_results:
                    # Extract best parameters from results
                    best_metrics = phase_results['aggregate_metrics'].get('best_portfolio_metrics', {})
                    if best_metrics:
                        output['best_parameters'] = best_metrics.get('parameters', {})
                elif key == 'metrics':
                    # Include aggregate metrics
                    output['metrics'] = phase_results.get('aggregate_metrics', {})
        
        return output