"""
Sequencer - handles execution of individual sequences.

This separates sequence execution from workflow orchestration,
keeping each component focused on a single responsibility.
"""

from typing import Dict, Any, Optional, List
import logging
import json
import os
from datetime import datetime

from .protocols import SequenceProtocol, PhaseConfig, TopologyBuilderProtocol
from .topology import TopologyBuilder
from ..containers.protocols import ContainerRole

logger = logging.getLogger(__name__)


class Sequencer:
    """
    Executes sequences according to their patterns.
    
    This is responsible for:
    1. Managing sequence instances
    2. Providing topology builder to sequences that need it
    3. Executing sequences with proper context
    4. Handling sequence-specific concerns
    """
    
    def __init__(self, sequences: Dict[str, SequenceProtocol], 
                 topology_builder: Optional[TopologyBuilderProtocol] = None):
        """
        Initialize sequencer.
        
        Args:
            sequences: Available sequence implementations
            topology_builder: Builder for creating topologies
        """
        self.sequences = sequences
        self.topology_builder = topology_builder or TopologyBuilder()
    
    def execute_sequence(
        self,
        phase_config: PhaseConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a sequence for a phase with results collection.
        
        Args:
            phase_config: Phase configuration with sequence name
            context: Execution context
            
        Returns:
            Sequence execution results
        """
        sequence_name = phase_config.sequence
        
        if sequence_name not in self.sequences:
            raise ValueError(f"Unknown sequence: {sequence_name}")
        
        sequence = self.sequences[sequence_name]
        
        # Ensure topology builder is available in context
        if not context.get('topology_builder'):
            context['topology_builder'] = self.topology_builder
        
        logger.info(f"Executing sequence '{sequence_name}' for phase '{phase_config.name}'")
        
        try:
            # Build and execute topology
            topology_definition = {
                'mode': phase_config.topology,
                'config': phase_config.config,
                'metadata': {
                    'workflow_id': context.get('workflow_name', 'unknown'),
                    'phase_name': phase_config.name
                }
            }
            
            # Extract tracing configuration from execution settings
            execution_config = phase_config.config.get('execution', {})
            if execution_config.get('enable_event_tracing', False):
                topology_definition['tracing_config'] = {
                    'enabled': True,
                    'trace_id': f"{context.get('workflow_name', 'unknown')}_{phase_config.name}",
                    'trace_dir': execution_config.get('trace_settings', {}).get('trace_dir', './traces'),
                    'max_events': execution_config.get('trace_settings', {}).get('max_events', 10000)
                }
                
                # Pass through container-specific settings if any
                trace_settings = execution_config.get('trace_settings', {})
                if 'container_settings' in trace_settings:
                    topology_definition['tracing_config']['container_settings'] = trace_settings['container_settings']
            
            topology = self.topology_builder.build_topology(topology_definition)
            
            # Execute the topology (actual implementation would start containers)
            result = self._execute_topology(topology, phase_config, context)
            
            # Collect results from containers
            phase_results = self._collect_phase_results(topology)
            
            # Handle storage based on phase config
            results_storage = phase_config.config.get('results_storage', 'memory')
            
            if results_storage == 'disk':
                results_path = self._save_results_to_disk(phase_results, phase_config, context)
                # Return minimal info to save memory
                result['results_saved'] = True
                result['results_path'] = results_path
                result['summary'] = self._create_summary(phase_results)
                result['aggregate_metrics'] = phase_results.get('aggregate_metrics', {})
            elif results_storage == 'hybrid':
                # Hybrid: Save large data to disk, keep summary in memory
                results_path = self._save_results_to_disk(phase_results, phase_config, context)
                result['results_saved'] = True
                result['results_path'] = results_path
                result['summary'] = self._create_summary(phase_results)
                result['aggregate_metrics'] = phase_results.get('aggregate_metrics', {})
                # Keep only essential data in memory
                result['phase_results'] = {
                    'aggregate_metrics': phase_results.get('aggregate_metrics', {}),
                    'container_count': len(phase_results.get('container_results', {})),
                    'total_trades': len(phase_results.get('trades', []))
                }
            else:
                # Default 'memory': Keep everything in memory
                result['phase_results'] = phase_results
                result['aggregate_metrics'] = phase_results.get('aggregate_metrics', {})
            
            # Add phase metadata
            result['sequence_name'] = sequence_name
            result['phase_name'] = phase_config.name
            result['success'] = True
            
            # Collect outputs as specified in phase config
            output = {}
            for key, should_collect in phase_config.output.items():
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
            result['output'] = output
            
            return result
            
        except Exception as e:
            logger.error(f"Sequence '{sequence_name}' failed for phase '{phase_config.name}': {e}")
            return {
                'success': False,
                'sequence_name': sequence_name,
                'phase_name': phase_config.name,
                'error': str(e)
            }
    
    def get_sequence_info(self, sequence_name: str) -> Dict[str, Any]:
        """Get information about a sequence."""
        if sequence_name not in self.sequences:
            return {'error': f'Unknown sequence: {sequence_name}'}
        
        sequence = self.sequences[sequence_name]
        
        return {
            'name': sequence_name,
            'class': sequence.__class__.__name__,
            'needs_topology': getattr(sequence, 'needs_topology', True),
            'description': sequence.__doc__ or 'No description available'
        }
    
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
            if hasattr(self, '_collect_phase_results'):
                results = self._collect_phase_results(topology)
            else:
                results = {}
            
            return {
                'containers_executed': len(containers),
                'success': True,
                'execution_result': execution_result,
                'results': results,
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
        
        For backtest: Stream data through the system
        For optimization: Run parameter combinations
        For signal generation: Generate and save signals
        """
        mode = phase_config.topology
        containers = topology.get('containers', {})
        
        if mode == 'backtest':
            # Find data containers and start streaming
            data_containers = [c for c in containers.values() 
                             if hasattr(c, 'role') and c.role == ContainerRole.DATA]
            
            if not data_containers:
                logger.warning("No data containers found in backtest topology")
                return {'bars_processed': 0}
            
            # Simple implementation: stream data from each data container
            total_bars = 0
            for data_container in data_containers:
                if hasattr(data_container, 'stream_data'):
                    # Stream all available data
                    bars_streamed = data_container.stream_data()
                    total_bars += bars_streamed
                    logger.info(f"Streamed {bars_streamed} bars from {data_container.container_id}")
            
            return {'bars_processed': total_bars}
            
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
        return {'signals_generated': 0}
    
    def _run_optimization(self, topology: Dict[str, Any],
                         phase_config: PhaseConfig) -> Dict[str, Any]:
        """Run optimization mode with multiple parameter combinations."""
        # Implementation would run multiple parameter sets
        return {'combinations_tested': 0}
    
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
            # Collect from portfolio containers
            if hasattr(container, 'streaming_metrics') and container.streaming_metrics:
                container_results = container.streaming_metrics.get_results()
                results['container_results'][container_id] = container_results
                
                # Aggregate portfolio data
                if hasattr(container, 'role') and container.role == ContainerRole.PORTFOLIO:
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
        
        for container_id, container_results in results['container_results'].items():
            filepath = os.path.join(container_dir, f"{container_id}_results.json")
            with open(filepath, 'w') as f:
                json.dump(container_results, f, indent=2, default=str)
        
        # Save aggregate results
        aggregate_path = os.path.join(results_dir, 'aggregate_results.json')
        with open(aggregate_path, 'w') as f:
            json.dump({
                'aggregate_metrics': results['aggregate_metrics'],
                'total_trades': len(results['trades']),
                'containers_tracked': len(results['container_results'])
            }, f, indent=2)
        
        # Save trades if present
        if results['trades']:
            trades_path = os.path.join(results_dir, 'all_trades.json')
            with open(trades_path, 'w') as f:
                json.dump(results['trades'], f, indent=2, default=str)
        
        # Save phase summary
        summary = {
            'phase': phase_name,
            'timestamp': datetime.now().isoformat(),
            'containers': list(results['container_results'].keys()),
            'metrics_summary': results['aggregate_metrics'],
            'config': {
                'results_storage': phase_config.config.get('results_storage'),
                'event_tracing': phase_config.config.get('event_tracing')
            }
        }
        
        summary_path = os.path.join(results_dir, 'phase_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved phase results to {results_dir}")
        return results_dir
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal summary for memory efficiency."""
        return {
            'best_sharpe': results['aggregate_metrics'].get('best_sharpe_ratio', 0),
            'avg_return': results['aggregate_metrics'].get('avg_total_return', 0),
            'total_trades': len(results.get('trades', [])),
            'containers': len(results['container_results'])
        }
    
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