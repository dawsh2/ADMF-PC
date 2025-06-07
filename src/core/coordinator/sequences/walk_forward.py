"""
Walk-Forward Sequence

Implements walk-forward analysis by executing a phase multiple times
with rolling train/test windows.
"""

from typing import Dict, Any, List, Tuple, Optional, Protocol
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TopologyBuilderProtocol(Protocol):
    """Protocol for topology builders."""
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        ...


class WalkForwardSequence:
    """
    Execute phase using walk-forward analysis.
    
    Splits data into rolling train/test windows and executes
    the phase for each window.
    """
    
    def __init__(self, topology_builder: TopologyBuilderProtocol):
        self.topology_builder = topology_builder
    
    def execute(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute walk-forward analysis.
        
        Args:
            phase_config: Phase-specific configuration
            base_config: Base workflow configuration  
            context: Execution context
            
        Returns:
            Aggregated results from all windows
        """
        phase_name = phase_config.get('name', 'unnamed_phase')
        topology_mode = phase_config.get('topology', 'backtest')
        
        # Get walk-forward parameters
        wf_config = self._get_walk_forward_config(phase_config, base_config)
        
        # Generate windows
        windows = self._generate_windows(
            start_date=wf_config['start_date'],
            end_date=wf_config['end_date'],
            train_periods=wf_config['train_periods'],
            test_periods=wf_config['test_periods'],
            step_size=wf_config['step_size']
        )
        
        logger.info(f"Executing walk-forward with {len(windows)} windows for phase: {phase_name}")
        
        # Execute each window
        window_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: "
                       f"Train {train_start} to {train_end}, "
                       f"Test {test_start} to {test_end}")
            
            # Execute window
            window_result = self._execute_window(
                phase_config=phase_config,
                base_config=base_config,
                context=context,
                window_index=i,
                train_period=(train_start, train_end),
                test_period=(test_start, test_end),
                topology_mode=topology_mode
            )
            
            window_results.append(window_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(window_results)
        
        return {
            'phase_name': phase_name,
            'sequence_type': 'walk_forward',
            'success': all(r.get('success', True) for r in window_results),
            'windows': len(windows),
            'window_results': window_results,
            'aggregated_results': aggregated
        }
    
    def _get_walk_forward_config(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract walk-forward configuration."""
        # Check phase override first, then base config
        config_override = phase_config.get('config_override', {})
        wf_config = config_override.get('walk_forward', base_config.get('walk_forward', {}))
        
        return {
            'start_date': base_config.get('start_date', '2020-01-01'),
            'end_date': base_config.get('end_date', '2023-12-31'),
            'train_periods': wf_config.get('train_periods', 252),  # Days
            'test_periods': wf_config.get('test_periods', 63),
            'step_size': wf_config.get('step_size', 21),
            'optimization_metric': wf_config.get('optimization_metric', 'sharpe_ratio')
        }
    
    def _generate_windows(
        self,
        start_date: str,
        end_date: str,
        train_periods: int,
        test_periods: int,
        step_size: int
    ) -> List[Tuple[str, str, str, str]]:
        """Generate train/test windows."""
        windows = []
        
        # Convert to datetime
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        final_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while True:
            # Calculate window dates
            train_end = current_start + timedelta(days=train_periods - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_periods - 1)
            
            # Check if we've exceeded the final date
            if test_end > final_end:
                break
            
            windows.append((
                current_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            # Step forward
            current_start += timedelta(days=step_size)
        
        return windows
    
    def _execute_window(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any],
        window_index: int,
        train_period: Tuple[str, str],
        test_period: Tuple[str, str],
        topology_mode: str
    ) -> Dict[str, Any]:
        """Execute a single window."""
        # Merge configurations
        merged_config = base_config.copy()
        if 'config_override' in phase_config:
            merged_config.update(phase_config['config_override'])
        
        # Add window-specific dates
        merged_config['train_start_date'] = train_period[0]
        merged_config['train_end_date'] = train_period[1]
        merged_config['test_start_date'] = test_period[0]
        merged_config['test_end_date'] = test_period[1]
        merged_config['window_index'] = window_index
        
        # Build topology for training
        train_topology_def = {
            'mode': topology_mode,
            'config': {
                **merged_config,
                'start_date': train_period[0],
                'end_date': train_period[1],
                'phase': 'train'
            },
            'event_tracer': context.get('event_tracer')
        }
        train_topology = self.topology_builder.build_topology(train_topology_def)
        
        # Execute training
        train_result = self._execute_topology(train_topology, merged_config, context)
        
        # Build topology for testing with optimal parameters
        test_topology_def = {
            'mode': topology_mode,
            'config': {
                **merged_config,
                'start_date': test_period[0],
                'end_date': test_period[1],
                'phase': 'test',
                'parameters': train_result.get('optimal_parameters', {})
            },
            'event_tracer': context.get('event_tracer')
        }
        test_topology = self.topology_builder.build_topology(test_topology_def)
        
        # Execute testing
        test_result = self._execute_topology(test_topology, merged_config, context)
        
        return {
            'window_index': window_index,
            'train_period': train_period,
            'test_period': test_period,
            'train_result': train_result,
            'test_result': test_result,
            'success': True
        }
    
    def _execute_topology(
        self,
        topology: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the topology and return results."""
        # Mock implementation
        return {
            'containers_executed': len(topology.get('containers', {})),
            'metrics': {
                'sharpe_ratio': 1.5 + (context.get('window_index', 0) * 0.1),
                'total_return': 0.15,
                'max_drawdown': 0.08
            },
            'optimal_parameters': {
                'momentum_threshold': 0.02,
                'position_size': 0.1
            }
        }
    
    def _aggregate_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        # Collect all test metrics
        test_metrics = []
        for window in window_results:
            if window.get('success') and 'test_result' in window:
                test_metrics.append(window['test_result'].get('metrics', {}))
        
        if not test_metrics:
            return {}
        
        # Calculate aggregated metrics
        aggregated = {}
        metric_names = test_metrics[0].keys()
        
        for metric in metric_names:
            values = [m.get(metric, 0) for m in test_metrics]
            aggregated[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': self._calculate_std(values)
            }
        
        return aggregated
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
