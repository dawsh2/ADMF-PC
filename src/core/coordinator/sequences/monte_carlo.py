"""
Monte Carlo Sequence

Executes a phase multiple times with randomized parameters
for robustness testing and confidence intervals.
"""

from typing import Dict, Any, List, Optional, Protocol
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


class TopologyBuilderProtocol(Protocol):
    """Protocol for topology builders."""
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        ...


class MonteCarloSequence:
    """
    Execute phase with Monte Carlo simulation.
    
    Runs multiple iterations with randomized parameters to test
    strategy robustness and generate confidence intervals.
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
        Execute Monte Carlo simulation.
        
        Args:
            phase_config: Phase-specific configuration
            base_config: Base workflow configuration
            context: Execution context
            
        Returns:
            Results with statistics and confidence intervals
        """
        phase_name = phase_config.get('name', 'unnamed_phase')
        topology_mode = phase_config.get('topology', 'backtest')
        
        # Get Monte Carlo parameters
        mc_config = self._get_monte_carlo_config(phase_config, base_config)
        iterations = mc_config['iterations']
        
        logger.info(f"Executing Monte Carlo with {iterations} iterations for phase: {phase_name}")
        
        # Run iterations
        iteration_results = []
        for i in range(iterations):
            logger.debug(f"Running iteration {i+1}/{iterations}")
            
            # Randomize parameters
            randomized_config = self._randomize_config(base_config, mc_config)
            
            # Execute iteration
            iteration_result = self._execute_iteration(
                phase_config=phase_config,
                config=randomized_config,
                context=context,
                iteration=i,
                topology_mode=topology_mode
            )
            
            iteration_results.append(iteration_result)
        
        # Calculate statistics
        statistics = self._calculate_statistics(iteration_results)
        
        return {
            'phase_name': phase_name,
            'sequence_type': 'monte_carlo',
            'success': True,
            'iterations': iterations,
            'iteration_results': iteration_results,
            'statistics': statistics,
            'confidence_intervals': self._calculate_confidence_intervals(iteration_results)
        }
    
    def _get_monte_carlo_config(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract Monte Carlo configuration."""
        config_override = phase_config.get('config_override', {})
        mc_config = config_override.get('monte_carlo', base_config.get('monte_carlo', {}))
        
        return {
            'iterations': mc_config.get('iterations', 100),
            'random_seed': mc_config.get('random_seed'),
            'parameter_ranges': mc_config.get('parameter_ranges', {}),
            'randomization_method': mc_config.get('randomization_method', 'uniform'),
            'confidence_level': mc_config.get('confidence_level', 0.95)
        }
    
    def _randomize_config(
        self,
        base_config: Dict[str, Any],
        mc_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Randomize configuration parameters."""
        randomized = base_config.copy()
        param_ranges = mc_config['parameter_ranges']
        method = mc_config['randomization_method']
        
        # Set random seed if specified
        if mc_config.get('random_seed'):
            random.seed(mc_config['random_seed'])
            np.random.seed(mc_config['random_seed'])
        
        # Randomize each parameter
        for param_path, range_config in param_ranges.items():
            # Navigate to parameter location
            parts = param_path.split('.')
            target = randomized
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Apply randomization
            param_name = parts[-1]
            base_value = target.get(param_name, range_config.get('default', 0))
            
            if method == 'uniform':
                # Uniform distribution within range
                min_val = range_config.get('min', base_value * 0.8)
                max_val = range_config.get('max', base_value * 1.2)
                target[param_name] = random.uniform(min_val, max_val)
                
            elif method == 'normal':
                # Normal distribution around base value
                std_ratio = range_config.get('std_ratio', 0.1)
                target[param_name] = np.random.normal(base_value, base_value * std_ratio)
                
            elif method == 'choice':
                # Random choice from list
                choices = range_config.get('choices', [base_value])
                target[param_name] = random.choice(choices)
        
        return randomized
    
    def _execute_iteration(
        self,
        phase_config: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any],
        iteration: int,
        topology_mode: str
    ) -> Dict[str, Any]:
        """Execute a single Monte Carlo iteration."""
        # Merge configurations
        merged_config = config.copy()
        if 'config_override' in phase_config:
            merged_config.update(phase_config['config_override'])
        
        merged_config['monte_carlo_iteration'] = iteration
        
        # Build topology
        topology_definition = {
            'mode': topology_mode,
            'config': merged_config,
            'event_tracer': context.get('event_tracer')
        }
        topology = self.topology_builder.build_topology(topology_definition)
        
        # Execute topology
        result = self._execute_topology(topology, merged_config, context)
        
        return {
            'iteration': iteration,
            'parameters': self._extract_randomized_params(merged_config),
            'result': result,
            'success': True
        }
    
    def _execute_topology(
        self,
        topology: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the topology and return results."""
        # Mock implementation with some randomness
        base_sharpe = 1.5
        noise = np.random.normal(0, 0.3)
        
        return {
            'containers_executed': len(topology.get('containers', {})),
            'metrics': {
                'sharpe_ratio': max(0, base_sharpe + noise),
                'total_return': max(-0.5, 0.15 + noise * 0.1),
                'max_drawdown': min(1.0, abs(0.08 + noise * 0.05))
            }
        }
    
    def _extract_randomized_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the randomized parameters from config."""
        # In real implementation, would track which params were randomized
        return {
            'momentum_threshold': config.get('strategies', {}).get('momentum_threshold', 0.02),
            'position_size': config.get('risk', {}).get('position_size', 0.1)
        }
    
    def _calculate_statistics(self, iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics across all iterations."""
        # Extract metrics from all iterations
        all_metrics = {}
        
        for iteration in iteration_results:
            metrics = iteration.get('result', {}).get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        statistics = {}
        for metric_name, values in all_metrics.items():
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'skew': self._calculate_skew(values),
                    'kurtosis': self._calculate_kurtosis(values)
                }
        
        return statistics
    
    def _calculate_confidence_intervals(
        self,
        iteration_results: List[Dict[str, Any]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for metrics."""
        # Extract metrics
        all_metrics = {}
        
        for iteration in iteration_results:
            metrics = iteration.get('result', {}).get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for metric_name, values in all_metrics.items():
            if len(values) > 1:
                sorted_values = sorted(values)
                n = len(sorted_values)
                
                lower_idx = int(n * (alpha / 2))
                upper_idx = int(n * (1 - alpha / 2))
                
                confidence_intervals[metric_name] = {
                    'lower': sorted_values[lower_idx],
                    'upper': sorted_values[upper_idx],
                    'confidence_level': confidence_level
                }
        
        return confidence_intervals
    
    def _calculate_skew(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        n = len(values)
        skew = (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in values)
        return skew
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of distribution."""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        n = len(values)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mean) / std) ** 4 for x in values)
        kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurt
