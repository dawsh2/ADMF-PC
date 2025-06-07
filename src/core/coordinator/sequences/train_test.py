"""
Train-Test Sequence

Handles train/optimize then test pattern with parameter optimization.
This sequence manages the optimizer integration and parameter selection.
"""

from typing import Dict, Any, List, Optional, Protocol, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TopologyBuilderProtocol(Protocol):
    """Protocol for topology builders."""
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        ...


class OptimizerProtocol(Protocol):
    """Protocol for parameter optimizers."""
    def optimize(
        self,
        parameter_space: Dict[str, Any],
        objective_function: callable,
        method: str = 'grid'
    ) -> Dict[str, Any]:
        ...


class TrainTestSequence:
    """
    Execute train/optimize then test pattern.
    
    This sequence:
    1. Runs training/optimization phase to find best parameters
    2. Validates on test set with selected parameters
    3. Returns both training and test results
    
    Can use grid search, random search, or Bayesian optimization.
    """
    
    def __init__(
        self,
        topology_builder: TopologyBuilderProtocol,
        optimizer: Optional[OptimizerProtocol] = None
    ):
        self.topology_builder = topology_builder
        self.optimizer = optimizer or self._create_default_optimizer()
    
    def execute(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute train-test sequence.
        
        Args:
            phase_config: Phase-specific configuration
            base_config: Base workflow configuration
            context: Execution context with inter-phase data
            
        Returns:
            Results with optimal parameters and test performance
        """
        phase_name = phase_config.get('name', 'unnamed_phase')
        topology_mode = phase_config.get('topology', 'backtest')
        
        # Get train-test configuration
        tt_config = self._get_train_test_config(phase_config, base_config)
        
        logger.info(f"Executing train-test sequence for phase: {phase_name}")
        
        # Step 1: Training/Optimization Phase
        if tt_config['optimize_parameters']:
            # Run optimization
            optimization_result = self._run_optimization(
                phase_config=phase_config,
                base_config=base_config,
                context=context,
                topology_mode=topology_mode,
                train_period=tt_config['train_period'],
                parameter_space=tt_config['parameter_space'],
                optimization_method=tt_config['optimization_method']
            )
            
            optimal_params = optimization_result['optimal_parameters']
            train_result = optimization_result
        else:
            # Just run training with current parameters
            train_result = self._run_training(
                phase_config=phase_config,
                base_config=base_config,
                context=context,
                topology_mode=topology_mode,
                train_period=tt_config['train_period']
            )
            optimal_params = self._extract_current_parameters(base_config)
        
        # Step 2: Testing Phase with optimal parameters
        test_result = self._run_testing(
            phase_config=phase_config,
            base_config=base_config,
            context=context,
            topology_mode=topology_mode,
            test_period=tt_config['test_period'],
            parameters=optimal_params
        )
        
        # Step 3: Validation and Analysis
        validation_result = self._validate_results(
            train_result=train_result,
            test_result=test_result,
            validation_config=tt_config.get('validation', {})
        )
        
        # Prepare inter-phase data for coordinator
        inter_phase_data = {
            'optimal_parameters': optimal_params,
            'parameter_performance': train_result.get('parameter_performance', {}),
            'validation_metrics': validation_result,
            'should_continue': validation_result.get('passed', True)
        }
        
        return {
            'phase_name': phase_name,
            'sequence_type': 'train_test',
            'success': validation_result.get('passed', True),
            'train_result': train_result,
            'test_result': test_result,
            'validation_result': validation_result,
            'optimal_parameters': optimal_params,
            'inter_phase_data': inter_phase_data  # For coordinator to manage
        }
    
    def _get_train_test_config(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract train-test configuration."""
        config_override = phase_config.get('config_override', {})
        tt_config = config_override.get('train_test', base_config.get('train_test', {}))
        
        # Determine train/test periods
        if 'train_period' in tt_config and 'test_period' in tt_config:
            train_period = tt_config['train_period']
            test_period = tt_config['test_period']
        else:
            # Auto-split based on data
            train_ratio = tt_config.get('train_ratio', 0.8)
            train_period, test_period = self._auto_split_periods(
                base_config.get('start_date'),
                base_config.get('end_date'),
                train_ratio
            )
        
        return {
            'train_period': train_period,
            'test_period': test_period,
            'optimize_parameters': tt_config.get('optimize_parameters', True),
            'parameter_space': tt_config.get('parameter_space', {}),
            'optimization_method': tt_config.get('optimization_method', 'grid'),
            'optimization_metric': tt_config.get('optimization_metric', 'sharpe_ratio'),
            'validation': tt_config.get('validation', {
                'min_sharpe': 0.5,
                'max_drawdown': 0.20,
                'min_trades': 10
            })
        }
    
    def _run_optimization(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any],
        topology_mode: str,
        train_period: Tuple[str, str],
        parameter_space: Dict[str, Any],
        optimization_method: str
    ) -> Dict[str, Any]:
        """Run parameter optimization on training data."""
        logger.info(f"Running {optimization_method} optimization on training period")
        
        # Define objective function for optimizer
        def objective_function(parameters: Dict[str, Any]) -> float:
            # Merge parameters into config
            config = base_config.copy()
            self._apply_parameters(config, parameters)
            config['start_date'] = train_period[0]
            config['end_date'] = train_period[1]
            
            # Build and execute topology
            topology_def = {
                'mode': topology_mode,
                'config': config,
                'event_tracer': context.get('event_tracer')
            }
            topology = self.topology_builder.build_topology(topology_def)
            result = self._execute_topology(topology, config, context)
            
            # Return objective value (negative for minimization)
            metric = result.get('metrics', {}).get('sharpe_ratio', 0)
            return -metric  # Negative because optimizers minimize
        
        # Run optimization
        optimization_result = self.optimizer.optimize(
            parameter_space=parameter_space,
            objective_function=objective_function,
            method=optimization_method
        )
        
        # Get performance for all tested parameters
        parameter_performance = optimization_result.get('parameter_performance', {})
        
        return {
            'success': True,
            'optimal_parameters': optimization_result['optimal_parameters'],
            'optimal_metric': -optimization_result['optimal_value'],  # Convert back
            'parameter_performance': parameter_performance,
            'parameters_tested': optimization_result.get('iterations', 0)
        }
    
    def _run_training(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any],
        topology_mode: str,
        train_period: Tuple[str, str]
    ) -> Dict[str, Any]:
        """Run training without optimization."""
        config = base_config.copy()
        config['start_date'] = train_period[0]
        config['end_date'] = train_period[1]
        
        topology_def = {
            'mode': topology_mode,
            'config': config,
            'event_tracer': context.get('event_tracer')
        }
        topology = self.topology_builder.build_topology(topology_def)
        result = self._execute_topology(topology, config, context)
        
        return {
            'success': True,
            'metrics': result.get('metrics', {}),
            'period': train_period
        }
    
    def _run_testing(
        self,
        phase_config: Dict[str, Any],
        base_config: Dict[str, Any],
        context: Dict[str, Any],
        topology_mode: str,
        test_period: Tuple[str, str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run testing with specified parameters."""
        # Merge optimal parameters into config
        config = base_config.copy()
        self._apply_parameters(config, parameters)
        config['start_date'] = test_period[0]
        config['end_date'] = test_period[1]
        
        topology_def = {
            'mode': topology_mode,
            'config': config,
            'event_tracer': context.get('event_tracer')
        }
        topology = self.topology_builder.build_topology(topology_def)
        result = self._execute_topology(topology, config, context)
        
        return {
            'success': True,
            'metrics': result.get('metrics', {}),
            'period': test_period,
            'parameters_used': parameters
        }
    
    def _validate_results(
        self,
        train_result: Dict[str, Any],
        test_result: Dict[str, Any],
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate results against criteria."""
        validation_results = {
            'passed': True,
            'warnings': [],
            'metrics_comparison': {}
        }
        
        # Check test performance against thresholds
        test_metrics = test_result.get('metrics', {})
        
        # Sharpe ratio check
        min_sharpe = validation_config.get('min_sharpe', 0.5)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        if test_sharpe < min_sharpe:
            validation_results['passed'] = False
            validation_results['warnings'].append(
                f"Test Sharpe ratio {test_sharpe:.2f} below minimum {min_sharpe}"
            )
        
        # Drawdown check
        max_dd = validation_config.get('max_drawdown', 0.20)
        test_dd = test_metrics.get('max_drawdown', 1.0)
        if test_dd > max_dd:
            validation_results['passed'] = False
            validation_results['warnings'].append(
                f"Test drawdown {test_dd:.2%} exceeds maximum {max_dd:.2%}"
            )
        
        # Compare train vs test metrics
        train_metrics = train_result.get('metrics', {})
        for metric_name in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            if metric_name in train_metrics and metric_name in test_metrics:
                train_val = train_metrics[metric_name]
                test_val = test_metrics[metric_name]
                degradation = (test_val - train_val) / abs(train_val) if train_val != 0 else 0
                
                validation_results['metrics_comparison'][metric_name] = {
                    'train': train_val,
                    'test': test_val,
                    'degradation': degradation
                }
                
                # Warn on significant degradation
                if abs(degradation) > 0.5:  # 50% degradation
                    validation_results['warnings'].append(
                        f"{metric_name} degraded by {abs(degradation):.1%} from train to test"
                    )
        
        return validation_results
    
    def _auto_split_periods(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float
    ) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Automatically split date range into train/test periods."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end - start).days
        train_days = int(total_days * train_ratio)
        
        train_end = start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        
        return (
            (start_date, train_end.strftime('%Y-%m-%d')),
            (test_start.strftime('%Y-%m-%d'), end_date)
        )
    
    def _apply_parameters(self, config: Dict[str, Any], parameters: Dict[str, Any]):
        """Apply parameters to configuration."""
        for param_path, value in parameters.items():
            # Navigate to parameter location
            parts = param_path.split('.')
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
    
    def _extract_current_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract current parameters from config."""
        # In real implementation, would extract based on parameter definitions
        return {
            'strategies.momentum.threshold': config.get('strategies', {}).get('momentum', {}).get('threshold', 0.02),
            'risk.position_size': config.get('risk', {}).get('position_size', 0.1)
        }
    
    def _execute_topology(
        self,
        topology: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the topology and return results."""
        # Mock implementation
        import random
        return {
            'containers_executed': len(topology.get('containers', {})),
            'metrics': {
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'total_return': random.uniform(0.05, 0.25),
                'max_drawdown': random.uniform(0.05, 0.15),
                'total_trades': random.randint(50, 200)
            }
        }
    
    def _create_default_optimizer(self) -> OptimizerProtocol:
        """Create default optimizer."""
        # Import actual optimizer or create simple grid search
        try:
            from ...optimization import GridSearchOptimizer
            return GridSearchOptimizer()
        except ImportError:
            # Fallback to simple implementation
            return SimpleGridOptimizer()


class SimpleGridOptimizer:
    """Simple grid search optimizer for fallback."""
    
    def optimize(
        self,
        parameter_space: Dict[str, Any],
        objective_function: callable,
        method: str = 'grid'
    ) -> Dict[str, Any]:
        """Run simple grid search optimization."""
        best_params = None
        best_value = float('inf')
        parameter_performance = {}
        iterations = 0
        
        # Generate parameter combinations
        param_combinations = self._generate_combinations(parameter_space)
        
        # Test each combination
        for params in param_combinations:
            value = objective_function(params)
            param_key = str(params)
            parameter_performance[param_key] = value
            
            if value < best_value:
                best_value = value
                best_params = params
            
            iterations += 1
        
        return {
            'optimal_parameters': best_params,
            'optimal_value': best_value,
            'parameter_performance': parameter_performance,
            'iterations': iterations
        }
    
    def _generate_combinations(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        # Simple implementation - in reality would handle nested parameters
        combinations = []
        
        # For now, just return a few test combinations
        combinations.append({
            'strategies.momentum.threshold': 0.01,
            'risk.position_size': 0.05
        })
        combinations.append({
            'strategies.momentum.threshold': 0.02,
            'risk.position_size': 0.10
        })
        combinations.append({
            'strategies.momentum.threshold': 0.03,
            'risk.position_size': 0.15
        })
        
        return combinations
