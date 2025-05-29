"""
Optimization workflow implementations.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from ...core.containers import ContainerLifecycleManager
from .containers import OptimizationContainer
from .protocols import Optimizer, Objective

logger = logging.getLogger(__name__)


class ContainerizedComponentOptimizer:
    """Optimizes components using container isolation"""
    
    def __init__(self, optimizer: Optimizer, objective: Objective,
                 use_containers: bool = True):
        """
        Initialize containerized optimizer.
        
        Args:
            optimizer: Optimization algorithm
            objective: Objective function
            use_containers: Whether to use container isolation
        """
        self.optimizer = optimizer
        self.objective = objective
        self.use_containers = use_containers
        self.container_manager = ContainerLifecycleManager()
    
    def optimize_component(self, component_spec: Dict[str, Any],
                          backtest_runner: Callable,
                          n_trials: int = None) -> Dict[str, Any]:
        """
        Optimize a component with full isolation.
        
        Args:
            component_spec: Component specification
            backtest_runner: Function to run backtest
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        start_time = datetime.now()
        
        if self.use_containers:
            # Create optimization container directly
            container_id = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            container = OptimizationContainer(container_id, component_spec)
            
            # Initialize the container
            container.initialize_scope()
            container.start()
            
            try:
                
                # Define evaluation function
                def evaluate(params: Dict[str, Any]) -> float:
                    results = container.run_trial(params, backtest_runner)
                    return self.objective.calculate(results)
                
                # Get parameter space
                parameter_space = component_spec.get('parameter_space', {})
                
                # Run optimization
                best_params = self.optimizer.optimize(
                    evaluate, 
                    n_trials,
                    parameter_space=parameter_space
                )
                
                # Collect results
                optimization_results = {
                    'best_parameters': best_params,
                    'best_score': self.optimizer.get_best_score(),
                    'optimization_history': self.optimizer.get_optimization_history(),
                    'all_results': container.get_results(),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
                
                return optimization_results
                
            finally:
                # Clean up container
                container.stop()
                container.dispose()
        
        else:
            # Direct optimization without containers
            return self._optimize_direct(component_spec, backtest_runner, n_trials)
    
    def _optimize_direct(self, component_spec: Dict[str, Any],
                        backtest_runner: Callable,
                        n_trials: int) -> Dict[str, Any]:
        """Direct optimization without container isolation"""
        # This would create components directly without containers
        # Simplified for now
        raise NotImplementedError("Direct optimization not implemented yet")


class SequentialOptimizationWorkflow:
    """Multi-stage optimization workflow"""
    
    def __init__(self, stages: List[Dict[str, Any]]):
        """
        Initialize sequential workflow.
        
        Args:
            stages: List of stage configurations
        """
        self.stages = stages
        self.results = {}
        self.current_stage = None
        self.container_manager = ContainerLifecycleManager()
        self._cancelled = False
    
    def run(self) -> Dict[str, Any]:
        """Execute sequential optimization stages"""
        logger.info(f"Starting sequential optimization with {len(self.stages)} stages")
        
        for stage_idx, stage_config in enumerate(self.stages):
            if self._cancelled:
                logger.info("Workflow cancelled")
                break
            
            stage_name = stage_config.get('name', f'stage_{stage_idx}')
            self.current_stage = stage_name
            
            logger.info(f"Running optimization stage: {stage_name}")
            
            try:
                # Create optimizer and objective for this stage
                optimizer = self._create_optimizer(stage_config['optimizer'])
                objective = self._create_objective(stage_config['objective'])
                
                # Get component configuration
                component_config = self._prepare_component_config(
                    stage_config, 
                    self.results  # Previous results available
                )
                
                # Get backtest runner
                backtest_runner = self._create_backtest_runner(stage_config)
                
                # Run optimization
                component_optimizer = ContainerizedComponentOptimizer(
                    optimizer, objective
                )
                
                stage_results = component_optimizer.optimize_component(
                    component_config,
                    backtest_runner,
                    n_trials=stage_config.get('n_trials', 100)
                )
                
                # Store results
                self.results[stage_name] = stage_results
                
                # Apply results to next stage if needed
                if stage_config.get('feed_forward', True) and stage_idx < len(self.stages) - 1:
                    self._apply_results_to_next_stage(stage_results, stage_idx + 1)
                
                logger.info(f"Stage {stage_name} complete. Best score: {stage_results['best_score']}")
                
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                self.results[stage_name] = {'error': str(e)}
                
                if stage_config.get('critical', True):
                    raise
        
        self.current_stage = None
        return self.results
    
    def get_stages(self) -> List[str]:
        """Get list of workflow stages"""
        return [s.get('name', f'stage_{i}') for i, s in enumerate(self.stages)]
    
    def get_current_stage(self) -> Optional[str]:
        """Get currently executing stage"""
        return self.current_stage
    
    def cancel(self) -> None:
        """Cancel the running workflow"""
        self._cancelled = True
    
    def _create_optimizer(self, config: Dict[str, Any]) -> Optimizer:
        """Create optimizer from configuration"""
        optimizer_type = config.get('type', 'grid')
        
        if optimizer_type == 'grid':
            from .optimizers import GridOptimizer
            return GridOptimizer()
        elif optimizer_type == 'bayesian':
            from .optimizers import BayesianOptimizer
            return BayesianOptimizer(
                acquisition_function=config.get('acquisition', 'expected_improvement')
            )
        elif optimizer_type == 'genetic':
            from .optimizers import GeneticOptimizer
            return GeneticOptimizer(
                population_size=config.get('population_size', 50),
                mutation_rate=config.get('mutation_rate', 0.1)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_objective(self, config: Dict[str, Any]) -> Objective:
        """Create objective from configuration"""
        objective_type = config.get('type', 'sharpe')
        
        if objective_type == 'sharpe':
            from .objectives import SharpeObjective
            return SharpeObjective()
        elif objective_type == 'return':
            from .objectives import MaxReturnObjective
            return MaxReturnObjective()
        elif objective_type == 'drawdown':
            from .objectives import MinDrawdownObjective
            return MinDrawdownObjective()
        elif objective_type == 'composite':
            from .objectives import CompositeObjective
            # Create sub-objectives
            components = []
            for comp_config in config.get('components', []):
                sub_obj = self._create_objective(comp_config)
                weight = comp_config.get('weight', 1.0)
                components.append((sub_obj, weight))
            return CompositeObjective(components)
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    def _prepare_component_config(self, stage_config: Dict[str, Any],
                                 previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare component configuration for stage"""
        # Start with base configuration
        component_config = stage_config.get('component', {}).copy()
        
        # Apply any parameter overrides from previous stages
        if stage_config.get('use_previous_best', False) and previous_results:
            # Find most recent stage with best parameters
            for stage_name in reversed(list(previous_results.keys())):
                if 'best_parameters' in previous_results[stage_name]:
                    best_params = previous_results[stage_name]['best_parameters']
                    if best_params:  # Check if best_params is not None
                        if 'params' not in component_config:
                            component_config['params'] = {}
                        component_config['params'].update(best_params)
                        break
        
        return component_config
    
    def _create_backtest_runner(self, stage_config: Dict[str, Any]) -> Callable:
        """Create backtest runner for stage"""
        # This would integrate with your backtesting system
        # For now, return a mock function
        def mock_backtest(component) -> Dict[str, Any]:
            # Simulate backtest results
            import random
            return {
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'total_return': random.uniform(-0.1, 0.5),
                'max_drawdown': random.uniform(0.05, 0.3),
                'num_trades': random.randint(10, 100)
            }
        
        return mock_backtest
    
    def _apply_results_to_next_stage(self, stage_results: Dict[str, Any],
                                    next_stage_idx: int) -> None:
        """Apply stage results to next stage configuration"""
        if next_stage_idx < len(self.stages):
            next_stage = self.stages[next_stage_idx]
            
            # Apply best parameters if configured
            if next_stage.get('inherit_parameters', False):
                if 'component' not in next_stage:
                    next_stage['component'] = {}
                if 'params' not in next_stage['component']:
                    next_stage['component']['params'] = {}
                
                if 'best_parameters' in stage_results:
                    next_stage['component']['params'].update(
                        stage_results['best_parameters']
                    )


class RegimeBasedOptimizationWorkflow:
    """Optimize separately for each market regime"""
    
    def __init__(self, regime_detector_config: Dict[str, Any],
                 component_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any]):
        """
        Initialize regime-based workflow.
        
        Args:
            regime_detector_config: Configuration for regime detection
            component_config: Base component configuration
            optimizer_config: Optimizer configuration
        """
        self.regime_detector_config = regime_detector_config
        self.component_config = component_config
        self.optimizer_config = optimizer_config
        self.results = {}
        self.current_stage = None
        self._cancelled = False
    
    def run(self) -> Dict[str, Any]:
        """Run regime-specific optimization"""
        logger.info("Starting regime-based optimization workflow")
        
        # Step 1: Detect regimes in training data
        self.current_stage = "regime_detection"
        regimes = self._detect_regimes()
        self.results['detected_regimes'] = list(regimes.keys())
        
        # Step 2: Optimize for each regime
        self.current_stage = "regime_optimization"
        for regime_name, regime_data in regimes.items():
            if self._cancelled:
                break
            
            logger.info(f"Optimizing for regime: {regime_name}")
            
            # Create optimizer and objective
            optimizer = self._create_optimizer(self.optimizer_config)
            objective = self._create_objective(self.optimizer_config)
            
            # Create regime-specific backtest runner
            regime_runner = self._create_regime_backtest_runner(regime_data)
            
            # Optimize
            component_optimizer = ContainerizedComponentOptimizer(
                optimizer, objective
            )
            
            regime_results = component_optimizer.optimize_component(
                self.component_config,
                regime_runner,
                n_trials=self.optimizer_config.get('n_trials_per_regime', 50)
            )
            
            self.results[regime_name] = regime_results
            
            logger.info(f"Regime {regime_name} optimization complete. "
                       f"Best score: {regime_results['best_score']}")
        
        # Step 3: Create adaptive strategy configuration
        self.current_stage = "adaptive_config"
        self.results['adaptive_config'] = self._create_adaptive_config()
        
        self.current_stage = None
        return self.results
    
    def get_stages(self) -> List[str]:
        """Get list of workflow stages"""
        return ["regime_detection", "regime_optimization", "adaptive_config"]
    
    def get_current_stage(self) -> Optional[str]:
        """Get currently executing stage"""
        return self.current_stage
    
    def cancel(self) -> None:
        """Cancel the running workflow"""
        self._cancelled = True
    
    def _detect_regimes(self) -> Dict[str, Any]:
        """Detect regimes in training data"""
        # This would use actual regime detection
        # For now, return mock regimes
        return {
            'TRENDING_UP': {'periods': [(0, 300)]},
            'HIGH_VOLATILITY': {'periods': [(300, 600)]},
            'TRENDING_DOWN': {'periods': [(600, 1000)]}
        }
    
    def _create_optimizer(self, config: Dict[str, Any]) -> Optimizer:
        """Create optimizer from configuration"""
        # Same as SequentialOptimizationWorkflow
        optimizer_type = config.get('type', 'grid')
        
        if optimizer_type == 'grid':
            from .optimizers import GridOptimizer
            return GridOptimizer()
        elif optimizer_type == 'bayesian':
            from .optimizers import BayesianOptimizer
            return BayesianOptimizer()
        elif optimizer_type == 'genetic':
            from .optimizers import GeneticOptimizer
            return GeneticOptimizer()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_objective(self, config: Dict[str, Any]) -> Objective:
        """Create objective from configuration"""
        # Same as SequentialOptimizationWorkflow
        objective_type = config.get('objective', 'sharpe')
        
        if objective_type == 'sharpe':
            from .objectives import SharpeObjective
            return SharpeObjective()
        elif objective_type == 'return':
            from .objectives import MaxReturnObjective
            return MaxReturnObjective()
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    def _create_regime_backtest_runner(self, regime_data: Dict[str, Any]) -> Callable:
        """Create backtest runner for specific regime"""
        # This would filter data for the regime
        def regime_backtest(component) -> Dict[str, Any]:
            # Simulate regime-specific backtest
            import random
            return {
                'sharpe_ratio': random.uniform(0.5, 2.0),
                'total_return': random.uniform(-0.1, 0.5),
                'max_drawdown': random.uniform(0.05, 0.3),
                'num_trades': random.randint(10, 100),
                'regime': regime_data
            }
        
        return regime_backtest
    
    def _create_adaptive_config(self) -> Dict[str, Any]:
        """Create configuration for regime-adaptive strategy"""
        return {
            'regime_detector': self.regime_detector_config,
            'regime_parameters': {
                regime: results['best_parameters']
                for regime, results in self.results.items()
                if regime not in ['detected_regimes', 'adaptive_config']
            }
        }