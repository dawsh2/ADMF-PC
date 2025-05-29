"""
Optimization workflow implementations with phase management support.

These workflows implement the critical architectural decisions from TEST_WORKFLOW.MD
and integrate with the Coordinator's phase management system.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import logging
import hashlib
import json
from pathlib import Path

from ...core.containers import ContainerLifecycleManager, UniversalScopedContainer
from ...core.coordinator import (
    Coordinator,
    PhaseTransition,
    ContainerNamingStrategy,
    ResultAggregator,
    StrategyIdentity,
    CheckpointManager,
    WalkForwardValidator
)
from ..protocols import Strategy, Classifier
from ..components.signal_replay import SignalCapture, SignalReplayer
from .containers import OptimizationContainer, RegimeAwareOptimizationContainer
from .protocols import Optimizer, Objective, Constraint

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


class PhaseAwareOptimizationWorkflow:
    """
    Multi-phase optimization workflow with full phase management support.
    
    Implements all critical architectural decisions:
    1. Clear phase transitions with data flow
    2. Consistent container naming
    3. Result streaming to avoid memory issues
    4. Cross-regime strategy tracking
    5. Checkpointing for resumability
    6. Walk-forward validation support
    """
    
    def __init__(self,
                 coordinator: Coordinator,
                 workflow_config: Dict[str, Any]):
        """
        Initialize phase-aware workflow.
        
        Args:
            coordinator: Enhanced coordinator with phase management
            workflow_config: Complete workflow configuration
        """
        self.coordinator = coordinator
        self.config = workflow_config
        
        # Ensure coordinator has phase management
        if not hasattr(coordinator, 'phase_transitions'):
            from ...core.coordinator import integrate_phase_management
            integrate_phase_management(coordinator)
        
        # Phase management components
        self.phase_transitions = coordinator.phase_transitions
        self.container_naming = coordinator.container_naming
        self.checkpointing = coordinator.checkpointing
        self.result_aggregator = ResultAggregator(
            workflow_config.get('output_dir', './results')
        )
        
        # Strategy tracking
        self.strategy_identities: Dict[str, StrategyIdentity] = {}
        
        # Walk-forward validation
        self.walk_forward = coordinator.walk_forward_validator
        
        # Workflow state
        self.workflow_id = workflow_config.get('workflow_id', self._generate_workflow_id())
        self.current_phase = None
        self.completed_phases = set()
        
    def _generate_workflow_id(self) -> str:
        """Generate unique workflow ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.md5(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"workflow_{timestamp}_{config_hash}"
    
    async def run(self) -> Dict[str, Any]:
        """
        Run complete multi-phase optimization workflow.
        
        Returns:
            Complete workflow results with all phase outputs
        """
        logger.info(f"Starting phase-aware optimization workflow {self.workflow_id}")
        
        try:
            # Phase 1: Parameter Optimization with Parallel Regimes
            phase1_results = await self._run_phase1_optimization()
            
            # Phase 2: Regime Analysis
            phase2_results = await self._run_phase2_analysis()
            
            # Phase 3: Weight Optimization
            phase3_results = await self._run_phase3_weights()
            
            # Phase 4: Walk-Forward Validation
            phase4_results = await self._run_phase4_validation()
            
            # Aggregate final results
            final_results = self._aggregate_results()
            
            return final_results
            
        finally:
            # Clean up
            self.result_aggregator.close()
    
    async def _run_phase1_optimization(self) -> Dict[str, Any]:
        """
        Phase 1: Grid search optimization with parallel regime environments.
        """
        logger.info("Phase 1: Starting parameter optimization")
        self.current_phase = "phase1"
        
        # Get phase configuration
        phase1_config = self.config['phases']['phase1']
        parameter_space = phase1_config['parameter_space']
        regime_classifiers = phase1_config['regime_classifiers']
        
        # Track results by classifier
        classifier_results = {}
        
        # Run optimization for each regime classifier in parallel
        for classifier_type in regime_classifiers:
            logger.info(f"Running optimization with {classifier_type} classifier")
            
            # Create regime classifier
            classifier = self._create_classifier(classifier_type)
            
            # Process each strategy
            for strategy_config in phase1_config['strategies']:
                strategy_class = strategy_config['class']
                base_params = strategy_config.get('base_params', {})
                
                # Track strategy identity
                identity = StrategyIdentity(strategy_class, base_params)
                self.strategy_identities[identity.canonical_id] = identity
                
                # Optimize for each regime
                regime_results = await self._optimize_strategy_by_regime(
                    strategy_class,
                    base_params,
                    parameter_space,
                    classifier,
                    classifier_type
                )
                
                # Store results
                if classifier_type not in classifier_results:
                    classifier_results[classifier_type] = {}
                
                classifier_results[classifier_type][identity.canonical_id] = regime_results
        
        # Record phase outputs
        self.phase_transitions.record_phase_output("1", "classifier_results", classifier_results)
        self.phase_transitions.record_phase_output("1", "parameter_performance", 
            self._extract_parameter_performance(classifier_results))
        
        self.completed_phases.add("phase1")
        return classifier_results
    
    async def _optimize_strategy_by_regime(self,
                                         strategy_class: str,
                                         base_params: Dict[str, Any],
                                         parameter_space: Dict[str, Any],
                                         classifier: Classifier,
                                         classifier_type: str) -> Dict[str, Any]:
        """Optimize strategy parameters for each regime."""
        results_by_regime = {}
        
        # Get market data and classify regimes
        market_data = self.config['market_data']
        regime_data = self._classify_market_data(market_data, classifier)
        
        for regime, regime_periods in regime_data.items():
            logger.info(f"Optimizing {strategy_class} for regime {regime}")
            
            # Generate container ID with full context
            container_id = self.container_naming.generate_container_id(
                phase="phase1",
                regime=f"{classifier_type}_{regime}",
                strategy=strategy_class,
                params=base_params
            )
            
            # Create optimization container
            container = RegimeAwareOptimizationContainer(
                scope_id=container_id,
                parent_container=self.coordinator.coordinator_container
            )
            container.current_regime = regime
            
            try:
                # Run optimization trials
                optimizer = self._create_optimizer(self.config['phases']['phase1']['optimizer'])
                objective = self._create_objective(self.config['phases']['phase1']['objective'])
                
                best_params = None
                best_score = -float('inf')
                
                # Evaluate each parameter combination
                for params in self._generate_parameter_combinations(parameter_space):
                    # Combine with base params
                    full_params = {**base_params, **params}
                    
                    # Run backtest in container
                    with container.create_trial_scope() as trial_scope:
                        # Create strategy
                        strategy = self._create_strategy(strategy_class, full_params)
                        
                        # Run backtest on regime data
                        results = self._run_backtest(strategy, regime_periods, market_data)
                        
                        # Calculate objective
                        score = objective.calculate(results)
                        
                        # Stream result to disk
                        self.result_aggregator.handle_container_result(
                            container_id,
                            {
                                'params': full_params,
                                'score': score,
                                'results': results,
                                'regime': regime,
                                'classifier': classifier_type
                            }
                        )
                        
                        # Track best
                        if score > best_score:
                            best_score = score
                            best_params = full_params
                
                # Record regime results
                results_by_regime[regime] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'container_id': container_id,
                    'num_periods': len(regime_periods)
                }
                
                # Update strategy identity
                identity = self.strategy_identities[
                    StrategyIdentity(strategy_class, base_params).canonical_id
                ]
                identity.add_regime_instance(f"{classifier_type}_{regime}", container_id)
                
            finally:
                # Clean up container
                container.cleanup()
        
        return results_by_regime
    
    async def _run_phase2_analysis(self) -> Dict[str, Any]:
        """
        Phase 2: Analyze regime-specific performance.
        """
        logger.info("Phase 2: Starting regime analysis")
        self.current_phase = "phase2"
        
        # Get phase 1 results
        phase1_results = self.phase_transitions.get_phase_input("2", "classifier_results")
        
        analysis_results = {}
        
        # Analyze each classifier's results
        for classifier_type, strategy_results in phase1_results.items():
            logger.info(f"Analyzing results for {classifier_type} classifier")
            
            classifier_analysis = {}
            
            # Analyze each strategy
            for strategy_id, regime_results in strategy_results.items():
                # Get strategy identity
                identity = self.strategy_identities[strategy_id]
                
                # Compare performance across regimes
                regime_comparison = self._compare_regime_performance(regime_results)
                
                # Identify parameter stability
                param_stability = self._analyze_parameter_stability(regime_results)
                
                classifier_analysis[strategy_id] = {
                    'regime_comparison': regime_comparison,
                    'param_stability': param_stability,
                    'best_overall_params': self._select_best_overall_params(regime_results)
                }
            
            analysis_results[classifier_type] = classifier_analysis
        
        # Cross-classifier comparison
        cross_classifier_analysis = self._compare_across_classifiers(analysis_results)
        
        # Record phase outputs
        self.phase_transitions.record_phase_output("2", "regime_best_params", 
            self._extract_regime_best_params(analysis_results))
        self.phase_transitions.record_phase_output("2", "classifier_comparison", 
            cross_classifier_analysis)
        
        self.completed_phases.add("phase2")
        return analysis_results
    
    async def _run_phase3_weights(self) -> Dict[str, Any]:
        """
        Phase 3: Optimize ensemble weights using signal replay.
        """
        logger.info("Phase 3: Starting weight optimization")
        self.current_phase = "phase3"
        
        # Get signals from phase 1
        phase1_signals = self._load_phase1_signals()
        
        weight_results = {}
        
        # Optimize weights for each regime
        for regime in self._get_all_regimes():
            logger.info(f"Optimizing weights for regime {regime}")
            
            # Filter signals for regime
            regime_signals = self._filter_signals_by_regime(phase1_signals, regime)
            
            # Create signal replayer
            replayer = SignalReplayer(regime_signals)
            
            # Optimize weights
            optimal_weights = self._optimize_signal_weights(replayer, regime)
            
            weight_results[regime] = {
                'weights': optimal_weights,
                'performance': self._evaluate_weights(replayer, optimal_weights)
            }
        
        # Record phase outputs
        self.phase_transitions.record_phase_output("3", "optimal_weights", weight_results)
        
        self.completed_phases.add("phase3")
        return weight_results
    
    async def _run_phase4_validation(self) -> Dict[str, Any]:
        """
        Phase 4: Walk-forward validation on test data.
        """
        logger.info("Phase 4: Starting walk-forward validation")
        self.current_phase = "phase4"
        
        # Get optimal configurations from previous phases
        regime_params = self.phase_transitions.get_phase_input("4", "regime_best_params")
        optimal_weights = self.phase_transitions.get_phase_input("4", "optimal_weights")
        
        # Split data for walk-forward
        test_periods = self._create_walk_forward_periods()
        
        validation_results = []
        
        for period in test_periods:
            logger.info(f"Validating period {period['start']} to {period['end']}")
            
            # Create adaptive strategy with regime switching
            adaptive_strategy = self._create_adaptive_strategy(
                regime_params,
                optimal_weights
            )
            
            # Run validation
            period_results = self._run_validation_period(
                adaptive_strategy,
                period
            )
            
            validation_results.append(period_results)
        
        # Aggregate validation results
        final_performance = self._aggregate_validation_results(validation_results)
        
        self.completed_phases.add("phase4")
        return final_performance
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate all phase results into final output."""
        return {
            'workflow_id': self.workflow_id,
            'completed_phases': list(self.completed_phases),
            'phase_results': {
                'phase1': self.phase_transitions.phase1_outputs,
                'phase2': self.phase_transitions.phase2_outputs,
                'phase3': self.phase_transitions.phase3_outputs
            },
            'top_strategies': self.result_aggregator.get_top_results(10),
            'strategy_identities': {
                sid: identity.regime_instances 
                for sid, identity in self.strategy_identities.items()
            }
        }
    
    # Helper methods would be implemented here...
    def _create_classifier(self, classifier_type: str) -> Classifier:
        """Create regime classifier based on type."""
        pass
    
    def _classify_market_data(self, data: Any, classifier: Classifier) -> Dict[str, List]:
        """Classify market data into regimes."""
        pass
    
    def _generate_parameter_combinations(self, space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from search space."""
        pass
    
    def _create_strategy(self, strategy_class: str, params: Dict[str, Any]) -> Strategy:
        """Create strategy instance with parameters."""
        pass
    
    def _run_backtest(self, strategy: Strategy, periods: List, data: Any) -> Dict[str, Any]:
        """Run backtest for specific periods."""
        pass
    
    def _extract_parameter_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameter performance metrics."""
        pass
    
    def _compare_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across regimes."""
        pass
    
    def _analyze_parameter_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter stability across regimes."""
        pass
    
    def _select_best_overall_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select best overall parameters."""
        pass
    
    def _compare_across_classifiers(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across different classifiers."""
        pass
    
    def _extract_regime_best_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best parameters per regime."""
        pass
    
    def _load_phase1_signals(self) -> List[Dict[str, Any]]:
        """Load captured signals from phase 1."""
        pass
    
    def _get_all_regimes(self) -> List[str]:
        """Get all detected regimes."""
        pass
    
    def _filter_signals_by_regime(self, signals: List, regime: str) -> List:
        """Filter signals for specific regime."""
        pass
    
    def _optimize_signal_weights(self, replayer: SignalReplayer, regime: str) -> Dict[str, float]:
        """Optimize signal weights for regime."""
        pass
    
    def _evaluate_weights(self, replayer: SignalReplayer, weights: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate performance with given weights."""
        pass
    
    def _create_walk_forward_periods(self) -> List[Dict[str, Any]]:
        """Create walk-forward validation periods."""
        pass
    
    def _create_adaptive_strategy(self, params: Dict, weights: Dict) -> Any:
        """Create adaptive strategy with regime switching."""
        pass
    
    def _run_validation_period(self, strategy: Any, period: Dict) -> Dict[str, Any]:
        """Run validation for specific period."""
        pass
    
    def _aggregate_validation_results(self, results: List) -> Dict[str, Any]:
        """Aggregate validation results."""
        pass


# Factory function
def create_phase_aware_workflow(
    coordinator: Coordinator,
    config: Dict[str, Any]
) -> PhaseAwareOptimizationWorkflow:
    """Create enhanced workflow with phase management."""
    return PhaseAwareOptimizationWorkflow(coordinator, config)