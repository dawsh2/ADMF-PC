"""
Clean workflow managers that orchestrate without executing.

Each manager knows HOW to orchestrate a workflow type,
but delegates all execution to specialized containers.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

from ..containers.backtest import BacktestPattern
from .types import (
    WorkflowConfig,
    WorkflowResult,
    WorkflowPhase,
    ExecutionContext,
    PhaseResult
)
from .protocols import WorkflowManager

logger = logging.getLogger(__name__)


class BaseWorkflowManager(WorkflowManager):
    """Base implementation for workflow managers."""
    
    def __init__(self, coordinator):
        """Initialize with reference to coordinator."""
        self.coordinator = coordinator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute workflow through phases."""
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            success=True,
            errors=[],
            results={}
        )
        
        try:
            # Execute each phase
            phases = self.get_execution_phases()
            
            for phase in phases:
                self.logger.info(f"Executing phase: {phase}")
                
                phase_result = await self.execute_phase(phase, config, context)
                
                if not phase_result.success and self.is_critical_phase(phase):
                    result.success = False
                    result.errors.extend(phase_result.errors)
                    break
                
                # Store phase results
                result.results[phase.value] = phase_result.data
            
            # Aggregate final results
            if result.success:
                result.results['final'] = await self.aggregate_results(
                    result.results
                )
            
        except Exception as e:
            self.logger.error(f"Workflow execution error: {e}")
            result.success = False
            result.errors.append(str(e))
        
        return result
    
    @abstractmethod
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get ordered list of phases to execute."""
        pass
    
    @abstractmethod
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute a specific phase."""
        pass
    
    @abstractmethod
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Check if phase failure should stop execution."""
        pass
    
    @abstractmethod
    async def aggregate_results(
        self,
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate phase results into final output."""
        pass


class BacktestWorkflowManager(BaseWorkflowManager):
    """Manager for backtest workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Backtest phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION,
            WorkflowPhase.AGGREGATION
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute backtest phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Create backtest container
                container_id = await self.coordinator.create_backtest_container(
                    workflow_id=context.workflow_id,
                    config=config,
                    pattern=BacktestPattern.FULL
                )
                
                # Store in context for other phases
                context.shared_resources['container_id'] = container_id
                result.data = {'container_id': container_id}
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Container handles its own data preparation
                container_id = context.shared_resources['container_id']
                
                prep_result = await self.coordinator.execute_container(
                    container_id,
                    method="prepare_data"
                )
                
                result.data = prep_result
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Execute the backtest
                container_id = context.shared_resources['container_id']
                
                backtest_result = await self.coordinator.execute_container(
                    container_id,
                    method="execute_backtest"
                )
                
                result.data = backtest_result
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Get final results
                container_id = context.shared_resources['container_id']
                
                final_results = await self.coordinator.execute_container(
                    container_id,
                    method="get_results"
                )
                
                result.data = final_results
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Phase {phase} error: {e}")
        
        return result
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """All phases except aggregation are critical."""
        return phase != WorkflowPhase.AGGREGATION
    
    async def aggregate_results(
        self,
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key metrics from backtest."""
        computation = phase_results.get(WorkflowPhase.COMPUTATION.value, {})
        
        return {
            'total_return': computation.get('total_return', 0),
            'sharpe_ratio': computation.get('sharpe_ratio', 0),
            'max_drawdown': computation.get('max_drawdown', 0),
            'win_rate': computation.get('win_rate', 0),
            'num_trades': computation.get('num_trades', 0)
        }


class OptimizationWorkflowManager(BaseWorkflowManager):
    """Manager for optimization workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Optimization phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION,  # Phase 1: Parameter search
            WorkflowPhase.VALIDATION,   # Phase 2: Ensemble optimization
            WorkflowPhase.AGGREGATION   # Phase 3: Final validation
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute optimization phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Setup optimization parameters
                result.data = {
                    'parameter_space': config.optimization_config.get('parameters', {}),
                    'objective': config.optimization_config.get('objective', 'sharpe_ratio'),
                    'method': config.optimization_config.get('method', 'grid_search')
                }
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Prepare data splits for walk-forward
                result.data = {
                    'train_periods': config.optimization_config.get('train_periods', []),
                    'test_periods': config.optimization_config.get('test_periods', [])
                }
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Phase 1: Parameter optimization
                # Create multiple backtest containers for parameter search
                param_results = await self._run_parameter_optimization(config, context)
                result.data = param_results
                
                # Store signals for Phase 2
                context.shared_resources['signals'] = param_results.get('signals', {})
                
            elif phase == WorkflowPhase.VALIDATION:
                # Phase 2: Ensemble optimization using signal replay
                signals = context.shared_resources.get('signals', {})
                
                ensemble_results = await self._run_ensemble_optimization(
                    config, context, signals
                )
                result.data = ensemble_results
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Phase 3: Final out-of-sample validation
                best_params = context.shared_resources.get('best_parameters', {})
                
                validation_results = await self._run_final_validation(
                    config, context, best_params
                )
                result.data = validation_results
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.logger.error(f"Phase {phase} error: {e}")
        
        return result
    
    async def _run_parameter_optimization(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run parameter optimization using signal generation pattern."""
        # Create signal generation containers for each parameter set
        results = []
        signals = {}
        
        param_space = config.optimization_config.get('parameters', {})
        
        # Generate parameter combinations
        param_sets = self._generate_parameter_combinations(param_space)
        
        for params in param_sets:
            # Create signal generation container
            container_id = await self.coordinator.create_backtest_container(
                workflow_id=f"{context.workflow_id}_param_{len(results)}",
                config=self._create_config_with_params(config, params),
                pattern=BacktestPattern.SIGNAL_GENERATION
            )
            
            # Generate signals
            signal_result = await self.coordinator.execute_container(
                container_id,
                method="generate_signals"
            )
            
            results.append({
                'parameters': params,
                'metrics': signal_result.get('metrics', {}),
                'signal_quality': signal_result.get('signal_quality', {})
            })
            
            # Store signals for ensemble phase
            signals[str(params)] = signal_result.get('signals', [])
        
        # Find best parameters
        best_result = max(results, key=lambda r: r['metrics'].get('sharpe_ratio', 0))
        context.shared_resources['best_parameters'] = best_result['parameters']
        
        return {
            'parameter_results': results,
            'best_parameters': best_result['parameters'],
            'signals': signals
        }
    
    async def _run_ensemble_optimization(
        self,
        config: WorkflowConfig,
        context: ExecutionContext,
        signals: Dict[str, List]
    ) -> Dict[str, Any]:
        """Run ensemble optimization using signal replay pattern."""
        # Test different weight combinations using signal replay
        results = []
        
        weight_space = config.optimization_config.get('ensemble_weights', {})
        weight_sets = self._generate_weight_combinations(weight_space)
        
        for weights in weight_sets:
            # Create signal replay container
            replay_config = config.copy()
            replay_config.parameters['ensemble_weights'] = weights
            
            container_id = await self.coordinator.create_backtest_container(
                workflow_id=f"{context.workflow_id}_ensemble_{len(results)}",
                config=replay_config,
                pattern=BacktestPattern.SIGNAL_REPLAY
            )
            
            # Replay signals with weights
            replay_result = await self.coordinator.execute_container(
                container_id,
                method="replay_signals",
                signals=signals,
                weights=weights
            )
            
            results.append({
                'weights': weights,
                'metrics': replay_result.get('metrics', {})
            })
        
        # Find best weights
        best_result = max(results, key=lambda r: r['metrics'].get('sharpe_ratio', 0))
        
        return {
            'weight_results': results,
            'best_weights': best_result['weights']
        }
    
    async def _run_final_validation(
        self,
        config: WorkflowConfig,
        context: ExecutionContext,
        best_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run final validation on out-of-sample data."""
        # Create full backtest with best parameters on test data
        test_config = self._create_config_with_params(config, best_params)
        test_config.data_config['dataset'] = 'test'
        
        container_id = await self.coordinator.create_backtest_container(
            workflow_id=f"{context.workflow_id}_validation",
            config=test_config,
            pattern=BacktestPattern.FULL
        )
        
        # Run full backtest
        validation_result = await self.coordinator.execute_container(
            container_id,
            method="execute_backtest"
        )
        
        return {
            'test_metrics': validation_result.get('metrics', {}),
            'validated': validation_result.get('sharpe_ratio', 0) > 0
        }
    
    def _generate_parameter_combinations(
        self,
        param_space: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from space."""
        # Simple grid search implementation
        import itertools
        
        keys = list(param_space.keys())
        values = [param_space[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _generate_weight_combinations(
        self,
        weight_space: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        """Generate weight combinations that sum to 1.0."""
        # This is simplified - real implementation would ensure weights sum to 1
        return self._generate_parameter_combinations(weight_space)
    
    def _create_config_with_params(
        self,
        base_config: WorkflowConfig,
        params: Dict[str, Any]
    ) -> WorkflowConfig:
        """Create new config with updated parameters."""
        new_config = base_config.copy()
        
        # Update strategy parameters
        for strategy in new_config.parameters.get('strategies', []):
            strategy['parameters'].update(params)
        
        return new_config
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Only initialization and computation are critical."""
        return phase in [WorkflowPhase.INITIALIZATION, WorkflowPhase.COMPUTATION]
    
    async def aggregate_results(
        self,
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate optimization results."""
        computation = phase_results.get(WorkflowPhase.COMPUTATION.value, {})
        validation = phase_results.get(WorkflowPhase.VALIDATION.value, {})
        aggregation = phase_results.get(WorkflowPhase.AGGREGATION.value, {})
        
        return {
            'best_parameters': computation.get('best_parameters', {}),
            'best_weights': validation.get('best_weights', {}),
            'in_sample_sharpe': computation.get('best_sharpe', 0),
            'out_sample_sharpe': aggregation.get('test_metrics', {}).get('sharpe_ratio', 0),
            'validated': aggregation.get('validated', False)
        }


class LiveWorkflowManager(BaseWorkflowManager):
    """Manager for live trading workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Live trading phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.COMPUTATION  # Continuous execution
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute live trading phase."""
        # Simplified for now - real implementation would handle live trading
        result = PhaseResult(phase=phase, success=True)
        result.data = {'status': 'not_implemented'}
        return result
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """All phases are critical in live trading."""
        return True
    
    async def aggregate_results(
        self,
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate live trading results."""
        return {'status': 'live_trading_active'}