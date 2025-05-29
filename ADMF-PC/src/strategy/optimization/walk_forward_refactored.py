"""
Refactored walk-forward validation with proper separation of concerns.

Responsibilities are split between:
- WalkForwardValidator: Manages period generation and data splitting
- Optimizer: Handles parameter search and selection
- BacktestContainer: Creates containers and executes backtests
- Coordinator: Orchestrates the entire walk-forward process
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from pathlib import Path
import logging

from ...core.coordinator import Coordinator
from ..protocols import Optimizer, Objective

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period."""
    period_id: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start
    
    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for data providers that can slice data for walk-forward."""
    
    def get_slice(self, start: int, end: int) -> Any:
        """Get data slice for specified range."""
        ...
    
    def get_length(self) -> int:
        """Get total data length."""
        ...


@runtime_checkable
class BacktestExecutor(Protocol):
    """Protocol for backtest execution."""
    
    def create_backtest_container(self, 
                                 container_id: str,
                                 config: Dict[str, Any]) -> Any:
        """Create backtest container with given configuration."""
        ...
    
    def execute_backtest(self,
                        container: Any,
                        strategy_config: Dict[str, Any],
                        data: Any) -> Dict[str, Any]:
        """Execute backtest in container and return results."""
        ...


class WalkForwardPeriodManager:
    """
    Manages walk-forward period generation and data splitting.
    
    This class is responsible ONLY for:
    - Generating walk-forward periods
    - Providing data slices for each period
    """
    
    def __init__(self,
                 data_provider: DataProvider,
                 train_size: int,
                 test_size: int,
                 step_size: int,
                 anchored: bool = False):
        """
        Initialize period manager.
        
        Args:
            data_provider: Provider that can slice data
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size between periods
            anchored: If True, training always starts from beginning
        """
        self.data_provider = data_provider
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored
        
        # Generate periods based on data length
        self.data_length = data_provider.get_length()
        self.periods = self._generate_periods()
    
    def _generate_periods(self) -> List[WalkForwardPeriod]:
        """Generate walk-forward periods based on configuration."""
        periods = []
        
        if self.anchored:
            # Anchored: expanding training window
            current_test_start = self.train_size
            period_num = 0
            
            while current_test_start + self.test_size <= self.data_length:
                periods.append(WalkForwardPeriod(
                    period_id=f"period_{period_num}",
                    train_start=0,
                    train_end=current_test_start,
                    test_start=current_test_start,
                    test_end=current_test_start + self.test_size
                ))
                current_test_start += self.step_size
                period_num += 1
        else:
            # Rolling: fixed-size moving window
            current_start = 0
            period_num = 0
            
            while current_start + self.train_size + self.test_size <= self.data_length:
                periods.append(WalkForwardPeriod(
                    period_id=f"period_{period_num}",
                    train_start=current_start,
                    train_end=current_start + self.train_size,
                    test_start=current_start + self.train_size,
                    test_end=current_start + self.train_size + self.test_size
                ))
                current_start += self.step_size
                period_num += 1
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods
    
    def get_periods(self) -> List[WalkForwardPeriod]:
        """Get all walk-forward periods."""
        return self.periods
    
    def get_period_data(self, period: WalkForwardPeriod) -> Dict[str, Any]:
        """Get training and test data for a specific period."""
        return {
            'train_data': self.data_provider.get_slice(
                period.train_start, 
                period.train_end
            ),
            'test_data': self.data_provider.get_slice(
                period.test_start,
                period.test_end
            )
        }


class WalkForwardOptimizer:
    """
    Handles optimization for walk-forward validation.
    
    This class is responsible ONLY for:
    - Running optimization on training data
    - Selecting best parameters
    - Evaluating on test data
    """
    
    def __init__(self,
                 optimizer: Optimizer,
                 objective: Objective):
        """
        Initialize walk-forward optimizer.
        
        Args:
            optimizer: Optimization algorithm
            objective: Objective function
        """
        self.optimizer = optimizer
        self.objective = objective
    
    def optimize_period(self,
                       period: WalkForwardPeriod,
                       parameter_space: Dict[str, Any],
                       evaluate_func: Any) -> Dict[str, Any]:
        """
        Optimize parameters for a single period.
        
        Args:
            period: Walk-forward period
            parameter_space: Parameter search space
            evaluate_func: Function to evaluate parameters
            
        Returns:
            Optimization results including best parameters
        """
        # Run optimization
        best_params = self.optimizer.optimize(
            evaluate_func,
            parameter_space
        )
        
        return {
            'period_id': period.period_id,
            'best_params': best_params,
            'best_score': self.optimizer.get_best_score(),
            'optimization_history': self.optimizer.get_optimization_history()
        }
    
    def calculate_objective(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate objective value from backtest results."""
        return self.objective.calculate(backtest_results)


class WalkForwardBacktestExecutor:
    """
    Handles backtest execution for walk-forward validation.
    
    This class is responsible ONLY for:
    - Creating backtest containers
    - Running backtests with given parameters
    - Collecting results
    """
    
    def __init__(self,
                 container_factory: Any,
                 backtest_engine: Any):
        """
        Initialize backtest executor.
        
        Args:
            container_factory: Factory to create backtest containers
            backtest_engine: Engine to run backtests
        """
        self.container_factory = container_factory
        self.backtest_engine = backtest_engine
    
    def execute_backtest(self,
                        container_id: str,
                        strategy_config: Dict[str, Any],
                        data: Any) -> Dict[str, Any]:
        """
        Execute backtest in isolated container.
        
        Args:
            container_id: Unique container identifier
            strategy_config: Strategy configuration with parameters
            data: Market data for backtest
            
        Returns:
            Backtest results
        """
        # Create container
        container = self.container_factory.create_instance({
            'container_id': container_id,
            'strategy_config': strategy_config,
            'data_config': {'data': data}
        })
        
        try:
            # Execute backtest
            results = self.backtest_engine.run(container)
            
            # Extract metrics
            return {
                'container_id': container_id,
                'metrics': results.get('metrics', {}),
                'returns': results.get('returns', []),
                'positions': results.get('positions', []),
                'trades': results.get('trades', [])
            }
            
        finally:
            # Clean up container
            container.dispose()


class WalkForwardCoordinator:
    """
    Coordinates the entire walk-forward validation process.
    
    This class is responsible for:
    - Orchestrating the walk-forward workflow
    - Managing phase transitions
    - Aggregating results
    - Handling checkpointing
    """
    
    def __init__(self,
                 coordinator: Coordinator,
                 period_manager: WalkForwardPeriodManager,
                 optimizer: WalkForwardOptimizer,
                 executor: WalkForwardBacktestExecutor):
        """
        Initialize walk-forward coordinator.
        
        Args:
            coordinator: Main system coordinator
            period_manager: Manages periods and data
            optimizer: Handles optimization
            executor: Handles backtest execution
        """
        self.coordinator = coordinator
        self.period_manager = period_manager
        self.optimizer = optimizer
        self.executor = executor
        
        # Results storage
        self.period_results: List[Dict[str, Any]] = []
    
    async def run_walk_forward(self,
                              strategy_class: str,
                              base_params: Dict[str, Any],
                              parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete walk-forward validation.
        
        Args:
            strategy_class: Strategy class to optimize
            base_params: Base strategy parameters
            parameter_space: Parameter search space
            
        Returns:
            Complete walk-forward results
        """
        logger.info(f"Starting walk-forward validation for {strategy_class}")
        
        # Process each period
        for period in self.period_manager.get_periods():
            logger.info(f"Processing {period.period_id}")
            
            # Phase 1: Get period data
            period_data = self.period_manager.get_period_data(period)
            
            # Phase 2: Optimize on training data
            optimization_result = await self._optimize_period(
                period,
                period_data['train_data'],
                strategy_class,
                base_params,
                parameter_space
            )
            
            # Phase 3: Test on out-of-sample data
            test_result = await self._test_period(
                period,
                period_data['test_data'],
                strategy_class,
                optimization_result['best_params']
            )
            
            # Phase 4: Aggregate results
            period_result = {
                'period': period,
                'optimization': optimization_result,
                'test': test_result
            }
            
            self.period_results.append(period_result)
            
            # Checkpoint after each period
            self.coordinator.checkpointing.save_checkpoint(
                f"walkforward_{period.period_id}",
                period_result
            )
        
        # Final aggregation
        return self._aggregate_results()
    
    async def _optimize_period(self,
                              period: WalkForwardPeriod,
                              train_data: Any,
                              strategy_class: str,
                              base_params: Dict[str, Any],
                              parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy for a single period."""
        
        # Create evaluation function
        def evaluate(params: Dict[str, Any]) -> float:
            # Combine base and trial parameters
            full_params = {**base_params, **params}
            
            # Create unique container ID
            container_id = self.coordinator.container_naming.generate_container_id(
                phase='walkforward_train',
                period=period.period_id,
                strategy=strategy_class,
                params=params
            )
            
            # Execute backtest
            results = self.executor.execute_backtest(
                container_id,
                {
                    'class': strategy_class,
                    'params': full_params
                },
                train_data
            )
            
            # Calculate objective
            return self.optimizer.calculate_objective(results['metrics'])
        
        # Run optimization
        return self.optimizer.optimize_period(
            period,
            parameter_space,
            evaluate
        )
    
    async def _test_period(self,
                          period: WalkForwardPeriod,
                          test_data: Any,
                          strategy_class: str,
                          optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test strategy on out-of-sample data."""
        
        # Create container ID for test
        container_id = self.coordinator.container_naming.generate_container_id(
            phase='walkforward_test',
            period=period.period_id,
            strategy=strategy_class,
            params=optimal_params
        )
        
        # Execute backtest
        results = self.executor.execute_backtest(
            container_id,
            {
                'class': strategy_class,
                'params': optimal_params
            },
            test_data
        )
        
        return {
            'container_id': container_id,
            'objective_score': self.optimizer.calculate_objective(results['metrics']),
            'metrics': results['metrics'],
            'trades': len(results.get('trades', []))
        }
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all periods."""
        
        # Extract scores
        train_scores = [r['optimization']['best_score'] for r in self.period_results]
        test_scores = [r['test']['objective_score'] for r in self.period_results]
        
        # Calculate statistics
        import statistics
        
        train_mean = statistics.mean(train_scores) if train_scores else 0
        test_mean = statistics.mean(test_scores) if test_scores else 0
        
        return {
            'periods': self.period_results,
            'summary': {
                'num_periods': len(self.period_results),
                'train_mean': train_mean,
                'train_std': statistics.stdev(train_scores) if len(train_scores) > 1 else 0,
                'test_mean': test_mean,
                'test_std': statistics.stdev(test_scores) if len(test_scores) > 1 else 0,
                'overfitting_ratio': train_mean / test_mean if test_mean > 0 else float('inf'),
                'robust': train_mean / test_mean < 1.5 if test_mean > 0 else False
            }
        }


# Factory function to create walk-forward validation setup
def create_walk_forward_validator(
    coordinator: Coordinator,
    data_provider: DataProvider,
    optimizer: Optimizer,
    objective: Objective,
    container_factory: Any,
    backtest_engine: Any,
    train_size: int,
    test_size: int,
    step_size: int,
    anchored: bool = False
) -> WalkForwardCoordinator:
    """
    Factory function to create complete walk-forward validation setup.
    
    This properly separates concerns:
    - Period management: WalkForwardPeriodManager
    - Optimization: WalkForwardOptimizer
    - Execution: WalkForwardBacktestExecutor
    - Coordination: WalkForwardCoordinator
    """
    
    # Create components
    period_manager = WalkForwardPeriodManager(
        data_provider=data_provider,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=anchored
    )
    
    wf_optimizer = WalkForwardOptimizer(
        optimizer=optimizer,
        objective=objective
    )
    
    executor = WalkForwardBacktestExecutor(
        container_factory=container_factory,
        backtest_engine=backtest_engine
    )
    
    # Create coordinator
    return WalkForwardCoordinator(
        coordinator=coordinator,
        period_manager=period_manager,
        optimizer=wf_optimizer,
        executor=executor
    )