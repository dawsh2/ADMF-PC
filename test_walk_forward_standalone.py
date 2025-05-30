"""
Standalone test for refactored walk-forward validation.
No external dependencies - demonstrates the architecture and separation of concerns.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ===== Core Data Structures =====

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


# ===== Protocols (Interfaces) =====

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
class Optimizer(Protocol):
    """Protocol for optimization algorithms."""
    
    def optimize(self, evaluate_func: Any, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization and return best parameters."""
        ...
    
    def get_best_score(self) -> float:
        """Get best score from optimization."""
        ...
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        ...


@runtime_checkable
class Objective(Protocol):
    """Protocol for objective functions."""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate objective value from results."""
        ...


# ===== Walk-Forward Components (Separated Concerns) =====

class WalkForwardPeriodManager:
    """
    Manages walk-forward period generation and data splitting.
    
    Responsibility: ONLY period generation and data slicing.
    """
    
    def __init__(self,
                 data_provider: DataProvider,
                 train_size: int,
                 test_size: int,
                 step_size: int,
                 anchored: bool = False):
        self.data_provider = data_provider
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored
        
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
    
    Responsibility: ONLY parameter optimization and objective calculation.
    """
    
    def __init__(self, optimizer: Optimizer, objective: Objective):
        self.optimizer = optimizer
        self.objective = objective
    
    def optimize_period(self,
                       period: WalkForwardPeriod,
                       parameter_space: Dict[str, Any],
                       evaluate_func: Any) -> Dict[str, Any]:
        """Optimize parameters for a single period."""
        best_params = self.optimizer.optimize(evaluate_func, parameter_space)
        
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
    
    Responsibility: ONLY container creation and backtest execution.
    """
    
    def __init__(self, container_factory: Any, backtest_engine: Any):
        self.container_factory = container_factory
        self.backtest_engine = backtest_engine
    
    def execute_backtest(self,
                        container_id: str,
                        strategy_config: Dict[str, Any],
                        data: Any) -> Dict[str, Any]:
        """Execute backtest in isolated container."""
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
    
    Responsibility: Orchestration, phase management, result aggregation.
    """
    
    def __init__(self,
                 coordinator: Any,
                 period_manager: WalkForwardPeriodManager,
                 optimizer: WalkForwardOptimizer,
                 executor: WalkForwardBacktestExecutor):
        self.coordinator = coordinator
        self.period_manager = period_manager
        self.optimizer = optimizer
        self.executor = executor
        
        self.period_results: List[Dict[str, Any]] = []
    
    def run_walk_forward(self,
                        strategy_class: str,
                        base_params: Dict[str, Any],
                        parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete walk-forward validation."""
        logger.info(f"Starting walk-forward validation for {strategy_class}")
        
        # Process each period
        for period in self.period_manager.get_periods():
            logger.info(f"Processing {period.period_id}")
            
            # Phase 1: Get period data
            period_data = self.period_manager.get_period_data(period)
            
            # Phase 2: Optimize on training data
            optimization_result = self._optimize_period(
                period,
                period_data['train_data'],
                strategy_class,
                base_params,
                parameter_space
            )
            
            # Phase 3: Test on out-of-sample data
            test_result = self._test_period(
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
            self.coordinator.save_checkpoint(
                f"walkforward_{period.period_id}",
                period_result
            )
        
        # Final aggregation
        return self._aggregate_results()
    
    def _optimize_period(self,
                        period: WalkForwardPeriod,
                        train_data: Any,
                        strategy_class: str,
                        base_params: Dict[str, Any],
                        parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy for a single period."""
        
        def evaluate(params: Dict[str, Any]) -> float:
            # Combine base and trial parameters
            full_params = {**base_params, **params}
            
            # Create unique container ID
            container_id = f"wf_train_{period.period_id}_{strategy_class}"
            
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
    
    def _test_period(self,
                    period: WalkForwardPeriod,
                    test_data: Any,
                    strategy_class: str,
                    optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test strategy on out-of-sample data."""
        
        # Create container ID for test
        container_id = f"wf_test_{period.period_id}_{strategy_class}"
        
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


# ===== Mock Implementations for Testing =====

class MockDataProvider:
    """Simple data provider for testing."""
    def __init__(self, data_length: int):
        self.data_length = data_length
        self.data = list(range(data_length))
    
    def get_slice(self, start: int, end: int) -> List[int]:
        return self.data[start:end]
    
    def get_length(self) -> int:
        return self.data_length


class MockOptimizer:
    """Simple optimizer for testing."""
    def __init__(self):
        self.best_score = 0
        self.history = []
    
    def optimize(self, evaluate_func, parameter_space):
        best_params = None
        best_score = -float('inf')
        
        # Simple grid search
        for lookback in parameter_space.get('lookback_period', [20]):
            for threshold in parameter_space.get('momentum_threshold', [0.02]):
                params = {'lookback_period': lookback, 'momentum_threshold': threshold}
                score = evaluate_func(params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        self.best_score = best_score
        self.history.append({'params': best_params, 'score': best_score})
        return best_params
    
    def get_best_score(self):
        return self.best_score
    
    def get_optimization_history(self):
        return self.history


class MockObjective:
    """Simple objective for testing."""
    def calculate(self, results):
        return results.get('sharpe_ratio', 1.0)


class MockContainer:
    """Mock container for testing."""
    def __init__(self, container_id, config):
        self.container_id = container_id
        self.config = config
        self.disposed = False
    
    def dispose(self):
        self.disposed = True


class MockContainerFactory:
    """Mock container factory for testing."""
    def __init__(self):
        self.containers = []
    
    def create_instance(self, config):
        container = MockContainer(config['container_id'], config)
        self.containers.append(container)
        return container


class MockBacktestEngine:
    """Mock backtest engine for testing."""
    def run(self, container):
        # Simulate results based on parameters
        params = container.config['strategy_config']['params']
        lookback = params.get('lookback_period', 20)
        threshold = params.get('momentum_threshold', 0.02)
        
        # Mock performance
        sharpe = 1.0 + (lookback - 10) * 0.01 + (0.03 - threshold) * 10
        
        return {
            'metrics': {
                'sharpe_ratio': sharpe,
                'total_return': 0.1,
                'max_drawdown': 0.05
            },
            'returns': [0.001] * 100,
            'trades': [{'id': i} for i in range(10)]
        }


class MockCoordinator:
    """Mock coordinator for testing."""
    def __init__(self):
        self.checkpoints = {}
    
    def save_checkpoint(self, name, data):
        self.checkpoints[name] = data


# ===== Demonstration =====

def demonstrate_architecture():
    """Demonstrate the refactored walk-forward architecture."""
    
    print("=" * 80)
    print("Refactored Walk-Forward Validation Architecture")
    print("=" * 80)
    
    # Create components
    print("\n1. Creating Components (Separated Concerns)")
    print("-" * 40)
    
    # Data provider - handles data
    data_provider = MockDataProvider(1000)
    print("✓ DataProvider: Manages data slicing")
    
    # Period manager - handles periods
    period_manager = WalkForwardPeriodManager(
        data_provider=data_provider,
        train_size=500,
        test_size=100,
        step_size=100,
        anchored=False
    )
    print("✓ PeriodManager: Manages walk-forward periods")
    
    # Optimizer - handles optimization
    wf_optimizer = WalkForwardOptimizer(
        optimizer=MockOptimizer(),
        objective=MockObjective()
    )
    print("✓ Optimizer: Handles parameter optimization")
    
    # Executor - handles execution
    executor = WalkForwardBacktestExecutor(
        container_factory=MockContainerFactory(),
        backtest_engine=MockBacktestEngine()
    )
    print("✓ Executor: Manages backtest execution")
    
    # Coordinator - orchestrates everything
    coordinator = WalkForwardCoordinator(
        coordinator=MockCoordinator(),
        period_manager=period_manager,
        optimizer=wf_optimizer,
        executor=executor
    )
    print("✓ Coordinator: Orchestrates the workflow")
    
    # Show periods
    print("\n2. Generated Walk-Forward Periods")
    print("-" * 40)
    periods = period_manager.get_periods()
    print(f"Total periods: {len(periods)}")
    for i, period in enumerate(periods[:3]):  # Show first 3
        print(f"\nPeriod {i}:")
        print(f"  Train: [{period.train_start}:{period.train_end}] (size={period.train_size})")
        print(f"  Test:  [{period.test_start}:{period.test_end}] (size={period.test_size})")
    
    # Run walk-forward
    print("\n\n3. Running Walk-Forward Validation")
    print("-" * 40)
    
    results = coordinator.run_walk_forward(
        strategy_class='MomentumStrategy',
        base_params={'signal_cooldown': 3600},
        parameter_space={
            'lookback_period': [10, 20, 30],
            'momentum_threshold': [0.01, 0.02, 0.03]
        }
    )
    
    # Show results
    print("\n\n4. Walk-Forward Results")
    print("-" * 40)
    summary = results['summary']
    print(f"Periods processed: {summary['num_periods']}")
    print(f"Average train score: {summary['train_mean']:.3f}")
    print(f"Average test score: {summary['test_mean']:.3f}")
    print(f"Overfitting ratio: {summary['overfitting_ratio']:.3f}")
    print(f"Strategy robust: {summary['robust']}")
    
    # Show architecture benefits
    print("\n\n5. Architecture Benefits")
    print("-" * 40)
    print("✓ Clear separation of concerns")
    print("✓ Each component has single responsibility")
    print("✓ Easy to test components independently")
    print("✓ Can swap implementations (e.g., different optimizers)")
    print("✓ Follows ADMF-PC protocol-based design")
    
    print("\n" + "=" * 80)
    print("Refactored Architecture Successfully Demonstrated! 🎯")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_architecture()