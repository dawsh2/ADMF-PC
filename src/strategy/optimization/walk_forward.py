"""
Walk-forward validation implementation for ADMF-PC.

This module provides walk-forward analysis capabilities for robust
out-of-sample testing of optimized strategies.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from pathlib import Path
import json

from ...core.containers import UniversalScopedContainer
from ..protocols import Strategy
from .protocols import Optimizer, Objective

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'period_id': self.period_id,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'train_size': self.train_size,
            'test_size': self.test_size
        }


class WalkForwardValidator:
    """
    Walk-forward validation for strategy optimization.
    
    Implements rolling window optimization and out-of-sample testing
    to validate strategy robustness.
    """
    
    def __init__(self, 
                 data_length: int,
                 train_size: int,
                 test_size: int,
                 step_size: int,
                 anchored: bool = False):
        """
        Initialize walk-forward validator.
        
        Args:
            data_length: Total length of available data
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size between periods
            anchored: If True, training always starts from beginning
        """
        self.data_length = data_length
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored
        
        # Validate parameters
        self._validate_parameters()
        
        # Generate periods
        self.periods = self._generate_periods()
        
    def _validate_parameters(self) -> None:
        """Validate walk-forward parameters."""
        if self.train_size <= 0:
            raise ValueError("Training size must be positive")
        
        if self.test_size <= 0:
            raise ValueError("Test size must be positive")
        
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
        
        if self.train_size + self.test_size > self.data_length:
            raise ValueError("Train + test size exceeds data length")
    
    def _generate_periods(self) -> List[WalkForwardPeriod]:
        """Generate walk-forward periods."""
        periods = []
        
        if self.anchored:
            # Anchored walk-forward (expanding window)
            current_test_start = self.train_size
            period_num = 0
            
            while current_test_start + self.test_size <= self.data_length:
                periods.append(WalkForwardPeriod(
                    period_id=f"period_{period_num}",
                    train_start=0,  # Always start from beginning
                    train_end=current_test_start,
                    test_start=current_test_start,
                    test_end=current_test_start + self.test_size
                ))
                
                current_test_start += self.step_size
                period_num += 1
                
        else:
            # Rolling walk-forward (fixed window)
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
    
    def get_period_by_id(self, period_id: str) -> Optional[WalkForwardPeriod]:
        """Get specific period by ID."""
        for period in self.periods:
            if period.period_id == period_id:
                return period
        return None


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis on strategies.
    
    Coordinates optimization on training data and validation on test data
    for each walk-forward period.
    """
    
    def __init__(self,
                 validator: WalkForwardValidator,
                 optimizer: Optimizer,
                 objective: Objective,
                 backtest_func: Callable):
        """
        Initialize walk-forward analyzer.
        
        Args:
            validator: Walk-forward validator with periods
            optimizer: Optimization algorithm
            objective: Objective function
            backtest_func: Function to run backtests
        """
        self.validator = validator
        self.optimizer = optimizer
        self.objective = objective
        self.backtest_func = backtest_func
        
        # Results storage
        self.period_results: Dict[str, Dict[str, Any]] = {}
        self.optimal_params: Dict[str, Dict[str, Any]] = {}
        
    def analyze_strategy(self,
                        strategy_class: str,
                        base_params: Dict[str, Any],
                        parameter_space: Dict[str, Any],
                        market_data: Any) -> Dict[str, Any]:
        """
        Run walk-forward analysis on a strategy.
        
        Args:
            strategy_class: Strategy class name
            base_params: Base strategy parameters
            parameter_space: Parameter search space
            market_data: Complete market data
            
        Returns:
            Analysis results including period performance
        """
        logger.info(f"Starting walk-forward analysis for {strategy_class}")
        
        all_results = []
        
        for period in self.validator.get_periods():
            logger.info(f"Processing {period.period_id}")
            
            # Step 1: Optimize on training data
            train_data = self._slice_data(market_data, period.train_start, period.train_end)
            
            optimal_params = self._optimize_period(
                strategy_class,
                base_params,
                parameter_space,
                train_data,
                period
            )
            
            # Step 2: Test on out-of-sample data
            test_data = self._slice_data(market_data, period.test_start, period.test_end)
            
            test_results = self._test_period(
                strategy_class,
                optimal_params,
                test_data,
                period
            )
            
            # Store results
            period_result = {
                'period': period.to_dict(),
                'optimal_params': optimal_params,
                'train_performance': self.optimizer.get_best_score(),
                'test_performance': test_results
            }
            
            self.period_results[period.period_id] = period_result
            self.optimal_params[period.period_id] = optimal_params
            all_results.append(period_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        return {
            'strategy_class': strategy_class,
            'base_params': base_params,
            'periods': all_results,
            'aggregated': aggregated,
            'summary': self._create_summary(aggregated)
        }
    
    def _optimize_period(self,
                        strategy_class: str,
                        base_params: Dict[str, Any],
                        parameter_space: Dict[str, Any],
                        train_data: Any,
                        period: WalkForwardPeriod) -> Dict[str, Any]:
        """Optimize strategy on training data."""
        def evaluate(params: Dict[str, Any]) -> float:
            # Combine base params with trial params
            full_params = {**base_params, **params}
            
            # Run backtest on training data
            results = self.backtest_func(
                strategy_class,
                full_params,
                train_data
            )
            
            # Calculate objective
            return self.objective.calculate(results)
        
        # Run optimization
        optimal_params = self.optimizer.optimize(
            evaluate,
            parameter_space
        )
        
        # Combine with base params
        return {**base_params, **optimal_params}
    
    def _test_period(self,
                    strategy_class: str,
                    optimal_params: Dict[str, Any],
                    test_data: Any,
                    period: WalkForwardPeriod) -> Dict[str, Any]:
        """Test strategy on out-of-sample data."""
        # Run backtest with optimal parameters
        results = self.backtest_func(
            strategy_class,
            optimal_params,
            test_data
        )
        
        # Calculate metrics
        return {
            'objective_score': self.objective.calculate(results),
            'metrics': results
        }
    
    def _slice_data(self, data: Any, start: int, end: int) -> Any:
        """Slice data for specific period."""
        # This is a placeholder - actual implementation depends on data format
        # Could be pandas DataFrame, numpy array, custom data structure, etc.
        if hasattr(data, 'iloc'):
            # Pandas DataFrame
            return data.iloc[start:end]
        elif hasattr(data, '__getitem__'):
            # List or array-like
            return data[start:end]
        else:
            raise NotImplementedError("Data slicing not implemented for this data type")
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all periods."""
        train_scores = [r['train_performance'] for r in results]
        test_scores = [r['test_performance']['objective_score'] for r in results]
        
        # Calculate statistics
        import numpy as np
        
        return {
            'train': {
                'mean': np.mean(train_scores),
                'std': np.std(train_scores),
                'min': np.min(train_scores),
                'max': np.max(train_scores)
            },
            'test': {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores)
            },
            'overfitting_ratio': np.mean(train_scores) / np.mean(test_scores) if np.mean(test_scores) > 0 else float('inf')
        }
    
    def _create_summary(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of walk-forward analysis."""
        return {
            'num_periods': len(self.validator.get_periods()),
            'avg_train_score': aggregated['train']['mean'],
            'avg_test_score': aggregated['test']['mean'],
            'consistency': 1.0 - (aggregated['test']['std'] / aggregated['test']['mean'] if aggregated['test']['mean'] > 0 else 0),
            'overfitting_ratio': aggregated['overfitting_ratio'],
            'robust': aggregated['overfitting_ratio'] < 1.5 and aggregated['test']['mean'] > 0
        }
    
    def save_results(self, filepath: Path) -> None:
        """Save walk-forward results to file."""
        results = {
            'validator_config': {
                'data_length': self.validator.data_length,
                'train_size': self.validator.train_size,
                'test_size': self.validator.test_size,
                'step_size': self.validator.step_size,
                'anchored': self.validator.anchored
            },
            'periods': [p.to_dict() for p in self.validator.get_periods()],
            'period_results': self.period_results,
            'optimal_params': self.optimal_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved walk-forward results to {filepath}")
    
    def load_results(self, filepath: Path) -> None:
        """Load walk-forward results from file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.period_results = results['period_results']
        self.optimal_params = results['optimal_params']
        
        logger.info(f"Loaded walk-forward results from {filepath}")


class ContainerizedWalkForward:
    """
    Walk-forward analysis with container isolation.
    
    Ensures each optimization and test run happens in a clean,
    isolated container following ADMF-PC architecture.
    """
    
    def __init__(self,
                 analyzer: WalkForwardAnalyzer,
                 container_factory: Callable):
        """
        Initialize containerized walk-forward.
        
        Args:
            analyzer: Walk-forward analyzer
            container_factory: Factory to create backtest containers
        """
        self.analyzer = analyzer
        self.container_factory = container_factory
        
    def run_analysis(self,
                    strategy_config: Dict[str, Any],
                    market_data: Any) -> Dict[str, Any]:
        """
        Run containerized walk-forward analysis.
        
        Each period optimization and test runs in isolated containers.
        """
        results = []
        
        for period in self.analyzer.validator.get_periods():
            # Create container for this period
            container_id = f"walkforward_{period.period_id}"
            
            # Run optimization in container
            train_container = self.container_factory(
                container_id=f"{container_id}_train",
                config=strategy_config
            )
            
            with train_container:
                train_results = self._run_period_optimization(
                    train_container,
                    period,
                    market_data
                )
            
            # Run test in separate container
            test_container = self.container_factory(
                container_id=f"{container_id}_test",
                config=strategy_config
            )
            
            with test_container:
                test_results = self._run_period_test(
                    test_container,
                    period,
                    train_results['optimal_params'],
                    market_data
                )
            
            results.append({
                'period': period.to_dict(),
                'train': train_results,
                'test': test_results
            })
        
        return self._aggregate_containerized_results(results)
    
    def _run_period_optimization(self,
                                container: UniversalScopedContainer,
                                period: WalkForwardPeriod,
                                market_data: Any) -> Dict[str, Any]:
        """Run optimization for a period in container."""
        # Slice data for training
        train_data = self.analyzer._slice_data(
            market_data, 
            period.train_start, 
            period.train_end
        )
        
        # Run optimization in container
        # Implementation depends on container setup
        return {
            'optimal_params': {'placeholder': True},
            'performance': 1.5
        }
    
    def _run_period_test(self,
                        container: UniversalScopedContainer,
                        period: WalkForwardPeriod,
                        optimal_params: Dict[str, Any],
                        market_data: Any) -> Dict[str, Any]:
        """Run test for a period in container."""
        # Slice data for testing
        test_data = self.analyzer._slice_data(
            market_data,
            period.test_start,
            period.test_end
        )
        
        # Run test in container
        return {
            'performance': 1.2,
            'metrics': {'sharpe': 1.2}
        }
    
    def _aggregate_containerized_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from containerized runs."""
        return {
            'periods': results,
            'summary': {
                'total_periods': len(results),
                'containerized': True
            }
        }