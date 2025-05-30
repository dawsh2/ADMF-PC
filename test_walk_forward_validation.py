"""
Comprehensive tests for walk-forward validation.

Tests the complete walk-forward analysis workflow including:
- Period generation (rolling and anchored)
- Optimization on training data
- Validation on test data
- Result aggregation
- Container isolation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List

from src.strategy.optimization.walk_forward import (
    WalkForwardPeriod,
    WalkForwardValidator,
    WalkForwardAnalyzer,
    ContainerizedWalkForward
)

from src.strategy.optimization import (
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    MaxReturnObjective
)

from src.strategy.strategies.momentum import MomentumStrategy
from src.core.containers import UniversalScopedContainer


class TestWalkForwardPeriod(unittest.TestCase):
    """Test WalkForwardPeriod data class."""
    
    def test_period_creation(self):
        """Test creating walk-forward period."""
        period = WalkForwardPeriod(
            period_id="period_0",
            train_start=0,
            train_end=500,
            test_start=500,
            test_end=600
        )
        
        self.assertEqual(period.train_size, 500)
        self.assertEqual(period.test_size, 100)
        
    def test_period_serialization(self):
        """Test period can be serialized."""
        period = WalkForwardPeriod(
            period_id="period_0",
            train_start=0,
            train_end=500,
            test_start=500,
            test_end=600
        )
        
        data = period.to_dict()
        
        self.assertEqual(data['period_id'], "period_0")
        self.assertEqual(data['train_start'], 0)
        self.assertEqual(data['train_end'], 500)
        self.assertEqual(data['test_start'], 500)
        self.assertEqual(data['test_end'], 600)
        self.assertEqual(data['train_size'], 500)
        self.assertEqual(data['test_size'], 100)


class TestWalkForwardValidator(unittest.TestCase):
    """Test WalkForwardValidator functionality."""
    
    def test_rolling_walk_forward(self):
        """Test rolling (non-anchored) walk-forward generation."""
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=100,
            anchored=False
        )
        
        periods = validator.get_periods()
        
        # Should generate 5 periods
        # Period 0: train[0:500], test[500:600]
        # Period 1: train[100:600], test[600:700]
        # Period 2: train[200:700], test[700:800]
        # Period 3: train[300:800], test[800:900]
        # Period 4: train[400:900], test[900:1000]
        self.assertEqual(len(periods), 5)
        
        # Check first period
        first = periods[0]
        self.assertEqual(first.train_start, 0)
        self.assertEqual(first.train_end, 500)
        self.assertEqual(first.test_start, 500)
        self.assertEqual(first.test_end, 600)
        
        # Check rolling window
        second = periods[1]
        self.assertEqual(second.train_start, 100)  # Rolled forward by step_size
        self.assertEqual(second.train_end, 600)
        
        # Check last period fits exactly
        last = periods[-1]
        self.assertEqual(last.test_end, 1000)
    
    def test_anchored_walk_forward(self):
        """Test anchored (expanding window) walk-forward generation."""
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=100,
            anchored=True
        )
        
        periods = validator.get_periods()
        
        # Should generate 5 periods
        # Period 0: train[0:500], test[500:600]
        # Period 1: train[0:600], test[600:700]  # Training expands
        # Period 2: train[0:700], test[700:800]
        # Period 3: train[0:800], test[800:900]
        # Period 4: train[0:900], test[900:1000]
        self.assertEqual(len(periods), 5)
        
        # All periods should start training from 0
        for period in periods:
            self.assertEqual(period.train_start, 0)
        
        # Training size should expand
        self.assertEqual(periods[0].train_size, 500)
        self.assertEqual(periods[1].train_size, 600)
        self.assertEqual(periods[2].train_size, 700)
        
        # Test windows should roll forward
        self.assertEqual(periods[0].test_start, 500)
        self.assertEqual(periods[1].test_start, 600)
        self.assertEqual(periods[2].test_start, 700)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid train size
        with self.assertRaises(ValueError):
            WalkForwardValidator(
                data_length=1000,
                train_size=0,
                test_size=100,
                step_size=100
            )
        
        # Train + test exceeds data
        with self.assertRaises(ValueError):
            WalkForwardValidator(
                data_length=1000,
                train_size=800,
                test_size=300,
                step_size=100
            )
    
    def test_get_period_by_id(self):
        """Test retrieving specific period."""
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=200,
            anchored=False
        )
        
        period = validator.get_period_by_id("period_1")
        self.assertIsNotNone(period)
        self.assertEqual(period.period_id, "period_1")
        
        # Non-existent period
        period = validator.get_period_by_id("period_999")
        self.assertIsNone(period)


class TestWalkForwardAnalyzer(unittest.TestCase):
    """Test WalkForwardAnalyzer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create validator
        self.validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=200,
            anchored=False
        )
        
        # Create optimizer and objective
        self.optimizer = GridOptimizer()
        self.objective = SharpeObjective()
        
        # Mock backtest function
        self.backtest_func = Mock()
        self.backtest_func.return_value = {
            'returns': [0.001, 0.002, -0.001, 0.003, 0.001],
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': 0.08
        }
        
        # Create analyzer
        self.analyzer = WalkForwardAnalyzer(
            validator=self.validator,
            optimizer=self.optimizer,
            objective=self.objective,
            backtest_func=self.backtest_func
        )
    
    def test_analyze_strategy(self):
        """Test complete strategy analysis."""
        # Mock data
        market_data = list(range(1000))  # Simple list as mock data
        
        # Define strategy and parameters
        strategy_class = 'MomentumStrategy'
        base_params = {'signal_cooldown': 3600}
        parameter_space = {
            'lookback_period': [10, 20, 30],
            'momentum_threshold': [0.01, 0.02, 0.03]
        }
        
        # Mock optimizer behavior
        self.optimizer.optimize = Mock(return_value={
            'lookback_period': 20,
            'momentum_threshold': 0.02
        })
        self.optimizer.get_best_score = Mock(return_value=1.5)
        
        # Run analysis
        results = self.analyzer.analyze_strategy(
            strategy_class,
            base_params,
            parameter_space,
            market_data
        )
        
        # Verify structure
        self.assertIn('strategy_class', results)
        self.assertIn('periods', results)
        self.assertIn('aggregated', results)
        self.assertIn('summary', results)
        
        # Should have analyzed all periods
        self.assertEqual(len(results['periods']), len(self.validator.get_periods()))
        
        # Each period should have results
        for period_result in results['periods']:
            self.assertIn('period', period_result)
            self.assertIn('optimal_params', period_result)
            self.assertIn('train_performance', period_result)
            self.assertIn('test_performance', period_result)
            
            # Optimal params should include base params
            self.assertIn('signal_cooldown', period_result['optimal_params'])
            self.assertIn('lookback_period', period_result['optimal_params'])
    
    def test_period_optimization(self):
        """Test optimization for single period."""
        train_data = list(range(500))
        period = self.validator.get_periods()[0]
        
        # Mock optimizer
        self.optimizer.optimize = Mock(return_value={
            'lookback_period': 20,
            'momentum_threshold': 0.02
        })
        
        optimal_params = self.analyzer._optimize_period(
            'MomentumStrategy',
            {'signal_cooldown': 3600},
            {'lookback_period': [10, 20, 30]},
            train_data,
            period
        )
        
        # Should combine base and optimal params
        self.assertEqual(optimal_params['signal_cooldown'], 3600)
        self.assertEqual(optimal_params['lookback_period'], 20)
        
        # Optimizer should have been called
        self.optimizer.optimize.assert_called_once()
    
    def test_period_testing(self):
        """Test out-of-sample testing for single period."""
        test_data = list(range(100))
        period = self.validator.get_periods()[0]
        optimal_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.02
        }
        
        test_results = self.analyzer._test_period(
            'MomentumStrategy',
            optimal_params,
            test_data,
            period
        )
        
        # Should have objective score and metrics
        self.assertIn('objective_score', test_results)
        self.assertIn('metrics', test_results)
        
        # Backtest should have been called with optimal params
        self.backtest_func.assert_called_with(
            'MomentumStrategy',
            optimal_params,
            test_data
        )
    
    def test_data_slicing(self):
        """Test data slicing for different data types."""
        # List data
        list_data = list(range(100))
        sliced = self.analyzer._slice_data(list_data, 10, 20)
        self.assertEqual(sliced, list(range(10, 20)))
        
        # Array data
        array_data = np.array(range(100))
        sliced = self.analyzer._slice_data(array_data, 10, 20)
        np.testing.assert_array_equal(sliced, np.array(range(10, 20)))
        
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.iloc = Mock()
        self.analyzer._slice_data(mock_df, 10, 20)
        mock_df.iloc.__getitem__.assert_called_with(slice(10, 20))
    
    def test_result_aggregation(self):
        """Test aggregation of results across periods."""
        results = [
            {
                'train_performance': 1.8,
                'test_performance': {'objective_score': 1.5}
            },
            {
                'train_performance': 2.0,
                'test_performance': {'objective_score': 1.4}
            },
            {
                'train_performance': 1.6,
                'test_performance': {'objective_score': 1.3}
            }
        ]
        
        aggregated = self.analyzer._aggregate_results(results)
        
        # Check structure
        self.assertIn('train', aggregated)
        self.assertIn('test', aggregated)
        self.assertIn('overfitting_ratio', aggregated)
        
        # Check calculations
        self.assertAlmostEqual(aggregated['train']['mean'], 1.8, places=1)
        self.assertAlmostEqual(aggregated['test']['mean'], 1.4, places=1)
        
        # Overfitting ratio should be > 1 (train better than test)
        self.assertGreater(aggregated['overfitting_ratio'], 1.0)
    
    def test_summary_creation(self):
        """Test summary statistics."""
        aggregated = {
            'train': {'mean': 1.8, 'std': 0.2, 'min': 1.6, 'max': 2.0},
            'test': {'mean': 1.4, 'std': 0.1, 'min': 1.3, 'max': 1.5},
            'overfitting_ratio': 1.29
        }
        
        summary = self.analyzer._create_summary(aggregated)
        
        # Check summary fields
        self.assertIn('num_periods', summary)
        self.assertIn('avg_train_score', summary)
        self.assertIn('avg_test_score', summary)
        self.assertIn('consistency', summary)
        self.assertIn('robust', summary)
        
        # Strategy should be considered robust (overfitting < 1.5)
        self.assertTrue(summary['robust'])
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        # Add some mock results
        self.analyzer.period_results = {
            'period_0': {'train_performance': 1.5, 'test_performance': 1.2},
            'period_1': {'train_performance': 1.6, 'test_performance': 1.3}
        }
        
        self.analyzer.optimal_params = {
            'period_0': {'lookback': 20},
            'period_1': {'lookback': 25}
        }
        
        # Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = Path(f.name)
        
        self.analyzer.save_results(filepath)
        
        # Create new analyzer and load
        new_analyzer = WalkForwardAnalyzer(
            self.validator,
            self.optimizer,
            self.objective,
            self.backtest_func
        )
        
        new_analyzer.load_results(filepath)
        
        # Verify loaded data
        self.assertEqual(new_analyzer.period_results, self.analyzer.period_results)
        self.assertEqual(new_analyzer.optimal_params, self.analyzer.optimal_params)
        
        # Clean up
        filepath.unlink()


class TestContainerizedWalkForward(unittest.TestCase):
    """Test containerized walk-forward validation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=300,
            anchored=False
        )
        
        optimizer = Mock()
        objective = Mock()
        backtest_func = Mock()
        
        self.analyzer = WalkForwardAnalyzer(
            validator,
            optimizer,
            objective,
            backtest_func
        )
        
        # Mock container factory
        self.container_factory = Mock()
        
        # Create containerized walk-forward
        self.containerized = ContainerizedWalkForward(
            self.analyzer,
            self.container_factory
        )
    
    def test_containerized_analysis(self):
        """Test walk-forward with container isolation."""
        strategy_config = {
            'class': 'MomentumStrategy',
            'params': {'lookback_period': 20}
        }
        
        market_data = list(range(1000))
        
        # Mock container creation
        mock_container = MagicMock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        
        self.container_factory.return_value = mock_container
        
        # Mock period optimization/test methods
        self.containerized._run_period_optimization = Mock(return_value={
            'optimal_params': {'lookback_period': 25},
            'performance': 1.5
        })
        
        self.containerized._run_period_test = Mock(return_value={
            'performance': 1.2,
            'metrics': {'sharpe': 1.2}
        })
        
        # Run analysis
        results = self.containerized.run_analysis(strategy_config, market_data)
        
        # Should have results for each period
        self.assertIn('periods', results)
        self.assertEqual(len(results['periods']), len(self.analyzer.validator.get_periods()))
        
        # Each period should have container results
        for period_result in results['periods']:
            self.assertIn('train', period_result)
            self.assertIn('test', period_result)
        
        # Containers should have been created for each period
        expected_calls = len(self.analyzer.validator.get_periods()) * 2  # train + test
        self.assertEqual(self.container_factory.call_count, expected_calls)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete walk-forward scenarios."""
    
    def test_momentum_strategy_walk_forward(self):
        """Test walk-forward on momentum strategy."""
        # Create 1000 days of mock price data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 1000)))
        
        market_data = [
            {
                'symbol': 'TEST',
                'close': price,
                'timestamp': datetime(2020, 1, 1) + timedelta(days=i)
            }
            for i, price in enumerate(prices)
        ]
        
        # Create walk-forward validator
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=600,
            test_size=100,
            step_size=100,
            anchored=False
        )
        
        # Create optimizer and objective
        optimizer = GridOptimizer()
        objective = SharpeObjective()
        
        # Mock backtest function
        def mock_backtest(strategy_class, params, data):
            # Simple mock: higher lookback = lower volatility = higher Sharpe
            lookback = params.get('lookback_period', 20)
            base_sharpe = 1.0
            sharpe = base_sharpe + (lookback - 10) * 0.05
            
            # Add some noise
            sharpe += np.random.normal(0, 0.1)
            
            returns = np.random.normal(0.001, 0.01, len(data))
            
            return {
                'returns': returns.tolist(),
                'sharpe_ratio': sharpe,
                'total_return': np.sum(returns),
                'max_drawdown': 0.1
            }
        
        # Create analyzer
        analyzer = WalkForwardAnalyzer(
            validator=validator,
            optimizer=optimizer,
            objective=objective,
            backtest_func=mock_backtest
        )
        
        # Run analysis
        results = analyzer.analyze_strategy(
            strategy_class='MomentumStrategy',
            base_params={'signal_cooldown': 3600},
            parameter_space={
                'lookback_period': [10, 20, 30, 40, 50]
            },
            market_data=market_data
        )
        
        # Verify results
        self.assertEqual(len(results['periods']), 4)  # 4 walk-forward periods
        
        # Check for overfitting
        summary = results['summary']
        print(f"Walk-forward summary: {summary}")
        
        # Average test score should be positive
        self.assertGreater(summary['avg_test_score'], 0)
        
        # Should not be severely overfit
        self.assertLess(summary['overfitting_ratio'], 2.0)
    
    def test_regime_aware_walk_forward(self):
        """Test walk-forward with regime awareness."""
        # Create data with regime changes
        data_length = 1000
        
        # First 400: trending market
        trend_returns = np.random.normal(0.002, 0.01, 400)
        
        # Next 300: volatile market
        volatile_returns = np.random.normal(0, 0.03, 300)
        
        # Last 300: ranging market
        range_returns = np.random.normal(0, 0.005, 300)
        
        all_returns = np.concatenate([trend_returns, volatile_returns, range_returns])
        prices = 100 * np.exp(np.cumsum(all_returns))
        
        market_data = [
            {
                'symbol': 'TEST',
                'close': price,
                'returns': ret,
                'regime': 'TREND' if i < 400 else 'VOLATILE' if i < 700 else 'RANGE',
                'timestamp': datetime(2020, 1, 1) + timedelta(days=i)
            }
            for i, (price, ret) in enumerate(zip(prices, all_returns))
        ]
        
        # Create walk-forward validator
        validator = WalkForwardValidator(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=150,
            anchored=False
        )
        
        # Mock regime-aware backtest
        def regime_aware_backtest(strategy_class, params, data):
            # Performance depends on regime
            regimes = [d['regime'] for d in data]
            regime_counts = {
                'TREND': regimes.count('TREND'),
                'VOLATILE': regimes.count('VOLATILE'),
                'RANGE': regimes.count('RANGE')
            }
            
            # Different parameters work better in different regimes
            lookback = params.get('lookback_period', 20)
            
            # Short lookback better for trending
            # Long lookback better for volatile
            trend_score = 1.5 - abs(lookback - 15) * 0.05
            volatile_score = 1.5 - abs(lookback - 40) * 0.05
            range_score = 1.5 - abs(lookback - 25) * 0.05
            
            # Weighted average based on regime prevalence
            total = sum(regime_counts.values())
            sharpe = (
                trend_score * regime_counts['TREND'] / total +
                volatile_score * regime_counts['VOLATILE'] / total +
                range_score * regime_counts['RANGE'] / total
            )
            
            returns = [d['returns'] for d in data]
            
            return {
                'returns': returns,
                'sharpe_ratio': sharpe,
                'total_return': sum(returns),
                'max_drawdown': 0.15
            }
        
        # Create analyzer
        analyzer = WalkForwardAnalyzer(
            validator=validator,
            optimizer=GridOptimizer(),
            objective=SharpeObjective(),
            backtest_func=regime_aware_backtest
        )
        
        # Run analysis
        results = analyzer.analyze_strategy(
            strategy_class='AdaptiveStrategy',
            base_params={'adapt_to_regime': True},
            parameter_space={
                'lookback_period': [15, 20, 25, 30, 35, 40]
            },
            market_data=market_data
        )
        
        # Periods should adapt to different regimes
        for i, period_result in enumerate(results['periods']):
            optimal_lookback = period_result['optimal_params']['lookback_period']
            print(f"Period {i}: Optimal lookback = {optimal_lookback}")
        
        # Summary should show regime adaptation
        summary = results['summary']
        self.assertGreater(summary['consistency'], 0.5)  # Should be somewhat consistent


if __name__ == '__main__':
    unittest.main()