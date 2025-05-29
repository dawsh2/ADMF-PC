# Enhanced Multi-Phase Optimization Workflow Testing Strategy

## Overview

This document outlines a comprehensive testing strategy for the multi-phase optimization workflow with parallel regime classifier environments. The approach emphasizes unit testing of individual phases, integration testing of the complete workflow, and specific tests for the parallel regime architecture.

## Testing Architecture

### 1. Test Data Fixtures

```python
# test/fixtures/market_data.py
class MarketDataFixtures:
    """Provides consistent test data for all workflow tests"""
    
    @staticmethod
    def create_regime_aware_data() -> pd.DataFrame:
        """Create synthetic data with clear regime transitions"""
        # Create 1000 bars with 3 distinct regimes
        data = []
        
        # Regime 1: Trending Up (bars 0-299)
        for i in range(300):
            data.append({
                'timestamp': datetime(2023, 1, 1) + timedelta(minutes=i),
                'open': 100 + i * 0.1 + random.uniform(-0.5, 0.5),
                'close': 100 + i * 0.1 + random.uniform(-0.3, 0.7),
                'high': 100 + i * 0.1 + random.uniform(0, 1),
                'low': 100 + i * 0.1 + random.uniform(-1, 0),
                'volume': 1000 + random.randint(-100, 100)
            })
        
        # Regime 2: High Volatility (bars 300-599)
        for i in range(300, 600):
            data.append({
                'timestamp': datetime(2023, 1, 1) + timedelta(minutes=i),
                'open': 130 + random.uniform(-5, 5),
                'close': 130 + random.uniform(-5, 5),
                'high': 135 + random.uniform(0, 5),
                'low': 125 + random.uniform(-5, 0),
                'volume': 2000 + random.randint(-500, 500)
            })
        
        # Regime 3: Trending Down (bars 600-999)
        for i in range(600, 1000):
            data.append({
                'timestamp': datetime(2023, 1, 1) + timedelta(minutes=i),
                'open': 130 - (i-600) * 0.05 + random.uniform(-0.5, 0.5),
                'close': 130 - (i-600) * 0.05 + random.uniform(-0.7, 0.3),
                'high': 130 - (i-600) * 0.05 + random.uniform(0, 1),
                'low': 130 - (i-600) * 0.05 + random.uniform(-1, 0),
                'volume': 1500 + random.randint(-200, 200)
            })
        
        df = pd.DataFrame(data)
        df['symbol'] = 'TEST'
        return df
    
    @staticmethod
    def create_simple_data(n_bars: int = 100) -> pd.DataFrame:
        """Create simple test data for unit tests"""
        # ... simplified version
    
    @staticmethod
    def create_multi_regime_transition_data() -> pd.DataFrame:
        """Create data with multiple rapid regime transitions for stress testing"""
        data = []
        regimes = ['BULL', 'BEAR', 'NEUTRAL', 'VOLATILE']
        bars_per_regime = 50
        
        for regime_idx, regime in enumerate(regimes * 3):  # 12 regime transitions
            base_price = 100 + regime_idx * 5
            for i in range(bars_per_regime):
                if regime == 'BULL':
                    price_drift = i * 0.1
                    volatility = 0.5
                elif regime == 'BEAR':
                    price_drift = -i * 0.1
                    volatility = 0.5
                elif regime == 'NEUTRAL':
                    price_drift = 0
                    volatility = 0.3
                else:  # VOLATILE
                    price_drift = 0
                    volatility = 3.0
                
                data.append({
                    'timestamp': datetime(2023, 1, 1) + timedelta(minutes=len(data)),
                    'open': base_price + price_drift + random.uniform(-volatility, volatility),
                    'close': base_price + price_drift + random.uniform(-volatility, volatility),
                    'high': base_price + price_drift + random.uniform(0, volatility * 1.5),
                    'low': base_price + price_drift + random.uniform(-volatility * 1.5, 0),
                    'volume': 1000 + random.randint(-100, 100),
                    'expected_regime': regime  # For validation
                })
        
        return pd.DataFrame(data)
```

### 2. Component Test Helpers

```python
# test/helpers/container_helpers.py
class ContainerTestHelpers:
    """Utilities for testing with containers"""
    
    @staticmethod
    def create_test_indicator_container(indicators: List[Dict]) -> UniversalScopedContainer:
        """Create indicator container for testing"""
        container = UniversalScopedContainer("test_indicators")
        
        for ind_spec in indicators:
            container.create_component({
                'name': f"{ind_spec['type']}_{ind_spec['period']}",
                'class': ind_spec['type'],
                'params': {'period': ind_spec['period']},
                'capabilities': []  # Minimal for testing
            })
        
        container.initialize_scope()
        return container
    
    @staticmethod
    def create_test_strategy_container(
        strategy_spec: Dict,
        shared_indicators: UniversalScopedContainer
    ) -> UniversalScopedContainer:
        """Create strategy container with shared indicators"""
        container = UniversalScopedContainer(f"test_strategy_{strategy_spec['id']}")
        
        # Register shared indicators
        container.register_shared_service('indicator_hub', shared_indicators)
        
        # Create strategy
        container.create_component({
            'name': 'strategy',
            'class': strategy_spec['class'],
            'params': strategy_spec['params'],
            'capabilities': ['events', 'optimization']
        })
        
        # Create isolated portfolio
        container.create_component({
            'name': 'portfolio',
            'class': 'Portfolio',
            'params': {'initial_cash': 100000}
        })
        
        container.initialize_scope()
        return container
    
    @staticmethod
    def create_regime_classifier_environment(
        classifier_type: str,
        classifier_config: Dict[str, Any],
        strategy_specs: List[Dict[str, Any]],
        shared_indicators: UniversalScopedContainer
    ) -> RegimeClassifierEnvironment:
        """Create a complete regime classifier environment for testing"""
        env = RegimeClassifierEnvironment(classifier_type, classifier_config)
        env.register_shared_service('indicator_hub', shared_indicators)
        env.setup_environment(strategy_specs)
        return env
```

## Phase-by-Phase Testing

### Phase 1: Grid Search Parameter Optimization with Parallel Regimes

```python
# test/test_phase1_grid_search.py
class TestPhase1GridSearch:
    """Test grid search with regime tracking"""
    
    def test_shared_indicator_calculation(self):
        """Test that indicators are calculated once and shared"""
        # Setup
        data = MarketDataFixtures.create_simple_data()
        indicator_container = ContainerTestHelpers.create_test_indicator_container([
            {'type': 'SMA', 'period': 10},
            {'type': 'SMA', 'period': 20}
        ])
        
        # Track calculation count
        calculation_count = 0
        original_calculate = indicator_container.resolve('SMA_10').calculate
        
        def counting_calculate(*args, **kwargs):
            nonlocal calculation_count
            calculation_count += 1
            return original_calculate(*args, **kwargs)
        
        indicator_container.resolve('SMA_10').calculate = counting_calculate
        
        # Process data through indicator hub
        for _, bar in data.iterrows():
            indicator_container.resolve('indicator_hub').process_bar(bar)
        
        # Verify calculated only once per bar
        assert calculation_count == len(data)
    
    def test_strategy_isolation(self):
        """Test that each strategy container maintains isolated state"""
        # Create shared indicators
        indicator_container = ContainerTestHelpers.create_test_indicator_container([
            {'type': 'SMA', 'period': 10}
        ])
        
        # Create two strategy containers with different parameters
        strategy1 = ContainerTestHelpers.create_test_strategy_container(
            {'id': '1', 'class': 'MAStrategy', 'params': {'threshold': 0.01}},
            indicator_container
        )
        strategy2 = ContainerTestHelpers.create_test_strategy_container(
            {'id': '2', 'class': 'MAStrategy', 'params': {'threshold': 0.02}},
            indicator_container
        )
        
        # Process same data through both
        data = MarketDataFixtures.create_simple_data()
        for _, bar in data.iterrows():
            # Indicators calculated once
            indicator_event = indicator_container.process_bar(bar)
            # Broadcast to both strategies
            strategy1.process_event(indicator_event)
            strategy2.process_event(indicator_event)
        
        # Verify different results due to different parameters
        portfolio1 = strategy1.resolve('portfolio')
        portfolio2 = strategy2.resolve('portfolio')
        
        assert portfolio1.get_total_trades() != portfolio2.get_total_trades()
        assert portfolio1.get_value() != portfolio2.get_value()
    
    def test_regime_tracking_during_optimization(self):
        """Test that regime classifications are properly tracked"""
        # Create regime-aware data
        data = MarketDataFixtures.create_regime_aware_data()
        
        # Create regime detector
        regime_detector = RegimeDetector(
            volatility_window=20,
            volatility_threshold=0.02
        )
        
        # Run optimization trial
        results = []
        for _, bar in data.iterrows():
            regime = regime_detector.classify(bar)
            # Store results with regime tag
            results.append({
                'timestamp': bar['timestamp'],
                'regime': regime,
                'price': bar['close']
            })
        
        # Verify all regimes detected
        regime_counts = pd.DataFrame(results)['regime'].value_counts()
        assert 'TRENDING_UP' in regime_counts
        assert 'HIGH_VOLATILITY' in regime_counts
        assert 'TRENDING_DOWN' in regime_counts
    
    def test_parallel_regime_environments(self):
        """Test parallel execution of different regime classifier environments"""
        # Setup shared infrastructure
        data = MarketDataFixtures.create_regime_aware_data()
        indicator_hub = ContainerTestHelpers.create_test_indicator_container([
            {'type': 'SMA', 'period': 10},
            {'type': 'RSI', 'period': 14}
        ])
        
        # Create strategy specs
        strategy_specs = [
            {'class': 'MAStrategy', 'params': {'fast': 5, 'slow': 20}},
            {'class': 'RSIStrategy', 'params': {'period': 14, 'threshold': 30}}
        ]
        
        # Create multiple regime environments
        hmm_env = ContainerTestHelpers.create_regime_classifier_environment(
            'HMM',
            {'states': ['BULL', 'BEAR', 'NEUTRAL']},
            strategy_specs,
            indicator_hub
        )
        
        vol_env = ContainerTestHelpers.create_regime_classifier_environment(
            'Volatility',
            {'thresholds': {'low': 0.01, 'high': 0.03}},
            strategy_specs,
            indicator_hub
        )
        
        # Process data through both environments
        hmm_results = []
        vol_results = []
        
        for _, bar in data.iterrows():
            # Calculate indicators once
            indicator_data = indicator_hub.process_bar(bar)
            
            # Each environment processes independently
            hmm_regime = hmm_env.process_indicator_event(indicator_data)
            vol_regime = vol_env.process_indicator_event(indicator_data)
            
            hmm_results.append(hmm_regime)
            vol_results.append(vol_regime)
        
        # Verify different regime classifications
        hmm_regimes = set(r['regime'] for r in hmm_results)
        vol_regimes = set(r['regime'] for r in vol_results)
        
        assert hmm_regimes != vol_regimes  # Different classifier paradigms
        assert len(hmm_regimes) >= 2  # Multiple regimes detected
        assert len(vol_regimes) >= 2
    
    def test_cross_classifier_strategy_consistency(self):
        """Test that same strategy produces consistent results across different classifiers"""
        # Create identical strategies in different classifier environments
        data = MarketDataFixtures.create_simple_data()
        indicator_hub = ContainerTestHelpers.create_test_indicator_container([
            {'type': 'SMA', 'period': 10}
        ])
        
        # Same strategy spec
        strategy_spec = {'class': 'MAStrategy', 'params': {'threshold': 0.01}}
        
        # Different classifier environments
        hmm_env = ContainerTestHelpers.create_regime_classifier_environment(
            'HMM', {}, [strategy_spec], indicator_hub
        )
        vol_env = ContainerTestHelpers.create_regime_classifier_environment(
            'Volatility', {}, [strategy_spec], indicator_hub
        )
        
        # Process same data
        for _, bar in data.iterrows():
            indicator_data = indicator_hub.process_bar(bar)
            hmm_env.process_indicator_event(indicator_data)
            vol_env.process_indicator_event(indicator_data)
        
        # Get portfolios from each environment
        hmm_portfolio = hmm_env.strategy_containers[0].resolve('portfolio')
        vol_portfolio = vol_env.strategy_containers[0].resolve('portfolio')
        
        # Both should have valid results (though different due to regime context)
        assert hmm_portfolio.get_total_trades() > 0
        assert vol_portfolio.get_total_trades() > 0
        
        # But results should differ due to regime influence
        assert hmm_portfolio.get_value() != vol_portfolio.get_value()
```

### Phase 2: Regime Analysis

```python
# test/test_phase2_regime_analysis.py
class TestPhase2RegimeAnalysis:
    """Test regime-specific parameter analysis"""
    
    def test_regime_analyzer_segmentation(self):
        """Test that RegimeAnalyzer correctly segments results by regime"""
        # Mock optimization results from Phase 1
        optimization_results = [
            {
                'params': {'fast': 10, 'slow': 20},
                'trades': [
                    {'timestamp': datetime(2023, 1, 1), 'pnl': 100, 'regime': 'TRENDING_UP'},
                    {'timestamp': datetime(2023, 1, 2), 'pnl': -50, 'regime': 'HIGH_VOLATILITY'},
                    {'timestamp': datetime(2023, 1, 3), 'pnl': 150, 'regime': 'TRENDING_UP'}
                ]
            },
            {
                'params': {'fast': 5, 'slow': 15},
                'trades': [
                    {'timestamp': datetime(2023, 1, 1), 'pnl': -20, 'regime': 'TRENDING_UP'},
                    {'timestamp': datetime(2023, 1, 2), 'pnl': 200, 'regime': 'HIGH_VOLATILITY'},
                    {'timestamp': datetime(2023, 1, 3), 'pnl': -30, 'regime': 'TRENDING_UP'}
                ]
            }
        ]
        
        # Analyze
        analyzer = RegimeAnalyzer()
        regime_results = analyzer.analyze(optimization_results)
        
        # Verify best parameters identified per regime
        assert regime_results['TRENDING_UP']['best_params'] == {'fast': 10, 'slow': 20}
        assert regime_results['HIGH_VOLATILITY']['best_params'] == {'fast': 5, 'slow': 15}
    
    def test_minimum_trade_requirement(self):
        """Test that regime analysis respects minimum trade requirements"""
        analyzer = RegimeAnalyzer(min_trades_per_regime=5)
        
        # Results with insufficient trades
        sparse_results = [{
            'params': {'fast': 10, 'slow': 20},
            'trades': [
                {'regime': 'RARE_REGIME', 'pnl': 1000}  # Only 1 trade
            ]
        }]
        
        regime_results = analyzer.analyze(sparse_results)
        
        # Should not include regime with insufficient data
        assert 'RARE_REGIME' not in regime_results
    
    def test_parallel_regime_analysis_comparison(self):
        """Test analysis of results from different regime classifiers"""
        # Mock results from parallel environments
        hmm_results = {
            'classifier': 'HMM',
            'optimization_results': [
                {
                    'params': {'fast': 10, 'slow': 20},
                    'performance_by_regime': {
                        'BULL': {'sharpe': 1.5, 'trades': 50},
                        'BEAR': {'sharpe': -0.3, 'trades': 30},
                        'NEUTRAL': {'sharpe': 0.8, 'trades': 20}
                    }
                }
            ]
        }
        
        vol_results = {
            'classifier': 'Volatility',
            'optimization_results': [
                {
                    'params': {'fast': 10, 'slow': 20},
                    'performance_by_regime': {
                        'LOW_VOL': {'sharpe': 1.2, 'trades': 60},
                        'HIGH_VOL': {'sharpe': 0.5, 'trades': 40}
                    }
                }
            ]
        }
        
        # Analyze cross-classifier performance
        cross_analyzer = CrossClassifierAnalyzer()
        comparison = cross_analyzer.compare_classifiers([hmm_results, vol_results])
        
        # Verify comparison metrics
        assert 'best_classifier_by_strategy' in comparison
        assert 'regime_stability_scores' in comparison
        assert 'parameter_consistency' in comparison
```

### Phase 3: Weight Optimization

```python
# test/test_phase3_weight_optimization.py
class TestPhase3WeightOptimization:
    """Test ensemble weight optimization using stored signals"""
    
    def test_signal_replay_weight_optimization(self):
        """Test that weight optimization works with stored signals"""
        # Mock stored signals from Phase 1
        stored_signals = [
            {
                'timestamp': datetime(2023, 1, 1),
                'regime': 'TRENDING_UP',
                'component_signals': {
                    'MA_strategy': 0.8,
                    'RSI_strategy': 0.3
                },
                'price': 100
            },
            {
                'timestamp': datetime(2023, 1, 2),
                'regime': 'TRENDING_UP',
                'component_signals': {
                    'MA_strategy': -0.5,
                    'RSI_strategy': 0.7
                },
                'price': 102
            }
        ]
        
        # Test different weight combinations
        weight_optimizer = WeightOptimizer()
        
        # Test weight set 1
        weights1 = {'MA_strategy': 0.7, 'RSI_strategy': 0.3}
        result1 = weight_optimizer.replay_with_weights(stored_signals, weights1)
        
        # Test weight set 2
        weights2 = {'MA_strategy': 0.3, 'RSI_strategy': 0.7}
        result2 = weight_optimizer.replay_with_weights(stored_signals, weights2)
        
        # Results should differ based on weights
        assert result1['total_return'] != result2['total_return']
        assert result1['signal_count'] != result2['signal_count']
    
    def test_regime_specific_weight_optimization(self):
        """Test that weights are optimized separately per regime"""
        # Setup genetic optimizer for weights
        genetic_optimizer = GeneticWeightOptimizer(
            population_size=20,
            generations=5
        )
        
        # Optimize for specific regime
        regime_signals = filter_signals_by_regime(stored_signals, 'TRENDING_UP')
        optimal_weights = genetic_optimizer.optimize(regime_signals)
        
        # Verify weights sum to 1 (or close)
        assert abs(sum(optimal_weights.values()) - 1.0) < 0.01
        
        # Verify different regimes get different weights
        volatile_signals = filter_signals_by_regime(stored_signals, 'HIGH_VOLATILITY')
        volatile_weights = genetic_optimizer.optimize(volatile_signals)
        
        assert optimal_weights != volatile_weights
    
    def test_classifier_specific_weight_optimization(self):
        """Test weight optimization for each classifier environment separately"""
        # Mock signals from different classifiers
        hmm_signals = [
            {
                'timestamp': datetime(2023, 1, 1),
                'regime': 'BULL',
                'classifier': 'HMM',
                'component_signals': {'MA': 0.8, 'RSI': 0.3}
            }
        ]
        
        vol_signals = [
            {
                'timestamp': datetime(2023, 1, 1),
                'regime': 'HIGH_VOL',
                'classifier': 'Volatility',
                'component_signals': {'MA': 0.2, 'RSI': 0.9}
            }
        ]
        
        # Optimize weights for each classifier
        optimizer = WeightOptimizer()
        hmm_weights = optimizer.optimize_for_classifier(hmm_signals, 'HMM')
        vol_weights = optimizer.optimize_for_classifier(vol_signals, 'Volatility')
        
        # Weights should be different for different classifiers
        assert hmm_weights != vol_weights
```

### Phase 4: Test Validation

```python
# test/test_phase4_validation.py
class TestPhase4Validation:
    """Test final validation on test set"""
    
    def test_adaptive_strategy_parameter_switching(self):
        """Test that adaptive strategy correctly switches parameters on regime change"""
        # Create adaptive strategy with regime-specific parameters
        regime_params = {
            'TRENDING_UP': {'fast': 10, 'slow': 20},
            'HIGH_VOLATILITY': {'fast': 5, 'slow': 15},
            'TRENDING_DOWN': {'fast': 15, 'slow': 30}
        }
        
        adaptive_strategy = RegimeAdaptiveStrategy(regime_params)
        parameter_history = []
        
        # Track parameter changes
        def on_parameters_changed(params):
            parameter_history.append({
                'timestamp': datetime.now(),
                'params': params.copy()
            })
        
        adaptive_strategy.on_parameters_changed = on_parameters_changed
        
        # Process test data with regime changes
        test_data = MarketDataFixtures.create_regime_aware_data()
        regime_detector = RegimeDetector()
        
        for _, bar in test_data.iterrows():
            # Detect regime
            regime = regime_detector.classify(bar)
            
            # Notify strategy of regime
            adaptive_strategy.on_regime_change(regime)
            
            # Process bar
            adaptive_strategy.on_bar(bar)
        
        # Verify parameters changed with regimes
        assert len(parameter_history) >= 2  # At least 2 regime changes
        
        # Verify correct parameters were applied
        for change in parameter_history:
            active_regime = adaptive_strategy.current_regime
            expected_params = regime_params[active_regime]
            assert change['params'] == expected_params
    
    def test_regime_transition_handling_across_environments(self):
        """Test that regime transitions are handled independently in each environment"""
        # Create data with known regime transitions
        transition_data = MarketDataFixtures.create_multi_regime_transition_data()
        
        # Set up parallel environments
        indicator_hub = ContainerTestHelpers.create_test_indicator_container([
            {'type': 'SMA', 'period': 10}
        ])
        
        hmm_env = ContainerTestHelpers.create_regime_classifier_environment(
            'HMM', {}, [{'class': 'TestStrategy', 'params': {}}], indicator_hub
        )
        vol_env = ContainerTestHelpers.create_regime_classifier_environment(
            'Volatility', {}, [{'class': 'TestStrategy', 'params': {}}], indicator_hub
        )
        
        # Track regime transitions
        hmm_transitions = []
        vol_transitions = []
        
        prev_hmm_regime = None
        prev_vol_regime = None
        
        for _, bar in transition_data.iterrows():
            indicator_data = indicator_hub.process_bar(bar)
            
            hmm_result = hmm_env.process_indicator_event(indicator_data)
            vol_result = vol_env.process_indicator_event(indicator_data)
            
            # Track transitions
            if prev_hmm_regime and hmm_result['regime'] != prev_hmm_regime:
                hmm_transitions.append({
                    'from': prev_hmm_regime,
                    'to': hmm_result['regime'],
                    'timestamp': bar['timestamp']
                })
            
            if prev_vol_regime and vol_result['regime'] != prev_vol_regime:
                vol_transitions.append({
                    'from': prev_vol_regime,
                    'to': vol_result['regime'],
                    'timestamp': bar['timestamp']
                })
            
            prev_hmm_regime = hmm_result['regime']
            prev_vol_regime = vol_result['regime']
        
        # Verify transitions detected
        assert len(hmm_transitions) > 0
        assert len(vol_transitions) > 0
        
        # Transitions should be independent
        hmm_timestamps = [t['timestamp'] for t in hmm_transitions]
        vol_timestamps = [t['timestamp'] for t in vol_transitions]
        assert hmm_timestamps != vol_timestamps  # Different transition points
```

## Integration Testing

### Full Workflow Test

```python
# test/test_full_workflow_integration.py
class TestFullWorkflowIntegration:
    """Test complete multi-phase workflow"""
    
    def test_complete_optimization_workflow(self):
        """Test all phases working together end-to-end"""
        # Configuration for full workflow
        workflow_config = {
            'phases': {
                'grid_search': {
                    'parameter_space': {
                        'fast': [5, 10, 15],
                        'slow': [20, 30, 40]
                    },
                    'regime_classifiers': ['HMM', 'Volatility']
                },
                'regime_analysis': {
                    'min_trades': 5
                },
                'weight_optimization': {
                    'method': 'genetic',
                    'per_regime': True
                },
                'test_validation': {
                    'test_split': 0.2
                }
            }
        }
        
        # Create coordinator
        coordinator = Coordinator(workflow_config)
        
        # Load data
        full_data = MarketDataFixtures.create_regime_aware_data()
        train_data, test_data = train_test_split(full_data, test_size=0.2)
        
        # Execute workflow
        results = coordinator.execute()
        
        # Verify all phases completed
        assert 'grid_search' in results
        assert 'regime_analysis' in results
        assert 'weight_optimization' in results
        assert 'test_validation' in results
        
        # Verify parallel classifier results
        assert 'HMM' in results['grid_search']['classifier_results']
        assert 'Volatility' in results['grid_search']['classifier_results']
        
        # Verify final test results
        test_results = results['test_validation']
        assert test_results['sharpe_ratio'] > 0
        assert test_results['regime_switches_handled'] > 0
    
    def test_workflow_reproducibility(self):
        """Test that workflow produces identical results when repeated"""
        config = load_test_config()
        config['random_seed'] = 42
        
        # Run workflow twice
        coordinator1 = Coordinator(config)
        results1 = coordinator1.execute()
        
        coordinator2 = Coordinator(config)
        results2 = coordinator2.execute()
        
        # Results should be identical
        assert results1['grid_search']['best_params'] == results2['grid_search']['best_params']
        assert results1['test_validation']['sharpe_ratio'] == results2['test_validation']['sharpe_ratio']
    
    def test_parallel_container_cleanup(self):
        """Test that all containers are properly cleaned up after parallel execution"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run workflow with multiple classifiers
        config = {
            'phases': {
                'grid_search': {
                    'parameter_space': {
                        'fast': list(range(5, 25, 5)),  # 4 values
                        'slow': list(range(20, 60, 10))  # 4 values
                    },
                    'regime_classifiers': ['HMM', 'Volatility', 'ML']  # 3 classifiers
                }
            }
        }
        
        coordinator = Coordinator(config)
        results = coordinator.execute()
        
        # Verify cleanup
        assert len(coordinator.active_containers) == 0  # All containers destroyed
        
        # Force garbage collection
        del results
        del coordinator
        gc.collect()
        
        # Check memory returned to near baseline
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_leak = final_memory - baseline_memory
        
        # Should have minimal memory leak (< 50MB)
        assert memory_leak < 50
```

## Performance Testing

```python
# test/test_workflow_performance.py
class TestWorkflowPerformance:
    """Test performance characteristics of the workflow"""
    
    def test_shared_indicator_performance(self):
        """Verify performance gains from shared indicators"""
        import time
        
        # Traditional approach - each strategy calculates indicators
        start = time.time()
        traditional_results = run_traditional_optimization(n_strategies=100)
        traditional_time = time.time() - start
        
        # Shared indicator approach
        start = time.time()
        shared_results = run_shared_indicator_optimization(n_strategies=100)
        shared_time = time.time() - start
        
        # Should be significantly faster
        speedup = traditional_time / shared_time
        assert speedup > 10  # At least 10x faster
        
        # Results should be identical
        assert traditional_results == shared_results
    
    def test_memory_usage(self):
        """Test memory efficiency of container approach"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run optimization with 100 strategies
        coordinator = Coordinator(large_optimization_config)
        results = coordinator.execute()
        
        # Check memory after
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        
        # Should use less than 1GB for 100 strategies
        assert memory_used < 1024
    
    def test_parallel_classifier_performance(self):
        """Test performance of parallel regime classifier execution"""
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        # Sequential execution
        start = time.time()
        seq_results = []
        for classifier in ['HMM', 'Volatility', 'ML']:
            result = run_classifier_optimization(classifier)
            seq_results.append(result)
        sequential_time = time.time() - start
        
        # Parallel execution
        start = time.time()
        with ThreadPoolExecutor() as executor:
            parallel_results = list(executor.map(
                run_classifier_optimization,
                ['HMM', 'Volatility', 'ML']
            ))
        parallel_time = time.time() - start
        
        # Should be faster (near 3x for 3 classifiers)
        speedup = sequential_time / parallel_time
        assert speedup > 2.5
        
        # Results should be identical
        assert seq_results == parallel_results
```

## Test Utilities

```python
# test/utils/workflow_assertions.py
class WorkflowAssertions:
    """Custom assertions for workflow testing"""
    
    @staticmethod
    def assert_regime_consistency(results: Dict, expected_regimes: List[str]):
        """Assert that all expected regimes were detected and optimized"""
        regime_results = results['regime_analysis']
        for regime in expected_regimes:
            assert regime in regime_results
            assert 'best_params' in regime_results[regime]
            assert 'performance' in regime_results[regime]
    
    @staticmethod
    def assert_parameter_validity(params: Dict, constraints: Dict):
        """Assert that optimized parameters meet constraints"""
        for param, value in params.items():
            if param in constraints:
                constraint = constraints[param]
                if 'min' in constraint:
                    assert value >= constraint['min']
                if 'max' in constraint:
                    assert value <= constraint['max']
                if 'values' in constraint:
                    assert value in constraint['values']
    
    @staticmethod
    def assert_performance_improvement(baseline: Dict, optimized: Dict):
        """Assert that optimization improved performance"""
        baseline_sharpe = baseline.get('sharpe_ratio', 0)
        optimized_sharpe = optimized.get('sharpe_ratio', 0)
        assert optimized_sharpe > baseline_sharpe
    
    @staticmethod
    def assert_classifier_independence(results: Dict):
        """Assert that classifier results are independent"""
        classifier_results = results['grid_search']['classifier_results']
        
        # Each classifier should have different optimal parameters
        hmm_best = classifier_results['HMM']['best_params']
        vol_best = classifier_results['Volatility']['best_params']
        
        # May be same by chance, but regime performance should differ
        hmm_regime_perf = classifier_results['HMM']['regime_performance']
        vol_regime_perf = classifier_results['Volatility']['regime_performance']
        
        # Different regime definitions should lead to different performance profiles
        assert hmm_regime_perf.keys() != vol_regime_perf.keys()
    
    @staticmethod
    def assert_regime_transition_independence(hmm_transitions: List, vol_transitions: List):
        """Assert that regime transitions are independent between classifiers"""
        # Transition timestamps should be different
        hmm_times = [t['timestamp'] for t in hmm_transitions]
        vol_times = [t['timestamp'] for t in vol_transitions]
        
        # Some overlap is okay, but not complete overlap
        overlap = set(hmm_times) & set(vol_times)
        overlap_ratio = len(overlap) / max(len(hmm_times), len(vol_times))
        
        assert overlap_ratio < 0.5  # Less than 50% overlap
```

## Testing Best Practices

1. **Use Consistent Test Data**: The `MarketDataFixtures` ensure all tests use the same data patterns
2. **Test Each Phase in Isolation**: Verify each phase works before testing integration
3. **Test Parallel Classifier Independence**: Ensure regime environments don't contaminate each other
4. **Mock External Dependencies**: Use mocks for data sources, brokers, etc.
5. **Test Edge Cases**: Empty data, single regime, no trades, rapid regime transitions
6. **Performance Benchmarks**: Ensure optimizations actually improve performance
7. **Reproducibility Tests**: Verify same config produces same results
8. **Memory Leak Tests**: Ensure containers are properly cleaned up
9. **Cross-Classifier Validation**: Verify strategies work across different regime paradigms
10. **Transition Stress Testing**: Test handling of rapid regime changes

## Additional Test Scenarios

### Stress Testing

```python
def test_rapid_regime_transitions():
    """Test system behavior with rapid regime changes"""
    # Create data with regime change every 10 bars
    rapid_transition_data = create_rapid_transition_data()
    
    # Run through system
    # Verify no crashes, memory leaks, or state corruption
```

### Edge Cases

```python
def test_single_regime_optimization():
    """Test when entire dataset is single regime"""
    # Should still work but with warnings

def test_no_trades_generated():
    """Test when strategies generate no trades"""
    # Should handle gracefully with appropriate warnings

def test_classifier_disagreement():
    """Test when classifiers completely disagree on regimes"""
    # System should still function with independent results
```

This comprehensive testing strategy ensures your multi-phase workflow with parallel regime classifier environments is robust, efficient, and produces reliable results across all execution scenarios.
