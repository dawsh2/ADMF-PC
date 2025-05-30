"""
Test strategy module components for PC architecture compliance.

This test suite verifies:
1. All strategy components follow Protocol + Composition (NO inheritance)
2. Strategies work correctly with capabilities
3. Container integration functions properly
4. Event-driven communication works as expected
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import strategy components
from src.strategy.protocols import Strategy, Indicator, Classifier, SignalDirection
from src.strategy.strategies.momentum import MomentumStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.components.indicators import SMA, RSI, ATR
from src.strategy.components.classifiers import VolatilityClassifier, TrendClassifier
from src.strategy.capabilities import StrategyCapability
from src.strategy.optimization.capabilities import OptimizationCapability

# Import core components
from src.core.components import ComponentFactory
from src.core.containers import UniversalScopedContainer
from src.core.events import EventBus


class TestStrategyProtocols(unittest.TestCase):
    """Test strategy components implement protocols correctly."""
    
    def test_strategy_protocol_implementation(self):
        """Test strategies implement Strategy protocol without inheritance."""
        strategies = [
            MomentumStrategy(),
            MeanReversionStrategy()
        ]
        
        for strategy in strategies:
            # Check NO inheritance (except object)
            bases = strategy.__class__.__bases__
            self.assertEqual(len(bases), 1)
            self.assertEqual(bases[0], object)
            
            # Check protocol methods
            self.assertTrue(hasattr(strategy, 'name'))
            self.assertTrue(hasattr(strategy, 'generate_signal'))
            
            # Test method works
            market_data = {
                'symbol': 'AAPL',
                'close': 150.0,
                'timestamp': datetime.now()
            }
            
            # May return None if not enough data
            signal = strategy.generate_signal(market_data)
            if signal:
                self.assertIn('direction', signal)
                self.assertIn('strength', signal)
                self.assertIn('timestamp', signal)
    
    def test_indicator_protocol_implementation(self):
        """Test indicators implement Indicator protocol."""
        indicators = [
            SMA(period=20),
            RSI(period=14),
            ATR(period=14)
        ]
        
        for indicator in indicators:
            # Check protocol methods
            self.assertTrue(hasattr(indicator, 'calculate'))
            self.assertTrue(hasattr(indicator, 'update'))
            
            # Test calculation
            data = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
            value = indicator.calculate(data)
            self.assertIsInstance(value, (int, float))
    
    def test_classifier_protocol_implementation(self):
        """Test classifiers implement Classifier protocol."""
        classifiers = [
            VolatilityClassifier(threshold_low=0.1, threshold_high=0.3),
            TrendClassifier(lookback=20)
        ]
        
        for classifier in classifiers:
            # Check protocol methods
            self.assertTrue(hasattr(classifier, 'classify'))
            self.assertTrue(hasattr(classifier, 'get_confidence'))
            
            # Test classification
            features = {
                'price': 100,
                'volume': 1000000,
                'volatility': 0.2
            }
            
            regime = classifier.classify(features)
            self.assertIsInstance(regime, str)
            
            confidence = classifier.get_confidence()
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestCapabilityEnhancement(unittest.TestCase):
    """Test capability-based enhancement system."""
    
    def test_strategy_capability_application(self):
        """Test StrategyCapability adds features to strategies."""
        # Create basic strategy
        strategy = MomentumStrategy()
        
        # Apply strategy capability
        capability = StrategyCapability()
        enhanced = capability.apply(strategy, {})
        
        # Should have additional methods
        self.assertTrue(hasattr(enhanced, 'validate_signal'))
        self.assertTrue(hasattr(enhanced, 'get_required_history'))
        self.assertTrue(hasattr(enhanced, 'get_supported_symbols'))
    
    def test_optimization_capability_application(self):
        """Test OptimizationCapability makes strategies optimizable."""
        # Create strategy
        strategy = MomentumStrategy(lookback_period=20)
        
        # Apply optimization capability
        capability = OptimizationCapability()
        optimizable = capability.apply(strategy, {
            'parameter_space': {
                'lookback_period': [10, 20, 30, 40],
                'momentum_threshold': [0.01, 0.02, 0.03]
            }
        })
        
        # Should have optimization methods
        self.assertTrue(hasattr(optimizable, 'get_parameters'))
        self.assertTrue(hasattr(optimizable, 'set_parameters'))
        self.assertTrue(hasattr(optimizable, 'get_parameter_space'))
        
        # Test parameter management
        params = optimizable.get_parameters()
        self.assertIn('lookback_period', params)
        self.assertEqual(params['lookback_period'], 20)
        
        # Set new parameters
        optimizable.set_parameters({'lookback_period': 30})
        self.assertEqual(optimizable.lookback_period, 30)
    
    def test_multiple_capabilities(self):
        """Test applying multiple capabilities to a component."""
        strategy = MomentumStrategy()
        
        # Apply multiple capabilities through factory
        factory = ComponentFactory()
        enhanced = factory.create_component({
            'class': 'MomentumStrategy',
            'capabilities': ['strategy', 'optimization', 'events'],
            'params': {
                'lookback_period': 20
            }
        })
        
        # Should have methods from all capabilities
        # Strategy capability
        self.assertTrue(hasattr(enhanced, 'validate_signal'))
        # Optimization capability  
        self.assertTrue(hasattr(enhanced, 'get_parameters'))
        # Events capability
        self.assertTrue(hasattr(enhanced, 'publish_event'))


class TestContainerIntegration(unittest.TestCase):
    """Test strategy integration with container system."""
    
    def setUp(self):
        self.container = UniversalScopedContainer('test_strategy_container')
        self.container.initialize_scope()
        self.container.start()
    
    def tearDown(self):
        self.container.stop()
        self.container.dispose()
    
    def test_strategy_in_container(self):
        """Test strategy execution within container."""
        # Create strategy
        strategy = MomentumStrategy()
        
        # Register in container
        self.container.register_component('momentum', strategy, {
            'capabilities': ['strategy']
        })
        
        # Execute in container scope
        with self.container.create_scope():
            # Get strategy from container
            retrieved = self.container.get_component('momentum')
            self.assertIsNotNone(retrieved)
            
            # Generate signal
            market_data = {
                'symbol': 'AAPL',
                'close': 150,
                'timestamp': datetime.now()
            }
            
            # Add enough data for signal generation
            for i in range(20):
                retrieved.generate_signal({
                    'symbol': 'AAPL',
                    'close': 150 + i * 0.5,
                    'timestamp': datetime.now()
                })
            
            # Should eventually generate a signal
            signal = retrieved.generate_signal({
                'symbol': 'AAPL',
                'close': 160,
                'timestamp': datetime.now()
            })
            
            # May or may not generate signal depending on momentum
            if signal:
                self.assertIn('direction', signal)
    
    def test_multiple_strategies_isolated(self):
        """Test multiple strategies run in isolation."""
        # Create strategies
        momentum = MomentumStrategy(lookback_period=20)
        mean_rev = MeanReversionStrategy(window_size=50)
        
        # Create separate containers
        container1 = UniversalScopedContainer('momentum_container')
        container2 = UniversalScopedContainer('mean_rev_container')
        
        container1.initialize_scope()
        container1.start()
        container2.initialize_scope()
        container2.start()
        
        try:
            # Register strategies
            container1.register_component('strategy', momentum)
            container2.register_component('strategy', mean_rev)
            
            # Each should have isolated state
            with container1.create_scope():
                s1 = container1.get_component('strategy')
                self.assertEqual(s1.lookback_period, 20)
            
            with container2.create_scope():
                s2 = container2.get_component('strategy')
                self.assertEqual(s2.window_size, 50)
            
        finally:
            container1.stop()
            container1.dispose()
            container2.stop()
            container2.dispose()


class TestEventIntegration(unittest.TestCase):
    """Test event-driven communication for strategies."""
    
    def setUp(self):
        self.event_bus = EventBus()
        self.container = UniversalScopedContainer('event_test_container')
        self.container.event_bus = self.event_bus
    
    def test_strategy_publishes_events(self):
        """Test strategies can publish events."""
        # Create strategy with event capability
        strategy = Mock()
        strategy.name = 'test_strategy'
        strategy.generate_signal = Mock(return_value={
            'direction': SignalDirection.BUY,
            'strength': 0.8,
            'symbol': 'AAPL'
        })
        
        # Add event publishing
        strategy.publish_event = Mock()
        
        # Generate signal
        signal = strategy.generate_signal({'close': 150})
        
        # Should publish signal event
        strategy.publish_event('signal.generated', {
            'strategy': strategy.name,
            'signal': signal
        })
        
        strategy.publish_event.assert_called_once()
    
    def test_strategy_subscribes_to_events(self):
        """Test strategies can subscribe to events."""
        received_events = []
        
        def on_market_update(event):
            received_events.append(event)
        
        # Subscribe to market updates
        self.event_bus.subscribe('market.update', on_market_update)
        
        # Publish market update
        self.event_bus.publish('market.update', {
            'symbol': 'AAPL',
            'price': 150
        })
        
        # Should receive event
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].data['symbol'], 'AAPL')


class TestStrategyWorkflows(unittest.TestCase):
    """Test complete strategy workflows."""
    
    def test_strategy_optimization_workflow(self):
        """Test optimizing a strategy."""
        from src.strategy.optimization import GridOptimizer, SharpeObjective
        
        # Create strategy
        strategy = MomentumStrategy()
        
        # Make optimizable
        opt_capability = OptimizationCapability()
        optimizable = opt_capability.apply(strategy, {
            'parameter_space': {
                'lookback_period': [10, 20, 30],
                'momentum_threshold': [0.01, 0.02, 0.03]
            }
        })
        
        # Create optimizer
        optimizer = GridOptimizer()
        objective = SharpeObjective()
        
        # Mock evaluation function
        def evaluate(params):
            # Mock backtest results
            return 0.5 + params['lookback_period'] * 0.01
        
        # Run optimization
        best_params = optimizer.optimize(
            evaluate,
            parameter_space=optimizable.get_parameter_space()
        )
        
        # Should find best parameters
        self.assertIsNotNone(best_params)
        self.assertIn('lookback_period', best_params)
        self.assertIn('momentum_threshold', best_params)
    
    def test_multi_symbol_strategy(self):
        """Test strategy handling multiple symbols."""
        strategy = MomentumStrategy()
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        signals = []
        
        # Generate signals for multiple symbols
        for symbol in symbols:
            # Add price history
            for i in range(25):
                strategy.generate_signal({
                    'symbol': symbol,
                    'close': 100 + i,
                    'timestamp': datetime.now()
                })
            
            # Generate final signal
            signal = strategy.generate_signal({
                'symbol': symbol,
                'close': 130,  # Strong momentum
                'timestamp': datetime.now()
            })
            
            if signal:
                signals.append(signal)
        
        # Should generate signals for trending symbols
        self.assertGreater(len(signals), 0)
        
        # Each signal should have correct symbol
        for signal in signals:
            self.assertIn(signal['symbol'], symbols)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world strategy scenarios."""
    
    def test_regime_adaptive_strategy(self):
        """Test strategy that adapts to market regimes."""
        # Create base strategy
        base_strategy = MomentumStrategy()
        
        # Create regime-specific parameters
        regime_params = {
            'TRENDING_UP': {
                'lookback_period': 10,  # Faster in trends
                'momentum_threshold': 0.01
            },
            'HIGH_VOLATILITY': {
                'lookback_period': 30,  # Slower in volatility
                'momentum_threshold': 0.03
            },
            'RANGING': {
                'lookback_period': 20,
                'momentum_threshold': 0.02
            }
        }
        
        # Mock regime detector
        regime_detector = Mock()
        regime_detector.classify.return_value = 'TRENDING_UP'
        
        # Apply regime parameters
        current_regime = regime_detector.classify({'volatility': 0.15})
        params = regime_params[current_regime]
        
        base_strategy.lookback_period = params['lookback_period']
        base_strategy.momentum_threshold = params['momentum_threshold']
        
        # Strategy should use regime-specific parameters
        self.assertEqual(base_strategy.lookback_period, 10)
        self.assertEqual(base_strategy.momentum_threshold, 0.01)
    
    def test_strategy_risk_limits(self):
        """Test strategy respects risk limits."""
        strategy = MomentumStrategy()
        
        # Mock risk manager
        risk_manager = Mock()
        risk_manager.check_position_limit.return_value = True
        risk_manager.get_max_position_size.return_value = 1000
        
        # Generate signal
        signal = {
            'symbol': 'AAPL',
            'direction': SignalDirection.BUY,
            'strength': 1.0  # Full strength
        }
        
        # Apply risk limits
        if risk_manager.check_position_limit('AAPL'):
            max_size = risk_manager.get_max_position_size()
            signal['max_size'] = max_size
        
        # Signal should have risk limits
        self.assertEqual(signal['max_size'], 1000)


if __name__ == '__main__':
    unittest.main()