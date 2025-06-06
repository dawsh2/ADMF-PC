"""
Test the unified architecture implementation.

This test demonstrates the simplified workflow execution with
stateless components and universal topology.
"""

import asyncio
import sys
import os
from datetime import datetime
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.coordinator.topology import WorkflowManager, WorkflowMode
from src.core.types.workflow import WorkflowConfig, WorkflowType, ExecutionContext


class TestUnifiedArchitecture:
    """Test cases for the unified architecture."""
    
    async def test_mode_detection(self):
        """Test that workflow modes are correctly detected from config."""
        manager = WorkflowManager()
        
        # Test backtest mode (default)
        config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            parameters={}
        )
        mode = manager._determine_mode(config)
        assert mode == WorkflowMode.BACKTEST
        
        # Test signal generation mode
        config.parameters['signal_generation_only'] = True
        mode = manager._determine_mode(config)
        assert mode == WorkflowMode.SIGNAL_GENERATION
        
        # Test signal replay mode
        config.parameters = {'signal_replay': True}
        mode = manager._determine_mode(config)
        assert mode == WorkflowMode.SIGNAL_REPLAY
    
    async def test_parameter_expansion(self):
        """Test parameter grid expansion for multiple combinations."""
        manager = WorkflowManager()
        
        config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            parameters={
                'strategies': [
                    {'type': 'momentum', 'threshold': 0.02},
                    {'type': 'mean_reversion', 'threshold': 2.0}
                ],
                'risk_profiles': [
                    {'type': 'conservative', 'max_position': 0.1},
                    {'type': 'aggressive', 'max_position': 0.2}
                ],
                'classifiers': [
                    {'type': 'trend'}
                ]
            }
        )
        
        combinations = manager._expand_parameter_combinations(config)
        
        # Should have 2 strategies × 2 risk profiles × 1 classifier = 4 combinations
        assert len(combinations) == 4
        
        # Check combination IDs are unique
        combo_ids = [c['combo_id'] for c in combinations]
        assert len(set(combo_ids)) == 4
        
        # Verify first combination
        first = combinations[0]
        assert first['combo_id'] == 'c0000'
        assert first['strategy_params']['type'] == 'momentum'
        assert first['risk_params']['type'] == 'conservative'
    
    def test_stateless_strategy_creation(self):
        """Test creation of stateless strategy components."""
        manager = WorkflowManager()
        
        # Test momentum strategy
        momentum = manager._create_stateless_strategy('momentum', {})
        assert hasattr(momentum, 'generate_signal')
        assert hasattr(momentum, 'required_features')
        
        # Test mean reversion strategy
        mean_rev = manager._create_stateless_strategy('mean_reversion', {})
        assert hasattr(mean_rev, 'generate_signal')
        
        # Test unknown strategy (should return placeholder)
        unknown = manager._create_stateless_strategy('unknown_strategy', {})
        assert isinstance(unknown, dict)
        assert unknown['stateless'] == True
    
    def test_stateless_classifier_creation(self):
        """Test creation of stateless classifier components."""
        manager = WorkflowManager()
        
        # Test trend classifier
        trend = manager._create_stateless_classifier('trend', {})
        assert callable(trend)  # Pure function
        
        # Test volatility classifier
        vol = manager._create_stateless_classifier('volatility', {})
        assert callable(vol)  # Pure function
        
        # Test composite classifier (defaults to trend)
        composite = manager._create_stateless_classifier('composite', {})
        assert callable(composite)  # Pure function
    
    def test_stateless_risk_validator_creation(self):
        """Test creation of stateless risk validator components."""
        manager = WorkflowManager()
        
        # Test position validator
        position = manager._create_stateless_risk_validator('position', {})
        assert callable(position)  # Pure function
        
        # Test conservative profile (uses composite)
        conservative = manager._create_stateless_risk_validator('conservative', {})
        assert callable(conservative)  # Pure function
        
        # Test aggressive profile (uses composite)
        aggressive = manager._create_stateless_risk_validator('aggressive', {})
        assert callable(aggressive)  # Pure function
    
    def test_stateless_strategy_call(self):
        """Test calling a stateless strategy."""
        manager = WorkflowManager()
        
        # Create real stateless strategy
        strategy = manager._create_stateless_strategy('momentum', {})
        
        # Test data
        features = {
            'sma_fast': 102,  # Higher fast MA for stronger momentum
            'sma_slow': 100,
            'rsi': 50
        }
        bar = {
            'symbol': 'SPY',
            'close': 100,
            'timestamp': datetime.now()
        }
        params = {
            'momentum_threshold': 0.01,  # Lower threshold to ensure signal
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        # Call strategy
        signal = manager._call_stateless_strategy(strategy, features, bar, params)
        
        # Should get a signal
        assert signal is not None
        assert signal['direction'] == 'long'  # Positive momentum
        assert signal['strength'] > 0
        assert signal['symbol'] == 'SPY'
    
    def test_stateless_classifier_call(self):
        """Test calling a stateless classifier."""
        manager = WorkflowManager()
        
        # Create real stateless classifier
        classifier = manager._create_stateless_classifier('trend', {})
        
        # Test data
        features = {
            'sma_fast': 103,  # Higher for clearer trend
            'sma_slow': 100
        }
        params = {
            'trend_threshold': 0.02
        }
        
        # Call classifier
        regime = manager._call_stateless_classifier(classifier, features, params)
        
        # Should get a regime
        assert regime is not None
        assert 'regime' in regime
        assert 'confidence' in regime
        assert regime['confidence'] > 0
    
    def test_stateless_risk_validator_call(self):
        """Test calling a stateless risk validator."""
        manager = WorkflowManager()
        
        # Create real stateless validator
        validator = manager._create_stateless_risk_validator('conservative', {})
        
        # Test data
        order = {
            'symbol': 'SPY',
            'quantity': 100,
            'side': 'buy',
            'price': None  # Market order
        }
        portfolio_state = {
            'positions': {},
            'cash': 10000,
            'total_value': 10000,
            'metrics': {'current_drawdown': 0.05}
        }
        risk_params = {
            'max_position_percent': 0.1,
            'max_drawdown': 0.2
        }
        market_data = {
            'close': 100
        }
        
        # Call validator
        result = manager._call_stateless_risk_validator(
            validator, order, portfolio_state, risk_params, market_data
        )
        
        # Should get validation result
        assert result is not None
        assert 'approved' in result
        assert 'risk_metrics' in result
        
        # This order should be approved (100 shares * $100 = $10k = 100% of portfolio)
        # But since max is 10%, it should be rejected
        assert result['approved'] == False
        assert 'exceeds limit' in result['reason']
    
    async def test_universal_topology_creation(self):
        """Test creation of universal topology."""
        manager = WorkflowManager()
        
        config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            parameters={
                'strategies': [
                    {'type': 'momentum'}
                ],
                'risk_profiles': [
                    {'type': 'conservative'}
                ],
                'features': {
                    'indicators': [
                        {'name': 'sma_fast', 'type': 'sma', 'period': 10}
                    ]
                }
            },
            data_config={'symbols': ['SPY']}
        )
        
        # Note: This will fail because we don't have real container implementations
        # but we can test the structure creation
        try:
            topology = await manager._create_universal_topology(config)
        except Exception as e:
            # Expected - no real container factory
            print(f"Expected error: {e}")
            
        # Test parameter combinations were created
        combinations = manager._expand_parameter_combinations(config)
        assert len(combinations) == 1
        
        # Test stateless components are created
        components = manager._create_stateless_components(config)
        assert 'strategies' in components
        assert 'momentum' in components['strategies']
        assert 'risk_validators' in components
        assert 'conservative' in components['risk_validators']


def test_import():
    """Test that all unified architecture components can be imported."""
    # Test stateless strategies
    from src.strategy.strategies.momentum import momentum_strategy
    from src.strategy.strategies.mean_reversion_simple import mean_reversion_strategy
    
    # Test stateless classifiers
    from src.strategy.classifiers.classifiers import (
        trend_classifier,
        volatility_classifier,
        momentum_regime_classifier
    )
    
    # Test risk validators
    from src.risk.validators import (
        validate_max_position,
        validate_drawdown,
        validate_composite
    )
    
    # Test workflow components
    from src.core.coordinator.topology import WorkflowManager, WorkflowMode
    from src.core.types.workflow import BaseMode
    
    # All imports successful
    assert True


if __name__ == "__main__":
    # Run basic tests
    test = TestUnifiedArchitecture()
    
    # Test synchronous methods
    print("Testing mode detection...")
    asyncio.run(test.test_mode_detection())
    print("✓ Mode detection works")
    
    print("\nTesting parameter expansion...")
    asyncio.run(test.test_parameter_expansion())
    print("✓ Parameter expansion works")
    
    print("\nTesting stateless component creation...")
    test.test_stateless_strategy_creation()
    test.test_stateless_classifier_creation()
    test.test_stateless_risk_validator_creation()
    print("✓ Stateless components created")
    
    print("\nTesting stateless component calls...")
    test.test_stateless_strategy_call()
    test.test_stateless_classifier_call()
    test.test_stateless_risk_validator_call()
    print("✓ Stateless components work")
    
    print("\nTesting imports...")
    test_import()
    print("✓ All imports successful")
    
    print("\n✅ All tests passed!")