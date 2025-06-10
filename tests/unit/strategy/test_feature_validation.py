"""
Test feature dependency validation for strategies and classifiers.

This test ensures that strategies fail properly when required features are missing
instead of silently returning no signals.
"""

import pytest
from typing import Dict, Any, Optional

from src.strategy.validation import (
    FeatureDependencyError,
    FeatureValidator,
    validate_strategy_features,
    create_validated_strategy,
    extract_required_features
)
from src.core.components.discovery import strategy, classifier


class TestFeatureValidation:
    """Test feature validation functionality."""
    
    def test_feature_validator_basic(self):
        """Test basic feature validation."""
        validator = FeatureValidator()
        
        # Test with all features present
        features = {'sma': 50.0, 'rsi': 30.0, 'volume': 1000}
        required = ['sma', 'rsi']
        
        # Should not raise
        validator.validate_features(features, required, 'test_strategy')
        
        # Test with missing features
        features = {'sma': 50.0}  # Missing RSI
        
        with pytest.raises(FeatureDependencyError) as exc_info:
            validator.validate_features(features, required, 'test_strategy')
        
        assert 'rsi' in str(exc_info.value)
        assert 'test_strategy' in str(exc_info.value)
    
    def test_feature_validator_with_none_values(self):
        """Test that None values are treated as missing."""
        validator = FeatureValidator()
        
        features = {'sma': 50.0, 'rsi': None}  # RSI is None
        required = ['sma', 'rsi']
        
        with pytest.raises(FeatureDependencyError) as exc_info:
            validator.validate_features(features, required, 'test_strategy')
        
        assert 'rsi' in str(exc_info.value)
    
    def test_strategy_decorator_with_validation(self):
        """Test @strategy decorator with automatic validation."""
        
        @strategy(
            features=['sma', 'rsi'],
            validate_features=True
        )
        def test_momentum(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # This should only execute if features are valid
            return {
                'symbol': bar.get('symbol'),
                'direction': 'long',
                'value': 1.0
            }
        
        # Test with valid features
        features = {'sma': 50.0, 'rsi': 30.0}
        bar = {'symbol': 'SPY', 'close': 100.0}
        params = {}
        
        signal = test_momentum(features, bar, params)
        assert signal is not None
        assert signal['direction'] == 'long'
        
        # Test with missing features
        features = {'sma': 50.0}  # Missing RSI
        
        with pytest.raises(FeatureDependencyError) as exc_info:
            test_momentum(features, bar, params)
        
        assert 'rsi' in str(exc_info.value)
        assert 'test_momentum' in str(exc_info.value)
    
    def test_strategy_decorator_feature_config(self):
        """Test @strategy decorator with feature_config."""
        
        @strategy(
            feature_config={
                'sma': {'params': ['period'], 'default': 20},
                'rsi': {'params': ['period'], 'default': 14}
            }
        )
        def config_based_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Features extracted from feature_config
            sma = features['sma']
            rsi = features['rsi']
            return {
                'symbol': bar.get('symbol'),
                'direction': 'long' if rsi < 30 else 'flat',
                'value': 1.0 if rsi < 30 else 0.0
            }
        
        # Check that required_features was set correctly
        assert hasattr(config_based_strategy, 'required_features')
        assert set(config_based_strategy.required_features) == {'sma', 'rsi'}
        
        # Test validation
        features = {'sma': 50.0}  # Missing RSI
        bar = {'symbol': 'SPY', 'close': 100.0}
        params = {}
        
        with pytest.raises(FeatureDependencyError):
            config_based_strategy(features, bar, params)
    
    def test_classifier_with_validation(self):
        """Test classifier with feature validation."""
        
        @classifier(
            regime_types=['bull', 'bear', 'sideways'],
            features=['sma', 'volume']
        )
        def trend_classifier(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
            sma = features['sma']
            volume = features['volume']
            
            # Simple regime detection
            if sma > params.get('sma_threshold', 50) and volume > params.get('volume_threshold', 1000000):
                regime = 'bull'
            elif sma < params.get('sma_threshold', 50):
                regime = 'bear'
            else:
                regime = 'sideways'
            
            return {
                'regime': regime,
                'confidence': 0.8
            }
        
        # Test with valid features
        features = {'sma': 55.0, 'volume': 1500000}
        params = {'sma_threshold': 50, 'volume_threshold': 1000000}
        
        result = trend_classifier(features, params)
        assert result['regime'] == 'bull'
        
        # Test with missing features
        features = {'sma': 55.0}  # Missing volume
        
        with pytest.raises(FeatureDependencyError) as exc_info:
            trend_classifier(features, params)
        
        assert 'volume' in str(exc_info.value)
    
    def test_create_validated_strategy(self):
        """Test creating a validated wrapper for existing strategies."""
        
        # Simple strategy without validation
        def simple_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            sma = features.get('sma', 0)
            return {'symbol': bar['symbol'], 'direction': 'long', 'value': 1.0} if sma > 50 else None
        
        # Create validated version
        validated = create_validated_strategy(
            simple_strategy,
            required_features=['sma', 'ema', 'rsi']
        )
        
        # Test with missing features
        features = {'sma': 55.0}  # Missing ema and rsi
        bar = {'symbol': 'SPY'}
        params = {}
        
        with pytest.raises(FeatureDependencyError) as exc_info:
            validated(features, bar, params)
        
        assert 'ema' in str(exc_info.value) or 'rsi' in str(exc_info.value)
    
    def test_extract_required_features(self):
        """Test extracting required features from decorated strategies."""
        
        @strategy(features=['sma', 'rsi', 'macd'])
        def decorated_strategy(features, bar, params):
            return None
        
        # Extract from decorated strategy
        required = extract_required_features(decorated_strategy)
        assert set(required) == {'sma', 'rsi', 'macd'}
        
        # Test with feature_config
        @strategy(
            feature_config={
                'ema': {'params': ['period']},
                'volume': {'params': []}
            }
        )
        def config_strategy(features, bar, params):
            return None
        
        required = extract_required_features(config_strategy)
        assert set(required) == {'ema', 'volume'}
    
    def test_validation_stats(self):
        """Test validation statistics tracking."""
        validator = FeatureValidator()
        
        # Reset stats
        initial_stats = validator.get_stats()
        
        # Perform some validations
        features = {'sma': 50.0, 'rsi': 30.0}
        validator.validate_features(features, ['sma', 'rsi'], 'test1')
        
        # Try with missing features
        try:
            validator.validate_features({'sma': 50.0}, ['sma', 'rsi', 'ema'], 'test2')
        except FeatureDependencyError:
            pass
        
        stats = validator.get_stats()
        assert stats['validations_performed'] >= initial_stats['validations_performed'] + 2
        assert stats['failures'] >= initial_stats['failures'] + 1
        assert stats['missing_features_total'] >= initial_stats['missing_features_total'] + 2  # rsi and ema
    
    def test_disable_validation(self):
        """Test that validation can be disabled."""
        
        @strategy(
            features=['sma', 'rsi'],
            validate_features=False  # Disable validation
        )
        def no_validation_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Should work even with missing features
            sma = features.get('sma', 0)
            rsi = features.get('rsi', 50)
            return {'symbol': bar['symbol'], 'direction': 'long', 'value': 1.0}
        
        # Call with missing features - should not raise
        features = {'sma': 50.0}  # Missing RSI
        bar = {'symbol': 'SPY'}
        params = {}
        
        signal = no_validation_strategy(features, bar, params)
        assert signal is not None  # Should execute without error