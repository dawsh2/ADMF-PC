"""
Demo of feature dependency validation preventing silent strategy failures.

This example shows:
1. The old behavior where strategies silently fail
2. The new behavior where missing features raise clear errors
"""

import logging
from typing import Dict, Any, Optional

from src.core.components.discovery import strategy
from src.strategy.validation import FeatureDependencyError, get_feature_validator

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Old-style strategy that silently fails
def old_sma_crossover_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Old strategy that silently returns None when features are missing."""
    
    # Get features - returns None if missing
    fast_sma = features.get('sma_10')
    slow_sma = features.get('sma_30')
    
    # Silent failure - just returns None
    if fast_sma is None or slow_sma is None:
        logger.warning("Missing SMA features, returning no signal")
        return None
    
    # Generate signal
    if fast_sma > slow_sma:
        return {
            'symbol': bar['symbol'],
            'direction': 'long',
            'value': 1.0,
            'reason': 'SMA crossover'
        }
    else:
        return None


# New-style strategy with validation
@strategy(
    features=['sma_10', 'sma_30'],
    validate_features=True
)
def new_sma_crossover_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """New strategy that validates features before execution."""
    
    # Can safely access features - validation ensures they exist
    fast_sma = features['sma_10']
    slow_sma = features['sma_30']
    
    # Generate signal
    if fast_sma > slow_sma:
        return {
            'symbol': bar['symbol'],
            'direction': 'long',
            'value': 1.0,
            'reason': 'SMA crossover'
        }
    else:
        return None


# Strategy with detailed feature requirements
@strategy(
    feature_config={
        'rsi': {'params': ['rsi_period'], 'default': 14},
        'sma': {'params': ['sma_period'], 'default': 20},
        'volume_sma': {'params': ['volume_period'], 'default': 20}
    }
)
def advanced_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Advanced strategy with multiple feature dependencies."""
    
    # All features are guaranteed to exist
    rsi = features['rsi']
    sma = features['sma']
    volume_sma = features['volume_sma']
    current_volume = bar.get('volume', 0)
    
    # Complex signal logic
    if rsi < 30 and current_volume > volume_sma * 1.5:
        return {
            'symbol': bar['symbol'],
            'direction': 'long',
            'value': 1.0,
            'reason': f'Oversold with high volume (RSI={rsi:.1f})'
        }
    elif rsi > 70:
        return {
            'symbol': bar['symbol'],
            'direction': 'short',
            'value': -1.0,
            'reason': f'Overbought (RSI={rsi:.1f})'
        }
    
    return None


def main():
    """Demonstrate feature validation."""
    
    # Sample bar data
    bar = {
        'symbol': 'SPY',
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000000
    }
    
    # Test scenario 1: Complete features
    print("\n=== Scenario 1: All features available ===")
    
    complete_features = {
        'sma_10': 100.2,
        'sma_30': 99.8,
        'rsi': 45.0,
        'sma': 100.0,
        'volume_sma': 800000
    }
    
    # Old strategy works
    signal = old_sma_crossover_strategy(complete_features, bar, {})
    print(f"Old strategy signal: {signal}")
    
    # New strategy works
    signal = new_sma_crossover_strategy(complete_features, bar, {})
    print(f"New strategy signal: {signal}")
    
    # Advanced strategy works
    signal = advanced_momentum_strategy(complete_features, bar, {})
    print(f"Advanced strategy signal: {signal}")
    
    # Test scenario 2: Missing features
    print("\n=== Scenario 2: Missing SMA features ===")
    
    incomplete_features = {
        'rsi': 45.0,
        'volume_sma': 800000
        # Missing sma_10, sma_30, sma
    }
    
    # Old strategy silently fails
    print("\nOld strategy behavior (silent failure):")
    signal = old_sma_crossover_strategy(incomplete_features, bar, {})
    print(f"Old strategy returned: {signal} (no error, just None)")
    
    # New strategy raises clear error
    print("\nNew strategy behavior (explicit error):")
    try:
        signal = new_sma_crossover_strategy(incomplete_features, bar, {})
        print(f"New strategy signal: {signal}")
    except FeatureDependencyError as e:
        print(f"ERROR: {e}")
        print(f"Missing features: {e.missing_features}")
        print(f"Strategy name: {e.strategy_name}")
    
    # Advanced strategy also fails clearly
    print("\nAdvanced strategy behavior:")
    try:
        signal = advanced_momentum_strategy(incomplete_features, bar, {})
        print(f"Advanced strategy signal: {signal}")
    except FeatureDependencyError as e:
        print(f"ERROR: {e}")
    
    # Test scenario 3: Partial features
    print("\n=== Scenario 3: Some features present ===")
    
    partial_features = {
        'sma_10': 100.2,
        # Missing sma_30
        'rsi': 45.0,
        'sma': 100.0,
        'volume_sma': 800000
    }
    
    print("\nNew strategy with partial features:")
    try:
        signal = new_sma_crossover_strategy(partial_features, bar, {})
        print(f"Signal: {signal}")
    except FeatureDependencyError as e:
        print(f"ERROR: {e}")
        print(f"Strategy '{e.strategy_name}' is missing: {e.missing_features}")
    
    # Show validation statistics
    print("\n=== Validation Statistics ===")
    validator = get_feature_validator()
    stats = validator.get_stats()
    print(f"Total validations performed: {stats['validations_performed']}")
    print(f"Total failures: {stats['failures']}")
    print(f"Total missing features: {stats['missing_features_total']}")
    
    # Demonstrate manual validation
    print("\n=== Manual Feature Validation ===")
    
    required_features = ['sma', 'ema', 'rsi', 'macd']
    available_features = {'sma': 100.0, 'rsi': 45.0}  # Missing ema and macd
    
    try:
        validator.validate_features(
            available_features,
            required_features,
            'manual_test_strategy'
        )
        print("Validation passed!")
    except FeatureDependencyError as e:
        print(f"Validation failed: {e}")
        print(f"You need to add these features: {e.missing_features}")


if __name__ == '__main__':
    main()