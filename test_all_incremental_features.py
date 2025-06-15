#!/usr/bin/env python3
"""
Test all incremental feature types to ensure they work correctly.
"""

import random
from src.strategy.components.features.incremental import IncrementalFeatureHub


def generate_random_bar(base_price: float = 100.0) -> dict:
    """Generate a random OHLCV bar."""
    change = random.uniform(-2, 2)
    open_price = base_price + random.uniform(-1, 1)
    close_price = open_price + change
    high_price = max(open_price, close_price) + random.uniform(0, 1)
    low_price = min(open_price, close_price) - random.uniform(0, 1)
    volume = random.uniform(100000, 1000000)
    
    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume
    }


def test_all_features():
    """Test all incremental feature types."""
    print("Testing all incremental feature types...\n")
    
    # Configure all feature types
    feature_configs = {
        # Trend features
        "sma_20": {"type": "sma", "period": 20},
        "ema_20": {"type": "ema", "period": 20},
        "dema_20": {"type": "dema", "period": 20},
        "tema_20": {"type": "tema", "period": 20},
        
        # Oscillators
        "rsi": {"type": "rsi", "period": 14},
        "stochastic": {"type": "stochastic", "k_period": 14, "d_period": 3},
        "williams_r": {"type": "williams_r", "period": 14},
        "cci": {"type": "cci", "period": 20},
        "ultimate": {"type": "ultimate", "fast": 7, "medium": 14, "slow": 28},
        "stoch_rsi": {"type": "stochastic_rsi", "rsi_period": 14, "stoch_period": 14},
        
        # Momentum
        "macd": {"type": "macd", "fast": 12, "slow": 26, "signal": 9},
        "adx": {"type": "adx", "period": 14},
        "momentum": {"type": "momentum", "period": 10},
        "vortex": {"type": "vortex", "period": 14},
        "roc": {"type": "roc", "period": 10},
        
        # Volatility
        "atr": {"type": "atr", "period": 14},
        "bollinger": {"type": "bollinger", "period": 20, "std_dev": 2.0},
        "keltner": {"type": "keltner", "period": 20, "multiplier": 2.0},
        "donchian": {"type": "donchian", "period": 20},
        "volatility": {"type": "volatility", "period": 20},
        
        # Volume
        "volume": {"type": "volume"},
        "obv": {"type": "obv"},
        "mfi": {"type": "mfi", "period": 14},
        "cmf": {"type": "cmf", "period": 20},
        "ad": {"type": "ad"},
        "vwap": {"type": "vwap"},
        "volume_sma": {"type": "volume_sma", "period": 20},
        "volume_ratio": {"type": "volume_ratio", "period": 20},
        
        # Trend advanced
        "aroon": {"type": "aroon", "period": 25},
        "supertrend": {"type": "supertrend", "period": 10, "multiplier": 3.0},
        "psar": {"type": "psar", "initial_af": 0.02, "max_af": 0.2},
        
        # Complex
        "ichimoku": {"type": "ichimoku", "tenkan": 9, "kijun": 26},
        "pivot": {"type": "pivot_points"},
        "linreg": {"type": "linear_regression", "period": 20},
        
        # Price features
        "high_20": {"type": "high", "period": 20},
        "low_20": {"type": "low", "period": 20},
        "atr_sma": {"type": "atr_sma", "atr_period": 14, "sma_period": 20},
        "vol_sma": {"type": "volatility_sma", "vol_period": 20, "sma_period": 20},
        
        # Pattern features
        "sr": {"type": "support_resistance", "lookback": 50},
        "swing": {"type": "swing_points", "lookback": 5},
        "fib": {"type": "fibonacci_retracement", "lookback": 50},
    }
    
    # Create hub and configure
    hub = IncrementalFeatureHub()
    hub.configure_features(feature_configs)
    
    # Generate and process bars
    symbol = "TEST"
    num_bars = 100
    errors = []
    
    print(f"Processing {num_bars} bars for {len(feature_configs)} features...")
    
    for i in range(num_bars):
        bar = generate_random_bar()
        
        try:
            features = hub.update_bar(symbol, bar)
            
            # On last bar, show what features we got
            if i == num_bars - 1:
                print(f"\nFeatures computed after {num_bars} bars:")
                print(f"Total features available: {len(features)}")
                
                # Group features by type
                by_type = {}
                for name, value in sorted(features.items()):
                    base_name = name.split('_')[0]
                    if base_name not in by_type:
                        by_type[base_name] = []
                    by_type[base_name].append((name, value))
                
                # Show sample from each type
                print("\nSample values by feature type:")
                for feature_type, values in sorted(by_type.items()):
                    print(f"\n{feature_type.upper()}:")
                    for name, value in values[:3]:  # Show up to 3 per type
                        if isinstance(value, float):
                            print(f"  {name}: {value:.4f}")
                        else:
                            print(f"  {name}: {value}")
                    if len(values) > 3:
                        print(f"  ... and {len(values) - 3} more")
                        
        except Exception as e:
            error_msg = f"Error at bar {i}: {str(e)}"
            if error_msg not in errors:
                errors.append(error_msg)
    
    # Report any errors
    if errors:
        print(f"\n⚠️  ERRORS ENCOUNTERED:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✅ All {len(feature_configs)} feature types processed successfully!")
    
    # Check which features are ready
    ready_features = [name for name in feature_configs if hub.has_sufficient_data(symbol)]
    print(f"\nFeatures ready: {len(ready_features)}/{len(feature_configs)}")
    
    if len(ready_features) < len(feature_configs):
        not_ready = set(feature_configs.keys()) - set(ready_features)
        print(f"Features not ready yet: {sorted(not_ready)}")


if __name__ == "__main__":
    test_all_features()