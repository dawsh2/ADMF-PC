#!/usr/bin/env python3
"""
Test and benchmark incremental features vs pandas-based features.

Demonstrates the performance improvement from O(n) to O(1) updates.
"""

import time
import random
from src.strategy.components.features.hub import create_feature_hub


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


def benchmark_feature_hub(use_incremental: bool, num_bars: int = 1000):
    """Benchmark feature hub performance."""
    print(f"\nBenchmarking {'INCREMENTAL' if use_incremental else 'PANDAS'} mode with {num_bars} bars...")
    
    # Configure features
    feature_configs = {
        "sma_10": {"feature": "sma", "period": 10},
        "sma_20": {"feature": "sma", "period": 20},
        "sma_50": {"feature": "sma", "period": 50},
        "ema_10": {"feature": "ema", "period": 10},
        "ema_20": {"feature": "ema", "period": 20},
        "rsi": {"feature": "rsi", "period": 14},
        "macd": {"feature": "macd", "fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"feature": "bollinger_bands", "period": 20, "std_dev": 2.0},
        "atr": {"feature": "atr", "period": 14},
        "stochastic": {"feature": "stochastic", "k_period": 14, "d_period": 3}
    }
    
    # Create feature hub
    hub = create_feature_hub(
        symbols=["TEST"],
        feature_configs=feature_configs,
        use_incremental=use_incremental
    )
    
    # Warmup phase
    print("Warming up...")
    base_price = 100.0
    for i in range(100):
        bar = generate_random_bar(base_price)
        hub.update_bar("TEST", bar)
        base_price = bar["close"]
    
    # Benchmark phase
    print("Benchmarking...")
    update_times = []
    
    for i in range(num_bars):
        bar = generate_random_bar(base_price)
        
        start_time = time.perf_counter()
        hub.update_bar("TEST", bar)
        update_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        update_times.append(update_time)
        base_price = bar["close"]
        
        # Print progress every 100 bars
        if (i + 1) % 100 == 0:
            avg_time = sum(update_times[-100:]) / 100
            print(f"  Bars {i+1}: avg update time = {avg_time:.2f}ms")
    
    # Calculate statistics
    avg_update_time = sum(update_times) / len(update_times)
    max_update_time = max(update_times)
    min_update_time = min(update_times)
    
    # Get final features
    features = hub.get_features("TEST")
    
    print(f"\nResults for {'INCREMENTAL' if use_incremental else 'PANDAS'} mode:")
    print(f"  Average update time: {avg_update_time:.2f}ms")
    print(f"  Min update time: {min_update_time:.2f}ms")
    print(f"  Max update time: {max_update_time:.2f}ms")
    print(f"  Features computed: {len(features)}")
    print(f"  Sample features: {list(features.keys())[:5]}...")
    
    return avg_update_time


def main():
    """Run the benchmark comparison."""
    print("=" * 60)
    print("Feature Hub Performance Comparison")
    print("=" * 60)
    
    # Test with 1000 bars
    pandas_time = benchmark_feature_hub(use_incremental=False, num_bars=1000)
    incremental_time = benchmark_feature_hub(use_incremental=True, num_bars=1000)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pandas-based average: {pandas_time:.2f}ms per bar")
    print(f"Incremental average: {incremental_time:.2f}ms per bar")
    print(f"Speedup: {pandas_time / incremental_time:.1f}x")
    print(f"Time saved per bar: {pandas_time - incremental_time:.2f}ms")
    
    # Extrapolate to real-world scenario
    bars_per_day = 390  # Regular trading hours
    print(f"\nFor a full trading day ({bars_per_day} bars):")
    print(f"  Pandas total time: {pandas_time * bars_per_day / 1000:.1f}s")
    print(f"  Incremental total time: {incremental_time * bars_per_day / 1000:.1f}s")
    print(f"  Time saved: {(pandas_time - incremental_time) * bars_per_day / 1000:.1f}s")


if __name__ == "__main__":
    main()