"""Final debug for swing_pivot_bounce - check if features are being passed correctly"""
import pandas as pd
import numpy as np
from src.strategy.strategies.indicators.structure import swing_pivot_bounce
from src.strategy.components.features.indicators.structure import SupportResistance

# Create test data
np.random.seed(42)
bars = []
prices = np.random.normal(520, 2, 100)

for i, price in enumerate(prices):
    bar = {
        'symbol': 'SPY',
        'open': price,
        'high': price + np.random.uniform(0, 0.5),
        'low': price - np.random.uniform(0, 0.5),
        'close': price,
        'volume': np.random.randint(1000, 10000),
        'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
    }
    bars.append(bar)

# Create S/R feature manually
sr = SupportResistance(lookback=20, min_touches=2)

print("Testing support/resistance feature calculation...")
sr_values = []
for i, bar in enumerate(bars[:50]):
    value = sr.update(
        price=bar['close'],
        high=bar['high'],
        low=bar['low']
    )
    if value:
        sr_values.append(value)
        print(f"Bar {i}: S/R values = {value}")

# Now test strategy with correct feature keys
print("\n\nTesting strategy with manually constructed features...")
params = {'sr_period': 20}

for i in range(30, 50):
    bar = bars[i]
    
    # Build features dict as it would appear from FeatureHub
    features = {
        'support_resistance_20_resistance': sr_values[-1]['resistance'] if sr_values and 'resistance' in sr_values[-1] else None,
        'support_resistance_20_support': sr_values[-1]['support'] if sr_values and 'support' in sr_values[-1] else None
    }
    
    result = swing_pivot_bounce(features, bar, params)
    
    if result:
        print(f"\nBar {i}: Close={bar['close']:.2f}")
        print(f"  Features: {features}")
        print(f"  Strategy result: signal={result['signal_value']}, metadata={result.get('metadata', {})}")