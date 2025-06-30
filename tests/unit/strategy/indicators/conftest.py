"""
Pytest fixtures for indicator strategy tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    base_price = 100
    trend = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 2, 100)
    prices = base_price + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

@pytest.fixture
def trending_data():
    """Create trending market data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 * (1.002 ** np.arange(100))  # 0.2% daily growth
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })

@pytest.fixture
def ranging_data():
    """Create ranging market data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + 5 * np.sin(np.arange(100) * 0.1)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, 100)
    })
