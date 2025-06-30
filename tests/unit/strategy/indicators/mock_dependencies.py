"""
Mock dependencies for testing without pandas/numpy.
"""

import sys

class MockDataFrame:
    """Mock pandas DataFrame."""
    def __init__(self, *args, **kwargs):
        pass

class MockSeries:
    """Mock pandas Series."""
    def __init__(self, *args, **kwargs):
        pass

class MockPandas:
    """Mock pandas module."""
    DataFrame = MockDataFrame
    Series = MockSeries
    
    @staticmethod
    def concat(*args, **kwargs):
        return MockDataFrame()
    
    @staticmethod
    def merge(*args, **kwargs):
        return MockDataFrame()

class MockNumpy:
    """Mock numpy module."""
    @staticmethod
    def array(*args, **kwargs):
        return []
    
    @staticmethod
    def zeros(*args, **kwargs):
        return []
    
    @staticmethod
    def ones(*args, **kwargs):
        return []
    
    @staticmethod
    def mean(*args, **kwargs):
        return 0
    
    @staticmethod
    def std(*args, **kwargs):
        return 1
    
    float64 = float
    nan = float('nan')
    inf = float('inf')

# Install mocks
sys.modules['pandas'] = MockPandas()
sys.modules['pd'] = MockPandas()
sys.modules['numpy'] = MockNumpy()
sys.modules['np'] = MockNumpy()