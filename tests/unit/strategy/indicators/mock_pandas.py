"""
Mock pandas module for testing without pandas dependency.
"""

import sys

class MockDataFrame:
    """Mock pandas DataFrame."""
    pass

class MockSeries:
    """Mock pandas Series."""
    pass

class MockPandas:
    """Mock pandas module."""
    DataFrame = MockDataFrame
    Series = MockSeries

# Install mock
sys.modules['pandas'] = MockPandas()
sys.modules['pd'] = MockPandas()