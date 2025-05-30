"""
Test configuration for ADMF-PC test suite.

Provides common test fixtures, utilities, and configuration.
"""

import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any


class TestConfig:
    """Common test configuration."""
    
    # Test data directory
    TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
    
    # Default test dates
    DEFAULT_START_DATE = datetime(2023, 1, 1)
    DEFAULT_END_DATE = datetime(2023, 12, 31)
    
    # Default test capital
    DEFAULT_CAPITAL = Decimal("100000")
    
    # Test symbols
    TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    # Market data settings
    DEFAULT_FREQUENCY = "1d"
    
    # Risk settings for tests
    DEFAULT_POSITION_SIZE_PCT = Decimal("0.02")  # 2%
    DEFAULT_MAX_EXPOSURE_PCT = Decimal("0.20")   # 20%
    DEFAULT_MAX_DRAWDOWN_PCT = Decimal("0.10")   # 10%
    
    # Execution settings
    DEFAULT_COMMISSION = Decimal("0.001")  # 0.1%
    DEFAULT_SLIPPAGE = Decimal("0.0005")   # 0.05%
    
    @classmethod
    def create_temp_dir(cls):
        """Create a temporary directory for test outputs."""
        return tempfile.mkdtemp(prefix="admfpc_test_")
    
    @classmethod
    def get_test_data_path(cls, filename):
        """Get path to test data file."""
        return os.path.join(cls.TEST_DATA_DIR, filename)


class MockMarketData:
    """Generate mock market data for testing."""
    
    @staticmethod
    def generate_price_series(
        start_price: float,
        num_days: int,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: int = None
    ) -> List[float]:
        """
        Generate a price series with random walk.
        
        Args:
            start_price: Initial price
            num_days: Number of days to generate
            volatility: Daily volatility (e.g., 0.02 for 2%)
            trend: Daily trend (e.g., 0.0001 for 0.01% daily drift)
            seed: Random seed for reproducibility
            
        Returns:
            List of prices
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        prices = [start_price]
        
        for _ in range(num_days - 1):
            # Random return with trend
            daily_return = random.gauss(trend, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        return prices
    
    @staticmethod
    def generate_ohlcv_data(
        symbol: str,
        start_date: datetime,
        prices: List[float],
        base_volume: int = 1000000
    ) -> Dict[datetime, Dict[str, float]]:
        """
        Generate OHLCV data from price series.
        
        Args:
            symbol: Symbol name
            start_date: Starting date
            prices: List of closing prices
            base_volume: Base daily volume
            
        Returns:
            Dict mapping dates to OHLCV data
        """
        import random
        
        data = {}
        current_date = start_date
        
        for i, close_price in enumerate(prices):
            # Generate OHLCV
            daily_range = close_price * 0.02  # 2% daily range
            
            high = close_price + random.uniform(0, daily_range/2)
            low = close_price - random.uniform(0, daily_range/2)
            
            # Open is previous close with gap
            if i > 0:
                gap = random.uniform(-0.005, 0.005) * prices[i-1]
                open_price = prices[i-1] + gap
            else:
                open_price = close_price
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Random volume
            volume = int(base_volume * random.uniform(0.7, 1.3))
            
            data[current_date] = {
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            }
            
            current_date += timedelta(days=1)
        
        return data
    
    @staticmethod
    def create_test_market_snapshot(
        symbols: List[str],
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """
        Create a market data snapshot for testing.
        
        Args:
            symbols: List of symbols
            timestamp: Snapshot timestamp
            
        Returns:
            Market data dict
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        market_data = {
            'timestamp': timestamp,
            'prices': {}
        }
        
        # Define some test prices
        test_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 380.0,
            'AMZN': 3500.0,
            'TSLA': 1000.0,
            'SPY': 450.0,
            'QQQ': 370.0
        }
        
        for symbol in symbols:
            if symbol in test_prices:
                price = test_prices[symbol]
            else:
                price = 100.0  # Default price
            
            market_data['prices'][symbol] = price
            
            # Add detailed data
            market_data[symbol] = {
                'bid': price - 0.01,
                'ask': price + 0.01,
                'last': price,
                'volume': 1000000,
                'open': price * 0.99,
                'high': price * 1.01,
                'low': price * 0.98,
                'close': price
            }
        
        return market_data


class TestSignals:
    """Generate test signals for strategies."""
    
    @staticmethod
    def create_buy_signal(
        symbol: str,
        strategy_id: str = "test_strategy",
        strength: Decimal = Decimal("0.8"),
        timestamp: datetime = None
    ):
        """Create a buy signal for testing."""
        from src.risk.protocols import Signal, SignalType, OrderSide
        
        if timestamp is None:
            timestamp = datetime.now()
        
        return Signal(
            signal_id=f"BUY_{symbol}_{timestamp.timestamp()}",
            strategy_id=strategy_id,
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=strength,
            timestamp=timestamp,
            metadata={'test': True}
        )
    
    @staticmethod
    def create_sell_signal(
        symbol: str,
        strategy_id: str = "test_strategy",
        strength: Decimal = Decimal("1.0"),
        timestamp: datetime = None
    ):
        """Create a sell signal for testing."""
        from src.risk.protocols import Signal, SignalType, OrderSide
        
        if timestamp is None:
            timestamp = datetime.now()
        
        return Signal(
            signal_id=f"SELL_{symbol}_{timestamp.timestamp()}",
            strategy_id=strategy_id,
            symbol=symbol,
            signal_type=SignalType.EXIT,
            side=OrderSide.SELL,
            strength=strength,
            timestamp=timestamp,
            metadata={'test': True}
        )


class TestAssertions:
    """Custom assertions for financial data."""
    
    @staticmethod
    def assert_decimal_equal(actual: Decimal, expected: Decimal, places: int = 4):
        """Assert two decimals are equal to specified decimal places."""
        scale = Decimal(10) ** -places
        diff = abs(actual - expected)
        
        assert diff < scale, (
            f"Decimal values not equal to {places} places: "
            f"{actual} != {expected} (diff: {diff})"
        )
    
    @staticmethod
    def assert_decimal_range(
        value: Decimal,
        min_value: Decimal,
        max_value: Decimal
    ):
        """Assert decimal is within range."""
        assert min_value <= value <= max_value, (
            f"Value {value} not in range [{min_value}, {max_value}]"
        )
    
    @staticmethod
    def assert_positive_decimal(value: Decimal):
        """Assert decimal is positive."""
        assert value > Decimal("0"), f"Expected positive value, got {value}"
    
    @staticmethod
    def assert_percentage(value: Decimal):
        """Assert value is a valid percentage (0-1)."""
        assert Decimal("0") <= value <= Decimal("1"), (
            f"Expected percentage in [0, 1], got {value}"
        )


# Test data fixtures
TEST_MARKET_DATA_1D = MockMarketData.create_test_market_snapshot(
    TestConfig.TEST_SYMBOLS
)

TEST_PRICE_SERIES = {
    'AAPL': MockMarketData.generate_price_series(
        150.0, 252, volatility=0.02, trend=0.0003, seed=42
    ),
    'GOOGL': MockMarketData.generate_price_series(
        2800.0, 252, volatility=0.025, trend=0.0002, seed=43
    ),
    'MSFT': MockMarketData.generate_price_series(
        380.0, 252, volatility=0.018, trend=0.0004, seed=44
    )
}


if __name__ == "__main__":
    # Example usage
    print("Test Configuration")
    print("-" * 50)
    print(f"Default start date: {TestConfig.DEFAULT_START_DATE}")
    print(f"Default end date: {TestConfig.DEFAULT_END_DATE}")
    print(f"Default capital: ${TestConfig.DEFAULT_CAPITAL:,}")
    print(f"Test symbols: {', '.join(TestConfig.TEST_SYMBOLS)}")
    print()
    
    print("Sample Market Data")
    print("-" * 50)
    for symbol, data in TEST_MARKET_DATA_1D.items():
        if symbol != 'timestamp' and symbol != 'prices':
            print(f"{symbol}: ${data['last']:.2f}")
    print()
    
    print("Sample Price Series (first 10 days)")
    print("-" * 50)
    for symbol, prices in TEST_PRICE_SERIES.items():
        print(f"{symbol}: {[f'${p:.2f}' for p in prices[:10]]}")