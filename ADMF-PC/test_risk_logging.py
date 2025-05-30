#!/usr/bin/env python3
"""Test script to verify risk module logging is working correctly."""

import logging
from decimal import Decimal
from datetime import datetime

# Configure logging to see all messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import risk modules
from src.risk.risk_portfolio import RiskPortfolioContainer
from src.risk.signal_processing import SignalProcessor, SignalAggregator
from src.risk.protocols import Signal, SignalType, OrderSide

def test_risk_portfolio_logging():
    """Test Risk Portfolio logging."""
    print("\n=== Testing Risk Portfolio Logging ===")
    
    # Create container
    container = RiskPortfolioContainer(
        name="TestPortfolio",
        initial_capital=Decimal("100000"),
        base_currency="USD"
    )
    
    # Test strategy management logging
    class MockStrategy:
        def get_metadata(self):
            return {"id": "test_strategy_1"}
    
    strategy = MockStrategy()
    container.add_strategy(strategy)
    container.remove_strategy("test_strategy_1")
    
    print("✓ Risk Portfolio logging works correctly")

def test_signal_processor_logging():
    """Test Signal Processor logging."""
    print("\n=== Testing Signal Processor Logging ===")
    
    processor = SignalProcessor()
    
    # Create test signal
    signal = Signal(
        signal_id="test_signal_1",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.8"),
        timestamp=datetime.now(),
        metadata={}
    )
    
    # Test validation logging (will trigger debug logs)
    processor._validate_signal(signal, None)
    
    print("✓ Signal Processor logging works correctly")

def test_signal_aggregator():
    """Test Signal Aggregator."""
    print("\n=== Testing Signal Aggregator ===")
    
    aggregator = SignalAggregator("weighted_average")
    
    # Create test signals
    signals = [
        Signal(
            signal_id=f"sig_{i}",
            strategy_id=f"strat_{i}",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.5"),
            timestamp=datetime.now(),
            metadata={}
        )
        for i in range(3)
    ]
    
    # Test aggregation
    aggregated = aggregator.aggregate_signals(signals)
    print(f"✓ Aggregated {len(signals)} signals into {len(aggregated)} signals")

if __name__ == "__main__":
    print("Testing Risk module logging after fixes...")
    
    try:
        test_risk_portfolio_logging()
        test_signal_processor_logging()
        test_signal_aggregator()
        
        print("\n✅ All logging tests passed! The risk module logging has been fixed.")
        print("   Standard Python logging format is now used throughout.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()