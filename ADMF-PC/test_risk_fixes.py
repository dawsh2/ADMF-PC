#!/usr/bin/env python3
"""
Test script to verify PC architecture fixes in the risk module.
"""

import sys
sys.path.append('/Users/daws/ADMF-PC/ADMF-PC')

from decimal import Decimal
from datetime import datetime

# Test 1: Signal construction with signal_id
print("Test 1: Signal with signal_id field")
print("-" * 40)
try:
    from src.risk.protocols import Signal, SignalType, OrderSide
    
    signal = Signal(
        signal_id="TEST-001",
        strategy_id="test_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.85"),
        timestamp=datetime.now(),
        metadata={"test": True}
    )
    print("✅ Signal created successfully with signal_id")
    print(f"   Signal ID: {signal.signal_id}")
    print(f"   Strategy: {signal.strategy_id}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: MaxExposureLimit exists
print("\nTest 2: MaxExposureLimit class")
print("-" * 40)
try:
    from src.risk.risk_limits import MaxExposureLimit
    
    limit = MaxExposureLimit(max_exposure_pct=Decimal("20"))
    print("✅ MaxExposureLimit created successfully")
    print(f"   Limit info: {limit.get_limit_info()}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Risk Portfolio Container event emission
print("\nTest 3: RiskPortfolioContainer initialization")
print("-" * 40)
try:
    from src.risk.risk_portfolio import RiskPortfolioContainer
    
    container = RiskPortfolioContainer(
        name="test_risk_portfolio",
        initial_capital=Decimal("100000")
    )
    print("✅ RiskPortfolioContainer created successfully")
    print(f"   Name: {container.name}")
    print(f"   Initial capital: {container._portfolio_state.get_cash_balance()}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Signal aggregation with signal_id
print("\nTest 4: Signal aggregation with proper IDs")
print("-" * 40)
try:
    from src.risk.signal_processing import SignalAggregator
    
    aggregator = SignalAggregator("weighted_average")
    
    signals = [
        Signal(
            signal_id="SIG-001",
            strategy_id="strat1",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        ),
        Signal(
            signal_id="SIG-002",
            strategy_id="strat2",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.6"),
            timestamp=datetime.now(),
            metadata={}
        )
    ]
    
    aggregated = aggregator.aggregate_signals(signals)
    if aggregated and aggregated[0].signal_id.startswith("AGG-"):
        print("✅ Signal aggregation works with proper IDs")
        print(f"   Aggregated signal ID: {aggregated[0].signal_id}")
        print(f"   Average strength: {aggregated[0].strength}")
    else:
        print("❌ Signal aggregation failed")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Complete integration
print("\nTest 5: Complete integration test")
print("-" * 40)
try:
    from src.risk.position_sizing import PercentagePositionSizer
    from src.risk.risk_limits import MaxPositionLimit
    
    # Create components
    portfolio = RiskPortfolioContainer(initial_capital=Decimal("100000"))
    portfolio.set_position_sizer(PercentagePositionSizer(percentage=Decimal("0.02")))
    portfolio.add_risk_limit(MaxPositionLimit(max_position_value=Decimal("5000")))
    portfolio.add_risk_limit(MaxExposureLimit(max_exposure_pct=Decimal("20")))
    
    # Process a signal
    test_signal = Signal(
        signal_id="INT-TEST-001",
        strategy_id="integration_test",
        symbol="MSFT",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=Decimal("0.75"),
        timestamp=datetime.now(),
        metadata={"price": 300.0}
    )
    
    market_data = {
        "prices": {"MSFT": 300.0},
        "timestamp": datetime.now()
    }
    
    orders = portfolio.process_signals([test_signal], market_data)
    
    if orders:
        print("✅ Complete integration test passed")
        print(f"   Generated {len(orders)} order(s)")
        print(f"   Order ID: {orders[0].order_id}")
        print(f"   Quantity: {orders[0].quantity}")
    else:
        print("⚠️  No orders generated (might be correct based on risk limits)")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("PC Architecture Fix Verification Complete!")
print("=" * 50)