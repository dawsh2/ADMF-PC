"""Examples of how to properly create Order objects from different modules."""

from datetime import datetime
from decimal import Decimal

# Example 1: Using Order from execution.protocols
# This Order class does NOT have a timestamp parameter
from src.execution.protocols import Order as ExecutionOrder
from src.core.types import OrderType, OrderSide

# Creating an execution Order (no timestamp parameter)
execution_order = ExecutionOrder(
    order_id="EXE-001",
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=100.0,
    price=None,  # None for market orders
    stop_price=None,
    time_in_force="DAY",
    created_at=datetime.now(),  # Note: it's created_at, not timestamp
    metadata={"strategy": "momentum"}
)

print("Execution Order created successfully!")
print(f"Order ID: {execution_order.order_id}")
print(f"Created at: {execution_order.created_at}")


# Example 2: Using Order from risk.protocols
# This Order class HAS a timestamp parameter
from src.risk.protocols import Order as RiskOrder, Signal, SignalType

# First create a signal (required for risk Order)
signal = Signal(
    signal_id="SIG-001",
    strategy_id="MOMENTUM-1",
    symbol="AAPL",
    signal_type=SignalType.ENTRY,
    side=OrderSide.BUY,
    strength=Decimal("0.8"),
    timestamp=datetime.now(),
    metadata={}
)

# Creating a risk Order (has timestamp parameter)
risk_order = RiskOrder(
    order_id="RISK-001",
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("100"),
    price=None,
    stop_price=None,
    time_in_force="DAY",
    source_signal=signal,
    risk_checks_passed=["position_size", "max_exposure"],
    timestamp=datetime.now(),  # This Order HAS timestamp
    metadata={"strategy": "momentum"}
)

print("\nRisk Order created successfully!")
print(f"Order ID: {risk_order.order_id}")
print(f"Timestamp: {risk_order.timestamp}")


# Example 3: Common mistake - using timestamp with ExecutionOrder
try:
    # This will fail!
    wrong_order = ExecutionOrder(
        order_id="WRONG-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        timestamp=datetime.now()  # ERROR: ExecutionOrder doesn't have timestamp
    )
except TypeError as e:
    print(f"\nError as expected: {e}")
    print("Solution: Use 'created_at' instead of 'timestamp' for ExecutionOrder")


# Summary of differences:
print("\n=== SUMMARY ===")
print("1. ExecutionOrder (src.execution.protocols):")
print("   - Uses 'created_at' parameter")
print("   - No 'timestamp' parameter")
print("   - No 'source_signal' or 'risk_checks_passed' parameters")

print("\n2. RiskOrder (src.risk.protocols):")
print("   - Uses 'timestamp' parameter")
print("   - Requires 'source_signal' (Signal object)")
print("   - Requires 'risk_checks_passed' (list of strings)")
print("   - Uses Decimal for quantity instead of float")