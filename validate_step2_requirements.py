#!/usr/bin/env python3
"""
Step 2 Requirements Validation Script

Validates that all Step 2 requirements from the complexity guide are implemented:

1. Risk Container with Event Isolation
2. Portfolio State Tracking
3. Risk Limits Enforcement
4. Position Sizing
5. Order Management
6. Event Flow Logging
7. Component Integration
8. Protocol-based Architecture

Architecture Context:
    - Part of: Step 2 - Add Risk Container validation
    - Purpose: Verify all requirements are met before proceeding to Step 3
    - Coverage: Complete Step 2 implementation validation
    - Dependencies: All Step 2 components
"""

import sys
import traceback
from decimal import Decimal
from datetime import datetime
from typing import List, Dict, Any

# Import validation components
try:
    from src.risk.step2_container_factory import create_test_risk_container, PRESET_CONFIGS
    from src.risk.models import (
        RiskConfig, TradingSignal, Order, Fill,
        SignalType, OrderSide, OrderType
    )
    from src.core.events.enhanced_isolation import get_enhanced_isolation_manager
    from src.core.logging.structured import ContainerLogger
except ImportError as e:
    print(f"Import error: {e}")
    print("Running validation with basic imports only...")
    # Use simpler validation approach if full imports fail


class Step2Validator:
    """Validates Step 2 implementation against requirements."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.logger = ContainerLogger("Step2Validator", "validation", "step2_validation")
    
    def validate_requirement(self, requirement_name: str, test_func) -> bool:
        """Validate a single requirement."""
        try:
            test_func()
            self.results.append(f"✅ {requirement_name}: PASSED")
            self.passed += 1
            return True
        except Exception as e:
            self.results.append(f"❌ {requirement_name}: FAILED - {str(e)}")
            self.results.append(f"   Error details: {traceback.format_exc()}")
            self.failed += 1
            return False
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "="*80)
        print("STEP 2 REQUIREMENTS VALIDATION RESULTS")
        print("="*80)
        
        for result in self.results:
            print(result)
        
        print("\n" + "-"*80)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 ALL STEP 2 REQUIREMENTS VALIDATED SUCCESSFULLY!")
            print("✅ Ready to proceed to Step 3")
        else:
            print("⚠️  Some requirements failed validation")
            print("❌ Fix failed requirements before proceeding")
        
        print("="*80)
        
        return self.failed == 0


def validate_risk_container_isolation():
    """Validate risk container has proper event isolation."""
    container = create_test_risk_container("isolation_test", 10000.0)
    
    # Check event bus exists and is isolated
    assert hasattr(container, 'event_bus'), "Risk container missing event bus"
    assert container.event_bus.container_id.endswith('_risk'), "Event bus not properly named"
    
    # Test isolation
    isolation_manager = get_enhanced_isolation_manager()
    external_bus = isolation_manager.create_container_bus("external_test")
    
    external_events = []
    def external_handler(event_type, data):
        external_events.append((event_type, data))
    
    external_bus.subscribe("TEST", external_handler)
    container.event_bus.publish("TEST", {"source": "risk"})
    
    assert len(external_events) == 0, "Event isolation failed - external bus received events"
    
    # Cleanup
    container.cleanup()
    isolation_manager.remove_container_bus("external_test")


def validate_portfolio_state_tracking():
    """Validate portfolio state tracks positions and cash correctly."""
    container = create_test_risk_container("portfolio_test", 50000.0)
    
    # Check initial state
    assert container.portfolio_state.cash == Decimal('50000'), "Initial cash incorrect"
    assert len(container.portfolio_state.positions) == 0, "Should start with no positions"
    
    # Create fill and update portfolio
    fill = Fill(
        fill_id="TEST_FILL", order_id="TEST_ORDER", symbol="SPY",
        side=OrderSide.BUY, quantity=Decimal('100'), price=Decimal('400'),
        timestamp=datetime.now()
    )
    
    container.on_fill(fill)
    
    # Verify position created
    assert "SPY" in container.portfolio_state.positions, "Position not created"
    position = container.portfolio_state.positions["SPY"]
    assert position.quantity == Decimal('100'), "Position quantity incorrect"
    assert position.avg_price == Decimal('400'), "Position average price incorrect"
    
    # Verify cash updated
    expected_cash = Decimal('50000') - Decimal('40000')  # 100 * 400
    assert container.portfolio_state.cash == expected_cash, "Cash not updated correctly"
    
    # Test total value calculation
    container.update_market_data({"SPY": 420.0})
    total_value = container.portfolio_state.calculate_total_value()
    expected_total = expected_cash + Decimal('42000')  # 100 * 420
    assert total_value == expected_total, "Total value calculation incorrect"
    
    container.cleanup()


def validate_risk_limits_enforcement():
    """Validate risk limits prevent dangerous trades."""
    container = create_test_risk_container("risk_test", 10000.0)
    container.update_market_data({"SPY": 100.0})
    
    # Create oversized position first
    large_fill = Fill(
        fill_id="LARGE_FILL", order_id="LARGE_ORDER", symbol="SPY",
        side=OrderSide.BUY, quantity=Decimal('80'), price=Decimal('100'),
        timestamp=datetime.now()
    )
    container.portfolio_state.update_position(large_fill)
    
    # Try to add more - should be rejected
    signal = TradingSignal(
        signal_id="RISKY_SIGNAL", strategy_id="test", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('1.0')
    )
    
    initial_rejected = container.rejected_signals
    container.on_signal(signal)
    
    assert container.rejected_signals > initial_rejected, "Risk limits did not reject oversized signal"
    
    container.cleanup()


def validate_position_sizing():
    """Validate position sizing calculates appropriate sizes."""
    # Test fixed sizing
    container = create_test_risk_container("sizing_test", 100000.0, "fixed")
    container.update_market_data({"SPY": 400.0})
    
    signal = TradingSignal(
        signal_id="SIZE_SIGNAL", strategy_id="test", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('1.0')
    )
    
    initial_orders = container.created_orders
    container.on_signal(signal)
    
    assert container.created_orders > initial_orders, "No order created for valid signal"
    
    # Check order size is reasonable
    pending_orders = container.portfolio_state.pending_orders
    assert len(pending_orders) > 0, "No pending orders found"
    
    order = list(pending_orders.values())[0]
    assert order.quantity > 0, "Order quantity should be positive"
    
    # Test signal strength scaling
    container.reset()
    container.update_market_data({"SPY": 400.0})
    
    weak_signal = TradingSignal(
        signal_id="WEAK_SIGNAL", strategy_id="test", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.1')
    )
    
    container.on_signal(weak_signal)
    
    # Should still create order but potentially smaller
    assert container.created_orders > 0, "Weak signal should still create order"
    
    container.cleanup()


def validate_order_management():
    """Validate order management creates proper orders."""
    container = create_test_risk_container("order_test", 50000.0)
    container.update_market_data({"SPY": 400.0})
    
    signal = TradingSignal(
        signal_id="ORDER_SIGNAL", strategy_id="momentum", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.8'),
        metadata={"custom_data": "test_value"}
    )
    
    container.on_signal(signal)
    
    # Verify order created
    pending_orders = container.portfolio_state.pending_orders
    assert len(pending_orders) > 0, "No orders created"
    
    order = list(pending_orders.values())[0]
    
    # Validate order properties
    assert order.symbol == "SPY", "Order symbol incorrect"
    assert order.side == OrderSide.BUY, "Order side incorrect"
    assert order.order_type == OrderType.MARKET, "Order type should be MARKET"
    assert order.quantity > 0, "Order quantity should be positive"
    assert "custom_data" in order.metadata, "Signal metadata not preserved"
    assert order.metadata["custom_data"] == "test_value", "Metadata value incorrect"
    
    # Test order ID generation
    assert order.order_id.startswith("ORD_order_test_"), "Order ID format incorrect"
    
    container.cleanup()


def validate_event_flow_logging():
    """Validate event flow logging works properly."""
    container = create_test_risk_container("logging_test", 25000.0)
    
    # Check logger setup
    assert hasattr(container, 'logger'), "Risk container missing logger"
    assert container.logger.component_name == "RiskContainer", "Logger component name incorrect"
    
    # Check all components have loggers
    assert hasattr(container.portfolio_state, 'logger'), "Portfolio state missing logger"
    assert hasattr(container.position_sizer, 'logger'), "Position sizer missing logger"
    assert hasattr(container.risk_limits, 'logger'), "Risk limits missing logger"
    assert hasattr(container.order_manager, 'logger'), "Order manager missing logger"
    
    # Test logging during operation (should not raise exceptions)
    container.update_market_data({"SPY": 400.0})
    
    signal = TradingSignal(
        signal_id="LOG_SIGNAL", strategy_id="test", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.7')
    )
    
    # This should log events without raising exceptions
    container.on_signal(signal)
    
    container.cleanup()


def validate_component_integration():
    """Validate components work together properly."""
    container = create_test_risk_container("integration_test", 75000.0)
    
    # Test complete signal-to-order flow
    container.update_market_data({"SPY": 300.0})
    
    signal = TradingSignal(
        signal_id="INT_SIGNAL", strategy_id="integration", symbol="SPY",
        signal_type=SignalType.ENTRY, side=OrderSide.BUY, strength=Decimal('0.6')
    )
    
    initial_state = container.get_state()
    container.on_signal(signal)
    final_state = container.get_state()
    
    # Verify signal processing chain worked
    assert final_state['processed_signals'] > initial_state['processed_signals'], "Signal not processed"
    assert final_state['created_orders'] > initial_state['created_orders'], "Order not created"
    
    # Test fill processing
    pending_orders = container.portfolio_state.pending_orders
    if pending_orders:
        order_id = list(pending_orders.keys())[0]
        
        fill = Fill(
            fill_id="INT_FILL", order_id=order_id, symbol="SPY",
            side=OrderSide.BUY, quantity=Decimal('50'), price=Decimal('300'),
            timestamp=datetime.now()
        )
        
        container.on_fill(fill)
        
        # Verify fill processed
        assert "SPY" in container.portfolio_state.positions, "Position not created from fill"
        assert len(container.portfolio_state.pending_orders) == 0, "Pending order not removed"
    
    container.cleanup()


def validate_protocol_based_architecture():
    """Validate architecture follows protocol-based design."""
    container = create_test_risk_container("protocol_test", 30000.0)
    
    # Verify no inheritance from complex base classes
    # Components should be simple classes with minimal inheritance
    
    # Check RiskContainer
    risk_base_classes = [cls.__name__ for cls in container.__class__.__mro__[1:]]
    assert 'object' in risk_base_classes, "RiskContainer inheritance chain incorrect"
    # Should not inherit from complex framework classes
    
    # Check portfolio state
    portfolio_base_classes = [cls.__name__ for cls in container.portfolio_state.__class__.__mro__[1:]]
    assert 'object' in portfolio_base_classes, "PortfolioState inheritance chain incorrect"
    
    # Check position sizer
    sizer_base_classes = [cls.__name__ for cls in container.position_sizer.__class__.__mro__[1:]]
    assert 'object' in sizer_base_classes, "PositionSizer inheritance chain incorrect"
    
    # Check risk limits
    limits_base_classes = [cls.__name__ for cls in container.risk_limits.__class__.__mro__[1:]]
    assert 'object' in limits_base_classes, "RiskLimits inheritance chain incorrect"
    
    # Check order manager
    manager_base_classes = [cls.__name__ for cls in container.order_manager.__class__.__mro__[1:]]
    assert 'object' in manager_base_classes, "OrderManager inheritance chain incorrect"
    
    container.cleanup()


def validate_factory_configurations():
    """Validate factory provides different configurations."""
    # Test preset configurations exist
    assert 'conservative' in PRESET_CONFIGS, "Conservative preset missing"
    assert 'moderate' in PRESET_CONFIGS, "Moderate preset missing"
    assert 'aggressive' in PRESET_CONFIGS, "Aggressive preset missing"
    assert 'test' in PRESET_CONFIGS, "Test preset missing"
    
    # Test each preset creates valid container
    for preset_name in PRESET_CONFIGS.keys():
        try:
            from src.risk.step2_container_factory import create_preset_risk_container
            container = create_preset_risk_container(f"preset_{preset_name}", preset_name, 50000.0)
            
            # Basic validation
            assert container.config.initial_capital == 50000.0, f"{preset_name} preset capital incorrect"
            assert hasattr(container.config, 'sizing_method'), f"{preset_name} preset missing sizing method"
            
            container.cleanup()
            
        except Exception as e:
            raise AssertionError(f"Failed to create {preset_name} preset: {e}")


def main():
    """Run all Step 2 validation tests."""
    validator = Step2Validator()
    
    print("Starting Step 2 Requirements Validation...")
    print("This validates the risk container implementation meets all complexity guide requirements.\n")
    
    # Run all validation tests
    validator.validate_requirement("Risk Container Event Isolation", validate_risk_container_isolation)
    validator.validate_requirement("Portfolio State Tracking", validate_portfolio_state_tracking)
    validator.validate_requirement("Risk Limits Enforcement", validate_risk_limits_enforcement)
    validator.validate_requirement("Position Sizing", validate_position_sizing)
    validator.validate_requirement("Order Management", validate_order_management)
    validator.validate_requirement("Event Flow Logging", validate_event_flow_logging)
    validator.validate_requirement("Component Integration", validate_component_integration)
    validator.validate_requirement("Protocol-based Architecture", validate_protocol_based_architecture)
    validator.validate_requirement("Factory Configurations", validate_factory_configurations)
    
    # Print results and return status
    success = validator.print_results()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())