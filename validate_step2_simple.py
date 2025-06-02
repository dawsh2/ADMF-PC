#!/usr/bin/env python3
"""
Simple Step 2 Requirements Validation

Validates Step 2 components without circular import issues.
Tests the Step 2 specific modules directly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/daws/ADMF-PC')

def test_models_import():
    """Test that models can be imported."""
    try:
        from src.risk.models import OrderSide, OrderType, SignalType, RiskConfig
        print("✅ Models import successful")
        return True
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        return False

def test_step2_components():
    """Test that Step 2 components can be imported."""
    components = [
        'src.risk.step2_portfolio_state',
        'src.risk.step2_position_sizer', 
        'src.risk.step2_risk_limits',
        'src.risk.step2_order_manager'
    ]
    
    success = True
    for component in components:
        try:
            __import__(component)
            print(f"✅ {component} import successful")
        except Exception as e:
            print(f"❌ {component} import failed: {e}")
            success = False
    
    return success

def test_container_factory():
    """Test container factory without circular imports."""
    try:
        # Import specific modules needed
        from src.risk.models import RiskConfig
        from src.risk.step2_portfolio_state import PortfolioState
        from src.risk.step2_position_sizer import PositionSizer
        from src.risk.step2_risk_limits import RiskLimits
        from src.risk.step2_order_manager import OrderManager
        
        # Test basic functionality
        config = RiskConfig(initial_capital=10000.0)
        portfolio = PortfolioState("test", 10000.0)
        sizer = PositionSizer("fixed", config)
        limits = RiskLimits(config)
        manager = OrderManager("test")
        
        print("✅ Step 2 components instantiation successful")
        print(f"  - Portfolio initial cash: {portfolio.cash}")
        print(f"  - Position sizer method: {sizer.sizing_method}")
        print(f"  - Risk limits max position: {limits.max_position_size}")
        print(f"  - Order manager container: {manager.container_id}")
        
        return True
    except Exception as e:
        print(f"❌ Step 2 components test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic Step 2 functionality."""
    try:
        from src.risk.models import OrderSide, SignalType, TradingSignal, Fill, RiskConfig
        from src.risk.step2_portfolio_state import PortfolioState
        from decimal import Decimal
        from datetime import datetime
        
        # Create test components
        portfolio = PortfolioState("test", 50000.0)
        
        # Test fill processing
        fill = Fill(
            fill_id="TEST001",
            order_id="ORDER001", 
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            price=Decimal('400'),
            timestamp=datetime.now()
        )
        
        portfolio.update_position(fill)
        
        # Verify position created
        assert "SPY" in portfolio.positions
        position = portfolio.positions["SPY"] 
        assert position.quantity == Decimal('100')
        assert position.avg_price == Decimal('400')
        
        # Verify cash updated
        expected_cash = Decimal('50000') - Decimal('40000')
        assert portfolio.cash == expected_cash
        
        print("✅ Basic functionality test passed")
        print(f"  - Position created: {position.quantity} shares at ${position.avg_price}")
        print(f"  - Cash remaining: ${portfolio.cash}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_position_sizing():
    """Test position sizing functionality."""
    try:
        from src.risk.models import OrderSide, SignalType, TradingSignal, RiskConfig
        from src.risk.step2_portfolio_state import PortfolioState
        from src.risk.step2_position_sizer import PositionSizer
        from decimal import Decimal
        from datetime import datetime
        
        # Create test components
        config = RiskConfig(
            initial_capital=100000.0,
            sizing_method='fixed',
            fixed_position_size=1000.0
        )
        
        portfolio = PortfolioState("test", 100000.0)
        portfolio.update_prices({"SPY": 400.0})
        
        sizer = PositionSizer('fixed', config)
        
        # Create test signal
        signal = TradingSignal(
            signal_id="TEST001",
            strategy_id="test",
            symbol="SPY", 
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('1.0'),
            timestamp=datetime.now()
        )
        
        # Calculate size
        size = sizer.calculate_size(signal, portfolio)
        
        # Should be $1000 / $400 = 2.5 shares, rounded to 2
        expected_size = Decimal('2')
        assert size == expected_size
        
        print("✅ Position sizing test passed")
        print(f"  - Signal strength: {signal.strength}")
        print(f"  - Calculated size: {size} shares")
        print(f"  - Expected size: {expected_size} shares")
        
        return True
        
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        return False

def test_risk_limits():
    """Test risk limits functionality."""
    try:
        from src.risk.models import OrderSide, SignalType, TradingSignal, RiskConfig
        from src.risk.step2_portfolio_state import PortfolioState
        from src.risk.step2_risk_limits import RiskLimits
        from decimal import Decimal
        from datetime import datetime
        
        # Create test components
        config = RiskConfig(
            max_position_size=0.1,  # 10%
            max_portfolio_risk=0.02  # 2%
        )
        
        portfolio = PortfolioState("test", 100000.0)
        limits = RiskLimits(config)
        
        # Create normal signal - should pass
        normal_signal = TradingSignal(
            signal_id="NORMAL001",
            strategy_id="test",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.1'),  # Low strength
            timestamp=datetime.now()
        )
        
        result = limits.can_trade(portfolio, normal_signal)
        assert result == True
        
        # Create high-risk signal - should fail
        risky_signal = TradingSignal(
            signal_id="RISKY001", 
            strategy_id="test",
            symbol="SPY",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal('0.9'),  # Very high strength
            timestamp=datetime.now()
        )
        
        result = limits.can_trade(portfolio, risky_signal)
        assert result == False
        
        print("✅ Risk limits test passed")
        print(f"  - Normal signal (strength 0.1): Allowed")
        print(f"  - Risky signal (strength 0.9): Rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk limits test failed: {e}")
        return False

def main():
    """Run all simple validation tests."""
    print("="*60)
    print("STEP 2 SIMPLE VALIDATION")
    print("="*60)
    
    tests = [
        ("Models Import", test_models_import),
        ("Step2 Components Import", test_step2_components), 
        ("Container Factory", test_container_factory),
        ("Basic Functionality", test_basic_functionality),
        ("Position Sizing", test_position_sizing),
        ("Risk Limits", test_risk_limits)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL STEP 2 BASIC VALIDATION PASSED!")
        print("✅ Step 2 components are working correctly")
    else:
        print("⚠️  Some tests failed")
    
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())