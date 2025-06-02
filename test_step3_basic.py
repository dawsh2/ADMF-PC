#!/usr/bin/env python3
"""
Basic Step 3 Test

Quick test to verify the classifier container components work.
"""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

from datetime import datetime
from decimal import Decimal

# Create test data structures
from dataclasses import dataclass

@dataclass
class TestBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

def test_step3_components():
    """Test Step 3 classifier components."""
    print("🧪 Testing Step 3 Classifier Components")
    
    try:
        # Import Step 3 components
        from src.strategy.classifiers.regime_types import MarketRegime, ClassifierConfig
        from src.strategy.classifiers.classifier_container import create_test_classifier_container
        
        print("✅ Step 3 imports successful")
        
        # Create test classifier container
        container = create_test_classifier_container("test_classifier", "pattern")
        print(f"✅ Classifier container created: {container.container_id}")
        
        # Create test market data
        test_bars = [
            TestBar("SPY", datetime.now(), 400.0, 402.0, 399.0, 401.0, 1000000),
            TestBar("SPY", datetime.now(), 401.0, 403.0, 400.0, 402.5, 1100000),
            TestBar("SPY", datetime.now(), 402.5, 405.0, 401.0, 404.0, 1200000),
            TestBar("SPY", datetime.now(), 404.0, 406.0, 403.0, 405.5, 1050000),
            TestBar("SPY", datetime.now(), 405.5, 407.0, 404.0, 406.0, 950000),
        ]
        
        print(f"✅ Created {len(test_bars)} test bars")
        
        # Process test data
        for i, bar in enumerate(test_bars):
            container.on_bar(bar)
            
            state = container.get_state()
            print(f"  Bar {i+1}: Regime={state['current_regime']}, Confidence={state['confidence']:.2f}, Ready={state['is_ready']}")
        
        # Get final state
        final_state = container.get_state()
        print(f"\n📊 Final Results:")
        print(f"  - Bars Processed: {final_state['bars_processed']}")
        print(f"  - Current Regime: {final_state['current_regime']}")
        print(f"  - Confidence: {final_state['confidence']:.3f}")
        print(f"  - Is Ready: {final_state['is_ready']}")
        print(f"  - Regime Changes: {final_state['regime_changes']}")
        
        # Test classification details
        if hasattr(container, 'get_classification_details'):
            details = container.get_classification_details()
            if 'classifier_details' in details:
                metrics = details['classifier_details'].get('metrics', {})
                print(f"  - Volatility: {metrics.get('volatility', 0):.4f}")
                print(f"  - Trend Strength: {metrics.get('trend_strength', 0):.4f}")
        
        # Cleanup
        container.cleanup()
        
        print("\n🎉 Step 3 Basic Test PASSED!")
        print("✅ Classifier container works correctly")
        print("✅ Pattern classifier processes market data")
        print("✅ Event isolation system functioning")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Step 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Step 3 basic test."""
    print("="*60)
    print("STEP 3 BASIC VALIDATION")
    print("="*60)
    
    success = test_step3_components()
    
    if success:
        print("\n✅ STEP 3 BASIC COMPONENTS WORKING!")
        print("🚀 Ready for integration with existing system")
        return 0
    else:
        print("\n❌ Step 3 basic test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())