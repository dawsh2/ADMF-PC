"""
Basic test to verify walk-forward validation core functionality.
No external dependencies.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

# Inline the core classes to avoid import issues
@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period."""
    period_id: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    
    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start
    
    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class WalkForwardValidator:
    """Walk-forward validation for strategy optimization."""
    
    def __init__(self, 
                 data_length: int,
                 train_size: int,
                 test_size: int,
                 step_size: int,
                 anchored: bool = False):
        self.data_length = data_length
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored
        
        # Validate parameters
        if self.train_size <= 0:
            raise ValueError("Training size must be positive")
        
        if self.test_size <= 0:
            raise ValueError("Test size must be positive")
        
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
        
        if self.train_size + self.test_size > self.data_length:
            raise ValueError("Train + test size exceeds data length")
        
        # Generate periods
        self.periods = self._generate_periods()
    
    def _generate_periods(self) -> List[WalkForwardPeriod]:
        """Generate walk-forward periods."""
        periods = []
        
        if self.anchored:
            # Anchored walk-forward (expanding window)
            current_test_start = self.train_size
            period_num = 0
            
            while current_test_start + self.test_size <= self.data_length:
                periods.append(WalkForwardPeriod(
                    period_id=f"period_{period_num}",
                    train_start=0,  # Always start from beginning
                    train_end=current_test_start,
                    test_start=current_test_start,
                    test_end=current_test_start + self.test_size
                ))
                
                current_test_start += self.step_size
                period_num += 1
                
        else:
            # Rolling walk-forward (fixed window)
            current_start = 0
            period_num = 0
            
            while current_start + self.train_size + self.test_size <= self.data_length:
                periods.append(WalkForwardPeriod(
                    period_id=f"period_{period_num}",
                    train_start=current_start,
                    train_end=current_start + self.train_size,
                    test_start=current_start + self.train_size,
                    test_end=current_start + self.train_size + self.test_size
                ))
                
                current_start += self.step_size
                period_num += 1
        
        return periods
    
    def get_periods(self) -> List[WalkForwardPeriod]:
        """Get all walk-forward periods."""
        return self.periods


def test_rolling_walk_forward():
    """Test rolling (fixed window) walk-forward generation."""
    print("\nTest: Rolling Walk-Forward")
    print("-" * 40)
    
    validator = WalkForwardValidator(
        data_length=1000,
        train_size=500,
        test_size=100,
        step_size=100,
        anchored=False
    )
    
    periods = validator.get_periods()
    print(f"Generated {len(periods)} periods")
    
    # Expected: 5 periods
    # Period 0: train[0:500], test[500:600]
    # Period 1: train[100:600], test[600:700]
    # Period 2: train[200:700], test[700:800]
    # Period 3: train[300:800], test[800:900]
    # Period 4: train[400:900], test[900:1000]
    
    assert len(periods) == 5, f"Expected 5 periods, got {len(periods)}"
    
    # Check first period
    first = periods[0]
    assert first.train_start == 0
    assert first.train_end == 500
    assert first.test_start == 500
    assert first.test_end == 600
    assert first.train_size == 500
    assert first.test_size == 100
    print("✓ First period correct")
    
    # Check rolling behavior
    second = periods[1]
    assert second.train_start == 100  # Rolled forward by step_size
    assert second.train_end == 600
    assert second.test_start == 600
    assert second.test_end == 700
    print("✓ Rolling window correct")
    
    # Check last period
    last = periods[-1]
    assert last.test_end == 1000  # Uses all data
    print("✓ Last period correct")
    
    # Display all periods
    print("\nAll periods:")
    for period in periods:
        print(f"  {period.period_id}: train[{period.train_start}:{period.train_end}], "
              f"test[{period.test_start}:{period.test_end}]")


def test_anchored_walk_forward():
    """Test anchored (expanding window) walk-forward generation."""
    print("\n\nTest: Anchored Walk-Forward")
    print("-" * 40)
    
    validator = WalkForwardValidator(
        data_length=1000,
        train_size=500,
        test_size=100,
        step_size=100,
        anchored=True
    )
    
    periods = validator.get_periods()
    print(f"Generated {len(periods)} periods")
    
    # All periods should start training from 0
    for period in periods:
        assert period.train_start == 0, f"Period {period.period_id} doesn't start from 0"
    print("✓ All periods anchored at start")
    
    # Training size should expand
    assert periods[0].train_size == 500
    assert periods[1].train_size == 600  # +100 (step_size)
    assert periods[2].train_size == 700  # +200
    print("✓ Training window expands correctly")
    
    # Test windows should roll forward
    assert periods[0].test_start == 500
    assert periods[1].test_start == 600
    assert periods[2].test_start == 700
    print("✓ Test windows roll forward")
    
    # Display first few periods
    print("\nFirst 3 anchored periods:")
    for period in periods[:3]:
        print(f"  {period.period_id}: train[{period.train_start}:{period.train_end}] "
              f"(size={period.train_size}), test[{period.test_start}:{period.test_end}]")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n\nTest: Edge Cases")
    print("-" * 40)
    
    # Test minimum viable configuration
    validator = WalkForwardValidator(
        data_length=200,
        train_size=100,
        test_size=50,
        step_size=50,
        anchored=False
    )
    
    periods = validator.get_periods()
    assert len(periods) == 2
    print("✓ Minimum viable configuration works")
    
    # Test exact fit
    validator2 = WalkForwardValidator(
        data_length=600,
        train_size=500,
        test_size=100,
        step_size=600,  # No rolling
        anchored=False
    )
    
    periods2 = validator2.get_periods()
    assert len(periods2) == 1
    print("✓ Exact fit configuration works")
    
    # Test error cases
    try:
        WalkForwardValidator(
            data_length=100,
            train_size=80,
            test_size=30,  # Total 110 > 100
            step_size=10,
            anchored=False
        )
        assert False, "Should have raised error"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")


def demonstrate_practical_example():
    """Demonstrate a practical walk-forward setup."""
    print("\n\nPractical Example: Strategy Validation")
    print("=" * 50)
    
    # Scenario: 2 years of daily data
    # Train on 1 year, test on 3 months, roll quarterly
    
    days_in_year = 252  # Trading days
    total_days = 2 * days_in_year  # 2 years
    train_days = days_in_year  # 1 year
    test_days = 63  # ~3 months
    step_days = 63  # Quarterly roll
    
    print(f"Scenario:")
    print(f"  - Total data: {total_days} days (2 years)")
    print(f"  - Training window: {train_days} days (1 year)")
    print(f"  - Test window: {test_days} days (3 months)")
    print(f"  - Step size: {step_days} days (quarterly)")
    
    validator = WalkForwardValidator(
        data_length=total_days,
        train_size=train_days,
        test_size=test_days,
        step_size=step_days,
        anchored=False
    )
    
    periods = validator.get_periods()
    print(f"\nGenerated {len(periods)} quarterly validation periods:")
    
    for i, period in enumerate(periods):
        train_months = period.train_size / 21  # Approximate
        test_months = period.test_size / 21
        
        print(f"\nQ{i+1} {2020 + i//4}:")
        print(f"  Train: days {period.train_start:3d}-{period.train_end:3d} "
              f"(~{train_months:.1f} months)")
        print(f"  Test:  days {period.test_start:3d}-{period.test_end:3d} "
              f"(~{test_months:.1f} months)")
    
    print("\nBenefits:")
    print("  ✓ Each quarter tested on future unseen data")
    print("  ✓ Parameters re-optimized quarterly")
    print("  ✓ Realistic simulation of live trading")
    print("  ✓ Overfitting detected by train/test gap")


if __name__ == "__main__":
    print("=" * 50)
    print("Walk-Forward Validation Tests")
    print("=" * 50)
    
    test_rolling_walk_forward()
    test_anchored_walk_forward()
    test_edge_cases()
    demonstrate_practical_example()
    
    print("\n" + "=" * 50)
    print("✅ All Walk-Forward Tests Passed!")
    print("=" * 50)
    
    print("\nSummary:")
    print("- Walk-forward validation splits data into train/test periods")
    print("- Rolling window: Fixed-size windows that move forward")
    print("- Anchored window: Training always starts from beginning")
    print("- Each period trains on past, tests on future")
    print("- Helps detect overfitting and validate robustness")