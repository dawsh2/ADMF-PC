"""
File: tests/system/test_step1_full_backtest.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#testing
Step: 1 - Core Pipeline Test
Dependencies: pytest, core.events, event_flow, data.models

System tests for Step 1 full backtest validation.
Tests complete system behavior with known results.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from src.core.events.event_flow import run_simple_pipeline_test
from src.data.models import Bar


class TestStep1FullBacktest:
    """System tests for complete backtest scenarios."""
    
    def test_basic_backtest_with_known_results(self):
        """Test full backtest produces expected results."""
        container_id = "system_test_backtest_001"
        
        # Use deterministic data with known crossover points
        data = create_deterministic_trend_data()
        
        # Pre-computed expected results based on our strategy logic
        expected = {
            'bars_processed': len(data),
            'min_signals': 2,  # Should generate at least 2 signals (buy and sell)
            'final_value_min': 100000,  # At least initial value
        }
        
        # Run backtest
        results = run_simple_pipeline_test(container_id, data)
        
        # Verify exact match for deterministic parts
        assert results['bars_processed'] == expected['bars_processed']
        assert results['final_portfolio_value'] >= expected['final_value_min']
        
        # Verify we generated signals (exact count depends on crossovers)
        assert results['signals_generated'] >= expected['min_signals']
        
        # Verify pipeline completeness
        assert results['orders_created'] == results['signals_generated']
        assert results['portfolio_updates'] == results['orders_created']
    
    def test_no_signal_scenario(self):
        """Test backtest with data that generates no signals."""
        container_id = "system_test_no_signals_002"
        
        # Create sideways market data (no clear trend)
        data = create_sideways_market_data()
        
        # Run backtest
        results = run_simple_pipeline_test(container_id, data)
        
        # Should process all bars but generate no signals
        assert results['bars_processed'] == len(data)
        assert results['signals_generated'] == 0
        assert results['orders_created'] == 0
        assert results['portfolio_updates'] == 0
        assert results['final_portfolio_value'] == 100000  # Unchanged
    
    def test_multiple_crossover_scenario(self):
        """Test backtest with multiple trend changes."""
        container_id = "system_test_multiple_003"
        
        # Create data with multiple crossovers
        data = create_multiple_crossover_data()
        
        # Run backtest
        results = run_simple_pipeline_test(container_id, data)
        
        # Should generate multiple signals
        assert results['bars_processed'] == len(data)
        assert results['signals_generated'] >= 4  # At least 4 crossovers expected
        
        # Verify pipeline consistency
        assert results['orders_created'] == results['signals_generated']
        assert results['portfolio_updates'] == results['orders_created']
    
    def test_performance_requirements(self):
        """Test system meets performance requirements."""
        container_id = "system_test_performance_004"
        
        # Create 1 year of daily data (252 trading days)
        data = create_large_dataset(252)
        
        # Measure execution time
        import time
        start_time = time.time()
        
        results = run_simple_pipeline_test(container_id, data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance requirements from Step 1
        assert execution_time < 1.0  # < 1 second for daily data
        assert results['bars_processed'] == 252
        
        # Verify system still works correctly with large dataset
        assert results['signals_generated'] >= 0  # May or may not generate signals
        assert results['final_portfolio_value'] > 0
    
    def test_memory_usage_validation(self):
        """Test memory usage stays within bounds."""
        container_id = "system_test_memory_005"
        
        # Create moderate dataset
        data = create_moderate_dataset(100)
        
        # Monitor memory before
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run backtest
        results = run_simple_pipeline_test(container_id, data)
        
        # Monitor memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Memory requirements from Step 1
        assert memory_used < 100  # < 100MB for test data
        
        # Verify functionality
        assert results['bars_processed'] == len(data)
    
    def test_error_recovery(self):
        """Test system handles edge cases gracefully."""
        container_id = "system_test_errors_006"
        
        # Create data with edge cases
        data = create_edge_case_data()
        
        # Should not crash even with edge cases
        results = run_simple_pipeline_test(container_id, data)
        
        # Verify basic functionality
        assert results['bars_processed'] == len(data)
        assert 'signals_generated' in results
        assert 'final_portfolio_value' in results
        
        # Should maintain non-negative portfolio value
        assert results['final_portfolio_value'] >= 0
    
    def test_isolation_validation_system_level(self):
        """Test container isolation at system level."""
        container_a = "system_isolation_a"
        container_b = "system_isolation_b"
        
        # Create different datasets
        data_a = create_deterministic_trend_data()
        data_b = create_sideways_market_data()
        
        # Run concurrent backtests
        results_a = run_simple_pipeline_test(container_a, data_a)
        results_b = run_simple_pipeline_test(container_b, data_b)
        
        # Verify results are independent
        assert results_a['container_id'] != results_b['container_id']
        assert results_a['signals_generated'] != results_b['signals_generated']
        
        # Verify each backtest worked correctly
        assert results_a['bars_processed'] == len(data_a)
        assert results_b['bars_processed'] == len(data_b)


def create_deterministic_trend_data() -> List[Bar]:
    """Create deterministic data with known crossover patterns."""
    base_time = datetime(2023, 1, 1, 9, 30)  # Fixed start time
    bars = []
    
    # Designed to create specific crossover pattern
    # Phase 1: Flat (no signal)
    # Phase 2: Rising trend (bullish crossover)
    # Phase 3: Falling trend (bearish crossover)
    
    phases = [
        ([100] * 15, "flat"),           # 15 bars flat
        ([100 + i for i in range(20)], "rising"),  # 20 bars rising
        ([119 - i for i in range(15)], "falling")  # 15 bars falling
    ]
    
    bar_count = 0
    for prices, phase in phases:
        for price in prices:
            bar = Bar(
                symbol="TEST",
                timestamp=base_time + timedelta(minutes=bar_count),
                open=price - 0.1,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000
            )
            bars.append(bar)
            bar_count += 1
    
    return bars


def create_sideways_market_data() -> List[Bar]:
    """Create sideways market data that should generate no signals."""
    base_time = datetime(2023, 1, 1, 9, 30)
    bars = []
    
    # Oscillate around 100 with small movements
    base_price = 100
    for i in range(30):
        # Small oscillation that shouldn't trigger crossovers
        price = base_price + (i % 4 - 2) * 0.5  # +/- 1 around base
        
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars


def create_multiple_crossover_data() -> List[Bar]:
    """Create data with multiple crossovers."""
    base_time = datetime(2023, 1, 1, 9, 30)
    bars = []
    
    # Create alternating trends
    segments = [
        list(range(100, 115)),      # Up trend
        list(range(114, 99, -1)),   # Down trend
        list(range(100, 110)),      # Up trend
        list(range(109, 94, -1)),   # Down trend
    ]
    
    bar_count = 0
    for segment in segments:
        for price in segment:
            bar = Bar(
                symbol="TEST",
                timestamp=base_time + timedelta(minutes=bar_count),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000
            )
            bars.append(bar)
            bar_count += 1
    
    return bars


def create_large_dataset(num_bars: int) -> List[Bar]:
    """Create large dataset for performance testing."""
    base_time = datetime(2023, 1, 1, 9, 30)
    bars = []
    
    base_price = 100
    for i in range(num_bars):
        # Gentle upward trend with noise
        trend = i * 0.1
        noise = (i % 10 - 5) * 0.2
        price = base_price + trend + noise
        
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(days=i),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars


def create_moderate_dataset(num_bars: int) -> List[Bar]:
    """Create moderate dataset for memory testing."""
    base_time = datetime(2023, 1, 1, 9, 30)
    bars = []
    
    for i in range(num_bars):
        price = 100 + i * 0.5  # Steady uptrend
        
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars


def create_edge_case_data() -> List[Bar]:
    """Create data with edge cases."""
    base_time = datetime(2023, 1, 1, 9, 30)
    bars = []
    
    # Include various edge cases
    edge_prices = [
        100,    # Normal
        100,    # Duplicate
        0.01,   # Very small
        1000,   # Large jump
        999.99, # High precision
        100,    # Return to normal
    ]
    
    for i, price in enumerate(edge_prices):
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars