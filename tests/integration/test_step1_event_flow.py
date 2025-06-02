"""
File: tests/integration/test_step1_event_flow.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#testing
Step: 1 - Core Pipeline Test
Dependencies: pytest, core.events, event_flow

Integration tests for Step 1 event flow.
Validates complete event pipeline with component interactions.
"""

import pytest
from datetime import datetime, timedelta
import time

from src.core.events.event_flow import setup_core_pipeline, run_simple_pipeline_test
from src.core.events.enhanced_isolation import get_enhanced_isolation_manager
from src.data.models import Bar


class TestEventFlowIntegration:
    """Integration tests for complete event flow."""
    
    def test_pipeline_setup(self):
        """Test pipeline sets up correctly."""
        container_id = "test_setup_001"
        
        try:
            pipeline = setup_core_pipeline(container_id)
            
            # Verify all components created
            assert 'event_bus' in pipeline
            assert 'components' in pipeline
            assert 'container_id' in pipeline
            
            components = pipeline['components']
            expected_components = [
                'data_source', 'indicator', 'strategy', 
                'risk_manager', 'execution', 'portfolio_state'
            ]
            
            for component_name in expected_components:
                assert component_name in components
                assert components[component_name] is not None
            
            # Verify event bus injection
            assert pipeline['components']['data_source'].event_bus is not None
            assert pipeline['components']['strategy'].event_bus is not None
            assert pipeline['components']['risk_manager'].event_bus is not None
            assert pipeline['components']['execution'].event_bus is not None
            
        finally:
            # Cleanup
            isolation_manager = get_enhanced_isolation_manager()
            isolation_manager.remove_container_bus(container_id)
    
    def test_signal_to_order_flow(self):
        """Test complete event flow from data to execution."""
        container_id = "test_flow_002"
        
        try:
            # Setup pipeline
            pipeline = setup_core_pipeline(container_id)
            components = pipeline['components']
            
            # Create test data that will trigger crossover
            test_bars = create_sma_crossover_scenario()
            
            # Process bars
            data_source = components['data_source']
            for bar in test_bars:
                data_source.emit_bar(bar)
                time.sleep(0.01)  # Small delay for event processing
            
            # Verify events flowed through pipeline
            risk_manager = components['risk_manager']
            execution = components['execution']
            portfolio_state = components['portfolio_state']
            
            # Should have generated at least one signal
            assert len(risk_manager.processed_signals) > 0
            
            # Should have created orders
            assert len(execution.processed_orders) > 0
            
            # Should have portfolio updates
            assert len(portfolio_state.updates) > 0
            
            # Verify data consistency
            assert len(risk_manager.processed_signals) == len(execution.processed_orders)
            assert len(execution.processed_orders) == len(portfolio_state.updates)
            
        finally:
            # Cleanup
            isolation_manager = get_enhanced_isolation_manager()
            isolation_manager.remove_container_bus(container_id)
    
    def test_event_isolation(self):
        """Test that events are properly isolated between containers."""
        container_a = "test_isolation_a"
        container_b = "test_isolation_b"
        
        try:
            # Setup two isolated pipelines
            pipeline_a = setup_core_pipeline(container_a)
            pipeline_b = setup_core_pipeline(container_b)
            
            # Create test data
            test_bars_a = create_simple_trend_data("STOCK_A")
            test_bars_b = create_simple_trend_data("STOCK_B")
            
            # Process data in both pipelines
            pipeline_a['components']['data_source'].emit_bar(test_bars_a[0])
            pipeline_b['components']['data_source'].emit_bar(test_bars_b[0])
            
            time.sleep(0.1)  # Allow processing
            
            # Get event bus stats
            isolation_manager = get_enhanced_isolation_manager()
            stats_a = isolation_manager.get_container_bus(container_a).get_isolation_stats()
            stats_b = isolation_manager.get_container_bus(container_b).get_isolation_stats()
            
            # Verify isolation (no violations)
            assert stats_a['isolation_violations'] == 0
            assert stats_b['isolation_violations'] == 0
            
            # Verify containers are isolated
            active_containers = isolation_manager._active_containers
            assert container_a in active_containers
            assert container_b in active_containers
            
        finally:
            # Cleanup
            isolation_manager = get_enhanced_isolation_manager()
            isolation_manager.remove_container_bus(container_a)
            isolation_manager.remove_container_bus(container_b)
    
    def test_indicator_strategy_integration(self):
        """Test integration between indicators and strategy."""
        container_id = "test_indicator_strategy_003"
        
        try:
            pipeline = setup_core_pipeline(container_id)
            strategy = pipeline['components']['strategy']
            indicator = pipeline['components']['indicator']
            
            # Verify indicator and strategy are properly connected
            assert strategy.fast_sma is not None
            assert strategy.slow_sma is not None
            
            # Create bars that will make indicators ready
            test_bars = create_sufficient_bars_for_sma()
            
            data_source = pipeline['components']['data_source']
            for bar in test_bars:
                data_source.emit_bar(bar)
                time.sleep(0.01)
            
            # Verify indicators are ready
            assert indicator.is_ready
            assert strategy.fast_sma.is_ready
            assert strategy.slow_sma.is_ready
            
            # Verify strategy can access indicator values
            assert indicator.current_value is not None
            assert strategy.fast_sma.current_value is not None
            assert strategy.slow_sma.current_value is not None
            
        finally:
            # Cleanup
            isolation_manager = get_enhanced_isolation_manager()
            isolation_manager.remove_container_bus(container_id)
    
    def test_fill_feedback_loop(self):
        """Test that fills properly flow back to risk manager."""
        container_id = "test_feedback_004"
        
        try:
            pipeline = setup_core_pipeline(container_id)
            components = pipeline['components']
            
            # Create crossover scenario
            test_bars = create_sma_crossover_scenario()
            
            # Process bars
            data_source = components['data_source']
            for bar in test_bars:
                data_source.emit_bar(bar)
                time.sleep(0.01)
            
            risk_manager = components['risk_manager']
            execution = components['execution']
            
            # Verify feedback loop
            if len(execution.processed_orders) > 0:
                # Should have equal number of portfolio updates and orders
                assert len(risk_manager.portfolio_updates) == len(execution.processed_orders)
                
                # Verify fill data is consistent
                for i, order in enumerate(execution.processed_orders):
                    portfolio_update = risk_manager.portfolio_updates[i]
                    assert portfolio_update['symbol'] == order['symbol']
                    assert portfolio_update['quantity'] == order['quantity']
            
        finally:
            # Cleanup
            isolation_manager = get_enhanced_isolation_manager()
            isolation_manager.remove_container_bus(container_id)
    
    def test_simple_pipeline_test_function(self):
        """Test the run_simple_pipeline_test helper function."""
        container_id = "test_helper_005"
        
        # Create test data
        test_bars = create_sma_crossover_scenario()
        
        # Run test
        results = run_simple_pipeline_test(container_id, test_bars)
        
        # Verify results structure
        assert 'container_id' in results
        assert 'bars_processed' in results
        assert 'signals_generated' in results
        assert 'orders_created' in results
        assert 'portfolio_updates' in results
        assert 'final_portfolio_value' in results
        
        # Verify data
        assert results['container_id'] == container_id
        assert results['bars_processed'] == len(test_bars)
        assert results['final_portfolio_value'] >= 100000  # Should be at least initial value


def create_sma_crossover_scenario():
    """Create test data that will trigger SMA crossover."""
    base_time = datetime.now()
    bars = []
    
    # Create rising trend that will trigger bullish crossover
    prices = [95, 96, 97, 99, 102, 105, 108, 110, 112, 115, 118, 120]
    
    for i, price in enumerate(prices):
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.5,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars


def create_simple_trend_data(symbol="TEST"):
    """Create simple trend data for testing."""
    base_time = datetime.now()
    bars = []
    
    prices = [100, 101, 102, 103, 104]
    
    for i, price in enumerate(prices):
        bar = Bar(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars


def create_sufficient_bars_for_sma():
    """Create enough bars to make SMAs ready."""
    base_time = datetime.now()
    bars = []
    
    # Need at least 20 bars for the indicator (period=20)
    for i in range(25):
        price = 100 + i * 0.5  # Gradual uptrend
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000
        )
        bars.append(bar)
    
    return bars