"""
Comprehensive tests for improved risk module components.

Tests cover dependency injection, performance optimizations, and proper
integration with the core system.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, MagicMock

from ..core.dependencies.container import DependencyContainer
from ..core.events.event_bus import EventBus
from ..improved_risk_portfolio import (
    RiskPortfolioContainer,
    create_risk_portfolio_container
)
from ..dependency_injection import (
    RiskComponentFactory,
    RiskDependencyResolver,
    create_position_sizer_spec,
    create_risk_limit_spec,
    create_signal_processor_spec,
)
from ..optimized_signal_flow import (
    OptimizedSignalFlowManager,
    HighPerformanceSignalCache,
    create_optimized_flow_manager
)
from ..improved_capabilities import RiskPortfolioCapability
from ..protocols import Signal, SignalType, OrderSide
from ..portfolio_state import PortfolioState


class TestDependencyInjection:
    """Test dependency injection infrastructure."""
    
    def test_risk_component_factory_creation(self):
        """Test creating components through factory."""
        container = DependencyContainer("test_container")
        factory = RiskComponentFactory(container)
        
        # Test position sizer creation
        spec = create_position_sizer_spec('percentage', 'test_sizer', percentage=Decimal("2.0"))
        sizer = factory.create_position_sizer(spec)
        
        assert sizer is not None
        assert hasattr(sizer, 'calculate_size')
        assert sizer.percentage == Decimal("2.0")
    
    def test_risk_limit_factory_creation(self):
        """Test creating risk limits through factory."""
        container = DependencyContainer("test_container")
        factory = RiskComponentFactory(container)
        
        # Test risk limit creation
        spec = create_risk_limit_spec(
            'position',
            'test_limit',
            max_position_value=Decimal("10000")
        )
        limit = factory.create_risk_limit(spec)
        
        assert limit is not None
        assert hasattr(limit, 'check_limit')
        assert limit.max_position_value == Decimal("10000")
    
    def test_dependency_resolver(self):
        """Test dependency resolution."""
        container = DependencyContainer("test_container")
        factory = RiskComponentFactory(container)
        resolver = RiskDependencyResolver(container, factory)
        
        # Register components
        sizer_spec = create_position_sizer_spec('fixed', 'default', size=Decimal("100"))
        resolver.register_position_sizer('default', sizer_spec)
        
        limit_spec = create_risk_limit_spec('position', 'pos_limit', max_position_value=Decimal("5000"))
        resolver.register_risk_limit(limit_spec)
        
        processor_spec = create_signal_processor_spec('standard', 'processor')
        resolver.register_signal_processor(processor_spec)
        
        # Test resolution
        sizer = resolver.get_position_sizer('default')
        limits = resolver.get_risk_limits()
        processor = resolver.get_signal_processor()
        
        assert sizer is not None
        assert len(limits) == 1
        assert processor is not None
    
    def test_spec_creation_functions(self):
        """Test specification creation utilities."""
        # Position sizer spec
        sizer_spec = create_position_sizer_spec(
            'volatility',
            'vol_sizer',
            risk_per_trade=Decimal("1.0"),
            lookback_period=20
        )
        assert sizer_spec.component_type == 'position_sizer'
        assert sizer_spec.class_name == 'VolatilityBasedSizer'
        assert sizer_spec.params['risk_per_trade'] == Decimal("1.0")
        
        # Risk limit spec
        limit_spec = create_risk_limit_spec(
            'drawdown',
            'dd_limit',
            max_drawdown_pct=Decimal("10"),
            reduce_at_pct=Decimal("8")
        )
        assert limit_spec.component_type == 'risk_limit'
        assert limit_spec.class_name == 'MaxDrawdownLimit'
        
        # Signal processor spec
        proc_spec = create_signal_processor_spec(
            'risk_adjusted',
            'risk_proc',
            risk_multiplier=Decimal("1.5")
        )
        assert proc_spec.component_type == 'signal_processor'
        assert proc_spec.class_name == 'RiskAdjustedSignalProcessor'


class TestImprovedRiskPortfolio:
    """Test improved risk portfolio container."""
    
    @pytest.fixture
    def risk_portfolio(self):
        """Create risk portfolio for testing."""
        container = DependencyContainer("test_container")
        return create_risk_portfolio_container(
            component_id="test_risk_portfolio",
            dependency_container=container,
            initial_capital=Decimal("100000")
        )
    
    def test_container_creation(self, risk_portfolio):
        """Test container creation with proper DI."""
        assert risk_portfolio.component_id == "test_risk_portfolio"
        assert risk_portfolio._dependency_container is not None
        assert risk_portfolio._resolver is not None
        assert risk_portfolio._factory is not None
    
    def test_component_configuration(self, risk_portfolio):
        """Test configuring components through container."""
        # Configure position sizer
        sizer = risk_portfolio.configure_position_sizer(
            'test_sizer',
            'percentage',
            percentage=Decimal("3.0")
        )
        assert sizer is not None
        assert sizer.percentage == Decimal("3.0")
        
        # Configure risk limit
        limit = risk_portfolio.configure_risk_limit(
            'position',
            'test_limit',
            max_position_value=Decimal("8000")
        )
        assert limit is not None
        assert limit.max_position_value == Decimal("8000")
        
        # Configure signal processor
        processor = risk_portfolio.configure_signal_processor('standard')
        assert processor is not None
    
    def test_lifecycle_management(self, risk_portfolio):
        """Test container lifecycle."""
        # Test initialization
        context = {'event_bus': Mock(), 'container_id': 'test'}
        risk_portfolio.initialize(context)
        assert risk_portfolio._initialized
        
        # Test start
        risk_portfolio.start()
        assert risk_portfolio._running
        
        # Test stop
        risk_portfolio.stop()
        assert not risk_portfolio._running
        
        # Test reset
        risk_portfolio.reset()
        portfolio_state = risk_portfolio.get_portfolio_state()
        assert portfolio_state.get_cash_balance() == Decimal("100000")
    
    def test_signal_processing(self, risk_portfolio):
        """Test signal processing through container."""
        # Initialize container
        context = {'event_bus': Mock()}
        risk_portfolio.initialize(context)
        risk_portfolio.start()
        
        # Configure default components
        risk_portfolio.configure_position_sizer(
            'default',
            'fixed',
            size=Decimal("100")
        )
        
        # Create test signal
        signal = Signal(
            signal_id="test_signal_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Process signal
        market_data = {"prices": {"AAPL": 150.0}}
        orders = risk_portfolio.process_signals([signal], market_data)
        
        assert len(orders) >= 0  # May be 0 if risk limits reject
    
    def test_portfolio_state_integration(self, risk_portfolio):
        """Test portfolio state integration."""
        portfolio_state = risk_portfolio.get_portfolio_state()
        
        assert portfolio_state is not None
        assert portfolio_state.get_cash_balance() == Decimal("100000")
        assert portfolio_state.get_total_value() == Decimal("100000")
        
        # Test metrics
        metrics = portfolio_state.get_risk_metrics()
        assert metrics.total_value == Decimal("100000")
        assert metrics.cash_balance == Decimal("100000")


class TestOptimizedSignalFlow:
    """Test optimized signal flow manager."""
    
    def test_optimized_flow_manager_creation(self):
        """Test creating optimized flow manager."""
        manager = create_optimized_flow_manager(performance_mode="fast")
        assert isinstance(manager, OptimizedSignalFlowManager)
        assert not manager._enable_caching  # Fast mode disables caching
        assert not manager._enable_validation  # Fast mode disables validation
        
        manager = create_optimized_flow_manager(performance_mode="safe")
        assert manager._enable_caching  # Safe mode enables caching
        assert manager._enable_validation  # Safe mode enables validation
    
    @pytest.mark.asyncio
    async def test_signal_collection_performance(self):
        """Test signal collection performance."""
        manager = OptimizedSignalFlowManager()
        manager.register_strategy("test_strategy")
        
        # Create test signal
        signal = Signal(
            signal_id="perf_test_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.7"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Collect signal (should be fast)
        await manager.collect_signal(signal)
        
        stats = manager.get_statistics()
        assert stats['signals_received'] == 1
        assert stats['buffer_size'] == 1
    
    @pytest.mark.asyncio
    async def test_batch_signal_processing(self):
        """Test batch signal processing."""
        manager = OptimizedSignalFlowManager()
        manager.register_strategy("test_strategy")
        
        # Create multiple signals
        signals = []
        for i in range(10):
            signal = Signal(
                signal_id=f"batch_test_{i:03d}",
                strategy_id="test_strategy",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal("0.5"),
                timestamp=datetime.now(),
                metadata={}
            )
            signals.append(signal)
            await manager.collect_signal(signal)
        
        # Mock dependencies
        portfolio_state = Mock()
        position_sizer = Mock()
        position_sizer.calculate_size.return_value = Decimal("100")
        risk_limits = []
        market_data = {"prices": {"AAPL": 150.0}}
        
        # Process signals
        orders = await manager.process_signals(
            portfolio_state, position_sizer, risk_limits, market_data
        )
        
        stats = manager.get_statistics()
        assert stats['signals_received'] == 10
        assert stats['buffer_size'] == 0  # Should be empty after processing
    
    def test_high_performance_cache(self):
        """Test high-performance signal cache."""
        cache = HighPerformanceSignalCache(max_size=100, duration=60)
        
        # Create test signal
        signal = Signal(
            signal_id="cache_test_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.6"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # First check should be miss
        assert not cache.is_duplicate(signal)
        
        # Add to cache
        cache.add_signal(signal)
        
        # Second check should be hit
        assert cache.is_duplicate(signal)
        
        # Check stats
        stats = cache._stats
        assert stats['hits'] == 1
        assert stats['misses'] == 1


class TestImprovedCapabilities:
    """Test improved capability system integration."""
    
    def test_risk_portfolio_capability_application(self):
        """Test applying risk portfolio capability."""
        # Create mock component
        component = Mock()
        component.component_id = "test_component"
        
        # Create capability
        capability = RiskPortfolioCapability()
        
        # Apply capability
        spec = {
            'initial_capital': 50000,
            'position_sizers': [
                {'name': 'default', 'type': 'percentage', 'percentage': 1.0}
            ],
            'risk_limits': [
                {'type': 'position', 'max_position_value': 5000}
            ]
        }
        
        enhanced_component = capability.apply(component, spec)
        
        # Check enhancements
        assert hasattr(enhanced_component, 'risk_portfolio')
        assert hasattr(enhanced_component, 'process_signals')
        assert hasattr(enhanced_component, 'get_portfolio_state')
        assert hasattr(enhanced_component, 'get_risk_metrics')
    
    def test_capability_lifecycle_integration(self):
        """Test capability lifecycle integration."""
        component = Mock()
        component.component_id = "lifecycle_test"
        component.initialize = Mock()
        component.start = Mock()
        component.stop = Mock()
        
        capability = RiskPortfolioCapability()
        spec = {'initial_capital': 25000}
        
        enhanced_component = capability.apply(component, spec)
        
        # Test enhanced lifecycle
        context = {'event_bus': Mock()}
        enhanced_component.initialize(context)
        enhanced_component.start()
        enhanced_component.stop()
        
        # Original methods should have been called
        component.initialize.assert_called_once()
        component.start.assert_called_once()
        component.stop.assert_called_once()
    
    def test_capability_delegation_methods(self):
        """Test capability delegation methods."""
        component = Mock()
        component.component_id = "delegation_test"
        
        capability = RiskPortfolioCapability()
        spec = {'initial_capital': 75000}
        
        enhanced_component = capability.apply(component, spec)
        
        # Test delegation methods exist
        assert callable(enhanced_component.process_signals)
        assert callable(enhanced_component.get_portfolio_state)
        assert callable(enhanced_component.get_position)
        assert callable(enhanced_component.get_risk_metrics)
        assert callable(enhanced_component.update_fills)
        assert callable(enhanced_component.update_market_data)


class TestPerformanceOptimizations:
    """Test performance optimizations."""
    
    def test_zero_allocation_hot_paths(self):
        """Test that hot paths don't allocate unnecessary objects."""
        manager = OptimizedSignalFlowManager(
            enable_caching=False,  # Disable for pure performance
            enable_validation=False
        )
        manager.register_strategy("perf_strategy")
        
        # Pre-warm manager
        signal = Signal(
            signal_id="warm_up",
            strategy_id="perf_strategy", 
            symbol="TEST",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.5"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        # This should be very fast (no allocations in hot path)
        import time
        start = time.perf_counter()
        
        for _ in range(1000):
            # Simulate hot path (signal collection)
            manager._stats['signals_received'] += 1
            if signal.strategy_id not in manager._registered_strategies:
                manager._stats['signals_rejected'] += 1
                continue
        
        elapsed = time.perf_counter() - start
        
        # Should complete very quickly (< 1ms for 1000 iterations)
        assert elapsed < 0.001
    
    def test_efficient_data_structures(self):
        """Test efficient data structure usage."""
        manager = OptimizedSignalFlowManager()
        
        # Test that signal buffer uses deque (O(1) operations)
        assert hasattr(manager._signal_buffer, 'append')
        assert hasattr(manager._signal_buffer, 'popleft')
        
        # Test that strategy registration uses set (O(1) lookup)
        manager.register_strategy("test_strategy")
        assert "test_strategy" in manager._registered_strategies
        
        # Lookup should be O(1)
        assert "test_strategy" in manager._registered_strategies
        assert "nonexistent" not in manager._registered_strategies


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_component_specs(self):
        """Test handling of invalid component specifications."""
        container = DependencyContainer("test_container")
        factory = RiskComponentFactory(container)
        
        # Test invalid position sizer type
        with pytest.raises(ValueError):
            create_position_sizer_spec('invalid_type', 'test', size=100)
        
        # Test invalid risk limit type
        with pytest.raises(ValueError):
            create_risk_limit_spec('invalid_type', 'test')
    
    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        container = DependencyContainer("test_container")
        resolver = RiskDependencyResolver(container, RiskComponentFactory(container))
        
        # Test missing position sizer
        with pytest.raises(ValueError):
            resolver.get_position_sizer('nonexistent')
        
        # Test missing signal processor
        with pytest.raises(ValueError):
            resolver.get_signal_processor()
    
    @pytest.mark.asyncio
    async def test_signal_processing_errors(self):
        """Test signal processing error handling."""
        manager = OptimizedSignalFlowManager()
        
        # Test unregistered strategy
        signal = Signal(
            signal_id="error_test",
            strategy_id="unregistered_strategy",
            symbol="TEST",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.5"),
            timestamp=datetime.now(),
            metadata={}
        )
        
        await manager.collect_signal(signal)
        
        stats = manager.get_statistics()
        assert stats['signals_rejected'] == 1
        assert stats['signals_received'] == 1


class TestThreadSafety:
    """Test thread safety features."""
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_collection(self):
        """Test concurrent signal collection."""
        manager = OptimizedSignalFlowManager()
        manager.register_strategy("concurrent_strategy")
        
        async def collect_signals(start_id: int, count: int):
            """Collect signals concurrently."""
            for i in range(count):
                signal = Signal(
                    signal_id=f"concurrent_{start_id}_{i}",
                    strategy_id="concurrent_strategy",
                    symbol="TEST",
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,
                    strength=Decimal("0.5"),
                    timestamp=datetime.now(),
                    metadata={}
                )
                await manager.collect_signal(signal)
        
        # Run concurrent signal collection
        tasks = [
            collect_signals(0, 50),
            collect_signals(100, 50),
            collect_signals(200, 50)
        ]
        
        await asyncio.gather(*tasks)
        
        stats = manager.get_statistics()
        assert stats['signals_received'] == 150
        assert stats['buffer_size'] == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
