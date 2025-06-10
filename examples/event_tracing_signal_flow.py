"""
Walk through how event tracing and signal performance tracking interact.

This demonstrates the current flow and identifies potential redundancies.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from src.core.events import Event, EventType, EventBus
from src.core.events.tracing.observers import EventTracer, MetricsObserver
from src.core.events.tracing.signal_performance import SignalPerformance, PerformanceAwareSignalIndex
from src.core.containers.components.signal_generator import SignalGeneratorComponent


def demonstrate_current_flow():
    """Show how events flow through the system currently."""
    
    print("=== Current Event Flow Architecture ===\n")
    
    # 1. Setup event bus with tracing
    event_bus = EventBus(bus_id="main_bus")
    
    # Create event tracer (tracks ALL events)
    tracer = EventTracer(
        correlation_id="demo_trace_001",
        config={
            'max_events': 10000,
            'retention_policy': 'trade_complete',  # Prune after trade closes
            'container_isolation': True
        }
    )
    event_bus.attach_observer(tracer)
    
    # Create metrics observer (also tracks trade events)
    from src.analytics.metrics_collection import BacktestMetricsCalculator
    calculator = BacktestMetricsCalculator()
    metrics_observer = MetricsObserver(
        calculator=calculator,
        retention_policy='trade_complete'
    )
    event_bus.attach_observer(metrics_observer)
    
    # 2. Signal generation with storage
    signal_gen = SignalGeneratorComponent(
        storage_enabled=True,
        storage_path=Path("./demo_signals"),
        workflow_id="demo_001"
    )
    
    # Register a simple strategy
    def momentum_strategy(features, bar, params):
        if features.get('sma_fast', 0) > features.get('sma_slow', 0):
            return {
                'symbol': bar['symbol'],
                'direction': 'long',
                'value': 1.0
            }
        return None
    
    signal_gen.register_strategy(
        name="momentum",
        strategy_id="mom_20_50",
        func=momentum_strategy,
        parameters={'fast': 20, 'slow': 50}
    )
    
    # 3. Simulate signal generation and event flow
    print("1. Signal Generation Phase:")
    print("-" * 40)
    
    # Generate a signal
    bars = {'SPY': {'symbol': 'SPY', 'close': 100, 'timestamp': datetime.now()}}
    features = {'SPY': {'sma_fast': 101, 'sma_slow': 99}}
    
    signal_events = signal_gen.process_synchronized_bars(bars, features)
    
    # Publish signal event
    if signal_events:
        signal_event = signal_events[0]
        print(f"Signal Generated: {signal_event.payload['direction']} {signal_event.payload['symbol']}")
        print(f"  - Event ID: {signal_event.event_id}")
        print(f"  - Strategy: {signal_event.payload['strategy_id']}")
        
        # Signal is stored in SignalGeneratorComponent's storage
        print("\n  Storage Locations:")
        print("    1. SignalGeneratorComponent.storage_manager (sparse signal index)")
        
        # Publish to event bus
        event_bus.publish(signal_event)
        
        # Now it's also in:
        print("    2. EventTracer.storage (full event trace)")
        print("    3. MetricsObserver.active_trades (for metrics)")
    
    # 4. Order and Fill flow
    print("\n\n2. Order Execution Phase:")
    print("-" * 40)
    
    # Create order event (would come from Portfolio)
    order_event = Event(
        event_type=EventType.ORDER_REQUEST.value,
        payload={
            'symbol': 'SPY',
            'quantity': 100,
            'side': 'buy',
            'strategy_id': 'mom_20_50'
        },
        correlation_id=signal_event.correlation_id,  # Same correlation
        causation_id=signal_event.event_id  # Caused by signal
    )
    event_bus.publish(order_event)
    print(f"Order Created: BUY 100 SPY")
    print("  - Stored in: EventTracer, MetricsObserver")
    
    # Create fill event (would come from ExecutionEngine)
    fill_event = Event(
        event_type=EventType.FILL.value,
        payload={
            'symbol': 'SPY',
            'quantity': 100,
            'price': 100.05,
            'side': 'buy'
        },
        correlation_id=signal_event.correlation_id,
        causation_id=order_event.event_id
    )
    event_bus.publish(fill_event)
    print(f"Order Filled: 100 @ $100.05")
    
    # 5. Position tracking
    print("\n\n3. Position Tracking Phase:")
    print("-" * 40)
    
    position_open_event = Event(
        event_type=EventType.POSITION_OPEN.value,
        payload={
            'symbol': 'SPY',
            'quantity': 100,
            'entry_price': 100.05,
            'direction': 'long',
            'strategy_id': 'mom_20_50'
        },
        correlation_id=signal_event.correlation_id,
        causation_id=fill_event.event_id
    )
    event_bus.publish(position_open_event)
    print("Position Opened: LONG 100 SPY @ $100.05")
    
    # 6. Later - position close
    print("\n\n4. Position Close Phase:")
    print("-" * 40)
    
    position_close_event = Event(
        event_type=EventType.POSITION_CLOSE.value,
        payload={
            'symbol': 'SPY',
            'quantity': 100,
            'exit_price': 101.50,
            'pnl': 145.00,
            'pnl_pct': 0.0145,
            'strategy_id': 'mom_20_50'
        },
        correlation_id=signal_event.correlation_id,
        causation_id=position_open_event.event_id
    )
    event_bus.publish(position_close_event)
    print("Position Closed: EXIT @ $101.50")
    print(f"  - P&L: $145.00 (1.45%)")
    
    # Show retention behavior
    print("\n  Retention Policy Effects:")
    print("    - EventTracer: Prunes all events except close (trade_complete policy)")
    print("    - MetricsObserver: Updates metrics then removes from active_trades")
    print("    - SignalGenerator storage: Keeps signal (no auto-pruning)")
    
    # 7. Performance tracking update
    print("\n\n5. Performance Update Phase:")
    print("-" * 40)
    
    # Create performance-aware signal index
    perf_index = PerformanceAwareSignalIndex(
        strategy_name="momentum",
        strategy_id="mom_20_50", 
        parameters={'fast': 20, 'slow': 50}
    )
    
    # Record the result
    perf_index.record_result(
        bar_idx=0,  # Original signal bar
        result={
            'entry_idx': 0,
            'exit_idx': 10,
            'pnl': 145.00,
            'pnl_pct': 0.0145
        }
    )
    
    print(f"Performance Updated:")
    print(f"  - Win rate: {perf_index.performance.win_rate:.1%}")
    print(f"  - Confidence: {perf_index.performance.confidence_score:.2f}")
    
    # Show all storage locations
    print("\n\n=== Data Storage Summary ===")
    print("-" * 40)
    
    print("1. EVENT TRACING (EventTracer):")
    print("   - Stores: ALL events with full context")
    print("   - Purpose: Debugging, audit trail, causation tracking")
    print("   - Retention: Configurable (trade_complete, sliding_window, etc.)")
    print("   - Contains: Events, timing, correlation chains")
    
    print("\n2. SIGNAL STORAGE (SignalGeneratorComponent):")
    print("   - Stores: Sparse signal indices")
    print("   - Purpose: Signal replay without recomputation")
    print("   - Retention: Permanent (for replay)")
    print("   - Contains: Signals, classifier states, minimal context")
    
    print("\n3. PERFORMANCE TRACKING (SignalPerformance):")
    print("   - Stores: Aggregated performance metrics")
    print("   - Purpose: Risk decisions, confidence scoring")
    print("   - Retention: Rolling window of recent trades")
    print("   - Contains: Win rates, P&L stats, regime performance")
    
    print("\n4. METRICS OBSERVER (MetricsObserver):")
    print("   - Stores: Temporary trade events")
    print("   - Purpose: Calculate metrics on trade close")
    print("   - Retention: Only active trades")
    print("   - Contains: Open positions awaiting close")
    
    # Identify redundancies
    print("\n\n=== Potential Redundancies ===")
    print("-" * 40)
    
    print("1. Signal Event Storage:")
    print("   - Stored in: EventTracer (full event)")
    print("   - Stored in: SignalGenerator (sparse index)")
    print("   - Overlap: Basic signal info duplicated")
    
    print("\n2. Trade Tracking:")
    print("   - EventTracer: Tracks full correlation chain")
    print("   - MetricsObserver: Tracks active trades")
    print("   - Overlap: Both store trade events temporarily")
    
    print("\n3. Performance Metrics:")
    print("   - MetricsObserver: Calculates overall metrics")
    print("   - SignalPerformance: Tracks per-strategy performance")
    print("   - Overlap: Some metrics calculated twice")
    
    # Show ideal architecture
    print("\n\n=== Ideal Architecture ===")
    print("-" * 40)
    
    print("1. EVENT TRACING as Source of Truth:")
    print("   - All events flow through EventBus with tracing")
    print("   - Tracer stores complete event history")
    print("   - Other components QUERY tracer instead of storing")
    
    print("\n2. Signal Storage References Events:")
    print("   - Store only: strategy_id, bar_idx, event_id")
    print("   - Query tracer for full signal details when needed")
    print("   - Eliminates duplication")
    
    print("\n3. Performance Tracking Uses Event Stream:")
    print("   - Subscribe to POSITION_CLOSE events")
    print("   - Update performance metrics in real-time")
    print("   - No need to store trade events")
    
    print("\n4. Unified Querying:")
    print("   - Single API to query events by:")
    print("     - Correlation (full trade)")
    print("     - Event type + filters (all signals)")
    print("     - Time range")
    print("     - Container/strategy")


def demonstrate_improved_flow():
    """Show improved architecture with reduced redundancy."""
    
    print("\n\n=== IMPROVED ARCHITECTURE ===")
    print("=" * 50)
    
    print("""
    The key insight: Event Tracing should be the source of truth!
    
    Current Flow:
    Signal → [Store in SignalGenerator] → Event → [Store in Tracer] → [Store in Observer]
                       ↓                              ↓                        ↓
                 (Duplication!)              (Source of Truth)         (Duplication!)
    
    Improved Flow:
    Signal → Event → [Store ONLY in Tracer] 
                            ↓
                    [Other components query]
                    
    Benefits:
    1. Single source of truth
    2. No duplication
    3. Consistent retention policies
    4. Better memory efficiency
    5. Simpler querying
    
    Implementation:
    - SignalPerformance subscribes to events, doesn't store
    - SignalStorage stores only references (event_id)
    - MetricsObserver queries tracer for trade events
    - All components use tracer.query() for historical data
    """)


if __name__ == '__main__':
    demonstrate_current_flow()
    demonstrate_improved_flow()