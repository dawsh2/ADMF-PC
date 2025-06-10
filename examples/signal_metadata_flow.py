"""
Demonstrates that signals still carry full metadata to risk functions.

The optimization is about STORAGE redundancy, not runtime data flow.
"""

from typing import Dict, Any
from datetime import datetime

from src.core.events import Event, EventType
from src.core.events.tracing.signal_performance import create_risk_aware_signal


def show_runtime_signal_flow():
    """Show how signals flow to risk with full metadata."""
    
    print("=== RUNTIME SIGNAL FLOW (Unchanged!) ===\n")
    
    # 1. Strategy generates basic signal
    basic_signal = {
        'symbol': 'SPY',
        'direction': 'long',
        'strength': 0.8,
        'price': 450.50
    }
    
    print("1. Strategy generates basic signal:")
    print(f"   {basic_signal}")
    
    # 2. Signal generator enhances with metadata
    enhanced_signal = {
        **basic_signal,
        'strategy_id': 'momentum_20_50',
        'strategy_name': 'momentum_crossover',
        'bar_idx': 1000,
        'timestamp': datetime.now(),
        'features': {
            'sma_20': 449.80,
            'sma_50': 448.20,
            'rsi': 45.0,
            'volume': 80_000_000
        },
        'classifier_states': {
            'trend': 'bull',
            'volatility': 'normal',
            'volume_regime': 'high'
        },
        'bar_data': {
            'open': 449.00,
            'high': 451.00,
            'low': 448.50,
            'close': 450.50,
            'volume': 80_000_000
        }
    }
    
    print("\n2. Signal Generator enhances with context:")
    print(f"   Added: features, classifier_states, bar_data")
    
    # 3. Performance tracker adds risk context
    # This happens IN MEMORY, not from storage!
    from src.core.events.tracing.signal_performance import SignalPerformance
    
    performance = SignalPerformance(
        strategy_id='momentum_20_50',
        strategy_name='momentum_crossover',
        parameters={'fast': 20, 'slow': 50}
    )
    
    # Simulate some history
    performance.total_signals = 100
    performance.winning_signals = 65
    performance.win_rate = 0.65
    performance.profit_factor = 2.1
    performance.confidence_score = 0.78
    performance.recent_win_rate = 0.75
    
    # Add regime-specific performance
    performance.regime_performance['bull'] = {
        'signals': 40,
        'wins': 30,
        'win_rate': 0.75,
        'avg_pnl': 0.018
    }
    
    # Create risk-aware signal
    risk_aware_signal = create_risk_aware_signal(
        enhanced_signal,
        performance,
        current_regime='bull'
    )
    
    print("\n3. Performance Tracker adds risk context:")
    print(f"   risk_context: {{")
    for key, value in risk_aware_signal['risk_context'].items():
        print(f"      {key}: {value}")
    print(f"   }}")
    
    # 4. Create event with full payload
    signal_event = Event(
        event_type=EventType.SIGNAL.value,
        payload=risk_aware_signal,  # FULL metadata included!
        source_id='strategy_container_001',
        metadata={
            'bar_idx': 1000,
            'strategy_id': 'momentum_20_50',
            'workflow_id': 'backtest_001'
        }
    )
    
    print("\n4. Signal Event created with FULL payload:")
    print(f"   Event ID: {signal_event.event_id}")
    print(f"   Payload keys: {list(signal_event.payload.keys())}")
    
    # 5. Risk function receives complete signal
    print("\n5. Risk Function receives:")
    
    def example_risk_function(event: Event) -> Dict[str, Any]:
        """Example risk function showing available data."""
        signal = event.payload
        
        # All original signal data
        symbol = signal['symbol']
        direction = signal['direction']
        
        # All context data
        features = signal['features']
        classifier_states = signal['classifier_states']
        
        # All performance data
        risk_context = signal['risk_context']
        confidence = risk_context['confidence']
        win_rate = risk_context['win_rate']
        regime_perf = risk_context['regime_performance']
        
        print(f"   Symbol: {symbol}")
        print(f"   Direction: {direction}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Strategy Win Rate: {win_rate:.1%}")
        print(f"   Recent Win Rate: {risk_context['recent_win_rate']:.1%}")
        print(f"   Regime Performance: {regime_perf['win_rate']:.1%} in {classifier_states['trend']}")
        print(f"   Features available: {list(features.keys())}")
        print(f"   Suggested size multiplier: {risk_context['suggested_size_multiplier']:.2f}x")
        
        # Make sophisticated decision
        if confidence < 0.3:
            return {'accept': False, 'reason': 'Low confidence'}
        
        if regime_perf['win_rate'] < 0.4:
            return {'accept': False, 'reason': 'Poor regime performance'}
        
        # Adjust size based on all factors
        base_size = 0.02
        size = base_size * risk_context['suggested_size_multiplier']
        
        return {
            'accept': True,
            'size': size,
            'confidence': confidence
        }
    
    result = example_risk_function(signal_event)
    print(f"\n   Decision: {result}")
    
    # Show what changes with the optimization
    print("\n\n=== WHAT CHANGES WITH OPTIMIZATION ===")
    print("-" * 50)
    
    print("RUNTIME FLOW: UNCHANGED!")
    print("- Signals still carry full metadata")
    print("- Risk functions receive complete context")
    print("- Performance data still included")
    print("- All decision-making data available")
    
    print("\nSTORAGE: OPTIMIZED!")
    print("- Signal stored ONCE in EventTracer")
    print("- SignalStorage stores only (strategy_id, bar_idx, event_id)")
    print("- Performance metrics updated from event stream")
    print("- No duplicate storage of same data")
    
    print("\nBENEFITS:")
    print("- Risk functions work exactly the same")
    print("- Less memory usage")
    print("- Single source of truth")
    print("- Easier to query historical data")


def show_performance_lookup_flow():
    """Show how performance data is retrieved efficiently."""
    
    print("\n\n=== PERFORMANCE DATA RETRIEVAL ===")
    print("-" * 50)
    
    print("""
    Current Architecture:
    1. SignalGenerator stores signal
    2. PerformanceTracker stores results
    3. When generating new signal:
       - Load performance from storage
       - Add to signal
       - Forward to risk
    
    Optimized Architecture:
    1. EventTracer stores all events
    2. PerformanceTracker maintains live metrics (in memory)
    3. When generating new signal:
       - Get live performance from tracker (no disk I/O)
       - Add to signal  
       - Forward to risk
       
    Key: Performance metrics are LIVE in memory, not loaded from storage!
    """)
    
    # Example of live performance tracking
    class LivePerformanceTracker:
        """Maintains live performance metrics by subscribing to events."""
        
        def __init__(self):
            # Live metrics - always in memory
            self.strategy_metrics: Dict[str, SignalPerformance] = {}
        
        def subscribe_to_events(self, event_bus):
            """Subscribe to relevant events."""
            event_bus.subscribe(EventType.SIGNAL.value, self.on_signal)
            event_bus.subscribe(EventType.POSITION_CLOSE.value, self.on_position_close)
        
        def on_signal(self, event: Event):
            """Track new signal."""
            strategy_id = event.payload.get('strategy_id')
            if strategy_id not in self.strategy_metrics:
                self.strategy_metrics[strategy_id] = SignalPerformance(
                    strategy_id=strategy_id,
                    strategy_name=event.payload.get('strategy_name', ''),
                    parameters={}
                )
        
        def on_position_close(self, event: Event):
            """Update performance on trade close."""
            strategy_id = event.payload.get('strategy_id')
            if strategy_id in self.strategy_metrics:
                # Update metrics
                perf = self.strategy_metrics[strategy_id]
                perf.total_signals += 1
                # ... update win rate, etc.
        
        def get_performance(self, strategy_id: str) -> SignalPerformance:
            """Get live performance - no disk I/O!"""
            return self.strategy_metrics.get(strategy_id)
    
    print("LivePerformanceTracker maintains metrics in memory")
    print("- No disk I/O when generating signals")
    print("- Instant access to current performance")
    print("- Updates in real-time from event stream")


if __name__ == '__main__':
    show_runtime_signal_flow()
    show_performance_lookup_flow()