#!/usr/bin/env python3
"""
Debug indicator calculation with sufficient price history.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def debug_indicator_with_history():
    """Debug indicator calculation with multiple bars"""
    print("🔧 Debugging Indicator Calculation with Price History...")
    
    try:
        from src.execution.containers import IndicatorContainer, StrategyContainer
        from src.core.events.types import Event, EventType
        
        # Create containers
        indicator_config = {"required_indicators": ["SMA_10", "SMA_20"]}
        indicator_container = IndicatorContainer(indicator_config, "test_indicator")
        
        strategy_config = {"type": "momentum", "parameters": {"lookback_period": 5}}
        strategy_container = StrategyContainer(strategy_config, "test_strategy")
        
        # Set up hierarchy
        indicator_container.add_child_container(strategy_container)
        
        # Initialize properly
        await indicator_container.initialize()
        print("✅ Containers initialized")
        
        # Track INDICATORS events
        indicators_events = []
        
        def track_indicators(event):
            if event.event_type == EventType.INDICATORS:
                indicators_events.append(event)
                print(f"   📤 INDICATORS event!")
                indicators = event.payload.get('indicators', {})
                for symbol, values in indicators.items():
                    print(f"      {symbol}: {values}")
        
        # Subscribe to INDICATORS events on strategy container
        strategy_container.event_bus.subscribe(EventType.INDICATORS, track_indicators)
        
        # Send multiple BAR events to build history
        print(f"\n🧪 Sending multiple BAR events to build history...")
        
        base_time = datetime.now()
        prices = [400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 
                  410.0, 411.0, 412.0, 413.0, 414.0, 415.0, 416.0, 417.0, 418.0, 419.0, 420.0]
        
        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(minutes=i)
            
            bar_event = Event(
                event_type=EventType.BAR,
                payload={
                    'timestamp': timestamp,
                    'market_data': {
                        'SPY': {
                            'open': price - 0.5,
                            'high': price + 0.5,
                            'low': price - 0.5,
                            'close': price,
                            'volume': 100000
                        }
                    },
                    'bar_objects': {}
                },
                timestamp=timestamp
            )
            
            # Process through indicator container
            await indicator_container.process_event(bar_event)
            
            # Log progress
            if i < 3 or i >= 9:  # First 3 and last ones
                print(f"   Bar {i+1}: SPY @ ${price}")
                
                # Check indicator values directly after 10 bars
                if i >= 9:  # After 10 bars, SMA_10 should work
                    value = indicator_container.indicator_hub.get_latest_value('SMA_10', 'SPY')
                    if value:
                        print(f"      ✅ SMA_10: {value.value}")
                    else:
                        print(f"      ❌ SMA_10: No value")
        
        print(f"\n📊 Results:")
        print(f"   Bars processed: {len(prices)}")
        print(f"   INDICATORS events: {len(indicators_events)}")
        
        # Check final indicator values
        print(f"\n🔍 Final indicator values:")
        for indicator_name in ["SMA_10", "SMA_20"]:
            value = indicator_container.indicator_hub.get_latest_value(indicator_name, 'SPY')
            if value:
                print(f"   ✅ {indicator_name}: {value.value}")
            else:
                print(f"   ❌ {indicator_name}: No value")
        
        # Check if subscriptions are working
        print(f"\n📋 Subscription check:")
        print(f"   IndicatorContainer subscriptions: {indicator_container._subscriptions}")
        
        return len(indicators_events) > 0
        
    except Exception as e:
        print(f"💥 Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_indicator_with_history())
    
    if success:
        print("\n✅ Indicators are being calculated and events generated!")
    else:
        print("\n❌ Indicator calculation still has issues.")
        
    sys.exit(0 if success else 1)