#!/usr/bin/env python3
"""
Debug why IndicatorContainer isn't generating INDICATORS events.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def debug_indicator_hub():
    """Debug indicator hub and INDICATORS event generation"""
    print("🔧 Debugging IndicatorContainer and INDICATORS events...")
    
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
        
        print("✅ Containers created and connected")
        
        # Initialize properly
        await indicator_container.initialize()
        print("✅ Containers initialized")
        
        # Check indicator hub
        print(f"\n📊 Indicator Hub Status:")
        print(f"   Hub exists: {indicator_container.indicator_hub is not None}")
        print(f"   Subscriptions: {indicator_container._subscriptions}")
        
        # Track events
        indicators_events = []
        original_publish = indicator_container.publish_event
        
        def track_publish(event, target_scope=None):
            event_type = event.event_type
            if event_type == EventType.INDICATORS:
                indicators_events.append(event)
                print(f"   📤 INDICATORS event published to {target_scope}")
                print(f"      Payload keys: {list(event.payload.keys())}")
                indicators = event.payload.get('indicators', {})
                print(f"      Indicators: {indicators}")
            return original_publish(event, target_scope)
        
        indicator_container.publish_event = track_publish
        
        # Test with a BAR event
        print(f"\n🧪 Testing BAR event processing...")
        
        test_bar_event = Event(
            event_type=EventType.BAR,
            payload={
                'timestamp': datetime.now(),
                'market_data': {
                    'SPY': {
                        'open': 400.0,
                        'high': 401.0,
                        'low': 399.0,
                        'close': 400.5,
                        'volume': 100000
                    }
                },
                'bar_objects': {}
            },
            timestamp=datetime.now()
        )
        
        # Process event
        await indicator_container.process_event(test_bar_event)
        
        print(f"\n📊 Results:")
        print(f"   INDICATORS events generated: {len(indicators_events)}")
        
        # Check indicator hub directly
        if indicator_container.indicator_hub:
            print(f"\n🔍 Checking indicator hub directly...")
            
            # Check if indicator hub has any data
            for indicator_name in ["SMA_10", "SMA_20"]:
                value = indicator_container.indicator_hub.get_latest_value(indicator_name, 'SPY')
                if value:
                    print(f"   ✅ {indicator_name}: {value.value}")
                else:
                    print(f"   ❌ {indicator_name}: No value")
            
            # Check indicator hub cache
            cache = getattr(indicator_container.indicator_hub, '_cache', {})
            print(f"   Cache entries: {len(cache)}")
            
            # Check indicators configuration
            indicators = getattr(indicator_container.indicator_hub, 'indicators', {})
            print(f"   Configured indicators: {indicators}")
        
        return len(indicators_events) > 0
        
    except Exception as e:
        print(f"💥 Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_indicator_hub())
    
    if success:
        print("\n✅ IndicatorContainer generates INDICATORS events!")
    else:
        print("\n❌ IndicatorContainer has issues with INDICATORS events.")
        
    sys.exit(0 if success else 1)