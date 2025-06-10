#!/usr/bin/env python3
"""
Simple script to show events from a running workflow.
We'll monkey-patch the data handler to print events as they're published.
"""

import sys
sys.path.append('.')

from src.core.coordinator.coordinator import Coordinator
from src.core.cli.args import load_yaml_config

# Monkey patch the EventBus publish method to show events
original_publish = None

def debug_publish(self, event):
    """Print events as they're published."""
    print(f"📡 EVENT: {event.event_type}")
    print(f"   📦 Container: {event.container_id}")
    print(f"   📄 Payload: {event.payload}")
    print(f"   🔗 Source: {event.source_id}")
    print("   " + "="*50)
    
    # Call original publish
    return original_publish(self, event)

def main():
    global original_publish
    
    print("🔍 Monkey-patching EventBus to show events...")
    
    # Import and patch EventBus
    from src.core.events.bus import EventBus
    original_publish = EventBus.publish
    EventBus.publish = debug_publish
    
    # Load config
    config = load_yaml_config('config/test_bar_streaming_console.yaml')
    config['data']['max_bars'] = 3  # Force limit
    
    print(f"📋 Config: {config['name']}")
    print(f"📊 Max bars: {config['data'].get('max_bars', 'unlimited')}")
    
    # Create coordinator and run
    coordinator = Coordinator()
    
    print("🚀 Running workflow (events will show below)...")
    print("="*60)
    
    result = coordinator.run_workflow(config)
    
    print("="*60)
    print(f"✅ Workflow complete: {result.get('success')}")
    
    # Restore original method
    EventBus.publish = original_publish

if __name__ == '__main__':
    main()