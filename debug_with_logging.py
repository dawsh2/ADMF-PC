#!/usr/bin/env python3
"""Debug with enhanced logging."""

import logging
import sys

# Configure logging before imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Patch ComponentState to log signal generation
import src.strategy.state as state_module

original_publish = state_module.ComponentState._publish_output

def debug_publish(self, output, symbol):
    """Log signal publishing."""
    if output and output.get('signal_value', 0) != 0:
        print(f"\n🚨 SIGNAL GENERATED: {output['signal_value']} for {symbol}")
        print(f"   Details: {output}")
    return original_publish(self, output, symbol)

state_module.ComponentState._publish_output = debug_publish

# Also patch the event bus to see if events are published
from src.core.events.bus import EventBus
original_publish_event = EventBus.publish_event

def debug_event_publish(self, event):
    if hasattr(event, 'event_type') and event.event_type == 'SIGNAL':
        print(f"\n📢 SIGNAL EVENT PUBLISHED: {event.payload}")
    return original_publish_event(self, event)

EventBus.publish_event = debug_event_publish

# Run main with simple config
sys.argv = ['main.py', '--config', 'config/bollinger/test_simple.yaml', '--signal-generation', '--dataset', 'train', '--bars', '100']

from main import main
main()