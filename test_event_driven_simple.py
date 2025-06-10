#!/usr/bin/env python3
"""Simple test of event-driven execution without CLI complexity."""

import sys
import os
sys.path.insert(0, '/Users/daws/ADMF-PC')

from src.core.events.types import Event, EventType

def test_event_creation():
    """Test basic event creation to verify imports work."""
    try:
        event = Event(
            event_type=EventType.BAR.value,
            payload={'test': 'data'},
            source_id='test_source'
        )
        print(f"✓ Event created successfully: {event.event_type}")
        return True
    except Exception as e:
        print(f"✗ Event creation failed: {e}")
        return False

def test_data_handler():
    """Test data handler basic functionality."""
    try:
        from src.data.handlers import SimpleHistoricalDataHandler
        handler = SimpleHistoricalDataHandler("test_handler", "./data")
        print("✓ Data handler created successfully")
        return True
    except Exception as e:
        print(f"✗ Data handler creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing event-driven architecture components...")
    
    success = True
    success &= test_event_creation()
    success &= test_data_handler()
    
    if success:
        print("\n✓ All basic tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)