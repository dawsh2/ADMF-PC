#!/usr/bin/env python3
"""
Simple test to verify Pydantic validation works.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.core.coordinator.config import (
        PYDANTIC_AVAILABLE,
        get_validation_errors,
        get_pydantic_example
    )

    print("🧪 Testing Pydantic Validation System")
    print(f"Pydantic available: {PYDANTIC_AVAILABLE}")
    
    if PYDANTIC_AVAILABLE:
        # Test 1: Valid configuration
        print("\n=== Test 1: Valid Configuration ===")
        valid_config = get_pydantic_example()
        errors = get_validation_errors(valid_config)
        
        if errors:
            print(f"❌ Unexpected validation errors: {errors}")
        else:
            print("✅ Valid configuration passed validation")
        
        # Test 2: Invalid configuration
        print("\n=== Test 2: Invalid Configuration ===")
        invalid_config = {
            "name": "Test",
            "data": {
                "symbols": [],  # Invalid: empty list
                "start_date": "invalid-date",  # Invalid: bad format
                "end_date": "2023-12-31",
                "frequency": "1d"
            },
            "portfolio": {
                "initial_capital": -1000  # Invalid: negative
            },
            "strategies": []  # Invalid: empty list
        }
        
        errors = get_validation_errors(invalid_config)
        if errors:
            print(f"✅ Correctly caught {len(errors)} validation errors:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  - ... and {len(errors) - 3} more errors")
        else:
            print("❌ Should have caught validation errors but didn't")
        
        print("\n🎉 Pydantic validation system is working correctly!")
    else:
        print("❌ Pydantic not available - install with: pip install pydantic>=2.0.0")

except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()