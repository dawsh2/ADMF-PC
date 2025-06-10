#!/usr/bin/env python3
"""
Test script to verify Pydantic validation integration.

Run this to test that the new validation system works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.coordinator.config import (
    PYDANTIC_AVAILABLE,
    WorkflowConfig,
    get_validation_errors,
    get_pydantic_example
)

def test_valid_config():
    """Test with a valid configuration."""
    print("=== Testing Valid Configuration ===")
    
    if not PYDANTIC_AVAILABLE:
        print("❌ Pydantic not available - install with: pip install pydantic>=2.0.0")
        return False
    
    # Get example config
    config = get_pydantic_example()
    print(f"✅ Example config loaded with {len(config)} fields")
    
    # Validate
    errors = get_validation_errors(config)
    if errors:
        print(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration validation passed!")
        return True

def test_invalid_config():
    """Test with invalid configuration."""
    print("\n=== Testing Invalid Configuration ===")
    
    if not PYDANTIC_AVAILABLE:
        print("❌ Pydantic not available")
        return False
    
    # Create invalid config
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
        "strategies": [
            {
                "name": "test_strategy",
                "type": "unknown_type",  # Invalid: unknown type
                "allocation": 1.5  # Invalid: > 1.0
            }
        ]
    }
    
    # Validate
    errors = get_validation_errors(invalid_config)
    if errors:
        print(f"✅ Correctly caught {len(errors)} validation errors:")
        for error in errors:
            print(f"  - {error}")
        return True
    else:
        print("❌ Validation should have failed but didn't!")
        return False

def test_coordinator_integration():
    """Test coordinator integration."""
    print("\n=== Testing Coordinator Integration ===")
    
    try:
        from src.core.coordinator.coordinator import Coordinator
        
        coordinator = Coordinator()
        
        # Test with invalid config
        invalid_config = {
            "name": "Test",
            "data": {"symbols": []},  # Invalid
            "portfolio": {"initial_capital": -100},  # Invalid
            "strategies": []  # Invalid: empty
        }
        
        result = coordinator.run_workflow(invalid_config)
        
        if result.get('success') == False and 'validation_errors' in result:
            print(f"✅ Coordinator correctly rejected invalid config with {len(result['validation_errors'])} errors")
            return True
        else:
            print("❌ Coordinator should have rejected invalid config")
            return False
            
    except Exception as e:
        print(f"❌ Coordinator integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Pydantic Validation Integration\n")
    
    results = []
    results.append(test_valid_config())
    results.append(test_invalid_config()) 
    results.append(test_coordinator_integration())
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Pydantic validation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())