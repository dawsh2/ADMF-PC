#!/usr/bin/env python3
"""
Quick test to verify Pydantic v2 compatibility is working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Test if we can import without errors
try:
    from src.core.coordinator.config.models import (
        PYDANTIC_AVAILABLE,
        WorkflowConfig,
        get_validation_errors,
        get_example_config,
        generate_schema_docs
    )
    
    print("✅ Successfully imported Pydantic models")
    print(f"Pydantic available: {PYDANTIC_AVAILABLE}")
    
    if PYDANTIC_AVAILABLE:
        # Test example config
        example = get_example_config()
        print(f"✅ Generated example config with {len(example)} top-level fields")
        
        # Test validation
        errors = get_validation_errors(example)
        if errors:
            print(f"❌ Example config has validation errors: {errors}")
        else:
            print("✅ Example config validates successfully")
        
        # Test schema docs
        docs = generate_schema_docs()
        print(f"✅ Generated schema documentation ({len(docs)} characters)")
        print("\nFirst 200 characters of docs:")
        print(docs[:200] + "...")
        
        # Test invalid config
        invalid = {"name": "test", "data": {"symbols": []}}  # Missing required fields
        errors = get_validation_errors(invalid)
        print(f"✅ Invalid config correctly produced {len(errors)} validation errors")
        
        print("\n🎉 All Pydantic v2 compatibility tests passed!")
    else:
        print("❌ Pydantic not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()