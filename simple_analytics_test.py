#!/usr/bin/env python3
"""
Simple test for analytics implementation to verify basic functionality
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import duckdb
        print(f"✅ DuckDB available: {duckdb.__version__}")
    except ImportError as e:
        print(f"❌ DuckDB not available: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas available: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas not available: {e}")
        return False
    
    try:
        from analytics import AnalyticsWorkspace
        print("✅ AnalyticsWorkspace import successful")
    except ImportError as e:
        print(f"❌ AnalyticsWorkspace import failed: {e}")
        return False
    
    try:
        from analytics import setup_workspace, migrate_workspace
        print("✅ Migration utilities import successful")
    except ImportError as e:
        print(f"❌ Migration utilities import failed: {e}")
        return False
    
    try:
        from analytics.exceptions import AnalyticsError, WorkspaceNotFoundError
        print("✅ Exception classes import successful")
    except ImportError as e:
        print(f"❌ Exception classes import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic workspace creation"""
    print("\nTesting basic functionality...")
    
    try:
        import tempfile
        from analytics import setup_workspace
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / 'test_workspace'
            
            # Try to create workspace
            workspace = setup_workspace(workspace_path)
            print("✅ Workspace creation successful")
            
            # Test basic query
            tables = workspace.describe()
            print(f"✅ Tables query successful: {len(tables)} tables found")
            
            # Test summary
            summary = workspace.summary()
            print(f"✅ Summary query successful: {summary}")
            
            workspace.close()
            return True
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Simple Analytics Implementation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed - check dependencies")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return False
    
    print("\n✅ All tests passed! Analytics implementation appears to be working.")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)