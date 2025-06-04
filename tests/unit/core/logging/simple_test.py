"""
Simple Test for Logging System v3
Quick validation of core functionality
"""

import tempfile
import json
from pathlib import Path

def test_basic_logging():
    """Test basic logging functionality."""
    print("🧪 Testing basic logging...")
    
    from .container_logger import ContainerLogger
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = ContainerLogger("test_container", "test_component", base_log_dir=temp_dir)
        
        # Test basic logging
        logger.info("Test message", key="value")
        logger.error("Test error", error_code=500)
        
        # Check log file was created
        log_file = Path(temp_dir) / "containers" / "test_container" / "test_component.log"
        assert log_file.exists(), "Log file should exist"
        
        # Check log content
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 2, "Should have 2 log entries"
            
            entry = json.loads(lines[0])
            assert entry['container_id'] == "test_container"
            assert entry['component_name'] == "test_component"
            assert entry['message'] == "Test message"
        
        logger.close()
        print("✅ Basic logging works!")


def test_component_enhancement():
    """Test adding logging to components."""
    print("🧪 Testing component enhancement...")
    
    from .capabilities import add_logging_to_any_component
    
    class TestClass:
        def __init__(self):
            self.value = 42
    
    obj = TestClass()
    enhanced = add_logging_to_any_component(obj, "test_container", "test_component")
    
    # Check that logging methods were added
    assert hasattr(enhanced, 'log_info'), "Should have log_info method"
    assert hasattr(enhanced, 'log_error'), "Should have log_error method"
    assert hasattr(enhanced, 'set_correlation_id'), "Should have correlation methods"
    
    # Test that original functionality is preserved
    assert enhanced.value == 42, "Original attributes should be preserved"
    
    print("✅ Component enhancement works!")


def test_log_manager():
    """Test log manager functionality."""
    print("🧪 Testing log manager...")
    
    from .log_manager import LogManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_manager = LogManager("test_coordinator", temp_dir)
        
        # Test container registration
        registry = log_manager.register_container("test_container")
        assert "test_container" in log_manager.active_containers
        
        # Test logger creation
        logger = registry.create_component_logger("test_component")
        logger.info("Test log from managed container")
        
        # Test summary
        summary = log_manager.get_log_summary()
        assert summary['coordinator_id'] == "test_coordinator"
        assert summary['active_containers'] == 1
        
        # Test cleanup
        log_manager.unregister_container("test_container")
        assert "test_container" not in log_manager.active_containers
        
        print("✅ Log manager works!")


def main():
    """Run simple tests."""
    print("🚀 Running simple Logging System v3 tests...")
    
    try:
        test_basic_logging()
        test_component_enhancement()
        test_log_manager()
        
        print("\n🎉 All simple tests passed!")
        print("✅ Logging System v3 core functionality is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()