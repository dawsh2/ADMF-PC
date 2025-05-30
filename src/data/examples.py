"""
Usage examples for the new Protocol+Composition data module.

Shows how to use the data module without ANY inheritance.
"""

import asyncio
from datetime import datetime, timedelta

# Import the new implementation
from . import (
    create_enhanced_data_handler,
    create_enhanced_data_loader,
    create_enhanced_streamer,
    apply_capabilities,
    SimpleHistoricalDataHandler,
    SimpleCSVLoader
)


async def example_basic_usage():
    """Basic usage example - simple data handler."""
    print("=== Basic Data Handler Example ===")
    
    # Create simple data handler - NO INHERITANCE!
    handler = SimpleHistoricalDataHandler(
        handler_id="basic_handler",
        data_dir="data"
    )
    
    # Load data
    success = handler.load_data(['AAPL', 'GOOGL'])
    print(f"Data loaded: {success}")
    
    # Stream some bars
    handler.start()
    for i in range(5):
        if handler.update_bars():
            print(f"Bar {i+1} emitted")
        else:
            break
    
    handler.stop()


async def example_enhanced_handler():
    """Enhanced handler with capabilities."""
    print("\n=== Enhanced Handler Example ===")
    
    # Create handler with capabilities
    handler = create_enhanced_data_handler(
        handler_type='historical',
        handler_id='enhanced_handler',
        data_dir='data',
        capabilities=[
            'logging',
            'monitoring', 
            'events',
            'validation',
            'splitting'
        ],
        # Capability-specific config
        logging={'logger_name': 'data.enhanced'},
        monitoring={'track_performance': ['load_data', 'update_bars']},
        events={'auto_emit': ['data_loaded', 'bar_updated']},
        validation={'validate_on_load': True},
        splitting={'default_ratio': 0.8, 'auto_split': True}
    )
    
    # Set up event handler
    def on_data_loaded(event):
        print(f"Event: {event['type']} - {event['payload']}")
    
    handler.subscribe_to_event('data_loaded', on_data_loaded)
    
    # Load data (automatically validates and splits)
    success = handler.load_data(['AAPL'])
    print(f"Enhanced data loaded: {success}")
    
    # Check metrics
    metrics = handler.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Use train split
    handler.set_active_split('train')
    split_info = handler.get_split_info()
    print(f"Split info: {split_info}")
    
    # Stream training data
    handler.start()
    bars_processed = 0
    while handler.update_bars() and bars_processed < 5:
        bars_processed += 1
    
    print(f"Processed {bars_processed} training bars")
    handler.stop()


async def example_custom_loader():
    """Custom loader with capabilities."""
    print("\n=== Custom Loader Example ===")
    
    # Create simple loader
    loader = SimpleCSVLoader(data_dir="data")
    
    # Add capabilities through composition
    enhanced_loader = apply_capabilities(loader, [
        'logging',
        'validation',
        'memory_optimization'
    ], {
        'logging': {'logger_name': 'data.loader'},
        'validation': {'validate_on_load': True},
        'memory_optimization': {'auto_optimize': True}
    })
    
    # Load and validate data
    try:
        df = enhanced_loader.load('AAPL')
        print(f"Loaded {len(df)} rows")
        
        # Check memory usage
        memory_info = enhanced_loader.get_memory_usage()
        print(f"Memory usage: {memory_info}")
        
        # Optimize memory
        optimization_results = enhanced_loader.optimize_memory()
        print(f"Memory optimization: {optimization_results}")
        
    except Exception as e:
        print(f"Loading failed: {e}")


async def example_streaming():
    """Streaming data example."""
    print("\n=== Streaming Example ===")
    
    # Create historical streamer
    streamer = create_enhanced_streamer(
        streamer_type='historical',
        config={
            'data_dir': 'data',
            'symbols': ['AAPL'],
            'max_bars': 10
        },
        capabilities=['logging', 'events'],
        logging={'logger_name': 'data.streamer'}
    )
    
    # Set up event handler
    def on_bars_streamed(event):
        payload = event['payload']
        print(f"Streamed {payload['symbol_count']} symbols at {payload['timestamp']}")
    
    streamer.subscribe_to_event('bars_streamed', on_bars_streamed)
    
    # Stream data
    bar_count = 0
    async for timestamp, bars in streamer.stream_bars():
        print(f"Timestamp: {timestamp}, Symbols: {list(bars.keys())}")
        bar_count += 1
        if bar_count >= 5:  # Limit for demo
            break


async def example_protocol_compliance():
    """Show how components implement protocols through duck typing."""
    print("\n=== Protocol Compliance Example ===")
    
    # Create various components
    handler = SimpleHistoricalDataHandler()
    loader = SimpleCSVLoader()
    
    # Check protocol compliance through duck typing
    from .protocols import DataProvider, DataLoader, BarStreamer
    
    def check_protocol_compliance(obj, protocol, protocol_name):
        """Check if object implements protocol."""
        try:
            # Check if object has required methods
            required_methods = [method for method in dir(protocol) 
                              if not method.startswith('_')]
            
            has_methods = []
            for method in required_methods:
                if hasattr(obj, method):
                    has_methods.append(method)
            
            print(f"{obj.__class__.__name__} implements {protocol_name}: {len(has_methods)} methods")
            
            # Try isinstance check (works with @runtime_checkable)
            is_instance = isinstance(obj, protocol)
            print(f"  isinstance check: {is_instance}")
            
        except Exception as e:
            print(f"  Error checking {protocol_name}: {e}")
    
    # Check protocols
    check_protocol_compliance(handler, DataProvider, "DataProvider")
    check_protocol_compliance(handler, BarStreamer, "BarStreamer") 
    check_protocol_compliance(loader, DataLoader, "DataLoader")


async def example_composition_vs_inheritance():
    """Show composition vs inheritance approach."""
    print("\n=== Composition vs Inheritance Example ===")
    
    print("❌ OLD WAY (Inheritance):")
    print("class DataHandler(Component, Lifecycle, EventCapable, ABC):")
    print("    # Complex inheritance hierarchy")
    print("    # Forced to inherit unwanted functionality")
    print("    # Hard to test and mock")
    print("")
    
    print("✅ NEW WAY (Composition):")
    
    # Start with simple class
    handler = SimpleHistoricalDataHandler(handler_id="demo")
    print(f"1. Simple class: {handler.__class__.__name__}")
    print(f"   Methods: {[m for m in dir(handler) if not m.startswith('_')][:5]}...")
    
    # Add capabilities one by one
    handler = apply_capabilities(handler, ['logging'], {
        'logging': {'logger_name': 'demo'}
    })
    print(f"2. + Logging: Now has log_info, log_error methods")
    
    handler = apply_capabilities(handler, ['monitoring'], {
        'monitoring': {'track_performance': ['load_data']}
    })
    print(f"3. + Monitoring: Now has get_metrics, record_metric methods")
    
    handler = apply_capabilities(handler, ['events'], {
        'events': {'auto_emit': ['data_loaded']}
    })
    print(f"4. + Events: Now has emit_event, subscribe_to_event methods")
    
    print("\n✅ Result: Full functionality through composition, zero inheritance!")


async def main():
    """Run all examples."""
    print("ADMF-PC Data Module - Protocol+Composition Examples")
    print("=" * 50)
    
    await example_basic_usage()
    await example_enhanced_handler()
    await example_custom_loader()
    await example_streaming()
    await example_protocol_compliance()
    await example_composition_vs_inheritance()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("✅ Zero inheritance used anywhere!")
    print("✅ All functionality through Protocol+Composition!")


if __name__ == "__main__":
    asyncio.run(main())
