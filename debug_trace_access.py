#!/usr/bin/env python3
"""
Debug script to access event trace data after workflow execution.
"""

import sys
sys.path.append('.')

from src.core.coordinator.coordinator import Coordinator
from src.core.cli.args import load_yaml_config

def main():
    print("🔍 Debugging Event Trace Access")
    
    # Load config with limited bars
    config = load_yaml_config('config/test_bar_streaming_console.yaml')
    config['data']['max_bars'] = 3  # Force limit
    
    print(f"📋 Config loaded: {config['name']}")
    print(f"📊 Max bars: {config['data'].get('max_bars')}")
    
    # Create coordinator  
    coordinator = Coordinator()
    
    # Run workflow
    print("🚀 Running workflow...")
    result = coordinator.run_workflow(config)
    
    print(f"✅ Workflow complete: {result.get('success')}")
    
    # Try to access trace data from different places
    print("\n🔍 Looking for trace data...")
    
    # Check if coordinator has access to containers
    if hasattr(coordinator, 'topology_builder'):
        topology_builder = coordinator.topology_builder
        print(f"📦 Topology builder: {topology_builder}")
        
        if hasattr(topology_builder, 'containers'):
            containers = topology_builder.containers
            print(f"📦 Found {len(containers)} containers")
            
            for container_id, container in containers.items():
                print(f"\n📦 Container: {container_id}")
                
                # Check if container has event bus
                if hasattr(container, 'event_bus'):
                    bus = container.event_bus
                    print(f"  🚌 Event bus: {bus.bus_id}")
                    
                    # Check if bus has tracer
                    if hasattr(bus, '_tracer'):
                        tracer = bus._tracer
                        print(f"  📊 Tracer found!")
                        
                        # Get trace summary
                        summary = tracer.get_summary()
                        print(f"  📈 Events traced: {summary.get('events_traced', 0)}")
                        print(f"  📈 Event counts: {summary.get('event_counts', {})}")
                        
                        # Get recent events
                        if hasattr(tracer, 'recent_events'):
                            events = list(tracer.recent_events)
                            print(f"  📝 Recent events: {len(events)}")
                            
                            for i, event in enumerate(events[:5]):  # Show first 5
                                print(f"    {i+1}. {event.event_type}: {event.payload}")
                    else:
                        print(f"  ❌ No tracer found")
                else:
                    print(f"  ❌ No event bus found")
    
    print("\n✅ Debug complete")

if __name__ == '__main__':
    main()