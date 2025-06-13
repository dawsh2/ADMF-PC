#!/usr/bin/env python3
"""
Simpler test demonstrating the signal flow using TopologyBuilder.
"""

import logging
import json
from pathlib import Path
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_signal_flow():
    """Test signal generation flow using TopologyBuilder."""
    
    print("\n" + "="*80)
    print("SIGNAL FLOW TEST USING TOPOLOGY BUILDER")
    print("="*80)
    
    # 1. BUILD TOPOLOGY
    print("\n1. TOPOLOGY CREATION")
    print("-" * 40)
    
    from src.core.coordinator.topology import TopologyBuilder
    from src.core.coordinator.coordinator import Coordinator
    
    builder = TopologyBuilder()
    
    # Define topology for signal generation
    topology_definition = {
        'mode': 'signal_generation',
        'config': {
            'symbols': ['SPY'],
            'timeframes': ['1D'],  # Add timeframe
            'max_bars': 30,
            'strategies': [{
                'name': 'ma_crossover',
                'type': 'ma_crossover',
                'params': {
                    'fast_period': 5,
                    'slow_period': 20
                }
            }],
            'initial_capital': 100000,
            'data_source': 'file',
            'data_path': 'data/SPY_1d.csv'
        },
        'tracing_config': {
            'enable_event_tracing': True,
            'trace_settings': {
                'storage_backend': 'hierarchical',
                'enable_console_output': True,
                'console_filter': ['SIGNAL'],
                'container_settings': {
                    'portfolio*': {'enabled': True}
                }
            }
        },
        'metadata': {
            'workflow_id': 'signal_flow_test',
            'phase_name': 'signal_generation'
        }
    }
    
    # Build topology
    topology = builder.build_topology(topology_definition)
    print(f"\nTopology built:")
    print(f"  - Name: {topology['name']}")
    print(f"  - Containers: {len(topology['containers'])}")
    print(f"  - Components: {len(topology['components'])}")
    
    # Show containers
    print("\nContainers created:")
    for container_id, container in topology['containers'].items():
        if hasattr(container, 'container_type'):
            print(f"  - {container_id} ({container.container_type})")
        else:
            print(f"  - {container_id}")
    
    # 2. CHECK FEATURE INFERENCE
    print("\n2. FEATURE INFERENCE")
    print("-" * 40)
    
    # Check strategy containers for inferred features
    for container_id, container in topology['containers'].items():
        if hasattr(container, 'container_type') and container.container_type == 'strategy':
            config = container.config.config
            if 'feature_configs' in config:
                print(f"\nFeatures for {container_id}:")
                for feat_id, feat_cfg in config['feature_configs'].items():
                    print(f"  - {feat_id}: {feat_cfg}")
    
    # 3. EXECUTE TOPOLOGY USING COORDINATOR
    print("\n3. TOPOLOGY EXECUTION")
    print("-" * 40)
    
    from src.core.coordinator.coordinator import Coordinator
    
    # Create coordinator
    coordinator = Coordinator()
    print("Created coordinator")
    
    # Configure topology execution
    config = {
        'symbols': ['SPY'],
        'timeframes': ['1D'],
        'max_bars': 30,
        'strategies': [{
            'name': 'ma_crossover',
            'type': 'ma_crossover',
            'params': {
                'fast_period': 5,
                'slow_period': 20
            }
        }],
        'initial_capital': 100000,
        'data_source': 'file',
        'data_path': 'data/SPY_1d.csv',
        'execution': {
            'max_duration': 5.0,  # seconds
            'enable_event_tracing': True,
            'trace_settings': {
                'storage_backend': 'hierarchical',
                'enable_console_output': True,
                'console_filter': ['SIGNAL'],
                'container_settings': {
                    'portfolio*': {'enabled': True}
                }
            }
        },
        'metadata': {
            'workflow_id': 'signal_flow_test'
        }
    }
    
    # 4. EXECUTE SIGNAL GENERATION
    print("\n4. SIGNAL GENERATION EXECUTION")
    print("-" * 40)
    
    print("Executing signal_generation topology...")
    result = coordinator.run_topology('signal_generation', config)
    
    # 5. CHECK EXECUTION RESULTS
    print("\n5. EXECUTION RESULTS")
    print("-" * 40)
    
    if result.get('success'):
        print("✅ Topology execution completed successfully")
        print(f"Execution ID: {result.get('execution_id')}")
        print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
        
        # Print container metrics
        metrics = result.get('metrics', {})
        if metrics:
            print("\nContainer Metrics:")
            for container_name, container_metrics in metrics.items():
                print(f"  {container_name}:")
                for metric_name, metric_value in container_metrics.items():
                    print(f"    - {metric_name}: {metric_value}")
        
        # Print outputs
        outputs = result.get('outputs', {})
        if outputs:
            print("\nOutputs:")
            for output_name, output_data in outputs.items():
                print(f"  {output_name}: {output_data}")
                
    else:
        print("❌ Topology execution failed")
        errors = result.get('errors', [])
        for error in errors:
            print(f"  Error: {error}")
    
    # 7. CHECK STORAGE
    print("\n7. SIGNAL STORAGE CHECK")
    print("-" * 40)
    
    workspace_path = Path('workspaces/signal_flow_test')
    if workspace_path.exists():
        for container_dir in sorted(workspace_path.iterdir()):
            if container_dir.is_dir():
                print(f"\nContainer: {container_dir.name}")
                
                # Check for event files
                events_file = container_dir / 'events.jsonl'
                if events_file.exists():
                    with open(events_file, 'r') as f:
                        events = [json.loads(line) for line in f]
                    
                    # Count event types
                    event_counts = {}
                    signal_events = []
                    for event in events:
                        event_type = event.get('event_type', 'UNKNOWN')
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                        
                        if event_type == 'SIGNAL':
                            signal_events.append(event)
                    
                    print(f"  Events stored: {len(events)}")
                    for event_type, count in sorted(event_counts.items()):
                        print(f"    - {event_type}: {count}")
                    
                    # Show sample signals
                    if signal_events:
                        print(f"\n  Sample signals:")
                        for i, signal in enumerate(signal_events[:3]):
                            payload = signal.get('payload', {})
                            print(f"    {i+1}. {payload.get('direction')} signal, "
                                  f"strength: {payload.get('strength', 0):.6f}, "
                                  f"strategy: {payload.get('strategy_id')}")
    
    # 8. PERFORMANCE ANALYSIS
    print("\n8. SIGNAL PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Try to load signals and analyze
    try:
        from src.analytics.mining.two_layer_mining import TwoLayerMining
        
        mining = TwoLayerMining(base_dir='workspaces')
        signals_df = mining.load_signals('signal_flow_test')
        
        if signals_df is not None and not signals_df.empty:
            print(f"\nLoaded {len(signals_df)} signals")
            print("\nSignal Analysis:")
            print(f"  - Direction distribution:")
            for direction, count in signals_df['direction'].value_counts().items():
                print(f"    - {direction}: {count}")
            print(f"  - Average strength: {signals_df['strength'].mean():.6f}")
            print(f"  - Strength range: [{signals_df['strength'].min():.6f}, {signals_df['strength'].max():.6f}]")
        else:
            print("No signals found for analysis")
    except Exception as e:
        print(f"Could not perform signal analysis: {e}")
    
    # 9. FINAL SUMMARY
    print("\n9. FINAL SUMMARY")
    print("-" * 40)
    
    # Coordinator automatically handles cleanup
    success = result.get('success', False)
    
    print("\nFlow Summary:")
    print(f"  ✓ Topology built with feature inference")
    print(f"  ✓ Signal generation executed")
    print(f"  ✓ Results: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print(f"  ✓ Duration: {result.get('duration_seconds', 0):.2f}s")
        if metrics:
            total_signals = sum(
                m.get('signals_generated', 0) 
                for m in metrics.values() 
                if isinstance(m, dict)
            )
            print(f"  ✓ Signals generated: {total_signals}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return success

if __name__ == '__main__':
    success = test_signal_flow()
    exit(0 if success else 1)