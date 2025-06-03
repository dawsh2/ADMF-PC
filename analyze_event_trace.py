"""
Quick script to analyze traced events from a backtest run

This script demonstrates how to analyze events that have been traced
during a backtest execution.
"""

from src.core.events.tracing.event_tracer import EventTracer
from collections import defaultdict
from typing import List, Dict, Any


def analyze_latest_run(tracer: EventTracer):
    """Analyze events from the tracer"""
    print(f"\nEvent Trace Analysis")
    print(f"Correlation ID: {tracer.correlation_id}")
    print(f"Total Events: {len(tracer.traced_events)}")
    
    # Count by type
    event_types = defaultdict(int)
    for event in tracer.traced_events:
        event_types[event.event_type] += 1
    
    print("\nEvent Counts by Type:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    
    # Find signal->fill chains
    signals = [e for e in tracer.traced_events if e.event_type == "SIGNAL"]
    print(f"\nSignals Generated: {len(signals)}")
    
    # Trace each signal
    complete_chains = 0
    for signal in signals:
        chain = trace_forward(signal, tracer)
        if any(e.event_type == "FILL" for e in chain):
            complete_chains += 1
            print(f"\nSignal {signal.event_id} -> Fill (chain length: {len(chain)})")
            for event in chain:
                print(f"  -> {event.event_type} [{event.event_id}] from {event.source_container}")
    
    print(f"\nComplete signal->fill chains: {complete_chains}/{len(signals)}")
    
    # Analyze latencies
    latency_stats = tracer.calculate_latency_stats()
    if latency_stats:
        print("\nLatency Statistics by Event Type:")
        for event_type, stats in latency_stats.items():
            print(f"  {event_type}:")
            print(f"    Average: {stats['avg_ms']:.2f}ms")
            print(f"    Min: {stats['min_ms']:.2f}ms")
            print(f"    Max: {stats['max_ms']:.2f}ms")
    
    # Find longest causation chains
    print("\nLongest Event Chains:")
    chain_lengths = []
    for event in tracer.traced_events:
        chain = tracer.trace_causation_chain(event.event_id)
        chain_lengths.append((len(chain), event))
    
    chain_lengths.sort(reverse=True)
    for length, event in chain_lengths[:5]:
        print(f"  {event.event_type} [{event.event_id}]: {length} events in chain")


def trace_forward(event, tracer) -> List[Any]:
    """Trace events forward from given event"""
    chain = [event]
    
    # Find all events caused by this one
    for traced in tracer.traced_events:
        if traced.causation_id == event.event_id:
            chain.extend(trace_forward(traced, tracer))
            
    return chain


def analyze_by_container(tracer: EventTracer):
    """Analyze events by source container"""
    print("\nEvent Analysis by Container:")
    
    container_stats = defaultdict(lambda: defaultdict(int))
    
    for event in tracer.traced_events:
        container = event.source_container
        container_stats[container][event.event_type] += 1
    
    for container, event_counts in sorted(container_stats.items()):
        total = sum(event_counts.values())
        print(f"\n{container}: {total} events")
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")


def find_patterns(tracer: EventTracer):
    """Look for common patterns in event sequences"""
    print("\nCommon Event Patterns:")
    
    # Look for SIGNAL -> ORDER -> FILL patterns
    pattern_count = 0
    for i, event in enumerate(tracer.traced_events):
        if event.event_type == "SIGNAL":
            # Look for ORDER within next 10 events
            for j in range(i+1, min(i+10, len(tracer.traced_events))):
                if (tracer.traced_events[j].event_type == "ORDER" and 
                    tracer.traced_events[j].causation_id == event.event_id):
                    # Look for FILL
                    for k in range(j+1, min(j+10, len(tracer.traced_events))):
                        if (tracer.traced_events[k].event_type == "FILL" and
                            tracer.traced_events[k].causation_id == tracer.traced_events[j].event_id):
                            pattern_count += 1
                            break
                    break
    
    print(f"  SIGNAL -> ORDER -> FILL patterns: {pattern_count}")


if __name__ == "__main__":
    # This will be integrated with backtest results
    print("Event Trace Analysis Tool")
    print("========================")
    print("\nRun a backtest first, then we'll analyze its events")
    print("\nExample usage after backtest:")
    print("  tracer = backtest_container.event_tracer")
    print("  analyze_latest_run(tracer)")
    print("  analyze_by_container(tracer)")
    print("  find_patterns(tracer)")