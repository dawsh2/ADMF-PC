"""
Event routing debugging and visualization tools.

This module provides tools for understanding and debugging event flow
in the system, including visualization and tracing capabilities.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .router import EventRouter


class EventFlowVisualizer:
    """Generate visual representations of event flow."""
    
    def __init__(self, router: EventRouter):
        """
        Initialize visualizer with router.
        
        Args:
            router: The event router to visualize
        """
        self.router = router
    
    def generate_mermaid_diagram(self) -> str:
        """
        Generate Mermaid diagram of event topology.
        
        Returns:
            Mermaid diagram code as string
        """
        topology = self.router.get_topology()
        diagram_lines = ["graph TD"]
        
        # Style definitions
        diagram_lines.extend([
            "    classDef dataContainer fill:#e8f4fd,stroke:#1976d2,stroke-width:2px",
            "    classDef indicatorContainer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
            "    classDef strategyContainer fill:#e8f5e9,stroke:#388e3c,stroke-width:2px",
            "    classDef riskContainer fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "    classDef executionContainer fill:#ffebee,stroke:#d32f2f,stroke-width:2px",
            ""
        ])
        
        # Add nodes
        node_styles = {
            'data': 'dataContainer',
            'indicator': 'indicatorContainer',
            'strategy': 'strategyContainer',
            'risk': 'riskContainer',
            'execution': 'executionContainer'
        }
        
        for node in topology['nodes']:
            node_id = node['id']
            # Extract role from node ID if possible
            role = 'default'
            for key in node_styles:
                if key in node_id.lower():
                    role = key
                    break
            
            # Node label with events
            label_parts = [node_id]
            if node['publishes']:
                label_parts.append(f"Publishes: {', '.join(node['publishes'])}")
            if node['subscribes']:
                label_parts.append(f"Subscribes: {', '.join(node['subscribes'])}")
            
            label = "<br/>".join(label_parts)
            diagram_lines.append(f'    {node_id}["{label}"]')
            
            # Apply style
            if role in node_styles:
                diagram_lines.append(f'    class {node_id} {node_styles[role]}')
        
        diagram_lines.append("")
        
        # Add edges with event labels
        for edge in topology['edges']:
            source = edge['source']
            target = edge['target']
            event = edge['event']
            
            diagram_lines.append(f'    {source} -->|{event}| {target}')
        
        return "\n".join(diagram_lines)
    
    def generate_dot_diagram(self) -> str:
        """
        Generate Graphviz DOT diagram of event topology.
        
        Returns:
            DOT diagram code as string
        """
        topology = self.router.get_topology()
        dot_lines = ['digraph EventFlow {']
        dot_lines.extend([
            '    rankdir=TB;',
            '    node [shape=box, style=rounded];',
            ''
        ])
        
        # Add nodes
        for node in topology['nodes']:
            node_id = node['id']
            label_parts = [node_id]
            
            if node['publishes']:
                label_parts.append(f"Pub: {', '.join(node['publishes'])}")
            if node['subscribes']:
                label_parts.append(f"Sub: {', '.join(node['subscribes'])}")
            
            label = "\\n".join(label_parts)
            
            # Color based on role
            color = "lightblue"
            if "indicator" in node_id.lower():
                color = "lavender"
            elif "strategy" in node_id.lower():
                color = "lightgreen"
            elif "risk" in node_id.lower():
                color = "lightyellow"
            elif "execution" in node_id.lower():
                color = "lightcoral"
            
            dot_lines.append(
                f'    "{node_id}" [label="{label}", fillcolor={color}, style=filled];'
            )
        
        dot_lines.append("")
        
        # Add edges
        for edge in topology['edges']:
            dot_lines.append(
                f'    "{edge["source"]}" -> "{edge["target"]}" '
                f'[label="{edge["event"]}"];'
            )
        
        dot_lines.append('}')
        return "\n".join(dot_lines)
    
    def save_visualization(
        self, 
        output_path: str, 
        format: str = "mermaid"
    ) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Path to save the visualization
            format: Format to use ("mermaid" or "dot")
        """
        if format == "mermaid":
            content = self.generate_mermaid_diagram()
        elif format == "dot":
            content = self.generate_dot_diagram()
        else:
            raise ValueError(f"Unknown format: {format}")
        
        Path(output_path).write_text(content)
        print(f"Visualization saved to {output_path}")


class EventTracer:
    """Trace and analyze event flow through the system."""
    
    def __init__(self, router: EventRouter):
        """
        Initialize tracer with router.
        
        Args:
            router: The event router to trace
        """
        self.router = router
        self._trace_events = []
        self._trace_enabled = False
    
    def start_tracing(self) -> None:
        """Start event tracing."""
        self._trace_enabled = True
        self.router.enable_debugging(True)
        self._trace_events.clear()
    
    def stop_tracing(self) -> None:
        """Stop event tracing."""
        self._trace_enabled = False
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get current trace events."""
        return self._trace_events.copy()
    
    def analyze_trace(self) -> Dict[str, Any]:
        """
        Analyze trace for patterns and issues.
        
        Returns:
            Analysis results including:
            - Event counts by type and source
            - Average routing times
            - Longest routing paths
            - Potential bottlenecks
        """
        if not self._trace_events:
            return {"error": "No trace events collected"}
        
        # Event counts
        events_by_type = {}
        events_by_source = {}
        routing_times = []
        
        for event in self._trace_events:
            event_type = event.get('event_type', 'unknown')
            source = event.get('source', 'unknown')
            routing_time = event.get('routing_time_ms', 0)
            
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            events_by_source[source] = events_by_source.get(source, 0) + 1
            routing_times.append(routing_time)
        
        # Calculate statistics
        avg_routing_time = sum(routing_times) / len(routing_times) if routing_times else 0
        max_routing_time = max(routing_times) if routing_times else 0
        
        # Find potential bottlenecks (sources with high event rates)
        bottlenecks = []
        total_events = len(self._trace_events)
        for source, count in events_by_source.items():
            percentage = (count / total_events) * 100
            if percentage > 30:  # More than 30% of events from one source
                bottlenecks.append({
                    "source": source,
                    "event_count": count,
                    "percentage": percentage
                })
        
        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "events_by_source": events_by_source,
            "routing_performance": {
                "average_ms": avg_routing_time,
                "max_ms": max_routing_time,
                "total_routing_time_ms": sum(routing_times)
            },
            "potential_bottlenecks": bottlenecks
        }
    
    def save_trace(self, output_path: str) -> None:
        """
        Save trace to JSON file.
        
        Args:
            output_path: Path to save the trace
        """
        trace_data = {
            "trace_start": self._trace_events[0]['timestamp'].isoformat() if self._trace_events else None,
            "trace_end": self._trace_events[-1]['timestamp'].isoformat() if self._trace_events else None,
            "event_count": len(self._trace_events),
            "events": [
                {
                    **event,
                    'timestamp': event['timestamp'].isoformat()
                }
                for event in self._trace_events
            ],
            "analysis": self.analyze_trace()
        }
        
        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"Trace saved to {output_path}")


class EventRoutingMonitor:
    """Real-time monitoring of event routing."""
    
    def __init__(self, router: EventRouter):
        """
        Initialize monitor with router.
        
        Args:
            router: The event router to monitor
        """
        self.router = router
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current routing status.
        
        Returns:
            Current status including metrics and health
        """
        metrics = self.router.get_metrics()
        topology = self.router.get_topology()
        
        # Calculate health indicators
        total_containers = len(set(
            node['id'] for node in topology['nodes']
        ))
        total_connections = len(topology['edges'])
        
        # Check for issues
        issues = []
        
        # High failure rate
        if metrics['total_events'] > 0:
            failure_rate = metrics['delivery_failures'] / metrics['total_events']
            if failure_rate > 0.05:  # More than 5% failures
                issues.append({
                    "type": "high_failure_rate",
                    "severity": "warning",
                    "message": f"Delivery failure rate is {failure_rate:.1%}"
                })
        
        # Orphaned containers
        orphaned = []
        for node in topology['nodes']:
            if not node['publishes'] and not node['subscribes']:
                orphaned.append(node['id'])
        
        if orphaned:
            issues.append({
                "type": "orphaned_containers",
                "severity": "info",
                "message": f"Containers with no connections: {', '.join(orphaned)}"
            })
        
        return {
            "status": "healthy" if not issues else "degraded",
            "metrics": metrics,
            "topology_summary": {
                "total_containers": total_containers,
                "total_connections": total_connections,
                "event_types": len(set(
                    edge['event'] for edge in topology['edges']
                ))
            },
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable status report.
        
        Returns:
            Formatted status report
        """
        status = self.get_status()
        
        lines = [
            "Event Routing Status Report",
            "=" * 40,
            f"Status: {status['status'].upper()}",
            f"Timestamp: {status['timestamp']}",
            "",
            "Metrics:",
            f"  Total Events Routed: {status['metrics']['total_events']:,}",
            f"  Delivery Failures: {status['metrics']['delivery_failures']:,}",
            f"  Active Subscriptions: {status['metrics']['active_subscriptions']:,}",
            f"  Avg Routing Latency: {status['metrics']['average_routing_latency_ms']:.2f}ms",
            "",
            "Topology:",
            f"  Total Containers: {status['topology_summary']['total_containers']}",
            f"  Total Connections: {status['topology_summary']['total_connections']}",
            f"  Event Types: {status['topology_summary']['event_types']}",
            ""
        ]
        
        if status['issues']:
            lines.extend([
                "Issues Detected:",
                "-" * 20
            ])
            for issue in status['issues']:
                lines.append(
                    f"  [{issue['severity'].upper()}] {issue['message']}"
                )
        else:
            lines.append("No issues detected.")
        
        return "\n".join(lines)