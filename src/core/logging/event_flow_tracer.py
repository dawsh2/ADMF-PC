"""
EventFlowTracer - Cross-Container Event Flow Tracking for Logging System v3
Composable component for tracing events across container boundaries
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .log_writer import LogWriter
from .correlation_tracker import CorrelationTracker
from .protocols import EventTrackable


@dataclass
class FlowEvent:
    """Represents a single event in a flow trace."""
    timestamp: str
    event_id: str
    event_type: str
    source: str
    target: str
    flow_type: str  # 'internal' or 'external'
    container_id: Optional[str] = None
    tier: Optional[str] = None
    latency_ms: Optional[float] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EventFlowTracer:
    """
    Event flow tracer built through composition.
    Tracks events across container boundaries.
    
    This component provides comprehensive event flow tracking for debugging
    multi-container architectures and performance analysis.
    """
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs"):
        """
        Initialize event flow tracer.
        
        Args:
            coordinator_id: Unique coordinator identifier
            base_log_dir: Base directory for log files
        """
        self.coordinator_id = coordinator_id
        
        # Compose with log writer
        flow_log_path = Path(base_log_dir) / "flows" / f"{coordinator_id}_event_flows.log"
        correlation_log_path = Path(base_log_dir) / "correlations" / f"{coordinator_id}_correlations.log"
        
        self.flow_writer = LogWriter(flow_log_path)
        self.correlation_writer = LogWriter(correlation_log_path)
        
        # Compose with correlation tracker
        self.correlation_tracker = CorrelationTracker()
        
        # Performance tracking
        self._events_traced = 0
        self._start_time = datetime.utcnow()
    
    # Implement EventTrackable protocol
    def trace_event(self, event_id: str, source: str, target: str, **context) -> None:
        """
        Main event tracing method - implements EventTrackable protocol.
        
        Args:
            event_id: Unique event identifier
            source: Source of the event
            target: Target of the event
            **context: Additional event context
        """
        timestamp = datetime.utcnow()
        
        flow_event = FlowEvent(
            timestamp=timestamp.isoformat() + 'Z',
            event_id=event_id,
            event_type=context.get('event_type', 'unknown'),
            source=source,
            target=target,
            flow_type=context.get('flow_type', 'unknown'),
            container_id=context.get('container_id'),
            tier=context.get('tier'),
            latency_ms=context.get('latency_ms'),
            correlation_id=self.correlation_tracker.get_correlation_id(),
            metadata=context.get('metadata', {})
        )
        
        # Write to flow log
        self.flow_writer.write(asdict(flow_event))
        
        # Update correlation tracking
        if flow_event.correlation_id:
            self.correlation_tracker.track_event(
                event_id, 
                f"{source}->{target}"
            )
            
            # Write correlation entry
            correlation_entry = {
                'timestamp': flow_event.timestamp,
                'correlation_id': flow_event.correlation_id,
                'event_id': event_id,
                'source': source,
                'target': target,
                'event_type': flow_event.event_type,
                'tier': flow_event.tier
            }
            self.correlation_writer.write(correlation_entry)
        
        self._events_traced += 1
    
    def trace_internal_event(self, container_id: str, event_id: str,
                           source: str, target: str, **context):
        """
        Trace internal container events.
        
        Args:
            container_id: Container where event occurs
            event_id: Unique event identifier
            source: Source component
            target: Target component
            **context: Additional context
        """
        self.trace_event(
            event_id=event_id,
            source=source,
            target=target,
            flow_type='internal',
            container_id=container_id,
            **context
        )
    
    def trace_external_event(self, event_id: str, source_container: str,
                           target_container: str, tier: str, **context):
        """
        Trace cross-container events.
        
        Args:
            event_id: Unique event identifier
            source_container: Source container
            target_container: Target container
            tier: Communication tier (fast, standard, reliable)
            **context: Additional context
        """
        self.trace_event(
            event_id=event_id,
            source=source_container,
            target=target_container,
            flow_type='external',
            tier=tier,
            **context
        )
    
    def trace_signal_flow(self, signal_id: str, container_path: List[str], **context):
        """
        Trace a signal through multiple containers.
        
        Args:
            signal_id: Unique signal identifier
            container_path: List of containers in the signal path
            **context: Additional context
        """
        correlation_id = f"signal_{signal_id}"
        
        from .correlation_tracker import CorrelationContext
        with CorrelationContext(self.correlation_tracker, correlation_id):
            for i in range(len(container_path) - 1):
                source = container_path[i]
                target = container_path[i + 1]
                
                self.trace_external_event(
                    event_id=f"{signal_id}_step_{i}",
                    source_container=source,
                    target_container=target,
                    tier="standard",
                    signal_id=signal_id,
                    step=i,
                    total_steps=len(container_path) - 1,
                    **context
                )
    
    def get_event_chain(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Get full event chain for a correlation ID.
        
        Args:
            correlation_id: Correlation ID to trace
            
        Returns:
            List of events in the correlation chain
        """
        # Read correlation log to find matching events
        events = []
        correlation_log_path = Path(self.flow_writer.log_file).parent.parent / "correlations" / f"{self.coordinator_id}_correlations.log"
        
        if correlation_log_path.exists():
            try:
                with open(correlation_log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get('correlation_id') == correlation_id:
                                events.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        
        # Sort by timestamp
        events.sort(key=lambda x: x.get('timestamp', ''))
        return events
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """
        Get event flow tracing statistics.
        
        Returns:
            Dictionary with tracing statistics
        """
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            'coordinator_id': self.coordinator_id,
            'events_traced': self._events_traced,
            'uptime_seconds': uptime,
            'events_per_second': self._events_traced / uptime if uptime > 0 else 0,
            'correlation_stats': self.correlation_tracker.get_statistics(),
            'flow_writer_metrics': self.flow_writer.get_metrics(),
            'correlation_writer_metrics': self.correlation_writer.get_metrics()
        }
    
    def analyze_container_communication(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """
        Analyze container communication patterns.
        
        Args:
            time_window_minutes: Time window for analysis
            
        Returns:
            Communication analysis results
        """
        cutoff_time = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        
        # Read recent flow events
        recent_flows = []
        flow_log_path = self.flow_writer.log_file
        
        if flow_log_path.exists():
            try:
                with open(flow_log_path, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            event_time = datetime.fromisoformat(
                                event['timestamp'].replace('Z', '+00:00')
                            ).timestamp()
                            
                            if event_time > cutoff_time:
                                recent_flows.append(event)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except Exception:
                pass
        
        # Analyze communication patterns
        communication_matrix = {}
        tier_usage = {'fast': 0, 'standard': 0, 'reliable': 0}
        event_types = {}
        
        for flow in recent_flows:
            source = flow.get('source', 'unknown')
            target = flow.get('target', 'unknown')
            tier = flow.get('tier', 'unknown')
            event_type = flow.get('event_type', 'unknown')
            
            # Build communication matrix
            if source not in communication_matrix:
                communication_matrix[source] = {}
            if target not in communication_matrix[source]:
                communication_matrix[source][target] = 0
            communication_matrix[source][target] += 1
            
            # Track tier usage
            if tier in tier_usage:
                tier_usage[tier] += 1
            
            # Track event types
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'time_window_minutes': time_window_minutes,
            'total_events': len(recent_flows),
            'communication_matrix': communication_matrix,
            'tier_usage': tier_usage,
            'event_types': event_types,
            'most_active_source': max(communication_matrix.keys(), 
                                    key=lambda x: sum(communication_matrix[x].values()))
                                    if communication_matrix else None,
            'most_used_tier': max(tier_usage.keys(), key=tier_usage.get) if tier_usage else None,
            'most_common_event_type': max(event_types.keys(), key=event_types.get) if event_types else None
        }
    
    def close(self):
        """Close the event flow tracer and cleanup resources."""
        self.flow_writer.close()
        self.correlation_writer.close()


class ContainerDebugger:
    """
    Debugging tools composed from smaller components.
    No inheritance - just protocols!
    
    This debugger provides comprehensive debugging capabilities for
    multi-container architectures built through composition.
    """
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs"):
        """
        Initialize container debugger.
        
        Args:
            coordinator_id: Unique coordinator identifier
            base_log_dir: Base directory for log files
        """
        self.coordinator_id = coordinator_id
        self.base_log_dir = Path(base_log_dir)
        
        # Compose with various components
        self.flow_tracer = EventFlowTracer(coordinator_id, base_log_dir)
        self.container_loggers: Dict[str, Any] = {}
    
    def create_container_logger(self, container_id: str, component_name: str):
        """Factory method for container loggers."""
        from .container_logger import ContainerLogger
        
        logger = ContainerLogger(container_id, component_name, base_log_dir=str(self.base_log_dir))
        logger_key = f"{container_id}.{component_name}"
        self.container_loggers[logger_key] = logger
        return logger
    
    def trace_signal_flow(self, start_event_id: str) -> List[Dict]:
        """Trace a signal from start event through entire system."""
        return self.flow_tracer.get_event_chain(start_event_id)
    
    def debug_container_isolation(self, container_id: str) -> Dict:
        """Debug what's happening inside a specific container."""
        container_logs = self._read_container_logs(container_id)
        
        # Analyze logs
        error_logs = [log for log in container_logs if log.get('level') == 'ERROR']
        warning_logs = [log for log in container_logs if log.get('level') == 'WARNING']
        
        return {
            'container_id': container_id,
            'log_count': len(container_logs),
            'error_count': len(error_logs),
            'warning_count': len(warning_logs),
            'components': list(set(log.get('component_name') for log in container_logs if log.get('component_name'))),
            'event_scopes': list(set(log.get('event_scope') for log in container_logs if log.get('event_scope'))),
            'recent_activity': container_logs[-10:] if container_logs else [],
            'recent_errors': error_logs[-5:] if error_logs else [],
            'correlation_ids': list(set(log.get('correlation_id') for log in container_logs if log.get('correlation_id')))
        }
    
    def get_cross_container_flows(self, time_window_minutes: int = 5) -> List[Dict]:
        """Get all cross-container communication in time window."""
        return self.flow_tracer.analyze_container_communication(time_window_minutes)
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze system for performance bottlenecks."""
        flow_stats = self.flow_tracer.get_flow_statistics()
        communication_analysis = self.flow_tracer.analyze_container_communication()
        
        # Identify potential bottlenecks
        bottlenecks = []
        
        # Check for containers with high error rates
        for container_id in self._get_active_containers():
            debug_info = self.debug_container_isolation(container_id)
            error_rate = debug_info['error_count'] / max(debug_info['log_count'], 1)
            
            if error_rate > 0.1:  # More than 10% errors
                bottlenecks.append({
                    'type': 'high_error_rate',
                    'container_id': container_id,
                    'error_rate': error_rate,
                    'severity': 'high' if error_rate > 0.2 else 'medium'
                })
        
        # Check for communication hotspots
        comm_matrix = communication_analysis.get('communication_matrix', {})
        for source, targets in comm_matrix.items():
            total_outbound = sum(targets.values())
            if total_outbound > 1000:  # High communication volume
                bottlenecks.append({
                    'type': 'communication_hotspot',
                    'source': source,
                    'outbound_events': total_outbound,
                    'target_count': len(targets),
                    'severity': 'high' if total_outbound > 5000 else 'medium'
                })
        
        return {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'bottlenecks': bottlenecks,
            'flow_statistics': flow_stats,
            'communication_analysis': communication_analysis,
            'recommendations': self._generate_recommendations(bottlenecks)
        }
    
    def _read_container_logs(self, container_id: str) -> List[Dict]:
        """Read logs for a specific container."""
        logs = []
        container_log_dir = self.base_log_dir / "containers" / container_id
        
        if container_log_dir.exists():
            for log_file in container_log_dir.glob("*.log"):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                log_entry = json.loads(line.strip())
                                logs.append(log_entry)
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''))
        return logs
    
    def _get_active_containers(self) -> List[str]:
        """Get list of active containers from log directory structure."""
        containers = []
        containers_dir = self.base_log_dir / "containers"
        
        if containers_dir.exists():
            containers = [d.name for d in containers_dir.iterdir() if d.is_dir()]
        
        return containers
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate recommendations based on identified bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'high_error_rate':
                recommendations.append(
                    f"Investigate error sources in container {bottleneck['container_id']} "
                    f"(error rate: {bottleneck['error_rate']:.1%})"
                )
            elif bottleneck['type'] == 'communication_hotspot':
                recommendations.append(
                    f"Consider optimizing communication from {bottleneck['source']} "
                    f"({bottleneck['outbound_events']} events to {bottleneck['target_count']} targets)"
                )
        
        if not recommendations:
            recommendations.append("No significant bottlenecks detected - system appears healthy")
        
        return recommendations
    
    def close(self):
        """Close debugger and cleanup resources."""
        self.flow_tracer.close()
        
        for logger in self.container_loggers.values():
            if hasattr(logger, 'close'):
                logger.close()
        
        self.container_loggers.clear()