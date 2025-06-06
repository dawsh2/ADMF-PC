"""
End-to-End Execution Tracing for ADMF-PC

Traces the complete flow from data ingestion to portfolio updates
to ensure canonical implementations are being used.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TracePoint(str, Enum):
    """Key trace points in the execution flow."""
    DATA_LOAD = "data.load"
    FEATURE_CALC = "feature.calc"
    SIGNAL_GEN = "signal.gen"
    ORDER_CREATE = "order.create"
    ORDER_ROUTE = "order.route"
    ORDER_EXEC = "order.exec"
    FILL_CREATE = "fill.create"
    FILL_ROUTE = "fill.route"
    PORTFOLIO_UPDATE = "portfolio.update"


@dataclass
class TraceEntry:
    """Single trace entry."""
    timestamp: float
    trace_point: TracePoint
    component: str  # Which file/class is executing
    details: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


class ExecutionTracer:
    """Lightweight tracer for end-to-end flow verification."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.traces: List[TraceEntry] = []
        self._correlation_counter = 0
        
    def trace(self, 
              trace_point: TracePoint, 
              component: str, 
              details: Optional[Dict[str, Any]] = None,
              correlation_id: Optional[str] = None) -> str:
        """Add a trace entry and return correlation_id."""
        if not self.enabled:
            return correlation_id or ""
            
        if not correlation_id:
            self._correlation_counter += 1
            correlation_id = f"trace_{self._correlation_counter:04d}"
            
        entry = TraceEntry(
            timestamp=time.time(),
            trace_point=trace_point,
            component=component,
            details=details or {},
            correlation_id=correlation_id
        )
        
        self.traces.append(entry)
        
        # Log the trace
        logger.info(f"TRACE[{correlation_id}] {trace_point.value} in {component}: {details}")
        
        return correlation_id
    
    def get_flow_for_correlation(self, correlation_id: str) -> List[TraceEntry]:
        """Get complete flow for a specific correlation ID."""
        return [t for t in self.traces if t.correlation_id == correlation_id]
    
    def verify_canonical_flow(self) -> Dict[str, Any]:
        """Verify that execution followed canonical implementations."""
        violations = []
        canonical_components = {
            TracePoint.DATA_LOAD: "csv_handler.py",
            TracePoint.FEATURE_CALC: "symbol_timeframe_container.py", 
            TracePoint.SIGNAL_GEN: "stateless_momentum.py",
            TracePoint.ORDER_CREATE: "portfolio_container.py",
            TracePoint.ORDER_EXEC: "execution_container.py",
            TracePoint.PORTFOLIO_UPDATE: "portfolio_container.py"
        }
        
        for trace in self.traces:
            expected_component = canonical_components.get(trace.trace_point)
            if expected_component and expected_component not in trace.component:
                violations.append({
                    'trace_point': trace.trace_point.value,
                    'expected': expected_component,
                    'actual': trace.component,
                    'correlation_id': trace.correlation_id
                })
        
        return {
            'total_traces': len(self.traces),
            'violations': violations,
            'canonical_compliance': len(violations) == 0
        }
    
    def print_flow_summary(self):
        """Print a summary of the execution flow."""
        if not self.traces:
            print("No traces recorded")
            return
            
        print("\n" + "="*80)
        print("EXECUTION FLOW TRACE SUMMARY")
        print("="*80)
        
        # Group by correlation ID
        correlations = {}
        for trace in self.traces:
            cid = trace.correlation_id or "unknown"
            if cid not in correlations:
                correlations[cid] = []
            correlations[cid].append(trace)
        
        for cid, traces in correlations.items():
            print(f"\n📊 Flow {cid}:")
            for i, trace in enumerate(sorted(traces, key=lambda t: t.timestamp), 1):
                elapsed = trace.timestamp - traces[0].timestamp if traces else 0
                print(f"  {i}. {trace.trace_point.value:15} | {trace.component:30} | +{elapsed:.3f}s")
                if trace.details:
                    key_details = {k: v for k, v in trace.details.items() 
                                 if k in ['symbol', 'price', 'quantity', 'direction', 'side']}
                    if key_details:
                        print(f"     └─ {key_details}")
        
        # Verification summary
        verification = self.verify_canonical_flow()
        print(f"\n🔍 CANONICAL COMPLIANCE: {'✅ PASS' if verification['canonical_compliance'] else '❌ FAIL'}")
        if verification['violations']:
            print("Violations found:")
            for v in verification['violations']:
                print(f"  - {v['trace_point']}: expected {v['expected']}, got {v['actual']}")


# Global tracer instance
_global_tracer = ExecutionTracer()


def trace(trace_point: TracePoint, 
          component: str, 
          details: Optional[Dict[str, Any]] = None,
          correlation_id: Optional[str] = None) -> str:
    """Convenience function for tracing."""
    return _global_tracer.trace(trace_point, component, details, correlation_id)


def get_tracer() -> ExecutionTracer:
    """Get the global tracer instance."""
    return _global_tracer


def enable_tracing():
    """Enable execution tracing."""
    _global_tracer.enabled = True
    logger.info("🔍 Execution tracing ENABLED")


def disable_tracing():
    """Disable execution tracing."""
    _global_tracer.enabled = False
    logger.info("🔍 Execution tracing DISABLED")