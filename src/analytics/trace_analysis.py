"""
Event Trace Analysis Tools

Extracts business results and insights from event traces.
Part of the analytics module - can be used by any system that needs to analyze traces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EventExtractor(ABC):
    """Base class for extracting specific data from event streams."""
    
    @abstractmethod
    def can_extract(self, event: Dict[str, Any]) -> bool:
        """Check if this extractor can process this event."""
        pass
    
    @abstractmethod
    def extract(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from event."""
        pass
    
    @property
    @abstractmethod
    def category_name(self) -> str:
        """Name of the category for extracted data."""
        pass


class PortfolioMetricsExtractor(EventExtractor):
    """Extract portfolio performance metrics from events."""
    
    @property
    def category_name(self) -> str:
        return "portfolio_metrics"
    
    def can_extract(self, event: Dict[str, Any]) -> bool:
        return event.get('type') == 'PORTFOLIO_UPDATE'
    
    def extract(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.can_extract(event):
            return None
            
        data = event.get('data', {})
        return {
            'timestamp': event.get('timestamp'),
            'container_id': event.get('source'),
            'total_value': data.get('total_value'),
            'pnl': data.get('pnl'),
            'sharpe_ratio': data.get('metrics', {}).get('sharpe_ratio'),
            'max_drawdown': data.get('metrics', {}).get('max_drawdown'),
            'total_return': data.get('metrics', {}).get('total_return'),
            'positions': data.get('positions', {})
        }


class SignalExtractor(EventExtractor):
    """Extract trading signals from events."""
    
    @property
    def category_name(self) -> str:
        return "signals"
    
    def can_extract(self, event: Dict[str, Any]) -> bool:
        return event.get('type') == 'STRATEGY_SIGNAL'
    
    def extract(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.can_extract(event):
            return None
            
        data = event.get('data', {})
        return {
            'timestamp': event.get('timestamp'),
            'strategy': data.get('strategy_id'),
            'symbol': data.get('symbol'),
            'direction': data.get('direction'),
            'strength': data.get('strength'),
            'reason': data.get('reason'),
            'features': data.get('features', {})
        }


class TradeExtractor(EventExtractor):
    """Extract trade execution data from events."""
    
    @property
    def category_name(self) -> str:
        return "trades"
    
    def can_extract(self, event: Dict[str, Any]) -> bool:
        return event.get('type') in ['FILL', 'ORDER_FILLED']
    
    def extract(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.can_extract(event):
            return None
            
        data = event.get('data', {})
        return {
            'timestamp': event.get('timestamp'),
            'order_id': data.get('order_id'),
            'symbol': data.get('symbol'),
            'side': data.get('side'),
            'quantity': data.get('quantity'),
            'price': data.get('price'),
            'commission': data.get('commission', 0),
            'slippage': data.get('slippage', 0)
        }


class TraceAnalyzer:
    """
    Analyzes event traces to extract business results.
    
    This is a general-purpose tool that can be used by any system
    needing to extract results from event traces.
    """
    
    def __init__(self, extractors: Optional[List[EventExtractor]] = None):
        """
        Initialize analyzer with extractors.
        
        Args:
            extractors: List of extractors to use. If None, uses defaults.
        """
        self.extractors = extractors or [
            PortfolioMetricsExtractor(),
            SignalExtractor(),
            TradeExtractor()
        ]
        self._extractor_map = {e.category_name: e for e in self.extractors}
    
    def analyze_trace(self, trace_events: Union[List[Dict], Any]) -> Dict[str, Any]:
        """
        Analyze a trace and extract results.
        
        Args:
            trace_events: List of events or an event tracer object
            
        Returns:
            Dict with extracted results organized by category
        """
        # Handle different input types
        if hasattr(trace_events, 'get_events'):
            events = trace_events.get_events()
        elif hasattr(trace_events, 'events'):
            events = trace_events.events
        else:
            events = trace_events
        
        # Extract data from events
        results = defaultdict(list)
        
        for event in events:
            event_dict = self._normalize_event(event)
            
            for extractor in self.extractors:
                if extractor.can_extract(event_dict):
                    extracted = extractor.extract(event_dict)
                    if extracted:
                        results[extractor.category_name].append(extracted)
        
        # Generate summary
        analysis = {
            'event_count': len(events),
            'extracted_at': datetime.now().isoformat(),
            'results': dict(results),
            'summary': self._generate_summary(dict(results))
        }
        
        return analysis
    
    def analyze_trace_file(self, trace_file: Path) -> Dict[str, Any]:
        """Analyze a trace from a file."""
        events = []
        
        # Support both JSON and JSONL formats
        if trace_file.suffix == '.jsonl':
            with open(trace_file, 'r') as f:
                for line in f:
                    events.append(json.loads(line))
        else:
            with open(trace_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    events = data
                else:
                    events = data.get('events', [])
        
        return self.analyze_trace(events)
    
    def to_dataframes(self, analysis: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Convert analysis results to pandas DataFrames."""
        dataframes = {}
        
        for category, records in analysis['results'].items():
            if records:
                df = pd.DataFrame(records)
                # Convert timestamps if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                dataframes[category] = df
        
        return dataframes
    
    def _normalize_event(self, event: Any) -> Dict[str, Any]:
        """Normalize event to dictionary format."""
        if isinstance(event, dict):
            return event
        elif hasattr(event, '__dict__'):
            return event.__dict__
        else:
            return {
                'type': str(getattr(event, 'type', 'UNKNOWN')),
                'data': getattr(event, 'data', {}),
                'timestamp': getattr(event, 'timestamp', datetime.now()),
                'source': getattr(event, 'source', 'unknown')
            }
    
    def _generate_summary(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate summary statistics from extracted results."""
        summary = {}
        
        # Portfolio metrics summary
        if 'portfolio_metrics' in results and results['portfolio_metrics']:
            metrics_df = pd.DataFrame(results['portfolio_metrics'])
            
            # Get final metrics (last row)
            final_metrics = metrics_df.iloc[-1]
            
            summary['portfolio'] = {
                'final_value': final_metrics.get('total_value'),
                'total_return': final_metrics.get('total_return'),
                'sharpe_ratio': final_metrics.get('sharpe_ratio'),
                'max_drawdown': final_metrics.get('max_drawdown'),
                'update_count': len(metrics_df)
            }
            
            # Add time-series metrics if we have enough data
            if len(metrics_df) > 1:
                summary['portfolio']['avg_sharpe'] = metrics_df['sharpe_ratio'].mean()
                summary['portfolio']['return_volatility'] = metrics_df['total_return'].std()
        
        # Signal analysis
        if 'signals' in results and results['signals']:
            signals_df = pd.DataFrame(results['signals'])
            
            summary['signals'] = {
                'total_count': len(signals_df),
                'by_direction': signals_df['direction'].value_counts().to_dict() if 'direction' in signals_df else {},
                'by_strategy': signals_df['strategy'].value_counts().to_dict() if 'strategy' in signals_df else {},
                'avg_strength': signals_df['strength'].mean() if 'strength' in signals_df else None
            }
        
        # Trade analysis
        if 'trades' in results and results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            summary['trades'] = {
                'total_count': len(trades_df),
                'total_commission': trades_df['commission'].sum() if 'commission' in trades_df else 0,
                'avg_slippage': trades_df['slippage'].mean() if 'slippage' in trades_df else 0,
                'by_side': trades_df['side'].value_counts().to_dict() if 'side' in trades_df else {}
            }
        
        return summary


class TraceArchiver:
    """
    Archives event traces for later analysis.
    
    Part of the analytics infrastructure - provides standardized
    trace storage and retrieval.
    """
    
    def __init__(self, archive_path: str = "./trace_archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
    
    def archive_trace(
        self,
        trace_id: str,
        events: Union[List[Dict], Any],
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = False
    ) -> Path:
        """
        Archive a trace for later analysis.
        
        Args:
            trace_id: Unique identifier for the trace
            events: Events to archive (list or tracer object)
            metadata: Optional metadata about the trace
            compress: Whether to compress the output
            
        Returns:
            Path to archived trace file
        """
        # Create directory structure
        trace_dir = self.archive_path / trace_id
        trace_dir.mkdir(exist_ok=True)
        
        # Normalize events
        if hasattr(events, 'get_events'):
            event_list = events.get_events()
        elif hasattr(events, 'events'):
            event_list = events.events
        else:
            event_list = events
        
        # Save events
        trace_file = trace_dir / "events.jsonl"
        if compress:
            trace_file = trace_dir / "events.jsonl.gz"
            import gzip
            with gzip.open(trace_file, 'wt') as f:
                for event in event_list:
                    f.write(json.dumps(self._serialize_event(event)) + '\n')
        else:
            with open(trace_file, 'w') as f:
                for event in event_list:
                    f.write(json.dumps(self._serialize_event(event)) + '\n')
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'trace_id': trace_id,
            'archived_at': datetime.now().isoformat(),
            'event_count': len(event_list),
            'compressed': compress
        })
        
        metadata_file = trace_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Archived trace {trace_id} with {len(event_list)} events")
        return trace_file
    
    def load_trace(self, trace_id: str) -> Dict[str, Any]:
        """Load an archived trace."""
        trace_dir = self.archive_path / trace_id
        
        if not trace_dir.exists():
            raise ValueError(f"Trace {trace_id} not found")
        
        # Load metadata
        metadata_file = trace_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load events
        events = []
        
        # Check for compressed file first
        if (trace_dir / "events.jsonl.gz").exists():
            import gzip
            with gzip.open(trace_dir / "events.jsonl.gz", 'rt') as f:
                for line in f:
                    events.append(json.loads(line))
        elif (trace_dir / "events.jsonl").exists():
            with open(trace_dir / "events.jsonl", 'r') as f:
                for line in f:
                    events.append(json.loads(line))
        
        return {
            'metadata': metadata,
            'events': events
        }
    
    def list_traces(self) -> List[Dict[str, Any]]:
        """List all archived traces."""
        traces = []
        
        for trace_dir in self.archive_path.iterdir():
            if trace_dir.is_dir():
                metadata_file = trace_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    traces.append(metadata)
        
        return sorted(traces, key=lambda x: x.get('archived_at', ''), reverse=True)
    
    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """Serialize event for storage."""
        if isinstance(event, dict):
            return event
        
        # Convert object to dict
        result = {}
        for key in ['type', 'data', 'timestamp', 'source']:
            if hasattr(event, key):
                value = getattr(event, key)
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[key] = value
        
        return result


# Convenience function for quick analysis
def analyze_trace(trace_events: Union[List[Dict], Any]) -> Dict[str, Any]:
    """Quick function to analyze a trace with default extractors."""
    analyzer = TraceAnalyzer()
    return analyzer.analyze_trace(trace_events)
