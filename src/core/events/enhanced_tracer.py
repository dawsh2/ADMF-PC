"""
Enhanced Event Tracer with Result Extraction

Extends event tracing with integrated result extraction capabilities.
Extracts business results in real-time as events are traced.
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import logging

from ..types.events import Event, EventType
from .result_extraction import ResultExtractor

logger = logging.getLogger(__name__)


class EnhancedEventTracer:
    """
    Event tracer with integrated result extraction.
    
    This tracer:
    - Traces all events for debugging/analysis
    - Extracts business results in real-time
    - Buffers results for efficient storage
    - Provides extracted results before teardown
    """
    
    def __init__(
        self,
        trace_id: str,
        trace_file_path: Optional[str] = None,
        result_extractors: Optional[List[ResultExtractor]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced tracer.
        
        Args:
            trace_id: Unique identifier for this trace
            trace_file_path: Path to write trace events (None = memory only)
            result_extractors: List of extractors for result extraction
            config: Additional configuration options
        """
        self.trace_id = trace_id
        self.trace_file_path = trace_file_path
        self.result_extractors = result_extractors or []
        self.config = config or {}
        
        # Event storage
        self.events: List[Event] = []
        self.event_count = 0
        
        # Result extraction
        self.extracted_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.extraction_errors: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.extraction_times: Dict[str, float] = defaultdict(float)
        
        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.max_events_in_memory = self.config.get('max_events_in_memory', 10000)
        self.write_mode = self.config.get('write_mode', 'append')  # 'append' or 'batch'
        self.event_buffer: List[Event] = []
        self.buffer_size = self.config.get('buffer_size', 100)
        
        # Initialize trace file if needed
        if self.trace_file_path and self.enabled:
            self._init_trace_file()
        
        logger.debug(f"EnhancedEventTracer initialized: {trace_id} with {len(self.result_extractors)} extractors")
    
    def trace_event(self, event: Event) -> None:
        """
        Trace event and optionally extract results.
        
        Args:
            event: The event to trace
        """
        if not self.enabled:
            return
        
        self.event_count += 1
        
        # Store event in memory if under limit
        if len(self.events) < self.max_events_in_memory:
            self.events.append(event)
        
        # Write to file if configured
        if self.trace_file_path:
            if self.write_mode == 'append':
                self._write_event_to_file(event)
            else:
                self.event_buffer.append(event)
                if len(self.event_buffer) >= self.buffer_size:
                    self._flush_event_buffer()
        
        # Extract results
        self._extract_results_from_event(event)
    
    def get_extracted_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all extracted results before teardown.
        
        Returns:
            Dictionary mapping extractor names to lists of results
        """
        return dict(self.extracted_results)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of trace and extraction activity.
        
        Returns:
            Summary statistics
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'trace_id': self.trace_id,
            'duration_seconds': duration,
            'event_count': self.event_count,
            'events_in_memory': len(self.events),
            'extractors_used': len(self.result_extractors),
            'extraction_results': {
                name: len(results) 
                for name, results in self.extracted_results.items()
            },
            'extraction_errors': len(self.extraction_errors),
            'extraction_times': dict(self.extraction_times),
            'events_per_second': self.event_count / duration if duration > 0 else 0
        }
        
        # Add event type breakdown
        if self.events:
            event_types = defaultdict(int)
            for event in self.events:
                event_types[event.type.value] += 1
            summary['event_types'] = dict(event_types)
        
        return summary
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]
    
    def get_events_by_source(self, source: str) -> List[Event]:
        """Get all events from a specific source."""
        return [e for e in self.events if e.source == source]
    
    def get_events_in_timerange(self, start: datetime, end: datetime) -> List[Event]:
        """Get events within a time range."""
        return [e for e in self.events if start <= e.timestamp <= end]
    
    def close(self) -> None:
        """
        Close the tracer and flush any pending data.
        
        Should be called before teardown to ensure all data is written.
        """
        # Flush any remaining events
        if self.event_buffer:
            self._flush_event_buffer()
        
        # Log summary
        summary = self.get_summary()
        logger.info(f"Tracer {self.trace_id} closed: {summary['event_count']} events, "
                   f"{sum(len(r) for r in self.extracted_results.values())} results extracted")
    
    def _init_trace_file(self) -> None:
        """Initialize trace file for writing."""
        try:
            trace_path = Path(self.trace_file_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clear file if it exists (start fresh)
            if trace_path.exists():
                trace_path.unlink()
            
            logger.debug(f"Initialized trace file: {self.trace_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize trace file: {e}")
            self.trace_file_path = None
    
    def _write_event_to_file(self, event: Event) -> None:
        """Write a single event to trace file."""
        try:
            with open(self.trace_file_path, 'a') as f:
                event_dict = self._event_to_dict(event)
                f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write event to trace file: {e}")
    
    def _flush_event_buffer(self) -> None:
        """Flush buffered events to file."""
        if not self.event_buffer or not self.trace_file_path:
            return
        
        try:
            with open(self.trace_file_path, 'a') as f:
                for event in self.event_buffer:
                    event_dict = self._event_to_dict(event)
                    f.write(json.dumps(event_dict) + '\n')
            
            self.event_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush event buffer: {e}")
    
    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': event.event_id,
            'type': event.type.value,
            'timestamp': event.timestamp.isoformat(),
            'source': event.source,
            'target': event.target,
            'correlation_id': event.correlation_id,
            'data': event.data,
            'metadata': event.metadata or {}
        }
    
    def _extract_results_from_event(self, event: Event) -> None:
        """
        Extract results from event using registered extractors.
        
        Args:
            event: The event to extract from
        """
        for extractor in self.result_extractors:
            if not extractor.can_extract(event):
                continue
            
            try:
                # Time the extraction
                start_time = datetime.now()
                
                result = extractor.extract(event)
                
                extraction_time = (datetime.now() - start_time).total_seconds()
                extractor_name = extractor.__class__.__name__
                self.extraction_times[extractor_name] += extraction_time
                
                if result:
                    self.extracted_results[extractor_name].append(result)
                    
            except Exception as e:
                # Log extraction errors but don't stop processing
                error_info = {
                    'timestamp': datetime.now(),
                    'extractor': extractor.__class__.__name__,
                    'event_type': event.type.value,
                    'event_id': event.event_id,
                    'error': str(e)
                }
                self.extraction_errors.append(error_info)
                logger.error(f"Extraction error: {error_info}")


class StreamingResultProcessor:
    """
    Process events in real-time to extract and buffer results.
    
    This processor:
    - Extracts results from events as they arrive
    - Buffers results for efficient storage
    - Flushes to configured output format
    - Handles multiple extractors efficiently
    """
    
    def __init__(
        self,
        extractors: List[ResultExtractor],
        output_config: Dict[str, Any]
    ):
        """
        Initialize streaming result processor.
        
        Args:
            extractors: List of result extractors to use
            output_config: Configuration for output (format, directory, etc.)
        """
        self.extractors = extractors
        self.output_config = output_config
        
        # Result buffers by category
        self.buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Configuration
        self.buffer_size = output_config.get('buffer_size', 1000)
        self.output_format = output_config.get('format', 'parquet')
        self.output_dir = Path(output_config.get('directory', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = output_config.get('compression', 'snappy')
        
        # Statistics
        self.events_processed = 0
        self.results_extracted = defaultdict(int)
        self.last_flush_time = defaultdict(lambda: datetime.now())
        self.write_count = 0
        self.flush_count = 0
        
        # Metrics aggregation
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        
        logger.info(f"StreamingResultProcessor initialized with {len(extractors)} extractors")
    
    def process_event(self, event: Event) -> None:
        """
        Process event and extract results.
        
        Args:
            event: The event to process
        """
        self.events_processed += 1
        
        # Quick check if event might contain results
        if not event.metadata.get('contains_result', True):
            return
        
        # Try each extractor
        for extractor in self.extractors:
            if extractor.can_extract(event):
                try:
                    result = extractor.extract(event)
                    if result:
                        category = extractor.__class__.__name__
                        self.buffers[category].append(result)
                        self.results_extracted[category] += 1
                        self.write_count += 1
                        
                        # Update metrics if this is portfolio metrics
                        if category == 'PortfolioMetricsExtractor':
                            self._update_metrics(result)
                        
                        # Flush if buffer full
                        if len(self.buffers[category]) >= self.buffer_size:
                            self._flush_category(category)
                            
                except Exception as e:
                    logger.error(f"Failed to extract with {extractor.__class__.__name__}: {e}")
    
    def flush_all(self) -> None:
        """Flush all buffered results."""
        for category in list(self.buffers.keys()):
            if self.buffers[category]:
                self._flush_category(category)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'events_processed': self.events_processed,
            'results_extracted': dict(self.results_extracted),
            'pending_results': {
                cat: len(buffer) 
                for cat, buffer in self.buffers.items()
            },
            'write_count': self.write_count,
            'flush_count': self.flush_count,
            'metrics_summary': self._get_metrics_summary()
        }
    
    def _flush_category(self, category: str) -> None:
        """
        Flush results for a specific category.
        
        Args:
            category: The result category to flush
        """
        if not self.buffers[category]:
            return
        
        try:
            if self.output_format == 'parquet':
                self._flush_to_parquet(category)
            elif self.output_format == 'jsonl':
                self._flush_to_jsonl(category)
            elif self.output_format == 'csv':
                self._flush_to_csv(category)
            else:
                raise ValueError(f"Unknown output format: {self.output_format}")
            
            # Clear buffer and update flush time
            self.buffers[category].clear()
            self.last_flush_time[category] = datetime.now()
            self.flush_count += 1
            
        except Exception as e:
            logger.error(f"Failed to flush {category} results: {e}")
    
    def _flush_to_parquet(self, category: str) -> None:
        """Flush results to Parquet format."""
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pandas/pyarrow not available, falling back to JSONL")
            self._flush_to_jsonl(category)
            return
        
        df = pd.DataFrame(self.buffers[category])
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"{category}_{timestamp}.parquet"
        
        # Write with compression
        df.to_parquet(
            filename,
            compression=self.compression,
            index=False
        )
        
        logger.debug(f"Flushed {len(df)} {category} results to {filename}")
    
    def _flush_to_jsonl(self, category: str) -> None:
        """Flush results to JSON Lines format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"{category}_{timestamp}.jsonl"
        
        with open(filename, 'w') as f:
            for record in self.buffers[category]:
                # Convert datetime objects to ISO format
                record_clean = self._clean_for_json(record)
                f.write(json.dumps(record_clean) + '\n')
        
        logger.debug(f"Flushed {len(self.buffers[category])} {category} results to {filename}")
    
    def _flush_to_csv(self, category: str) -> None:
        """Flush results to CSV format."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not available, falling back to JSONL")
            self._flush_to_jsonl(category)
            return
        
        df = pd.DataFrame(self.buffers[category])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"{category}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        
        logger.debug(f"Flushed {len(df)} {category} results to {filename}")
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        else:
            return obj
    
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update aggregated metrics from portfolio results."""
        container_id = result.get('container_id', 'unknown')
        
        # Track key metrics
        if 'sharpe_ratio' in result:
            self.metrics['sharpe_ratios'][container_id].append(result['sharpe_ratio'])
        if 'total_value' in result:
            self.metrics['portfolio_values'][container_id].append(result['total_value'])
        if 'pnl' in result:
            self.metrics['pnl'][container_id].append(result['pnl'])
        if 'max_drawdown' in result:
            self.metrics['max_drawdowns'][container_id].append(result['max_drawdown'])
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of aggregated metrics."""
        summary = {}
        
        # Aggregate Sharpe ratios
        if self.metrics['sharpe_ratios']:
            all_sharpes = []
            best_container = None
            best_sharpe = -float('inf')
            
            for container_id, sharpes in self.metrics['sharpe_ratios'].items():
                if sharpes:
                    all_sharpes.extend(sharpes)
                    avg_sharpe = sum(sharpes) / len(sharpes)
                    if avg_sharpe > best_sharpe:
                        best_sharpe = avg_sharpe
                        best_container = container_id
            
            if all_sharpes:
                summary['avg_sharpe'] = sum(all_sharpes) / len(all_sharpes)
                summary['best_sharpe'] = best_sharpe
                summary['best_container'] = best_container
        
        # Aggregate returns
        if self.metrics['pnl']:
            all_pnls = []
            for pnls in self.metrics['pnl'].values():
                all_pnls.extend(pnls)
            
            if all_pnls:
                summary['total_pnl'] = sum(all_pnls)
                summary['avg_pnl'] = sum(all_pnls) / len(all_pnls)
        
        return summary
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """Get aggregated results with detailed metrics."""
        # Flush all buffers first
        self.flush_all()
        
        aggregated = {
            'total_results': self.write_count,
            'categories': list(self.results_extracted.keys()),
            'metrics_summary': self._get_metrics_summary(),
            'top_performers': self._get_top_performers(),
            'statistics': self.get_statistics()
        }
        
        # Save summary
        summary_path = self.output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        
        logger.info(f"Saved aggregated results to {summary_path}")
        
        return aggregated
    
    def _get_top_performers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N performers by Sharpe ratio."""
        performers = []
        
        for container_id, sharpes in self.metrics['sharpe_ratios'].items():
            if sharpes:
                avg_sharpe = sum(sharpes) / len(sharpes)
                performers.append({
                    'container_id': container_id,
                    'avg_sharpe': avg_sharpe,
                    'num_results': len(sharpes)
                })
        
        # Sort by Sharpe ratio
        performers.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        return performers[:n]