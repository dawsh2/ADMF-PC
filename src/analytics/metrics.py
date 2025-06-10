"""
Analytics Bridge to Container Event System

Extracts and processes metrics that containers already calculate
via their event observer system. No duplication - just aggregation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for metrics extracted from container observers."""
    container_id: str
    correlation_id: Optional[str]
    
    # Metrics already calculated by container observers
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    avg_trade_return: float
    portfolio_value: float
    cash: float
    
    # Observer metadata
    events_observed: int
    events_pruned: int
    active_trades: int
    retention_policy: str
    
    calculated_at: datetime


class ContainerMetricsExtractor:
    """
    Extract metrics from containers that already calculate them via observers.
    
    This is a bridge to the existing event-based metrics system,
    not a replacement for it.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_metrics(self, container: Any, container_id: str) -> Optional[MetricsResult]:
        """
        Extract metrics from container's observer system.
        
        Args:
            container: Container instance with event observers
            container_id: Unique identifier for the container
            
        Returns:
            MetricsResult extracted from container observers
        """
        self.logger.debug(f"Extracting metrics from container {container_id}")
        
        try:
            # Get metrics from container's observer
            metrics_data = self._get_container_metrics(container)
            
            if not metrics_data:
                self.logger.warning(f"No metrics available from container {container_id}")
                return None
            
            # Extract correlation_id for linking to event traces
            correlation_id = self._extract_correlation_id(container, metrics_data)
            
            # Extract metrics from observer
            metrics = metrics_data.get('metrics', {})
            observer_stats = metrics_data.get('observer_stats', {})
            
            return MetricsResult(
                container_id=container_id,
                correlation_id=correlation_id,
                total_return=metrics.get('total_return', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                trade_count=metrics.get('trade_count', 0),
                win_rate=metrics.get('win_rate', 0.0),
                avg_trade_return=metrics.get('avg_trade_return', 0.0),
                portfolio_value=metrics.get('portfolio_value', 0.0),
                cash=metrics.get('cash', 0.0),
                events_observed=observer_stats.get('events_observed', 0),
                events_pruned=observer_stats.get('events_pruned', 0),
                active_trades=observer_stats.get('active_trades', 0),
                retention_policy=observer_stats.get('retention_policy', 'unknown'),
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract metrics from {container_id}: {e}")
            return None
    
    def _get_container_metrics(self, container: Any) -> Optional[Dict[str, Any]]:
        """Get metrics from container's observer system."""
        
        # Try different ways containers might expose metrics
        if hasattr(container, 'get_metrics'):
            return container.get_metrics()
        
        if hasattr(container, 'metrics_observer') and hasattr(container.metrics_observer, 'get_metrics'):
            return container.metrics_observer.get_metrics()
        
        # Look for observer in event bus
        if hasattr(container, 'event_bus') and hasattr(container.event_bus, 'observers'):
            for observer in container.event_bus.observers:
                if hasattr(observer, 'get_metrics'):
                    return observer.get_metrics()
        
        # Try get_results method (alternative interface)
        if hasattr(container, 'get_results'):
            results = container.get_results()
            # Convert results format to metrics format if needed
            return self._convert_results_to_metrics(results)
        
        return None
    
    def _convert_results_to_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert container results format to metrics format."""
        
        # If results already in metrics format, return as-is
        if 'metrics' in results and 'observer_stats' in results:
            return results
        
        # Otherwise, assume results are the metrics themselves
        return {
            'metrics': results,
            'observer_stats': {
                'events_observed': 0,
                'events_pruned': 0,
                'active_trades': 0,
                'retention_policy': 'unknown'
            }
        }
    
    def _extract_correlation_id(self, container: Any, metrics_data: Dict[str, Any]) -> Optional[str]:
        """Extract correlation ID from container for linking to event traces."""
        
        # Try to get from container config
        if hasattr(container, 'config'):
            config = container.config
            if hasattr(config, 'combo_id'):
                return config.combo_id
            elif isinstance(config, dict) and 'combo_id' in config:
                return config['combo_id']
        
        # Try to get from container name/id
        if hasattr(container, 'name'):
            return container.name
        
        if hasattr(container, 'container_id'):
            return container.container_id
        
        # Try to extract from metrics data
        if 'correlation_id' in metrics_data:
            return metrics_data['correlation_id']
        
        return None
    
    def extract_batch_metrics(self, containers: Dict[str, Any]) -> List[MetricsResult]:
        """
        Extract metrics from multiple containers in parallel-friendly way.
        
        Args:
            containers: Dict mapping container_id -> container_instance
            
        Returns:
            List of MetricsResult objects
        """
        results = []
        
        for container_id, container in containers.items():
            try:
                metrics = self.extract_metrics(container, container_id)
                if metrics:
                    results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to extract metrics from {container_id}: {e}")
                # Continue processing other containers
        
        self.logger.info(f"Extracted metrics from {len(results)}/{len(containers)} containers")
        return results
    
    def to_dataframe(self, metrics_results: List[MetricsResult]) -> pd.DataFrame:
        """Convert metrics results to DataFrame for analysis."""
        
        data = []
        for result in metrics_results:
            data.append({
                'container_id': result.container_id,
                'correlation_id': result.correlation_id,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'trade_count': result.trade_count,
                'win_rate': result.win_rate,
                'avg_trade_return': result.avg_trade_return,
                'portfolio_value': result.portfolio_value,
                'cash': result.cash,
                'events_observed': result.events_observed,
                'events_pruned': result.events_pruned,
                'active_trades': result.active_trades,
                'retention_policy': result.retention_policy,
                'calculated_at': result.calculated_at
            })
        
        return pd.DataFrame(data)