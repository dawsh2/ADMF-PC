"""Protocols for data mining components."""

from typing import Protocol, Dict, List, Any
import pandas as pd

from src.core.events.types import Event


class MetricsAggregatorProtocol(Protocol):
    """Protocol for aggregating metrics into SQL database."""
    
    def aggregate(self, events: List[Event]) -> Dict[str, Any]: 
        """Aggregate events into metrics."""
        ...
    
    def store_metrics(self, metrics: Dict[str, Any], run_id: str) -> None: 
        """Store metrics in database."""
        ...
    
    def query_metrics(self, filters: Dict[str, Any]) -> pd.DataFrame: 
        """Query metrics from database."""
        ...