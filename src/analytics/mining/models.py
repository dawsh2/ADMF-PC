"""Data models for mining operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass
class OptimizationRun:
    """Metadata for an optimization run."""
    run_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, Any]