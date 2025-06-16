"""
Classification types for market regime classifiers.

This module defines the data structures for classifier outputs,
separate from trading signals for better type safety and clarity.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ClassificationConfidence(Enum):
    """Confidence levels for classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @classmethod
    def from_value(cls, confidence: float) -> 'ClassificationConfidence':
        """Convert numeric confidence to enum."""
        if confidence < 0.3:
            return cls.LOW
        elif confidence < 0.7:
            return cls.MEDIUM
        else:
            return cls.HIGH


@dataclass
class Classification:
    """
    Market regime classification from a classifier.
    
    Unlike Signal which represents trading decisions,
    Classification represents market state identification.
    
    Attributes:
        symbol: Symbol being classified
        regime: The identified market regime (e.g., "trending_up", "high_volatility")
        confidence: Confidence level of the classification (0.0-1.0)
        timestamp: When the classification was made
        classifier_id: ID of the classifier that generated this
        previous_regime: The regime before this classification (None for first)
        features: Snapshot of features used for classification
        metadata: Additional classifier-specific information
    """
    symbol: str
    regime: str
    confidence: float
    timestamp: datetime
    classifier_id: str
    previous_regime: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def is_regime_change(self) -> bool:
        """Check if this represents a regime change."""
        return self.previous_regime is not None and self.previous_regime != self.regime
    
    @property
    def confidence_level(self) -> ClassificationConfidence:
        """Get confidence as enum level."""
        return ClassificationConfidence.from_value(self.confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'regime': self.regime,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'classifier_id': self.classifier_id,
            'previous_regime': self.previous_regime,
            'features': self.features,
            'metadata': self.metadata,
            'is_regime_change': self.is_regime_change,
            'confidence_level': self.confidence_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Classification':
        """Create Classification from dictionary."""
        # Handle timestamp conversion
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        return cls(
            symbol=data['symbol'],
            regime=data['regime'],
            confidence=data['confidence'],
            timestamp=timestamp,
            classifier_id=data['classifier_id'],
            previous_regime=data.get('previous_regime'),
            features=data.get('features', {}),
            metadata=data.get('metadata', {})
        )


def create_classification_event(classification: Classification, 
                              source_id: str,
                              container_id: Optional[str] = None) -> 'Event':
    """
    Create a CLASSIFICATION event from a Classification object.
    
    Args:
        classification: The classification data
        source_id: ID of the component creating the event
        container_id: Optional container ID
        
    Returns:
        Event object with CLASSIFICATION type
    """
    from ..core.events.types import Event, EventType
    
    return Event(
        event_type=EventType.CLASSIFICATION.value,
        timestamp=classification.timestamp,
        payload=classification.to_dict(),
        source_id=source_id,
        container_id=container_id
    )