"""
File: src/strategy/classifiers/regime_types.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#classifier-container
Step: 3 - Classifier Container
Dependencies: dataclasses, enum, datetime

Market regime types and events for classifier container.
Defines regime states, confidence levels, and regime change events.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from decimal import Decimal


class MarketRegime(Enum):
    """
    Market regime enumeration.
    
    Represents different market conditions that can be detected
    by classification algorithms.
    """
    UNKNOWN = "unknown"
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"


class ClassificationConfidence(Enum):
    """
    Classification confidence levels.
    
    Indicates how confident the classifier is in its regime prediction.
    """
    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0
    
    @classmethod
    def from_float(cls, value: float) -> 'ClassificationConfidence':
        """Convert float confidence to enum."""
        if value >= 0.9:
            return cls.VERY_HIGH
        elif value >= 0.7:
            return cls.HIGH
        elif value >= 0.5:
            return cls.MEDIUM
        elif value >= 0.25:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class RegimeChangeEvent:
    """
    Event emitted when market regime changes.
    
    Contains information about the regime transition including
    confidence levels and classification metadata.
    """
    timestamp: datetime
    old_regime: MarketRegime
    new_regime: MarketRegime
    confidence: float
    classifier_id: str
    symbol: str = "UNKNOWN"
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ClassificationConfidence:
        """Get confidence as enum."""
        return ClassificationConfidence.from_float(self.confidence)
    
    @property
    def is_significant_change(self) -> bool:
        """Check if this is a significant regime change."""
        return (
            self.old_regime != self.new_regime and
            self.confidence >= 0.7 and
            self.old_regime != MarketRegime.UNKNOWN
        )


@dataclass
class RegimeState:
    """
    Current state of market regime classification.
    
    Tracks the current regime, confidence, duration, and
    additional state information.
    """
    regime: MarketRegime
    confidence: float
    started_at: datetime
    last_updated: datetime
    duration_bars: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get regime duration in seconds."""
        return (self.last_updated - self.started_at).total_seconds()
    
    @property
    def is_stable(self) -> bool:
        """Check if regime is stable (high confidence, sufficient duration)."""
        return self.confidence >= 0.7 and self.duration_bars >= 10
    
    def update(self, new_confidence: float, timestamp: datetime, 
               features: Optional[Dict[str, float]] = None) -> None:
        """Update regime state with new information."""
        self.confidence = new_confidence
        self.last_updated = timestamp
        self.duration_bars += 1
        
        if features:
            self.features.update(features)


@dataclass
class ClassificationFeatures:
    """
    Features used for market regime classification.
    
    Contains extracted features from market data that are
    used by classifiers to determine market regime.
    """
    returns: float
    volatility: float
    volume_ratio: float
    price_range: float
    momentum: float
    trend_strength: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "UNKNOWN"
    
    def to_array(self) -> list:
        """Convert features to array for ML algorithms."""
        return [
            self.returns,
            self.volatility,
            self.volume_ratio,
            self.price_range,
            self.momentum,
            self.trend_strength
        ]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary."""
        return {
            'returns': self.returns,
            'volatility': self.volatility,
            'volume_ratio': self.volume_ratio,
            'price_range': self.price_range,
            'momentum': self.momentum,
            'trend_strength': self.trend_strength
        }


@dataclass
class ClassifierConfig:
    """
    Configuration for classifier containers.
    
    Contains parameters for different types of classifiers
    and their behavioral settings.
    """
    classifier_type: str = "pattern"  # "hmm", "pattern", "ensemble"
    lookback_period: int = 50
    min_confidence: float = 0.6
    regime_change_threshold: float = 0.1
    feature_window: int = 20
    
    # HMM specific config
    hmm_states: int = 3
    hmm_iterations: int = 100
    hmm_covariance_type: str = "diag"
    
    # Pattern specific config
    atr_period: int = 14
    trend_period: int = 20
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.7
    
    # Ensemble config
    ensemble_classifiers: list = field(default_factory=lambda: ["hmm", "pattern"])
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {"hmm": 0.6, "pattern": 0.4})
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.classifier_type not in ["hmm", "pattern", "ensemble"]:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")
        
        if self.lookback_period < 10:
            raise ValueError("lookback_period must be at least 10")
        
        if self.hmm_states < 2:
            raise ValueError("hmm_states must be at least 2")


# Utility functions for creating test data

def create_trending_regime() -> RegimeState:
    """Create a trending regime state for testing."""
    return RegimeState(
        regime=MarketRegime.TRENDING,
        confidence=0.85,
        started_at=datetime.now(),
        last_updated=datetime.now(),
        duration_bars=25,
        features={'trend_strength': 0.8, 'volatility': 0.15}
    )


def create_ranging_regime() -> RegimeState:
    """Create a ranging regime state for testing."""
    return RegimeState(
        regime=MarketRegime.RANGING,
        confidence=0.75,
        started_at=datetime.now(),
        last_updated=datetime.now(),
        duration_bars=30,
        features={'trend_strength': 0.1, 'volatility': 0.08}
    )


def create_volatile_regime() -> RegimeState:
    """Create a volatile regime state for testing."""
    return RegimeState(
        regime=MarketRegime.VOLATILE,
        confidence=0.9,
        started_at=datetime.now(),
        last_updated=datetime.now(),
        duration_bars=15,
        features={'trend_strength': 0.3, 'volatility': 0.35}
    )