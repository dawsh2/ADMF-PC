"""
File: src/strategy/classifiers/classifier.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#classifier-protocol
Step: 3 - Classifier Container
Dependencies: abc, typing, dataclasses

Base classifier protocol and abstract implementation.
Defines the interface for market regime classifiers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, List
from datetime import datetime
from collections import deque

from .regime_types import (
    MarketRegime, 
    RegimeState, 
    ClassificationFeatures,
    ClassifierConfig
)
from ...data.models import Bar


class ClassifierProtocol(Protocol):
    """
    Protocol for market regime classifiers.
    
    Defines the interface that all classifiers must implement
    for regime detection and state management.
    """
    
    def update(self, bar: Bar) -> None:
        """
        Update classifier with new market data.
        
        Args:
            bar: New market data bar
        """
        ...
    
    def classify(self) -> MarketRegime:
        """
        Classify current market regime.
        
        Returns:
            Current market regime classification
        """
        ...
    
    @property
    def confidence(self) -> float:
        """Get classification confidence (0.0 to 1.0)."""
        ...
    
    @property
    def is_ready(self) -> bool:
        """Check if classifier has enough data to classify."""
        ...
    
    def reset(self) -> None:
        """Reset classifier state."""
        ...


class BaseClassifier(ABC):
    """
    Abstract base class for market regime classifiers.
    
    Provides common functionality and enforces the classifier interface.
    Handles feature extraction, state management, and logging.
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialize base classifier.
        
        Args:
            config: Classifier configuration
        """
        self.config = config
        self.config.validate()
        
        # State management
        self._confidence = 0.0
        self._current_regime = MarketRegime.UNKNOWN
        self._is_ready = False
        
        # Data windows
        self.bar_history = deque(maxlen=config.lookback_period)
        self.feature_history = deque(maxlen=config.feature_window)
        
        # Feature calculation state
        self._last_price = None
        self._volume_sma = 0.0
        self._volume_count = 0
        
        # Classification history
        self.regime_history: List[RegimeState] = []
        
    @property
    def confidence(self) -> float:
        """Get current classification confidence."""
        return self._confidence
    
    @property
    def current_regime(self) -> MarketRegime:
        """Get current regime classification."""
        return self._current_regime
    
    @property
    def is_ready(self) -> bool:
        """Check if classifier has enough data."""
        return self._is_ready
    
    def update(self, bar: Bar) -> None:
        """
        Update classifier with new bar data.
        
        Args:
            bar: New market data bar
        """
        # Store bar
        self.bar_history.append(bar)
        
        # Extract features
        features = self._extract_features(bar)
        if features:
            self.feature_history.append(features)
        
        # Update readiness
        self._update_readiness()
        
        # Perform classification if ready
        if self.is_ready:
            self._update_classification()
    
    def _extract_features(self, bar: Bar) -> Optional[ClassificationFeatures]:
        """
        Extract features from market bar.
        
        Args:
            bar: Market data bar
            
        Returns:
            Extracted features or None if insufficient data
        """
        if len(self.bar_history) < 2:
            return None
        
        prev_bar = self.bar_history[-2]
        
        # Calculate basic features
        returns = (bar.close - prev_bar.close) / prev_bar.close
        price_range = (bar.high - bar.low) / bar.open
        
        # Update volume statistics
        self._volume_count += 1
        self._volume_sma = (
            (self._volume_sma * (self._volume_count - 1) + bar.volume) 
            / self._volume_count
        )
        
        volume_ratio = bar.volume / max(self._volume_sma, 1.0)
        
        # Calculate volatility (rolling std of returns)
        volatility = self._calculate_volatility()
        
        # Calculate momentum
        momentum = self._calculate_momentum(bar)
        
        return ClassificationFeatures(
            returns=returns,
            volatility=volatility,
            volume_ratio=volume_ratio,
            price_range=price_range,
            momentum=momentum,
            timestamp=bar.timestamp,
            symbol=bar.symbol
        )
    
    def _calculate_volatility(self) -> float:
        """Calculate rolling volatility from recent returns."""
        if len(self.bar_history) < 5:
            return 0.0
        
        recent_bars = list(self.bar_history)[-5:]
        returns = []
        
        for i in range(1, len(recent_bars)):
            ret = (recent_bars[i].close - recent_bars[i-1].close) / recent_bars[i-1].close
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return variance ** 0.5
    
    def _calculate_momentum(self, bar: Bar) -> float:
        """Calculate price momentum."""
        if len(self.bar_history) < 10:
            return 0.0
        
        # Use 10-period momentum
        old_bar = self.bar_history[-10]
        momentum = (bar.close - old_bar.close) / old_bar.close
        
        return momentum
    
    def _update_readiness(self) -> None:
        """Update classifier readiness state."""
        min_bars_needed = max(self.config.lookback_period // 4, 10)
        self._is_ready = len(self.bar_history) >= min_bars_needed
    
    def _update_classification(self) -> None:
        """Update classification and confidence."""
        new_regime = self.classify()
        new_confidence = self._calculate_confidence()
        
        # Check for regime change
        if (new_regime != self._current_regime and 
            new_confidence >= self.config.min_confidence):
            
            self._handle_regime_change(new_regime, new_confidence)
        
        self._current_regime = new_regime
        self._confidence = new_confidence
    
    def _handle_regime_change(self, new_regime: MarketRegime, confidence: float) -> None:
        """Handle regime change event."""
        current_time = datetime.now()
        if self.bar_history:
            current_time = self.bar_history[-1].timestamp
        
        # Create new regime state
        regime_state = RegimeState(
            regime=new_regime,
            confidence=confidence,
            started_at=current_time,
            last_updated=current_time,
            duration_bars=1,
            features=self.feature_history[-1].to_dict() if self.feature_history else {}
        )
        
        self.regime_history.append(regime_state)
        
        # Keep only recent history
        if len(self.regime_history) > 50:
            self.regime_history = self.regime_history[-50:]
    
    @abstractmethod
    def classify(self) -> MarketRegime:
        """
        Perform regime classification.
        
        Must be implemented by concrete classifiers.
        
        Returns:
            Classified market regime
        """
        pass
    
    @abstractmethod
    def _calculate_confidence(self) -> float:
        """
        Calculate classification confidence.
        
        Must be implemented by concrete classifiers.
        
        Returns:
            Confidence level (0.0 to 1.0)
        """
        pass
    
    def reset(self) -> None:
        """Reset classifier to initial state."""
        self.bar_history.clear()
        self.feature_history.clear()
        self.regime_history.clear()
        
        self._confidence = 0.0
        self._current_regime = MarketRegime.UNKNOWN
        self._is_ready = False
        self._last_price = None
        self._volume_sma = 0.0
        self._volume_count = 0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current classifier state.
        
        Returns:
            Dictionary containing classifier state information
        """
        return {
            'classifier_type': self.__class__.__name__,
            'current_regime': self._current_regime.value,
            'confidence': self._confidence,
            'is_ready': self._is_ready,
            'bars_processed': len(self.bar_history),
            'features_extracted': len(self.feature_history),
            'regime_changes': len(self.regime_history),
            'last_regime_change': (
                self.regime_history[-1].started_at.isoformat() 
                if self.regime_history else None
            )
        }
    
    def get_recent_regimes(self, count: int = 5) -> List[RegimeState]:
        """
        Get recent regime states.
        
        Args:
            count: Number of recent regimes to return
            
        Returns:
            List of recent regime states
        """
        return self.regime_history[-count:] if self.regime_history else []


class DummyClassifier(BaseClassifier):
    """
    Dummy classifier for testing purposes.
    
    Always returns UNKNOWN regime with low confidence.
    Useful for testing classifier infrastructure without
    complex classification logic.
    """
    
    def classify(self) -> MarketRegime:
        """Always return UNKNOWN regime."""
        return MarketRegime.UNKNOWN
    
    def _calculate_confidence(self) -> float:
        """Always return low confidence."""
        return 0.1 if self.is_ready else 0.0