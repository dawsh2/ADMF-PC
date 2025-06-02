"""
File: src/strategy/classifiers/classifier_container.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#classifier-container
Step: 3 - Classifier Container
Dependencies: core.events, core.logging

Classifier container with event isolation for market regime detection.
Manages classifier lifecycle and emits regime change events.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid

from ...core.events.enhanced_isolation import get_enhanced_isolation_manager
from ...core.logging.structured import ContainerLogger
from ...data.models import Bar
from .regime_types import (
    MarketRegime, 
    RegimeChangeEvent, 
    RegimeState,
    ClassifierConfig
)
from .classifier import BaseClassifier, DummyClassifier
from .pattern_classifier import PatternClassifier


class ClassifierContainer:
    """
    Container for market regime classification.
    
    Manages classifier lifecycle with event isolation and emits
    regime change events when market conditions change.
    
    Architecture Context:
        - Part of: Step 3 - Classifier Container
        - Implements: Container isolation with regime detection
        - Provides: Market regime classification and change events
        - Dependencies: Enhanced event isolation, structured logging
    
    Example:
        config = ClassifierConfig(classifier_type='pattern')
        container = ClassifierContainer("classifier_001", config)
        container.on_bar(bar)  # Process market data
    """
    
    def __init__(self, container_id: str, config: ClassifierConfig):
        """
        Initialize classifier container.
        
        Args:
            container_id: Unique container identifier
            config: Classifier configuration
        """
        self.container_id = container_id
        self.config = config
        
        # Create isolated event bus
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(
            f"{container_id}_classifier"
        )
        
        # Initialize classifier
        self.classifier = self._create_classifier(config.classifier_type)
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[RegimeState] = []
        self.bars_processed = 0
        self.regime_changes = 0
        
        # Setup logging
        self.logger = ContainerLogger("ClassifierContainer", container_id, "classifier_container")
        
        # Event subscribers
        self.regime_change_subscribers: List[Callable[[RegimeChangeEvent], None]] = []
        
        # Wire internal events
        self._setup_internal_events()
        
        self.logger.info(
            "ClassifierContainer initialized",
            container_id=container_id,
            classifier_type=config.classifier_type,
            lookback_period=config.lookback_period
        )
    
    def _create_classifier(self, classifier_type: str) -> BaseClassifier:
        """
        Factory method for classifier creation.
        
        Args:
            classifier_type: Type of classifier to create
            
        Returns:
            Initialized classifier instance
            
        Raises:
            ValueError: If classifier type is unknown
        """
        if classifier_type == "pattern":
            return PatternClassifier(self.config)
        elif classifier_type == "dummy":
            return DummyClassifier(self.config)
        else:
            # For Step 3, we'll implement HMM classifier later
            # For now, fall back to pattern classifier
            self.logger.warning(
                f"Unknown classifier type '{classifier_type}', using pattern classifier"
            )
            return PatternClassifier(self.config)
    
    def on_bar(self, bar: Bar) -> None:
        """
        Process new market data bar.
        
        Args:
            bar: Market data bar to process
        """
        self.bars_processed += 1
        
        self.logger.trace(
            "Processing market bar",
            symbol=bar.symbol,
            timestamp=bar.timestamp.isoformat() if bar.timestamp else "unknown",
            price=float(bar.close),
            bars_processed=self.bars_processed
        )
        
        # Get previous regime
        old_regime = self.classifier.current_regime
        
        # Update classifier
        self.classifier.update(bar)
        
        # Check for regime change
        new_regime = self.classifier.current_regime
        if new_regime != old_regime and new_regime != MarketRegime.UNKNOWN:
            self._handle_regime_change(old_regime, new_regime, bar.timestamp)
    
    def _handle_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime, 
                            timestamp: datetime) -> None:
        """
        Handle market regime transition.
        
        Args:
            old_regime: Previous market regime
            new_regime: New market regime
            timestamp: Time of regime change
        """
        self.regime_changes += 1
        self.current_regime = new_regime
        
        # Log regime change
        self.logger.info(
            "Regime change detected",
            old_regime=old_regime.value,
            new_regime=new_regime.value,
            confidence=self.classifier.confidence,
            timestamp=timestamp.isoformat() if timestamp else "unknown",
            bars_processed=self.bars_processed
        )
        
        # Get current features if available
        features = {}
        if hasattr(self.classifier, 'get_classification_details'):
            details = self.classifier.get_classification_details()
            features = details.get('metrics', {})
        
        # Create regime change event
        event = RegimeChangeEvent(
            timestamp=timestamp,
            old_regime=old_regime,
            new_regime=new_regime,
            confidence=self.classifier.confidence,
            classifier_id=self.container_id,
            symbol=getattr(self.classifier.bar_history[-1], 'symbol', 'UNKNOWN') if self.classifier.bar_history else 'UNKNOWN',
            features=features,
            metadata={
                'bars_processed': self.bars_processed,
                'regime_changes': self.regime_changes,
                'classifier_type': self.config.classifier_type
            }
        )
        
        # Store in history
        regime_state = RegimeState(
            regime=new_regime,
            confidence=self.classifier.confidence,
            started_at=timestamp,
            last_updated=timestamp,
            duration_bars=1,
            features=features
        )
        
        self.regime_history.append(regime_state)
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        # Emit regime change event
        self.event_bus.publish("REGIME_CHANGE", event)
        
        # Notify direct subscribers
        for subscriber in self.regime_change_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                self.logger.error(
                    "Error notifying regime change subscriber",
                    error=str(e),
                    subscriber=str(subscriber)
                )
    
    def _setup_internal_events(self) -> None:
        """Setup internal event subscriptions."""
        # Subscribe to bar events if needed
        self.event_bus.subscribe("BAR", self._on_bar_event)
        
        self.logger.debug("Internal event subscriptions configured")
    
    def _on_bar_event(self, event_type: str, bar: Bar) -> None:
        """Handle bar events from event bus."""
        if event_type == "BAR":
            self.on_bar(bar)
    
    def subscribe_to_regime_changes(self, callback: Callable[[RegimeChangeEvent], None]) -> None:
        """
        Subscribe to regime change events.
        
        Args:
            callback: Function to call when regime changes
        """
        self.regime_change_subscribers.append(callback)
        
        self.logger.debug(
            "Regime change subscriber added",
            total_subscribers=len(self.regime_change_subscribers)
        )
    
    def get_current_regime(self) -> MarketRegime:
        """
        Get current market regime.
        
        Returns:
            Current market regime classification
        """
        return self.classifier.current_regime
    
    def get_confidence(self) -> float:
        """
        Get current classification confidence.
        
        Returns:
            Confidence level (0.0 to 1.0)
        """
        return self.classifier.confidence
    
    def is_ready(self) -> bool:
        """
        Check if classifier is ready.
        
        Returns:
            True if classifier has enough data
        """
        return self.classifier.is_ready
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current container state.
        
        Returns:
            Dictionary containing container state information
        """
        classifier_state = self.classifier.get_state()
        
        return {
            'container_id': self.container_id,
            'current_regime': self.current_regime.value,
            'confidence': self.classifier.confidence,
            'is_ready': self.classifier.is_ready,
            'bars_processed': self.bars_processed,
            'regime_changes': self.regime_changes,
            'regime_history_length': len(self.regime_history),
            'classifier_state': classifier_state,
            'config': {
                'classifier_type': self.config.classifier_type,
                'lookback_period': self.config.lookback_period,
                'min_confidence': self.config.min_confidence
            },
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
    
    def get_classification_details(self) -> Dict[str, Any]:
        """
        Get detailed classification information.
        
        Returns:
            Dictionary with detailed classification data
        """
        base_details = {
            'container_id': self.container_id,
            'current_regime': self.current_regime.value,
            'confidence': self.classifier.confidence,
            'is_ready': self.classifier.is_ready,
            'bars_processed': self.bars_processed,
            'regime_changes': self.regime_changes
        }
        
        # Add classifier-specific details if available
        if hasattr(self.classifier, 'get_classification_details'):
            classifier_details = self.classifier.get_classification_details()
            base_details['classifier_details'] = classifier_details
        
        return base_details
    
    def reset(self) -> None:
        """
        Reset container state.
        
        This method supports backtesting scenarios where containers
        need to be reset between test runs.
        """
        self.classifier.reset()
        self.regime_history.clear()
        
        self.current_regime = MarketRegime.UNKNOWN
        self.bars_processed = 0
        self.regime_changes = 0
        
        self.logger.info("ClassifierContainer reset")
    
    def cleanup(self) -> None:
        """
        Cleanup container resources.
        
        This method should be called when the container is no longer needed
        to properly release event bus resources and log final state.
        """
        self.logger.info(
            "ClassifierContainer cleanup",
            final_state=self.get_state()
        )
        
        # Clear subscribers
        self.regime_change_subscribers.clear()
        
        # Remove from isolation manager
        if self.isolation_manager:
            self.isolation_manager.remove_container_bus(f"{self.container_id}_classifier")


def create_test_classifier_container(
    container_id: str = "test_classifier",
    classifier_type: str = "pattern"
) -> ClassifierContainer:
    """
    Create a classifier container for testing.
    
    Args:
        container_id: Container identifier
        classifier_type: Type of classifier to use
        
    Returns:
        Configured classifier container for testing
    """
    config = ClassifierConfig(
        classifier_type=classifier_type,
        lookback_period=30,
        min_confidence=0.6,
        atr_period=14,
        trend_period=20,
        volatility_threshold=0.02,
        trend_threshold=0.5
    )
    
    return ClassifierContainer(container_id, config)


def create_conservative_classifier_container(
    container_id: str = "conservative_classifier"
) -> ClassifierContainer:
    """
    Create a conservative classifier container.
    
    Args:
        container_id: Container identifier
        
    Returns:
        Conservatively configured classifier container
    """
    config = ClassifierConfig(
        classifier_type="pattern",
        lookback_period=50,
        min_confidence=0.8,  # High confidence required
        atr_period=20,
        trend_period=30,
        volatility_threshold=0.015,  # Lower threshold = more sensitive
        trend_threshold=0.6  # Higher threshold = less sensitive
    )
    
    return ClassifierContainer(container_id, config)