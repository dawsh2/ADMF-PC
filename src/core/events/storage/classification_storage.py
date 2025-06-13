"""
Sparse storage for CLASSIFICATION events.

Only stores regime changes, not repeated classifications.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging

from ..types import Event, EventType
from ..protocols import EventObserverProtocol

logger = logging.getLogger(__name__)


class ClassificationStorage(EventObserverProtocol):
    """
    Sparse storage for classification events.
    
    Only stores:
    - Initial classification for each classifier/symbol pair
    - Regime changes (when classification changes)
    - Confidence threshold breaches (optional)
    
    This dramatically reduces storage compared to storing every classification.
    """
    
    def __init__(self, confidence_threshold: Optional[float] = None):
        """
        Initialize classification storage.
        
        Args:
            confidence_threshold: Optional threshold - store when confidence changes significantly
        """
        # Store regime changes: {(classifier_id, symbol): [(timestamp, regime, confidence, features)]}
        self._regime_history: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        
        # Current regime for each classifier/symbol
        self._current_regimes: Dict[Tuple[str, str], Dict] = {}
        
        # Optional confidence threshold for additional storage triggers
        self._confidence_threshold = confidence_threshold
        
        # Metrics
        self._events_received = 0
        self._events_stored = 0
    
    def on_event(self, event: Event) -> None:
        """Handle incoming events - only process CLASSIFICATION events."""
        if event.event_type != EventType.CLASSIFICATION.value:
            return
        
        self._events_received += 1
        
        payload = event.payload
        classifier_id = payload.get('classifier_id')
        symbol = payload.get('symbol')
        regime = payload.get('regime')
        confidence = payload.get('confidence', 0.0)
        features = payload.get('features', {})
        is_regime_change = payload.get('is_regime_change', False)
        
        if not all([classifier_id, symbol, regime]):
            logger.warning(f"Invalid classification event: {payload}")
            return
        
        key = (classifier_id, symbol)
        current = self._current_regimes.get(key)
        
        # Determine if we should store this event
        should_store = False
        store_reason = ""
        
        if current is None:
            # First classification for this classifier/symbol
            should_store = True
            store_reason = "initial"
        elif is_regime_change:
            # Regime changed
            should_store = True
            store_reason = "regime_change"
        elif self._confidence_threshold and current:
            # Check confidence threshold breach
            confidence_delta = abs(confidence - current.get('confidence', 0.0))
            if confidence_delta >= self._confidence_threshold:
                should_store = True
                store_reason = f"confidence_breach ({confidence_delta:.3f})"
        
        if should_store:
            # Store the classification
            classification_record = {
                'timestamp': event.timestamp,
                'regime': regime,
                'confidence': confidence,
                'features': features.copy(),  # Store snapshot of features
                'reason': store_reason,
                'previous_regime': payload.get('previous_regime')
            }
            
            self._regime_history[key].append(classification_record)
            self._current_regimes[key] = {
                'regime': regime,
                'confidence': confidence,
                'timestamp': event.timestamp
            }
            
            self._events_stored += 1
            
            logger.info(f"Stored classification: {classifier_id}/{symbol} -> {regime} "
                       f"(confidence: {confidence:.3f}, reason: {store_reason})")
        else:
            logger.debug(f"Skipped classification: {classifier_id}/{symbol} -> {regime} "
                        f"(unchanged, confidence: {confidence:.3f})")
    
    def get_regime_history(self, classifier_id: str, symbol: str) -> List[Dict]:
        """Get regime history for a classifier/symbol pair."""
        key = (classifier_id, symbol)
        return self._regime_history.get(key, [])
    
    def get_current_regime(self, classifier_id: str, symbol: str) -> Optional[Dict]:
        """Get current regime for a classifier/symbol pair."""
        key = (classifier_id, symbol)
        return self._current_regimes.get(key)
    
    def get_regime_at_time(self, classifier_id: str, symbol: str, 
                          timestamp: datetime) -> Optional[Dict]:
        """Get regime that was active at a specific time."""
        history = self.get_regime_history(classifier_id, symbol)
        
        if not history:
            return None
        
        # Find the regime active at the given timestamp
        active_regime = None
        for record in history:
            if record['timestamp'] <= timestamp:
                active_regime = record
            else:
                break
        
        return active_regime
    
    def get_regime_durations(self, classifier_id: str, symbol: str) -> List[Dict]:
        """Calculate duration of each regime period."""
        history = self.get_regime_history(classifier_id, symbol)
        
        if len(history) < 2:
            return []
        
        durations = []
        for i in range(len(history) - 1):
            current = history[i]
            next_change = history[i + 1]
            
            duration = (next_change['timestamp'] - current['timestamp']).total_seconds()
            
            durations.append({
                'regime': current['regime'],
                'start': current['timestamp'],
                'end': next_change['timestamp'],
                'duration_seconds': duration,
                'confidence': current['confidence']
            })
        
        # Add current regime (still active)
        if history:
            last = history[-1]
            durations.append({
                'regime': last['regime'],
                'start': last['timestamp'],
                'end': None,  # Still active
                'duration_seconds': None,
                'confidence': last['confidence']
            })
        
        return durations
    
    def get_metrics(self) -> Dict:
        """Get storage metrics."""
        return {
            'events_received': self._events_received,
            'events_stored': self._events_stored,
            'storage_ratio': self._events_stored / max(self._events_received, 1),
            'unique_pairs': len(self._current_regimes),
            'total_regime_changes': sum(len(history) for history in self._regime_history.values()),
            'classifiers': len(set(key[0] for key in self._current_regimes)),
            'symbols': len(set(key[1] for key in self._current_regimes))
        }
    
    def clear(self):
        """Clear all stored data."""
        self._regime_history.clear()
        self._current_regimes.clear()
        self._events_received = 0
        self._events_stored = 0