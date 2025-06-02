"""
Hidden Markov Model (HMM) regime classifier.

Implements market regime detection using HMM for identifying
bull, bear, and neutral market states.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum
import logging

from ...core.events import Event, EventType, EventBus
from ...data.models import MarketData
from .classifier import BaseClassifier
from .regime_types import MarketRegime, RegimeState


logger = logging.getLogger(__name__)


class HMMRegimeState(Enum):
    """HMM-specific regime states."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    
    def to_regime_state(self) -> RegimeState:
        """Convert to standard regime state."""
        mapping = {
            self.STRONG_BULL: RegimeState.BULL_MARKET,
            self.BULL: RegimeState.BULL_MARKET,
            self.NEUTRAL: RegimeState.NEUTRAL,
            self.BEAR: RegimeState.BEAR_MARKET,
            self.STRONG_BEAR: RegimeState.BEAR_MARKET
        }
        return mapping[self]


@dataclass
class HMMParameters:
    """Parameters for HMM model."""
    n_states: int = 5  # Number of hidden states
    lookback_period: int = 20  # Periods for observation window
    
    # Observation features
    use_returns: bool = True
    use_volume: bool = True
    use_volatility: bool = True
    
    # Model parameters
    transition_smoothing: float = 0.1
    emission_smoothing: float = 0.1
    
    # Thresholds
    confidence_threshold: float = 0.6
    min_data_points: int = 50


class HMMClassifier(BaseClassifier):
    """
    Hidden Markov Model classifier for regime detection.
    
    Uses HMM to identify market regimes based on:
    - Price returns
    - Volume patterns
    - Volatility clustering
    """
    
    def __init__(
        self,
        parameters: Optional[HMMParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """Initialize HMM classifier."""
        # Create a dummy config for the base class
        from .regime_types import ClassifierConfig
        config = ClassifierConfig(
            lookback_period=parameters.lookback_period if parameters else 20,
            min_confidence=parameters.confidence_threshold if parameters else 0.6
        )
        super().__init__(config=config)
        
        self.params = parameters or HMMParameters()
        
        # Model components
        self._transition_matrix: Optional[np.ndarray] = None
        self._emission_params: Dict[str, Any] = {}
        self._current_state: Optional[HMMRegimeState] = None
        self._state_probabilities: Optional[np.ndarray] = None
        
        # Data buffers
        self._price_buffer: List[Tuple[datetime, float]] = []
        self._volume_buffer: List[Tuple[datetime, float]] = []
        self._returns_buffer: List[float] = []
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize HMM model parameters."""
        # Initialize transition matrix (equal probability initially)
        n = self.params.n_states
        self._transition_matrix = np.ones((n, n)) / n
        
        # Add slight preference for staying in same state
        np.fill_diagonal(self._transition_matrix, 0.4)
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters for each state
        states = list(HMMRegimeState)[:n]
        for i, state in enumerate(states):
            self._emission_params[state.value] = {
                'mean_return': self._get_initial_mean_return(state),
                'std_return': self._get_initial_std_return(state),
                'volume_factor': self._get_initial_volume_factor(state)
            }
            
    def _get_initial_mean_return(self, state: HMMRegimeState) -> float:
        """Get initial mean return for state."""
        return {
            HMMRegimeState.STRONG_BULL: 0.002,
            HMMRegimeState.BULL: 0.001,
            HMMRegimeState.NEUTRAL: 0.0,
            HMMRegimeState.BEAR: -0.001,
            HMMRegimeState.STRONG_BEAR: -0.002
        }.get(state, 0.0)
        
    def _get_initial_std_return(self, state: HMMRegimeState) -> float:
        """Get initial return volatility for state."""
        return {
            HMMRegimeState.STRONG_BULL: 0.015,
            HMMRegimeState.BULL: 0.012,
            HMMRegimeState.NEUTRAL: 0.010,
            HMMRegimeState.BEAR: 0.013,
            HMMRegimeState.STRONG_BEAR: 0.018
        }.get(state, 0.01)
        
    def _get_initial_volume_factor(self, state: HMMRegimeState) -> float:
        """Get initial volume factor for state."""
        return {
            HMMRegimeState.STRONG_BULL: 1.3,
            HMMRegimeState.BULL: 1.1,
            HMMRegimeState.NEUTRAL: 1.0,
            HMMRegimeState.BEAR: 1.2,
            HMMRegimeState.STRONG_BEAR: 1.5
        }.get(state, 1.0)
        
    def classify_regime(
        self,
        market_data: Dict[str, MarketData],
        timestamp: datetime
    ) -> MarketRegime:
        """
        Classify current market regime using HMM.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            MarketRegime classification
        """
        # Update data buffers
        self._update_buffers(market_data, timestamp)
        
        # Check if we have enough data
        if len(self._price_buffer) < self.params.min_data_points:
            from .regime_types import MarketRegime
            return MarketRegime.UNKNOWN
            
        # Extract features
        features = self._extract_features()
        
        # Run HMM inference
        state_probs = self._forward_algorithm(features)
        
        # Get most likely state
        most_likely_idx = np.argmax(state_probs)
        states = list(HMMRegimeState)[:self.params.n_states]
        most_likely_state = states[most_likely_idx]
        confidence = state_probs[most_likely_idx]
        
        # Update current state
        self._current_state = most_likely_state
        self._state_probabilities = state_probs
        
        # Convert to standard regime
        regime = most_likely_state.to_regime_state()
        
        # Create a simple MarketRegime object for now
        from .regime_types import MarketRegime
        if regime == RegimeState.BULL_MARKET:
            result_regime = MarketRegime.BULL
        elif regime == RegimeState.BEAR_MARKET:
            result_regime = MarketRegime.BEAR
        else:
            result_regime = MarketRegime.NEUTRAL
        
        # Update model parameters online
        if confidence > self.params.confidence_threshold:
            self._update_model_parameters(features, most_likely_idx)
            
        return result_regime
        
    def _update_buffers(
        self,
        market_data: Dict[str, MarketData],
        timestamp: datetime
    ) -> None:
        """Update internal data buffers."""
        # For now, use first symbol as market proxy
        # In practice, could use index or aggregate
        if not market_data:
            return
            
        symbol = next(iter(market_data.keys()))
        data = market_data[symbol]
        
        # Update price buffer
        price = data.close if hasattr(data, 'close') else data.price
        self._price_buffer.append((timestamp, price))
        
        # Update volume buffer
        if hasattr(data, 'volume'):
            self._volume_buffer.append((timestamp, data.volume))
            
        # Calculate return
        if len(self._price_buffer) > 1:
            prev_price = self._price_buffer[-2][1]
            ret = (price - prev_price) / prev_price if prev_price > 0 else 0
            self._returns_buffer.append(ret)
            
        # Limit buffer sizes
        max_size = self.params.lookback_period * 2
        if len(self._price_buffer) > max_size:
            self._price_buffer.pop(0)
        if len(self._volume_buffer) > max_size:
            self._volume_buffer.pop(0)
        if len(self._returns_buffer) > max_size:
            self._returns_buffer.pop(0)
            
    def _extract_features(self) -> Dict[str, np.ndarray]:
        """Extract features for HMM observation."""
        features = {}
        
        # Return features
        if self.params.use_returns and self._returns_buffer:
            recent_returns = self._returns_buffer[-self.params.lookback_period:]
            features['return'] = np.mean(recent_returns)
            features['return_std'] = np.std(recent_returns)
            
        # Volume features
        if self.params.use_volume and self._volume_buffer:
            recent_volumes = [v for _, v in self._volume_buffer[-self.params.lookback_period:]]
            if len(recent_volumes) > 1:
                avg_volume = np.mean(recent_volumes)
                features['volume_ratio'] = recent_volumes[-1] / avg_volume if avg_volume > 0 else 1.0
                
        # Volatility features
        if self.params.use_volatility and len(self._returns_buffer) >= 5:
            # Calculate realized volatility
            recent_returns = self._returns_buffer[-5:]
            features['volatility'] = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
        return features
        
    def _forward_algorithm(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run forward algorithm for HMM inference.
        
        Args:
            features: Observation features
            
        Returns:
            State probabilities
        """
        n_states = self.params.n_states
        states = list(HMMRegimeState)[:n_states]
        
        # Calculate emission probabilities
        emission_probs = np.zeros(n_states)
        
        for i, state in enumerate(states):
            emission_probs[i] = self._calculate_emission_probability(
                features, state
            )
            
        # Normalize emission probabilities
        emission_probs /= emission_probs.sum()
        
        # If no previous state, use uniform prior
        if self._state_probabilities is None:
            prior = np.ones(n_states) / n_states
        else:
            prior = self._state_probabilities
            
        # Forward step: prior * transition * emission
        forward_probs = prior @ self._transition_matrix * emission_probs
        
        # Normalize
        forward_probs /= forward_probs.sum()
        
        return forward_probs
        
    def _calculate_emission_probability(
        self,
        features: Dict[str, np.ndarray],
        state: HMMRegimeState
    ) -> float:
        """Calculate emission probability for state given features."""
        params = self._emission_params[state.value]
        prob = 1.0
        
        # Return likelihood
        if 'return' in features:
            mean = params['mean_return']
            std = params['std_return']
            return_prob = self._gaussian_pdf(features['return'], mean, std)
            prob *= return_prob
            
        # Volume likelihood
        if 'volume_ratio' in features:
            expected_ratio = params['volume_factor']
            volume_prob = self._gaussian_pdf(
                features['volume_ratio'],
                expected_ratio,
                0.3  # Fixed std for volume ratio
            )
            prob *= volume_prob
            
        return prob
        
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Calculate Gaussian probability density."""
        if std == 0:
            return 1.0 if x == mean else 0.0
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        
    def _update_model_parameters(
        self,
        features: Dict[str, np.ndarray],
        state_idx: int
    ) -> None:
        """Update model parameters using online learning."""
        states = list(HMMRegimeState)[:self.params.n_states]
        state = states[state_idx]
        params = self._emission_params[state.value]
        
        # Update emission parameters with exponential smoothing
        alpha = 0.1  # Learning rate
        
        if 'return' in features:
            params['mean_return'] = (1 - alpha) * params['mean_return'] + alpha * features['return']
            
        if 'return_std' in features:
            params['std_return'] = (1 - alpha) * params['std_return'] + alpha * features['return_std']
            
        if 'volume_ratio' in features:
            params['volume_factor'] = (1 - alpha) * params['volume_factor'] + alpha * features['volume_ratio']
            
    def process_indicator_event(self, event: Event) -> None:
        """Process indicator events for additional features."""
        if event.event_type == EventType.INDICATOR:
            # Could incorporate technical indicators into HMM
            # For now, we rely on price/volume data
            pass
            
    def get_state_history(self) -> List[Tuple[datetime, HMMRegimeState, float]]:
        """Get history of state classifications."""
        # This would be implemented with proper state tracking
        # For now, return empty list
        return []
        
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return {
            'transition_matrix': self._transition_matrix.tolist() if self._transition_matrix is not None else None,
            'emission_params': self._emission_params,
            'current_state': self._current_state.value if self._current_state else None,
            'state_probabilities': self._state_probabilities.tolist() if self._state_probabilities is not None else None
        }
    
    def classify(self) -> 'MarketRegime':
        """
        Required abstract method implementation.
        
        Returns:
            Current market regime classification
        """
        if not self.is_ready or len(self.bar_history) < self.params.min_data_points:
            from .regime_types import MarketRegime
            return MarketRegime.UNKNOWN
        
        # Use the most recent bar for classification
        latest_bar = self.bar_history[-1]
        market_data = {latest_bar.symbol: latest_bar}
        
        # Call the main classify_regime method
        regime_result = self.classify_regime(market_data, latest_bar.timestamp)
        return regime_result
    
    def _calculate_confidence(self) -> float:
        """
        Required abstract method implementation.
        
        Returns:
            Classification confidence (0.0 to 1.0)
        """
        if self._state_probabilities is None:
            return 0.0
        
        # Return the probability of the most likely state
        return float(np.max(self._state_probabilities))