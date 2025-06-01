# Step 10.5: Regime Adaptation

**Status**: Intermediate Complexity Step
**Complexity**: Very High
**Prerequisites**: [Step 10.4: Market Making](step-10.4-market-making.md) completed
**Architecture Ref**: [Adaptive Systems Architecture](../architecture/adaptive-systems-architecture.md)

## 🎯 Objective

Implement adaptive trading systems that adjust to market regimes:
- Real-time regime detection and classification
- Regime-specific parameter sets and strategies
- Dynamic strategy switching mechanisms
- Performance decay detection and adaptation
- Adaptive position sizing and risk management
- Machine learning-based regime prediction

## 📋 Required Reading

Before starting:
1. [Market Regime Theory](../references/market-regimes.md)
2. [Adaptive Control Systems](../references/adaptive-control.md)
3. [Online Learning Algorithms](../references/online-learning.md)

## 🏗️ Implementation Tasks

### 1. Real-Time Regime Detection

```python
# src/adaptation/regime_detection.py
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"

@dataclass
class RegimeSignal:
    """Signal indicating regime characteristics"""
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    
    # Persistence metrics
    time_in_regime: timedelta
    regime_strength: float
    transition_probability: Dict[MarketRegime, float]
    
    # Market characteristics
    volatility_regime: str
    trend_regime: str
    liquidity_regime: str
    correlation_regime: str

@dataclass
class RegimeTransition:
    """Detected regime transition"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    confidence: float
    
    # Transition characteristics
    abrupt: bool  # True for sudden transitions
    probability: float
    expected_duration: timedelta

class RealTimeRegimeDetector:
    """
    Real-time market regime detection using multiple methods.
    Combines statistical and machine learning approaches.
    """
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        
        # Detection models
        self.hmm_detector = HMMRegimeDetector()
        self.statistical_detector = StatisticalRegimeDetector()
        self.ml_detector = MLRegimeDetector()
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[Tuple[datetime, MarketRegime, float]] = []
        self.feature_history: List[Dict[str, float]] = []
        
        # Performance tracking
        self.regime_performance: Dict[MarketRegime, Dict] = {}
        self.transition_accuracy: Dict[Tuple[MarketRegime, MarketRegime], float] = {}
        
        self.logger = ComponentLogger("RealTimeRegimeDetector", "adaptation")
    
    def detect_regime(self, market_data: Dict[str, pd.DataFrame],
                     current_time: datetime) -> RegimeSignal:
        """Detect current market regime using ensemble approach"""
        
        # Extract features for regime detection
        features = self._extract_regime_features(market_data, current_time)
        self.feature_history.append(features)
        
        # Keep only recent history
        if len(self.feature_history) > self.lookback_window:
            self.feature_history = self.feature_history[-self.lookback_window:]
        
        # Run multiple detection methods
        detections = {}
        
        # Method 1: HMM-based detection
        if len(self.feature_history) > 50:
            hmm_result = self.hmm_detector.detect(self.feature_history)
            detections['hmm'] = hmm_result
        
        # Method 2: Statistical rules
        stat_result = self.statistical_detector.detect(features, self.feature_history)
        detections['statistical'] = stat_result
        
        # Method 3: Machine learning
        if len(self.feature_history) > 100:
            ml_result = self.ml_detector.detect(features, self.feature_history)
            detections['ml'] = ml_result
        
        # Ensemble combination
        regime_signal = self._combine_detections(detections, features, current_time)
        
        # Update state
        self._update_regime_state(regime_signal, current_time)
        
        return regime_signal
    
    def _extract_regime_features(self, market_data: Dict[str, pd.DataFrame],
                                current_time: datetime) -> Dict[str, float]:
        """Extract features for regime classification"""
        
        features = {}
        
        # Assume we have data for primary asset
        primary_asset = list(market_data.keys())[0]
        data = market_data[primary_asset]
        
        if len(data) < 20:
            return features
        
        # Returns and volatility features
        returns = data['close'].pct_change().dropna()
        
        if len(returns) > 0:
            # Trend features
            features['return_1d'] = returns.iloc[-1] if len(returns) > 0 else 0
            features['return_5d'] = returns.tail(5).mean() if len(returns) >= 5 else 0
            features['return_20d'] = returns.tail(20).mean() if len(returns) >= 20 else 0
            
            # Volatility features
            features['vol_5d'] = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0
            features['vol_20d'] = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0
            
            # Volatility of volatility
            if len(returns) >= 20:
                rolling_vol = returns.rolling(5).std()
                features['vol_of_vol'] = rolling_vol.tail(10).std() if len(rolling_vol) >= 10 else 0
            
            # Skewness and kurtosis
            if len(returns) >= 20:
                features['skewness_20d'] = returns.tail(20).skew()
                features['kurtosis_20d'] = returns.tail(20).kurtosis()
            
            # Trend strength
            if len(data) >= 20:
                ma_5 = data['close'].rolling(5).mean()
                ma_20 = data['close'].rolling(20).mean()
                
                features['ma_trend'] = (ma_5.iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1] if not ma_20.empty else 0
                
                # Price momentum
                features['momentum_20d'] = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20] if len(data) >= 20 else 0
        
        # Volume features
        if 'volume' in data.columns and len(data) >= 20:
            volume = data['volume']
            features['volume_trend'] = (volume.tail(5).mean() - volume.tail(20).mean()) / volume.tail(20).mean()
            features['volume_volatility'] = volume.tail(20).std() / volume.tail(20).mean()
        
        # Multi-asset features (if available)
        if len(market_data) > 1:
            correlation_features = self._calculate_cross_asset_features(market_data)
            features.update(correlation_features)
        
        return features
    
    def _calculate_cross_asset_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate cross-asset regime features"""
        
        features = {}
        
        # Get returns for all assets
        returns_data = {}
        for asset, data in market_data.items():
            if len(data) >= 20:
                returns_data[asset] = data['close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return features
        
        # Align returns
        min_length = min(len(r) for r in returns_data.values())
        aligned_returns = pd.DataFrame({
            asset: returns.tail(min_length).values
            for asset, returns in returns_data.items()
        })
        
        if len(aligned_returns) >= 20:
            # Average correlation
            corr_matrix = aligned_returns.tail(20).corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            features['avg_correlation'] = corr_matrix.values[mask].mean()
            
            # Correlation stability
            if len(aligned_returns) >= 40:
                recent_corr = aligned_returns.tail(20).corr()
                older_corr = aligned_returns.tail(40).head(20).corr()
                
                corr_diff = np.abs(recent_corr.values - older_corr.values)
                features['correlation_stability'] = 1 - corr_diff[mask].mean()
            
            # Dispersion
            features['cross_sectional_vol'] = aligned_returns.tail(20).std(axis=1).mean()
        
        return features
    
    def _combine_detections(self, detections: Dict[str, Tuple[MarketRegime, float]],
                          features: Dict[str, float],
                          current_time: datetime) -> RegimeSignal:
        """Combine multiple detection methods"""
        
        if not detections:
            return RegimeSignal(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features=features,
                time_in_regime=timedelta(),
                regime_strength=0.0,
                transition_probability={},
                volatility_regime="unknown",
                trend_regime="unknown", 
                liquidity_regime="unknown",
                correlation_regime="unknown"
            )
        
        # Voting mechanism
        regime_votes = {}
        confidence_weights = {}
        
        for method, (regime, confidence) in detections.items():
            if regime not in regime_votes:
                regime_votes[regime] = 0
                confidence_weights[regime] = 0
            
            # Weight votes by confidence and method reliability
            method_weight = {
                'hmm': 0.4,
                'statistical': 0.3,
                'ml': 0.3
            }.get(method, 0.2)
            
            vote_weight = confidence * method_weight
            regime_votes[regime] += vote_weight
            confidence_weights[regime] += vote_weight
        
        # Select regime with highest weighted vote
        if regime_votes:
            best_regime = max(regime_votes, key=regime_votes.get)
            regime_confidence = confidence_weights[best_regime] / sum(confidence_weights.values())
        else:
            best_regime = MarketRegime.UNKNOWN
            regime_confidence = 0.0
        
        # Calculate additional metrics
        time_in_regime = self._calculate_time_in_regime(best_regime, current_time)
        regime_strength = self._calculate_regime_strength(features, best_regime)
        transition_probs = self._estimate_transition_probabilities(best_regime)
        
        # Classify sub-regimes
        vol_regime = self._classify_volatility_regime(features)
        trend_regime = self._classify_trend_regime(features)
        liquidity_regime = self._classify_liquidity_regime(features)
        correlation_regime = self._classify_correlation_regime(features)
        
        return RegimeSignal(
            regime=best_regime,
            confidence=regime_confidence,
            features=features,
            time_in_regime=time_in_regime,
            regime_strength=regime_strength,
            transition_probability=transition_probs,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            liquidity_regime=liquidity_regime,
            correlation_regime=correlation_regime
        )

class StatisticalRegimeDetector:
    """Rule-based regime detection using statistical measures"""
    
    def detect(self, current_features: Dict[str, float],
              feature_history: List[Dict[str, float]]) -> Tuple[MarketRegime, float]:
        """Detect regime using statistical rules"""
        
        if not current_features:
            return MarketRegime.UNKNOWN, 0.0
        
        # Extract key features
        vol_20d = current_features.get('vol_20d', 0)
        return_20d = current_features.get('return_20d', 0)
        vol_of_vol = current_features.get('vol_of_vol', 0)
        
        # Volatility thresholds
        low_vol_threshold = 0.15
        high_vol_threshold = 0.25
        crisis_vol_threshold = 0.4
        
        # Return thresholds (annualized)
        bull_threshold = 0.05
        bear_threshold = -0.05
        
        confidence = 0.5  # Base confidence
        
        # Crisis detection (highest priority)
        if vol_20d > crisis_vol_threshold or vol_of_vol > 0.1:
            return MarketRegime.CRISIS, 0.9
        
        # High volatility regimes
        if vol_20d > high_vol_threshold:
            if return_20d > bull_threshold:
                regime = MarketRegime.BULL_HIGH_VOL
                confidence = 0.8
            elif return_20d < bear_threshold:
                regime = MarketRegime.BEAR_HIGH_VOL
                confidence = 0.8
            else:
                regime = MarketRegime.SIDEWAYS_HIGH_VOL
                confidence = 0.7
        
        # Low volatility regimes
        elif vol_20d < low_vol_threshold:
            if return_20d > bull_threshold:
                regime = MarketRegime.BULL_LOW_VOL
                confidence = 0.8
            elif return_20d < bear_threshold:
                regime = MarketRegime.BEAR_LOW_VOL
                confidence = 0.8
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
                confidence = 0.7
        
        # Medium volatility regimes
        else:
            if return_20d > bull_threshold:
                regime = MarketRegime.BULL_LOW_VOL  # Default to low vol
                confidence = 0.6
            elif return_20d < bear_threshold:
                regime = MarketRegime.BEAR_LOW_VOL
                confidence = 0.6
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
                confidence = 0.5
        
        return regime, confidence

class HMMRegimeDetector:
    """Hidden Markov Model regime detection"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.last_fit_time = None
        self.refit_frequency = timedelta(days=7)  # Refit weekly
    
    def detect(self, feature_history: List[Dict[str, float]]) -> Tuple[MarketRegime, float]:
        """Detect regime using HMM"""
        
        if len(feature_history) < 50:
            return MarketRegime.UNKNOWN, 0.0
        
        # Convert features to array
        feature_array = self._features_to_array(feature_history)
        
        # Fit or update model
        if self._should_refit():
            self._fit_model(feature_array)
        
        if self.model is None:
            return MarketRegime.UNKNOWN, 0.0
        
        # Predict current state
        current_state = self.model.predict(feature_array)[-1]
        
        # Get state probabilities
        state_probs = self.model.predict_proba(feature_array)[-1]
        confidence = state_probs[current_state]
        
        # Map HMM state to market regime
        regime = self._map_state_to_regime(current_state, feature_history[-1])
        
        return regime, confidence
    
    def _features_to_array(self, feature_history: List[Dict[str, float]]) -> np.ndarray:
        """Convert feature history to numpy array"""
        
        # Select key features
        key_features = ['return_20d', 'vol_20d', 'vol_of_vol', 'momentum_20d']
        
        feature_array = []
        for features in feature_history:
            row = [features.get(f, 0.0) for f in key_features]
            feature_array.append(row)
        
        return np.array(feature_array)
    
    def _fit_model(self, feature_array: np.ndarray) -> None:
        """Fit HMM model"""
        
        try:
            from hmmlearn import hmm
            
            # Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.model.fit(feature_array)
            self.last_fit_time = datetime.now()
            
        except ImportError:
            # Fallback to simple clustering
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.model.fit(feature_array)
            self.last_fit_time = datetime.now()
    
    def _should_refit(self) -> bool:
        """Check if model should be refit"""
        if self.model is None:
            return True
        
        if self.last_fit_time is None:
            return True
        
        return datetime.now() - self.last_fit_time > self.refit_frequency
    
    def _map_state_to_regime(self, state: int, features: Dict[str, float]) -> MarketRegime:
        """Map HMM state to interpretable market regime"""
        
        # Use current features to interpret state
        vol_20d = features.get('vol_20d', 0)
        return_20d = features.get('return_20d', 0)
        
        # Simple mapping based on features
        if vol_20d > 0.3:
            return MarketRegime.CRISIS
        elif vol_20d > 0.2:
            if return_20d > 0:
                return MarketRegime.BULL_HIGH_VOL
            else:
                return MarketRegime.BEAR_HIGH_VOL
        else:
            if return_20d > 0.02:
                return MarketRegime.BULL_LOW_VOL
            elif return_20d < -0.02:
                return MarketRegime.BEAR_LOW_VOL
            else:
                return MarketRegime.SIDEWAYS_LOW_VOL
```

### 2. Adaptive Strategy Framework

```python
# src/adaptation/adaptive_strategy.py
class AdaptiveStrategy:
    """
    Base class for strategies that adapt to market regimes.
    Automatically switches parameters and behavior.
    """
    
    def __init__(self, base_strategy: TradingStrategy,
                 regime_detector: RealTimeRegimeDetector):
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector
        
        # Regime-specific configurations
        self.regime_configs: Dict[MarketRegime, Dict] = {}
        self.regime_strategies: Dict[MarketRegime, TradingStrategy] = {}
        
        # Adaptation tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_changes = []
        self.performance_by_regime: Dict[MarketRegime, PerformanceTracker] = {}
        
        # Adaptation parameters
        self.adaptation_sensitivity = 0.7  # Threshold for regime changes
        self.min_regime_duration = timedelta(minutes=30)  # Minimum time in regime
        self.last_regime_change = datetime.now()
        
        self.logger = ComponentLogger("AdaptiveStrategy", "adaptation")
    
    def configure_regime(self, regime: MarketRegime, 
                        config: Dict,
                        strategy: Optional[TradingStrategy] = None) -> None:
        """Configure parameters for specific regime"""
        
        self.regime_configs[regime] = config
        
        if strategy:
            self.regime_strategies[regime] = strategy
        
        # Initialize performance tracker
        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = PerformanceTracker()
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> Signal:
        """Generate signal with regime adaptation"""
        
        current_time = datetime.now()
        
        # Detect current regime
        regime_signal = self.regime_detector.detect_regime(market_data, current_time)
        
        # Check if regime change should trigger adaptation
        should_adapt = self._should_adapt_regime(regime_signal, current_time)
        
        if should_adapt:
            self._adapt_to_regime(regime_signal.regime, current_time)
        
        # Get strategy for current regime
        active_strategy = self._get_active_strategy(regime_signal.regime)
        
        # Generate base signal
        base_signal = active_strategy.generate_signal(market_data)
        
        # Apply regime-specific adjustments
        adapted_signal = self._apply_regime_adjustments(
            base_signal, regime_signal, market_data
        )
        
        # Track performance
        self._track_regime_performance(regime_signal.regime, adapted_signal)
        
        return adapted_signal
    
    def _should_adapt_regime(self, regime_signal: RegimeSignal,
                           current_time: datetime) -> bool:
        """Determine if regime adaptation should occur"""
        
        # Don't adapt too frequently
        time_since_change = current_time - self.last_regime_change
        if time_since_change < self.min_regime_duration:
            return False
        
        # Check confidence threshold
        if regime_signal.confidence < self.adaptation_sensitivity:
            return False
        
        # Check if regime actually changed
        if regime_signal.regime == self.current_regime:
            return False
        
        # Additional stability checks
        if regime_signal.time_in_regime < timedelta(minutes=5):
            return False  # Too new, wait for stability
        
        return True
    
    def _adapt_to_regime(self, new_regime: MarketRegime,
                        current_time: datetime) -> None:
        """Adapt strategy to new regime"""
        
        old_regime = self.current_regime
        
        self.logger.info(f"Regime change: {old_regime} -> {new_regime}")
        
        # Record regime change
        self.regime_changes.append({
            'timestamp': current_time,
            'from_regime': old_regime,
            'to_regime': new_regime
        })
        
        # Update current regime
        self.current_regime = new_regime
        self.last_regime_change = current_time
        
        # Apply regime-specific configuration
        if new_regime in self.regime_configs:
            config = self.regime_configs[new_regime]
            self._apply_regime_config(config)
        
        # Switch to regime-specific strategy if available
        if new_regime in self.regime_strategies:
            self.base_strategy = self.regime_strategies[new_regime]
    
    def _get_active_strategy(self, regime: MarketRegime) -> TradingStrategy:
        """Get strategy for current regime"""
        
        if regime in self.regime_strategies:
            return self.regime_strategies[regime]
        
        return self.base_strategy
    
    def _apply_regime_adjustments(self, base_signal: Signal,
                                regime_signal: RegimeSignal,
                                market_data: Dict[str, pd.DataFrame]) -> Signal:
        """Apply regime-specific signal adjustments"""
        
        adjusted_signal = base_signal.copy()
        
        # Volatility adjustments
        if regime_signal.volatility_regime == "high":
            # Reduce position size in high volatility
            adjusted_signal.strength *= 0.7
            adjusted_signal.metadata['volatility_adjustment'] = 0.7
        
        # Trend adjustments
        if regime_signal.trend_regime == "strong_trend":
            # Increase conviction in trending markets
            if base_signal.direction != SignalDirection.FLAT:
                adjusted_signal.strength *= 1.2
                adjusted_signal.metadata['trend_adjustment'] = 1.2
        
        # Crisis adjustments
        if regime_signal.regime == MarketRegime.CRISIS:
            # Defensive positioning in crisis
            adjusted_signal.strength *= 0.3
            adjusted_signal.metadata['crisis_adjustment'] = 0.3
        
        # Correlation adjustments
        if regime_signal.correlation_regime == "high_correlation":
            # Reduce diversification benefits
            adjusted_signal.strength *= 0.8
            adjusted_signal.metadata['correlation_adjustment'] = 0.8
        
        return adjusted_signal
    
    def _apply_regime_config(self, config: Dict) -> None:
        """Apply regime-specific configuration"""
        
        # Update base strategy parameters
        for param, value in config.items():
            if hasattr(self.base_strategy, param):
                setattr(self.base_strategy, param, value)
                self.logger.debug(f"Updated {param} = {value}")

class PerformanceDecayDetector:
    """
    Detects when strategy performance is decaying.
    Triggers parameter adaptation or strategy switching.
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.performance_history: List[Tuple[datetime, float]] = []
        self.decay_threshold = 0.3  # 30% decay triggers adaptation
        
    def add_performance_point(self, timestamp: datetime, pnl: float) -> None:
        """Add performance observation"""
        
        self.performance_history.append((timestamp, pnl))
        
        # Keep only recent history
        if len(self.performance_history) > self.lookback_window:
            self.performance_history = self.performance_history[-self.lookback_window:]
    
    def detect_decay(self) -> Tuple[bool, float, Dict]:
        """Detect performance decay"""
        
        if len(self.performance_history) < 20:
            return False, 0.0, {}
        
        # Extract recent performance
        recent_pnl = [pnl for _, pnl in self.performance_history]
        
        # Calculate rolling metrics
        decay_metrics = self._calculate_decay_metrics(recent_pnl)
        
        # Overall decay score
        decay_score = self._calculate_decay_score(decay_metrics)
        
        # Determine if decay threshold exceeded
        is_decaying = decay_score > self.decay_threshold
        
        return is_decaying, decay_score, decay_metrics
    
    def _calculate_decay_metrics(self, pnl_series: List[float]) -> Dict:
        """Calculate various decay metrics"""
        
        metrics = {}
        
        # Trend analysis
        x = np.arange(len(pnl_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, pnl_series)
        
        metrics['trend_slope'] = slope
        metrics['trend_r_squared'] = r_value ** 2
        metrics['trend_p_value'] = p_value
        
        # Rolling averages
        if len(pnl_series) >= 20:
            recent_avg = np.mean(pnl_series[-10:])  # Last 10 observations
            earlier_avg = np.mean(pnl_series[-20:-10])  # Previous 10 observations
            
            if earlier_avg != 0:
                metrics['recent_vs_earlier'] = (recent_avg - earlier_avg) / abs(earlier_avg)
            else:
                metrics['recent_vs_earlier'] = 0
        
        # Volatility analysis
        metrics['recent_volatility'] = np.std(pnl_series[-10:]) if len(pnl_series) >= 10 else 0
        metrics['overall_volatility'] = np.std(pnl_series)
        
        # Drawdown analysis
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-6)
        
        metrics['max_drawdown'] = abs(np.min(drawdown))
        metrics['current_drawdown'] = abs(drawdown[-1])
        
        return metrics
    
    def _calculate_decay_score(self, metrics: Dict) -> float:
        """Calculate overall decay score"""
        
        score = 0.0
        
        # Negative trend
        if metrics.get('trend_slope', 0) < 0:
            score += abs(metrics['trend_slope']) * 0.3
        
        # Recent performance worse than earlier
        if metrics.get('recent_vs_earlier', 0) < -0.1:
            score += abs(metrics['recent_vs_earlier']) * 0.4
        
        # High drawdown
        if metrics.get('current_drawdown', 0) > 0.1:
            score += metrics['current_drawdown'] * 0.3
        
        return min(score, 1.0)  # Cap at 1.0
```

### 3. Dynamic Parameter Optimization

```python
# src/adaptation/parameter_optimization.py
class DynamicParameterOptimizer:
    """
    Continuously optimizes strategy parameters.
    Uses online learning and bandit algorithms.
    """
    
    def __init__(self, parameter_space: Dict[str, Tuple[float, float]]):
        self.parameter_space = parameter_space
        self.parameter_names = list(parameter_space.keys())
        
        # Current best parameters
        self.current_params = {
            name: (bounds[0] + bounds[1]) / 2  # Start at midpoint
            for name, bounds in parameter_space.items()
        }
        
        # Optimization history
        self.evaluation_history: List[Tuple[Dict, float, datetime]] = []
        
        # Online optimizers
        self.bandit_optimizer = MultiArmedBandit(parameter_space)
        self.gradient_optimizer = OnlineGradientOptimizer(parameter_space)
        
        # Exploration parameters
        self.exploration_rate = 0.1
        self.exploitation_weight = 0.8
        
        self.logger = ComponentLogger("DynamicParameterOptimizer", "adaptation")
    
    def suggest_parameters(self, current_performance: float,
                         regime_context: RegimeSignal) -> Dict[str, float]:
        """Suggest next parameter set to try"""
        
        # Update optimizers with current performance
        if self.evaluation_history:
            self._update_optimizers(current_performance)
        
        # Choose optimization method
        if np.random.random() < self.exploration_rate:
            # Exploration: try bandit approach
            suggested_params = self.bandit_optimizer.suggest()
        else:
            # Exploitation: use gradient approach
            suggested_params = self.gradient_optimizer.suggest()
        
        # Apply regime-specific adjustments
        adjusted_params = self._adjust_for_regime(suggested_params, regime_context)
        
        # Record suggestion
        self.evaluation_history.append((
            adjusted_params.copy(),
            None,  # Performance will be updated later
            datetime.now()
        ))
        
        return adjusted_params
    
    def update_performance(self, performance: float) -> None:
        """Update performance for most recent parameter suggestion"""
        
        if self.evaluation_history:
            # Update most recent entry
            params, _, timestamp = self.evaluation_history[-1]
            self.evaluation_history[-1] = (params, performance, timestamp)
            
            # Update current best if better
            if len(self.evaluation_history) == 1 or performance > self._get_best_performance():
                self.current_params = params.copy()
                self.logger.info(f"New best parameters: {self.current_params}, performance: {performance}")
    
    def _update_optimizers(self, performance: float) -> None:
        """Update both optimizers with latest performance"""
        
        if not self.evaluation_history:
            return
        
        latest_params, _, _ = self.evaluation_history[-1]
        
        # Update bandit
        self.bandit_optimizer.update(latest_params, performance)
        
        # Update gradient optimizer
        self.gradient_optimizer.update(latest_params, performance)
    
    def _adjust_for_regime(self, base_params: Dict[str, float],
                          regime_context: RegimeSignal) -> Dict[str, float]:
        """Adjust parameters based on current regime"""
        
        adjusted = base_params.copy()
        
        # Volatility adjustments
        if regime_context.volatility_regime == "high":
            # More conservative parameters in high volatility
            for param in adjusted:
                if 'risk' in param.lower() or 'position' in param.lower():
                    adjusted[param] *= 0.7
        
        # Trend adjustments
        if regime_context.trend_regime == "strong_trend":
            # More aggressive trend following
            for param in adjusted:
                if 'momentum' in param.lower() or 'trend' in param.lower():
                    adjusted[param] *= 1.3
        
        return adjusted
    
    def _get_best_performance(self) -> float:
        """Get best performance so far"""
        
        performances = [perf for _, perf, _ in self.evaluation_history if perf is not None]
        
        return max(performances) if performances else float('-inf')

class MultiArmedBandit:
    """Multi-armed bandit for parameter optimization"""
    
    def __init__(self, parameter_space: Dict[str, Tuple[float, float]],
                 n_arms: int = 20):
        self.parameter_space = parameter_space
        self.n_arms = n_arms
        
        # Create discrete arms
        self.arms = self._create_arms()
        
        # Bandit state
        self.arm_counts = np.zeros(len(self.arms))
        self.arm_rewards = np.zeros(len(self.arms))
        
        # UCB parameters
        self.c = 2.0  # Exploration parameter
        self.total_pulls = 0
    
    def _create_arms(self) -> List[Dict[str, float]]:
        """Create discrete parameter combinations (arms)"""
        
        arms = []
        
        # Create random combinations
        for _ in range(self.n_arms):
            arm = {}
            for param, (low, high) in self.parameter_space.items():
                arm[param] = np.random.uniform(low, high)
            arms.append(arm)
        
        return arms
    
    def suggest(self) -> Dict[str, float]:
        """Suggest arm using UCB algorithm"""
        
        if self.total_pulls == 0:
            # First pull, choose randomly
            return self.arms[0].copy()
        
        # Calculate UCB values
        ucb_values = []
        
        for i in range(len(self.arms)):
            if self.arm_counts[i] == 0:
                # Unplayed arm gets highest priority
                ucb_values.append(float('inf'))
            else:
                # UCB formula
                avg_reward = self.arm_rewards[i] / self.arm_counts[i]
                confidence = self.c * np.sqrt(
                    np.log(self.total_pulls) / self.arm_counts[i]
                )
                ucb_values.append(avg_reward + confidence)
        
        # Select arm with highest UCB
        best_arm_idx = np.argmax(ucb_values)
        return self.arms[best_arm_idx].copy()
    
    def update(self, params: Dict[str, float], reward: float) -> None:
        """Update bandit with reward for parameter set"""
        
        # Find closest arm
        closest_arm_idx = self._find_closest_arm(params)
        
        # Update statistics
        self.arm_counts[closest_arm_idx] += 1
        self.arm_rewards[closest_arm_idx] += reward
        self.total_pulls += 1
    
    def _find_closest_arm(self, params: Dict[str, float]) -> int:
        """Find arm closest to given parameters"""
        
        min_distance = float('inf')
        closest_idx = 0
        
        for i, arm in enumerate(self.arms):
            # Calculate Euclidean distance
            distance = 0
            for param in self.parameter_space:
                if param in params and param in arm:
                    # Normalize by parameter range
                    param_range = self.parameter_space[param][1] - self.parameter_space[param][0]
                    normalized_diff = (params[param] - arm[param]) / param_range
                    distance += normalized_diff ** 2
            
            distance = np.sqrt(distance)
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx

class OnlineGradientOptimizer:
    """Online gradient-based parameter optimization"""
    
    def __init__(self, parameter_space: Dict[str, Tuple[float, float]],
                 learning_rate: float = 0.01):
        self.parameter_space = parameter_space
        self.learning_rate = learning_rate
        
        # Current parameters (normalized to [0, 1])
        self.normalized_params = {name: 0.5 for name in parameter_space}
        
        # Gradient estimates
        self.gradients = {name: 0.0 for name in parameter_space}
        
        # History for gradient estimation
        self.history: List[Tuple[Dict, float]] = []
        
    def suggest(self) -> Dict[str, float]:
        """Suggest parameters using gradient ascent"""
        
        if len(self.history) >= 2:
            self._update_gradients()
        
        # Gradient ascent step
        for param in self.normalized_params:
            self.normalized_params[param] += self.learning_rate * self.gradients[param]
            
            # Clip to [0, 1]
            self.normalized_params[param] = np.clip(self.normalized_params[param], 0, 1)
        
        # Convert to actual parameter values
        actual_params = self._denormalize_params(self.normalized_params)
        
        return actual_params
    
    def update(self, params: Dict[str, float], performance: float) -> None:
        """Update with new performance observation"""
        
        # Normalize parameters
        normalized = self._normalize_params(params)
        
        # Add to history
        self.history.append((normalized, performance))
        
        # Keep recent history only
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def _update_gradients(self) -> None:
        """Estimate gradients using finite differences"""
        
        if len(self.history) < 10:
            return
        
        # Use recent history
        recent_history = self.history[-10:]
        
        for param in self.parameter_space:
            # Calculate finite difference gradient
            gradient = 0.0
            count = 0
            
            for i in range(len(recent_history) - 1):
                params1, perf1 = recent_history[i]
                params2, perf2 = recent_history[i + 1]
                
                param_diff = params2[param] - params1[param]
                perf_diff = perf2 - perf1
                
                if abs(param_diff) > 1e-6:
                    gradient += perf_diff / param_diff
                    count += 1
            
            if count > 0:
                # Exponential moving average
                self.gradients[param] = 0.7 * self.gradients[param] + 0.3 * (gradient / count)
    
    def _normalize_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Normalize parameters to [0, 1]"""
        
        normalized = {}
        
        for param, value in params.items():
            if param in self.parameter_space:
                low, high = self.parameter_space[param]
                normalized[param] = (value - low) / (high - low)
            else:
                normalized[param] = 0.5  # Default
        
        return normalized
    
    def _denormalize_params(self, normalized_params: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized parameters back to actual values"""
        
        actual = {}
        
        for param, norm_value in normalized_params.items():
            if param in self.parameter_space:
                low, high = self.parameter_space[param]
                actual[param] = low + norm_value * (high - low)
        
        return actual
```

### 4. Adaptive Risk Management

```python
# src/adaptation/adaptive_risk.py
class AdaptiveRiskManager:
    """
    Risk manager that adapts to market regimes.
    Adjusts position sizes, stops, and limits dynamically.
    """
    
    def __init__(self, base_risk_manager: RiskManager):
        self.base_risk_manager = base_risk_manager
        
        # Regime-specific risk parameters
        self.regime_risk_configs: Dict[MarketRegime, Dict] = {}
        
        # Current risk state
        self.current_regime = MarketRegime.UNKNOWN
        self.risk_multipliers = {
            'position_size': 1.0,
            'stop_loss': 1.0,
            'max_drawdown': 1.0,
            'correlation_limit': 1.0
        }
        
        # Risk monitoring
        self.regime_drawdowns: Dict[MarketRegime, float] = {}
        self.regime_var: Dict[MarketRegime, float] = {}
        
        self.logger = ComponentLogger("AdaptiveRiskManager", "risk")
    
    def configure_regime_risk(self, regime: MarketRegime, config: Dict) -> None:
        """Configure risk parameters for specific regime"""
        
        self.regime_risk_configs[regime] = config
        
        self.logger.info(f"Configured risk for {regime}: {config}")
    
    def adapt_to_regime(self, regime_signal: RegimeSignal) -> None:
        """Adapt risk parameters to current regime"""
        
        new_regime = regime_signal.regime
        
        if new_regime != self.current_regime:
            self.logger.info(f"Risk adaptation: {self.current_regime} -> {new_regime}")
            
            self.current_regime = new_regime
            self._update_risk_multipliers(regime_signal)
    
    def _update_risk_multipliers(self, regime_signal: RegimeSignal) -> None:
        """Update risk multipliers based on regime"""
        
        regime = regime_signal.regime
        
        # Default conservative approach
        self.risk_multipliers = {
            'position_size': 0.5,
            'stop_loss': 0.8,
            'max_drawdown': 0.7,
            'correlation_limit': 0.6
        }
        
        # Regime-specific adjustments
        if regime == MarketRegime.BULL_LOW_VOL:
            # Favorable conditions - can take more risk
            self.risk_multipliers['position_size'] = 1.2
            self.risk_multipliers['max_drawdown'] = 1.0
            
        elif regime == MarketRegime.BULL_HIGH_VOL:
            # Positive but volatile - moderate risk
            self.risk_multipliers['position_size'] = 0.8
            self.risk_multipliers['stop_loss'] = 0.7
            
        elif regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
            # Bearish conditions - very conservative
            self.risk_multipliers['position_size'] = 0.4
            self.risk_multipliers['stop_loss'] = 0.6
            self.risk_multipliers['max_drawdown'] = 0.5
            
        elif regime == MarketRegime.CRISIS:
            # Crisis - minimal risk
            self.risk_multipliers['position_size'] = 0.2
            self.risk_multipliers['stop_loss'] = 0.5
            self.risk_multipliers['max_drawdown'] = 0.3
            self.risk_multipliers['correlation_limit'] = 0.3
        
        # Volatility regime adjustments
        if regime_signal.volatility_regime == "high":
            for key in self.risk_multipliers:
                self.risk_multipliers[key] *= 0.8  # Further reduction
        
        # Apply user-configured overrides
        if regime in self.regime_risk_configs:
            config = self.regime_risk_configs[regime]
            for param, multiplier in config.items():
                if param in self.risk_multipliers:
                    self.risk_multipliers[param] = multiplier
    
    def calculate_position_size(self, signal: Signal,
                              portfolio_value: float,
                              market_data: MarketData) -> float:
        """Calculate position size with regime adaptation"""
        
        # Get base position size
        base_size = self.base_risk_manager.calculate_position_size(
            signal, portfolio_value, market_data
        )
        
        # Apply regime adjustment
        regime_adjusted_size = base_size * self.risk_multipliers['position_size']
        
        # Additional volatility adjustment
        current_vol = market_data.volatility if hasattr(market_data, 'volatility') else 0.2
        vol_adjustment = min(1.0, 0.2 / max(current_vol, 0.05))  # Scale by volatility
        
        final_size = regime_adjusted_size * vol_adjustment
        
        self.logger.debug(
            f"Position size: base={base_size}, regime_adj={regime_adjusted_size}, "
            f"vol_adj={final_size}, multipliers={self.risk_multipliers}"
        )
        
        return final_size
    
    def calculate_stop_loss(self, entry_price: float,
                          signal_direction: SignalDirection,
                          market_data: MarketData) -> float:
        """Calculate stop loss with regime adaptation"""
        
        # Get base stop loss
        base_stop = self.base_risk_manager.calculate_stop_loss(
            entry_price, signal_direction, market_data
        )
        
        # Apply regime adjustment (tighter stops in volatile regimes)
        stop_multiplier = self.risk_multipliers['stop_loss']
        
        if signal_direction == SignalDirection.LONG:
            # For long positions, stop is below entry
            stop_distance = entry_price - base_stop
            adjusted_distance = stop_distance * stop_multiplier
            regime_stop = entry_price - adjusted_distance
        else:
            # For short positions, stop is above entry
            stop_distance = base_stop - entry_price
            adjusted_distance = stop_distance * stop_multiplier
            regime_stop = entry_price + adjusted_distance
        
        return regime_stop
    
    def check_portfolio_risk(self, portfolio: Portfolio,
                           regime_signal: RegimeSignal) -> RiskAssessment:
        """Assess portfolio risk with regime context"""
        
        # Get base risk assessment
        base_assessment = self.base_risk_manager.check_portfolio_risk(portfolio)
        
        # Calculate regime-specific metrics
        regime_var = self._calculate_regime_var(portfolio, regime_signal)
        correlation_risk = self._assess_correlation_risk(portfolio, regime_signal)
        concentration_risk = self._assess_concentration_risk(portfolio, regime_signal)
        
        # Combine assessments
        regime_assessment = RiskAssessment(
            overall_risk=base_assessment.overall_risk,
            position_risk=base_assessment.position_risk,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            var_1d=regime_var,
            max_drawdown_risk=base_assessment.max_drawdown_risk * self.risk_multipliers['max_drawdown'],
            regime_context={
                'current_regime': regime_signal.regime,
                'regime_confidence': regime_signal.confidence,
                'volatility_regime': regime_signal.volatility_regime,
                'risk_multipliers': self.risk_multipliers.copy()
            }
        )
        
        return regime_assessment
    
    def _calculate_regime_var(self, portfolio: Portfolio,
                            regime_signal: RegimeSignal) -> float:
        """Calculate Value at Risk adjusted for current regime"""
        
        # Base VaR calculation
        base_var = self.base_risk_manager.calculate_var(portfolio)
        
        # Regime adjustments
        if regime_signal.regime == MarketRegime.CRISIS:
            # Higher tail risk in crisis
            regime_var = base_var * 2.0
        elif regime_signal.volatility_regime == "high":
            # Higher volatility increases VaR
            regime_var = base_var * 1.5
        elif regime_signal.regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.SIDEWAYS_LOW_VOL]:
            # Lower risk in stable regimes
            regime_var = base_var * 0.8
        else:
            regime_var = base_var
        
        return regime_var
    
    def _assess_correlation_risk(self, portfolio: Portfolio,
                               regime_signal: RegimeSignal) -> float:
        """Assess correlation risk in current regime"""
        
        # In crisis regimes, correlations spike
        if regime_signal.regime == MarketRegime.CRISIS:
            correlation_multiplier = 2.0
        elif regime_signal.correlation_regime == "high_correlation":
            correlation_multiplier = 1.5
        else:
            correlation_multiplier = 1.0
        
        base_correlation_risk = 0.1  # Base assumption
        
        return base_correlation_risk * correlation_multiplier
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_5_regime_adaptation.py`:

```python
class TestRegimeDetection:
    """Test regime detection components"""
    
    def test_statistical_regime_detection(self):
        """Test rule-based regime detection"""
        detector = StatisticalRegimeDetector()
        
        # High volatility bull market
        features = {
            'vol_20d': 0.3,
            'return_20d': 0.1,
            'vol_of_vol': 0.05
        }
        
        regime, confidence = detector.detect(features, [])
        
        assert regime == MarketRegime.BULL_HIGH_VOL
        assert confidence > 0.5
    
    def test_crisis_detection(self):
        """Test crisis regime detection"""
        detector = StatisticalRegimeDetector()
        
        # Crisis conditions
        features = {
            'vol_20d': 0.5,  # Very high volatility
            'return_20d': -0.2,  # Large negative returns
            'vol_of_vol': 0.15  # High volatility of volatility
        }
        
        regime, confidence = detector.detect(features, [])
        
        assert regime == MarketRegime.CRISIS
        assert confidence > 0.8

class TestAdaptiveStrategy:
    """Test adaptive strategy functionality"""
    
    def test_regime_adaptation(self):
        """Test strategy adaptation to regime changes"""
        base_strategy = create_test_strategy()
        regime_detector = create_mock_regime_detector()
        
        adaptive = AdaptiveStrategy(base_strategy, regime_detector)
        
        # Configure different regimes
        adaptive.configure_regime(MarketRegime.BULL_LOW_VOL, {
            'position_size_multiplier': 1.2,
            'risk_threshold': 0.02
        })
        
        adaptive.configure_regime(MarketRegime.CRISIS, {
            'position_size_multiplier': 0.3,
            'risk_threshold': 0.005
        })
        
        # Test adaptation
        market_data = create_test_market_data()
        
        # Should adapt to crisis regime
        regime_detector.set_regime(MarketRegime.CRISIS, confidence=0.9)
        signal = adaptive.generate_signal(market_data)
        
        assert signal.strength < 0.5  # Should be reduced in crisis

class TestParameterOptimization:
    """Test dynamic parameter optimization"""
    
    def test_bandit_optimization(self):
        """Test multi-armed bandit optimization"""
        param_space = {
            'param1': (0.0, 1.0),
            'param2': (0.0, 2.0)
        }
        
        bandit = MultiArmedBandit(param_space, n_arms=10)
        
        # Test suggestions
        suggestions = []
        for i in range(20):
            suggestion = bandit.suggest()
            suggestions.append(suggestion)
            
            # Simulate performance (param1 closer to 0.7 is better)
            performance = 1.0 - abs(suggestion['param1'] - 0.7)
            bandit.update(suggestion, performance)
        
        # Should converge toward optimal
        final_suggestion = bandit.suggest()
        assert abs(final_suggestion['param1'] - 0.7) < 0.3
    
    def test_gradient_optimization(self):
        """Test gradient-based optimization"""
        param_space = {
            'x': (-2.0, 2.0)
        }
        
        optimizer = OnlineGradientOptimizer(param_space, learning_rate=0.1)
        
        # Test optimization of quadratic function
        for i in range(50):
            suggestion = optimizer.suggest()
            
            # Performance function: -(x-1)^2 (maximum at x=1)
            performance = -(suggestion['x'] - 1.0)**2
            optimizer.update(suggestion, performance)
        
        # Should converge toward x=1
        final_suggestion = optimizer.suggest()
        assert abs(final_suggestion['x'] - 1.0) < 0.5
```

### Integration Tests

Create `tests/integration/test_step10_5_adaptation_integration.py`:

```python
def test_full_adaptive_workflow():
    """Test complete adaptive trading workflow"""
    # Setup components
    base_strategy = MomentumStrategy()
    regime_detector = RealTimeRegimeDetector()
    adaptive_strategy = AdaptiveStrategy(base_strategy, regime_detector)
    adaptive_risk = AdaptiveRiskManager(BaseRiskManager())
    
    # Configure regimes
    adaptive_strategy.configure_regime(MarketRegime.BULL_LOW_VOL, {
        'lookback_period': 20,
        'momentum_threshold': 0.02
    })
    
    adaptive_strategy.configure_regime(MarketRegime.CRISIS, {
        'lookback_period': 5,
        'momentum_threshold': 0.05
    })
    
    adaptive_risk.configure_regime_risk(MarketRegime.CRISIS, {
        'position_size': 0.2,
        'stop_loss': 0.5
    })
    
    # Simulate regime changes
    performance_by_regime = {}
    
    for scenario in ['bull_market', 'crisis', 'recovery']:
        market_data = generate_regime_scenario(scenario, days=30)
        scenario_pnl = 0
        
        for day_data in market_data:
            # Generate signal with adaptation
            signal = adaptive_strategy.generate_signal(day_data)
            
            # Calculate position size with adaptive risk
            regime_signal = regime_detector.detect_regime(day_data, datetime.now())
            adaptive_risk.adapt_to_regime(regime_signal)
            
            position_size = adaptive_risk.calculate_position_size(
                signal, portfolio_value=100000, market_data=day_data['SPY'].iloc[-1]
            )
            
            # Simulate trade
            if signal.direction != SignalDirection.FLAT:
                entry_price = day_data['SPY']['close'].iloc[-1]
                next_price = entry_price * (1 + np.random.normal(0, 0.01))
                
                pnl = position_size * (next_price - entry_price) / entry_price
                if signal.direction == SignalDirection.SHORT:
                    pnl = -pnl
                
                scenario_pnl += pnl
        
        performance_by_regime[scenario] = scenario_pnl
    
    # Verify adaptation effectiveness
    # Crisis should have smaller losses due to adaptation
    assert performance_by_regime['crisis'] > -5000
    # Bull market should capture upside
    assert performance_by_regime['bull_market'] > 1000

def test_parameter_adaptation():
    """Test dynamic parameter adaptation"""
    param_space = {
        'momentum_lookback': (5, 50),
        'volatility_threshold': (0.01, 0.05),
        'position_size_factor': (0.5, 2.0)
    }
    
    optimizer = DynamicParameterOptimizer(param_space)
    strategy = ParameterizedStrategy()
    
    # Simulate optimization over time
    best_performance = float('-inf')
    performance_history = []
    
    for i in range(100):
        # Get parameter suggestion
        current_perf = np.random.normal(0, 100)  # Simulate performance
        params = optimizer.suggest_parameters(current_perf, create_test_regime_signal())
        
        # Apply parameters to strategy
        strategy.update_parameters(params)
        
        # Simulate strategy performance
        market_data = generate_test_market_data()
        signal = strategy.generate_signal(market_data)
        
        # Calculate performance (simple simulation)
        performance = simulate_strategy_performance(strategy, market_data)
        performance_history.append(performance)
        
        # Update optimizer
        optimizer.update_performance(performance)
        
        best_performance = max(best_performance, performance)
    
    # Should show improvement over time
    early_performance = np.mean(performance_history[:20])
    late_performance = np.mean(performance_history[-20:])
    
    assert late_performance > early_performance
```

### System Tests

Create `tests/system/test_step10_5_production_adaptation.py`:

```python
def test_real_time_adaptation_performance():
    """Test adaptation system under production loads"""
    # Setup production-scale system
    assets = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    strategies = {asset: create_adaptive_strategy(asset) for asset in assets}
    
    # Performance tracking
    adaptation_stats = {
        'regime_changes': 0,
        'parameter_updates': 0,
        'response_times': [],
        'memory_usage': []
    }
    
    # Run for extended period
    start_time = time.time()
    
    for i in range(5000):  # 5000 iterations
        iteration_start = time.time()
        
        # Generate market data update
        market_data = generate_multi_asset_update(assets)
        
        # Process each strategy
        for asset, strategy in strategies.items():
            # Detect regime
            regime_signal = strategy.regime_detector.detect_regime(
                {asset: market_data[asset]}, datetime.now()
            )
            
            # Generate signal with adaptation
            signal = strategy.generate_signal({asset: market_data[asset]})
            
            # Track adaptations
            if hasattr(strategy, '_last_adaptation') and strategy._last_adaptation > iteration_start - 1:
                adaptation_stats['regime_changes'] += 1
        
        # Track performance
        iteration_time = time.time() - iteration_start
        adaptation_stats['response_times'].append(iteration_time)
        
        if i % 100 == 0:  # Every 100 iterations
            memory_mb = get_memory_usage_mb()
            adaptation_stats['memory_usage'].append(memory_mb)
    
    total_time = time.time() - start_time
    
    # Performance requirements
    avg_response_time = np.mean(adaptation_stats['response_times'])
    p95_response_time = np.percentile(adaptation_stats['response_times'], 95)
    avg_memory = np.mean(adaptation_stats['memory_usage'])
    
    assert avg_response_time < 0.020  # 20ms average
    assert p95_response_time < 0.100  # 100ms p95
    assert avg_memory < 1000  # 1GB max memory
    
    # Adaptation frequency should be reasonable
    adaptations_per_minute = adaptation_stats['regime_changes'] / (total_time / 60)
    assert 0.1 < adaptations_per_minute < 10  # 0.1 to 10 adaptations per minute

def test_stress_adaptation():
    """Test adaptation under extreme market conditions"""
    stress_scenarios = [
        'flash_crash',
        'volatility_spike',
        'correlation_breakdown',
        'liquidity_crisis'
    ]
    
    adaptive_system = create_production_adaptive_system()
    
    for scenario in stress_scenarios:
        # Reset system
        adaptive_system.reset()
        
        # Generate stress scenario
        stress_data = generate_stress_scenario(scenario, duration_minutes=60)
        
        scenario_stats = {
            'max_drawdown': 0,
            'regime_changes': 0,
            'parameter_updates': 0,
            'risk_violations': 0
        }
        
        portfolio_value = 1000000
        peak_value = portfolio_value
        
        for minute_data in stress_data:
            # Process adaptation
            regime_signal = adaptive_system.detect_regime(minute_data)
            adapted_signals = adaptive_system.generate_signals(minute_data, regime_signal)
            
            # Calculate portfolio impact
            minute_pnl = simulate_portfolio_pnl(adapted_signals, minute_data)
            portfolio_value += minute_pnl
            
            # Track drawdown
            peak_value = max(peak_value, portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value
            scenario_stats['max_drawdown'] = max(scenario_stats['max_drawdown'], drawdown)
            
            # Track adaptations
            if adaptive_system.regime_changed:
                scenario_stats['regime_changes'] += 1
            
            if adaptive_system.parameters_updated:
                scenario_stats['parameter_updates'] += 1
        
        # Verify adaptation helped limit losses
        assert scenario_stats['max_drawdown'] < 0.20  # Max 20% drawdown
        assert scenario_stats['regime_changes'] > 0  # Should adapt during stress
        
        print(f"Scenario {scenario}: "
              f"Drawdown={scenario_stats['max_drawdown']:.2%}, "
              f"Regime changes={scenario_stats['regime_changes']}")

def test_adaptation_accuracy():
    """Test accuracy of regime detection and adaptation"""
    # Create synthetic data with known regime changes
    known_regimes = create_synthetic_regime_data(
        regimes=[
            (MarketRegime.BULL_LOW_VOL, 100),
            (MarketRegime.CRISIS, 30),
            (MarketRegime.RECOVERY, 50),
            (MarketRegime.SIDEWAYS_LOW_VOL, 80)
        ]
    )
    
    detector = RealTimeRegimeDetector()
    detected_regimes = []
    
    # Run detection
    for i, (data, true_regime) in enumerate(known_regimes):
        detected = detector.detect_regime(data, datetime.now())
        detected_regimes.append((detected.regime, detected.confidence))
    
    # Calculate accuracy
    correct_detections = sum(
        1 for (detected, _), (_, true) in zip(detected_regimes, known_regimes)
        if detected == true
    )
    
    accuracy = correct_detections / len(known_regimes)
    
    # Should achieve reasonable accuracy
    assert accuracy > 0.7  # 70% accuracy minimum
    
    # High confidence detections should be more accurate
    high_conf_correct = sum(
        1 for (detected, conf), (_, true) in zip(detected_regimes, known_regimes)
        if conf > 0.8 and detected == true
    )
    high_conf_total = sum(1 for _, conf in detected_regimes if conf > 0.8)
    
    if high_conf_total > 0:
        high_conf_accuracy = high_conf_correct / high_conf_total
        assert high_conf_accuracy > 0.85  # 85% accuracy for high confidence
```

## ✅ Validation Checklist

### Regime Detection
- [ ] Multiple detection methods implemented
- [ ] Real-time processing functional
- [ ] Feature extraction comprehensive
- [ ] Ensemble combination working
- [ ] Confidence scoring accurate

### Strategy Adaptation
- [ ] Regime-specific configurations
- [ ] Dynamic parameter switching
- [ ] Performance decay detection
- [ ] Adaptation frequency controlled
- [ ] Transition smoothing applied

### Parameter Optimization
- [ ] Multi-armed bandit working
- [ ] Gradient optimization functional
- [ ] Online learning effective
- [ ] Exploration/exploitation balanced
- [ ] Convergence demonstrated

### Risk Adaptation
- [ ] Regime-based risk scaling
- [ ] Position size adaptation
- [ ] Stop loss adjustment
- [ ] Portfolio risk monitoring
- [ ] Crisis mode protection

### Integration
- [ ] All components work together
- [ ] Performance monitoring active
- [ ] Memory usage controlled
- [ ] Response times acceptable
- [ ] Stress testing passed

## 📊 Performance Benchmarks

### Detection Performance
- Regime detection: < 50ms
- Feature extraction: < 20ms
- Model updates: < 100ms
- Confidence calculation: < 10ms

### Adaptation Performance
- Parameter switching: < 5ms
- Risk recalculation: < 10ms
- Strategy adjustment: < 15ms
- Portfolio rebalancing: < 30ms

### Accuracy Targets
- Regime detection accuracy: > 70%
- High confidence accuracy: > 85%
- Adaptation timing: < 2 minutes delay
- Parameter convergence: < 50 iterations

## 🐛 Common Issues

1. **Over-Adaptation**
   - Set minimum regime duration
   - Use confidence thresholds
   - Smooth transitions
   - Track adaptation frequency

2. **Detection Lag**
   - Use multiple time scales
   - Implement leading indicators
   - Reduce feature computation
   - Optimize model updates

3. **Parameter Instability**
   - Add regularization
   - Use exploration limits
   - Implement momentum
   - Track performance variance

## 🎯 Success Criteria

Step 10.5 is complete when:
1. ✅ Real-time regime detection operational
2. ✅ Strategy adaptation functional
3. ✅ Parameter optimization working
4. ✅ Risk adaptation implemented
5. ✅ Performance benchmarks met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.6: Custom Indicators](step-10.6-custom-indicators.md)

## 📚 Additional Resources

- [Regime Switching Models](../references/regime-switching.md)
- [Adaptive Control Theory](../references/adaptive-control-theory.md)
- [Online Learning Algorithms](../references/online-learning-algos.md)
- [Dynamic Risk Management](../references/dynamic-risk-mgmt.md)