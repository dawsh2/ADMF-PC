# Step 3: Classifier Container

**Status**: Container Architecture Step
**Complexity**: Medium
**Prerequisites**: [Step 2.5: Walk-Forward Foundation](../01-foundation-phase/step-02.5-walk-forward.md) completed
**Architecture Ref**: [CONTAINER-HIERARCHY.md](../../architecture/02-CONTAINER-HIERARCHY.md#classifier-container)

## 🎯 Objective

Implement a Classifier Container that:
- Detects market regimes (trending, ranging, volatile)
- Maintains its own isolated state and event bus
- Emits regime change events
- Enables regime-aware strategy selection
- Integrates with the container hierarchy

## 📋 Required Reading

Before starting:
1. [Container Hierarchy](../../architecture/02-CONTAINER-HIERARCHY.md)
2. [Classifier Documentation](../../strategy/classifiers/README.md)
3. [HMM Classifier Implementation](../../strategy/classifiers/hmm_classifier.py)

## 🏗️ Implementation Tasks

### 1. Base Classifier Container

```python
# src/strategy/classifiers/classifier_container.py
class ClassifierContainer:
    """
    Container for market regime classification.
    Manages classifier lifecycle and event isolation.
    """
    
    def __init__(self, container_id: str, config: ClassifierConfig):
        self.container_id = container_id
        self.config = config
        
        # Create isolated event bus
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_classifier"
        )
        
        # Initialize classifier
        self.classifier = self._create_classifier(config.classifier_type)
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        
        # Setup logging
        self.logger = ComponentLogger("ClassifierContainer", container_id)
        
        # Wire internal events
        self._setup_internal_events()
    
    def _create_classifier(self, classifier_type: str) -> BaseClassifier:
        """Factory method for classifier creation"""
        if classifier_type == "hmm":
            return HMMClassifier(self.config.hmm_config)
        elif classifier_type == "pattern":
            return PatternClassifier(self.config.pattern_config)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def on_bar(self, bar: Bar) -> None:
        """Process new market data"""
        # Update classifier
        self.classifier.update(bar)
        
        # Check for regime change
        new_regime = self.classifier.classify()
        
        if new_regime != self.current_regime:
            self._handle_regime_change(new_regime, bar.timestamp)
    
    def _handle_regime_change(self, new_regime: MarketRegime, 
                            timestamp: datetime) -> None:
        """Handle market regime transition"""
        old_regime = self.current_regime
        self.current_regime = new_regime
        
        # Log regime change
        self.logger.info(
            f"Regime change detected: {old_regime} → {new_regime} "
            f"at {timestamp}"
        )
        
        # Store in history
        self.regime_history.append({
            'timestamp': timestamp,
            'from_regime': old_regime,
            'to_regime': new_regime,
            'confidence': self.classifier.confidence
        })
        
        # Emit regime change event
        event = RegimeChangeEvent(
            timestamp=timestamp,
            old_regime=old_regime,
            new_regime=new_regime,
            confidence=self.classifier.confidence,
            classifier_id=self.container_id
        )
        
        self.event_bus.publish("REGIME_CHANGE", event)
```

### 2. HMM Classifier Implementation

```python
# src/strategy/classifiers/hmm_classifier.py
import numpy as np
from hmmlearn import hmm

class HMMClassifier(BaseClassifier):
    """
    Hidden Markov Model based market regime classifier.
    Detects trending, ranging, and volatile regimes.
    """
    
    def __init__(self, config: HMMConfig):
        self.n_states = config.n_states
        self.lookback = config.lookback
        self.features_window = deque(maxlen=self.lookback)
        
        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100
        )
        
        # Regime mapping
        self.regime_map = {
            0: MarketRegime.TRENDING,
            1: MarketRegime.RANGING,
            2: MarketRegime.VOLATILE
        }
        
        self.is_trained = False
        self.confidence = 0.0
    
    def update(self, bar: Bar) -> None:
        """Update classifier with new data"""
        # Calculate features
        features = self._calculate_features(bar)
        self.features_window.append(features)
        
        # Train model when we have enough data
        if len(self.features_window) == self.lookback and not self.is_trained:
            self._train_model()
    
    def _calculate_features(self, bar: Bar) -> np.ndarray:
        """Extract features for regime detection"""
        return np.array([
            bar.close / bar.open - 1,  # Return
            (bar.high - bar.low) / bar.open,  # Range
            bar.volume / 1e6,  # Normalized volume
        ])
    
    def _train_model(self) -> None:
        """Train HMM on historical features"""
        X = np.array(self.features_window)
        self.model.fit(X)
        self.is_trained = True
        
        self.logger.info("HMM model trained successfully")
    
    def classify(self) -> MarketRegime:
        """Classify current market regime"""
        if not self.is_trained or len(self.features_window) < self.lookback:
            return MarketRegime.UNKNOWN
        
        # Get current features
        X = np.array(self.features_window)
        
        # Predict states
        states = self.model.predict(X)
        current_state = states[-1]
        
        # Calculate confidence
        log_prob, posteriors = self.model.score_samples(X)
        self.confidence = posteriors[-1, current_state]
        
        return self.regime_map[current_state]
```

### 3. Pattern-Based Classifier

```python
# src/strategy/classifiers/pattern_classifier.py
class PatternClassifier(BaseClassifier):
    """
    Rule-based pattern classifier for market regimes.
    Uses technical indicators and price patterns.
    """
    
    def __init__(self, config: PatternConfig):
        self.atr_period = config.atr_period
        self.trend_period = config.trend_period
        self.vol_threshold = config.volatility_threshold
        
        # Indicators
        self.atr = AverageTrueRange(self.atr_period)
        self.sma = SimpleMovingAverage(self.trend_period)
        self.price_history = deque(maxlen=self.trend_period)
        
    def update(self, bar: Bar) -> None:
        """Update indicators with new data"""
        self.atr.update(bar)
        self.sma.update(bar)
        self.price_history.append(bar.close)
    
    def classify(self) -> MarketRegime:
        """Classify based on patterns"""
        if len(self.price_history) < self.trend_period:
            return MarketRegime.UNKNOWN
        
        # Calculate metrics
        volatility = self.atr.value / self.sma.value
        trend_strength = self._calculate_trend_strength()
        
        # Classify regime
        if volatility > self.vol_threshold:
            return MarketRegime.VOLATILE
        elif abs(trend_strength) > 0.7:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.RANGING
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength using linear regression"""
        prices = list(self.price_history)
        x = np.arange(len(prices))
        
        # Linear regression
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Normalize by price level
        normalized_slope = slope / np.mean(prices)
        
        return normalized_slope
```

### 4. Integration with Strategy Container

```python
# src/containers/integrated_container.py
class IntegratedStrategyContainer:
    """Strategy container that responds to regime changes"""
    
    def __init__(self, container_id: str, config: Dict):
        self.container_id = container_id
        
        # Create sub-containers
        self.classifier = ClassifierContainer(
            f"{container_id}_classifier",
            config['classifier']
        )
        
        self.strategies = {
            MarketRegime.TRENDING: TrendFollowingStrategy(),
            MarketRegime.RANGING: MeanReversionStrategy(),
            MarketRegime.VOLATILE: VolatilityStrategy()
        }
        
        self.active_strategy = None
        
        # Subscribe to regime changes
        self.classifier.event_bus.subscribe(
            "REGIME_CHANGE",
            self.on_regime_change
        )
    
    def on_regime_change(self, event: RegimeChangeEvent) -> None:
        """Switch strategy based on regime"""
        self.logger.info(
            f"Switching strategy for {event.new_regime} regime"
        )
        
        # Deactivate old strategy
        if self.active_strategy:
            self.active_strategy.deactivate()
        
        # Activate new strategy
        self.active_strategy = self.strategies[event.new_regime]
        self.active_strategy.activate()
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step3_classifier.py`:

```python
class TestHMMClassifier:
    """Test HMM classifier functionality"""
    
    def test_feature_calculation(self):
        """Test feature extraction"""
        classifier = HMMClassifier(HMMConfig(n_states=3))
        
        bar = Bar(
            open=100, high=102, low=99, close=101,
            volume=1000000
        )
        
        features = classifier._calculate_features(bar)
        
        assert len(features) == 3
        assert abs(features[0] - 0.01) < 0.001  # Return
        assert abs(features[1] - 0.03) < 0.001  # Range
        assert features[2] == 1.0  # Volume
    
    def test_regime_classification(self):
        """Test regime detection with synthetic data"""
        classifier = HMMClassifier(HMMConfig(n_states=3, lookback=50))
        
        # Create trending data
        trending_bars = create_trending_market(50)
        for bar in trending_bars:
            classifier.update(bar)
        
        regime = classifier.classify()
        assert regime == MarketRegime.TRENDING
        assert classifier.confidence > 0.7
```

### Integration Tests

Create `tests/integration/test_step3_classifier_integration.py`:

```python
def test_classifier_container_isolation():
    """Test classifier container event isolation"""
    # Create two classifier containers
    classifier1 = ClassifierContainer("classifier1", config1)
    classifier2 = ClassifierContainer("classifier2", config2)
    
    # Subscribe to events from classifier1
    regime_changes = []
    classifier1.event_bus.subscribe(
        "REGIME_CHANGE",
        lambda e: regime_changes.append(e)
    )
    
    # Send data that triggers regime change in classifier2
    volatile_data = create_volatile_market(100)
    for bar in volatile_data:
        classifier2.on_bar(bar)
    
    # Verify no leakage
    assert len(regime_changes) == 0

def test_strategy_regime_switching():
    """Test strategy switches on regime change"""
    container = IntegratedStrategyContainer("test", config)
    
    # Start with trending market
    trending_data = create_trending_market(100)
    for bar in trending_data:
        container.on_bar(bar)
    
    assert isinstance(container.active_strategy, TrendFollowingStrategy)
    
    # Switch to ranging market
    ranging_data = create_ranging_market(100)
    for bar in ranging_data:
        container.on_bar(bar)
    
    assert isinstance(container.active_strategy, MeanReversionStrategy)
```

### System Tests

Create `tests/system/test_step3_regime_aware_backtest.py`:

```python
def test_regime_aware_backtest():
    """Test complete backtest with regime detection"""
    # Create data with multiple regimes
    data = create_multi_regime_data()
    
    # Setup regime-aware system
    config = {
        'classifier': {
            'type': 'hmm',
            'n_states': 3,
            'lookback': 50
        },
        'strategies': {
            'trending': {'type': 'trend_following'},
            'ranging': {'type': 'mean_reversion'},
            'volatile': {'type': 'volatility'}
        }
    }
    
    system = create_regime_aware_system(config)
    results = system.run_backtest(data)
    
    # Verify regime detection worked
    assert len(results['regime_changes']) > 5
    
    # Verify strategy performance
    assert results['sharpe_ratio'] > 1.0
    assert results['max_drawdown'] < 0.15
    
    # Verify each regime had trades
    trades_by_regime = results['trades_by_regime']
    assert all(len(trades) > 0 for trades in trades_by_regime.values())
```

## ✅ Validation Checklist

### Classifier Validation
- [ ] HMM classifier trains correctly
- [ ] Pattern classifier detects regimes
- [ ] Regime transitions logged properly
- [ ] Confidence scores reasonable

### Container Validation
- [ ] Event bus properly isolated
- [ ] Lifecycle management working
- [ ] State maintained correctly
- [ ] No memory leaks

### Integration Validation
- [ ] Regime changes trigger strategy switches
- [ ] Multiple classifiers work independently
- [ ] Performance acceptable with classifier overhead

## 📊 Memory & Performance

### Memory Monitoring
```python
@profile
def process_market_data(classifier_container, data):
    """Profile memory usage during classification"""
    for bar in data:
        classifier_container.on_bar(bar)
        
    # Force garbage collection
    gc.collect()
    
    # Check memory usage
    memory_usage = get_memory_usage()
    assert memory_usage < 200 * 1024 * 1024  # 200MB limit
```

### Performance Benchmarks
- Classification latency: < 1ms per bar
- HMM training: < 100ms for 1000 bars
- Memory usage: < 50MB per classifier
- Regime detection accuracy: > 80%

## 🐛 Common Issues

1. **HMM Training Failures**
   - Ensure sufficient data before training
   - Check for NaN values in features
   - Validate feature scaling

2. **Regime Oscillation**
   - Add hysteresis to prevent rapid switching
   - Require minimum confidence for change
   - Use regime duration constraints

3. **Memory Growth**
   - Limit history buffer sizes
   - Clear old regime history periodically
   - Use rolling windows for features

## 🎯 Success Criteria

Step 3 is complete when:
1. ✅ Classifier container properly isolated
2. ✅ Multiple classifiers supported
3. ✅ Regime detection accurate
4. ✅ Strategy switching working
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 4: Multiple Strategies](step-04-multiple-strategies.md)

## 📚 Additional Resources

- [Market Regime Theory](../references/market-regimes.md)
- [HMM Tutorial](https://hmmlearn.readthedocs.io/)
- [Container Patterns](../../core/containers/patterns.md)