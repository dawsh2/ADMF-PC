# Step 6: Multiple Classifiers

**Status**: Container Architecture Step
**Complexity**: High
**Prerequisites**: [Step 5: Multiple Risk Containers](step-05-multiple-risk.md) completed
**Architecture Ref**: [CONTAINER-HIERARCHY.md](../../architecture/02-CONTAINER-HIERARCHY.md#classifier-ensemble)

## 🎯 Objective

Implement ensemble classifier system:
- Run multiple classifier types in parallel
- Consensus-based regime detection
- Classifier performance tracking
- A/B testing of classifier approaches
- Confidence-weighted regime determination

## 📋 Required Reading

Before starting:
1. [Classifier Container](step-03-classifier-container.md)
2. [Event Aggregation Patterns](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md#aggregation-pattern)
3. [Ensemble Methods Theory](../references/ensemble-methods.md)

## 🏗️ Implementation Tasks

### 1. Classifier Ensemble Container

```python
# src/strategy/classifiers/ensemble_classifier.py
class ClassifierEnsemble:
    """
    Manages multiple classifiers for robust regime detection.
    Uses weighted voting to determine market regime.
    """
    
    def __init__(self, container_id: str, config: EnsembleConfig):
        self.container_id = container_id
        self.config = config
        
        # Create master event bus
        self.isolation_manager = get_isolation_manager()
        self.event_bus = self.isolation_manager.create_isolated_bus(
            f"{container_id}_ensemble"
        )
        
        # Classifier containers
        self.classifiers: Dict[str, ClassifierContainer] = {}
        self.classifier_weights: Dict[str, float] = {}
        self.performance_tracker = ClassifierPerformanceTracker()
        
        # Consensus mechanism
        self.consensus_builder = RegimeConsensusBuilder(
            min_agreement=config.min_agreement,
            confidence_threshold=config.confidence_threshold
        )
        
        # Regime history for validation
        self.regime_history: List[RegimeDecision] = []
        
        # Setup logging
        self.logger = ComponentLogger("ClassifierEnsemble", container_id)
        
        # Initialize classifiers
        self._initialize_classifiers()
    
    def _initialize_classifiers(self) -> None:
        """Create diverse classifier containers"""
        classifier_configs = {
            'hmm': {
                'type': 'hmm',
                'config': HMMConfig(n_states=3, lookback=50),
                'weight': 0.3
            },
            'pattern': {
                'type': 'pattern',
                'config': PatternConfig(atr_period=14, trend_period=20),
                'weight': 0.2
            },
            'ml_based': {
                'type': 'ml',
                'config': MLConfig(model_type='random_forest'),
                'weight': 0.3
            },
            'statistical': {
                'type': 'statistical',
                'config': StatisticalConfig(window=100),
                'weight': 0.2
            }
        }
        
        for classifier_id, spec in classifier_configs.items():
            # Create classifier container
            container = ClassifierContainer(
                container_id=f"{self.container_id}_{classifier_id}",
                config=spec['config']
            )
            
            # Subscribe to regime changes
            container.event_bus.subscribe(
                "REGIME_CHANGE",
                lambda event, cid=classifier_id: self.on_classifier_regime(cid, event)
            )
            
            # Store container and weight
            self.classifiers[classifier_id] = container
            self.classifier_weights[classifier_id] = spec['weight']
            
            self.logger.info(f"Initialized classifier: {classifier_id}")
    
    def on_bar(self, bar: Bar) -> None:
        """Distribute market data to all classifiers"""
        # Clear previous classifications
        self.current_classifications = {}
        
        # Send to all classifiers
        for classifier_id, container in self.classifiers.items():
            try:
                container.on_bar(bar)
            except Exception as e:
                self.logger.error(
                    f"Classifier {classifier_id} failed: {e}"
                )
    
    def on_classifier_regime(self, classifier_id: str, 
                           event: RegimeChangeEvent) -> None:
        """Handle regime classification from individual classifier"""
        self.logger.log_event_flow(
            "REGIME_CLASSIFICATION", f"classifier_{classifier_id}", 
            "ensemble", f"Regime: {event.new_regime}"
        )
        
        # Store classification
        self.current_classifications[classifier_id] = {
            'regime': event.new_regime,
            'confidence': event.confidence,
            'timestamp': event.timestamp
        }
        
        # Check if we have enough classifications for consensus
        if len(self.current_classifications) >= self.config.min_classifiers:
            self._build_consensus(event.timestamp)
```

### 2. Regime Consensus Building

```python
# src/strategy/classifiers/consensus.py
class RegimeConsensusBuilder:
    """
    Builds consensus from multiple classifier outputs.
    Supports various aggregation methods.
    """
    
    def __init__(self, min_agreement: float = 0.6, 
                 confidence_threshold: float = 0.7):
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.aggregation_methods = {
            'weighted_vote': self._weighted_vote,
            'confidence_weighted': self._confidence_weighted,
            'bayesian': self._bayesian_aggregation,
            'max_confidence': self._max_confidence
        }
    
    def build_consensus(self, classifications: Dict[str, Dict], 
                       weights: Dict[str, float],
                       method: str = 'confidence_weighted') -> Optional[ConsensusRegime]:
        """Build consensus regime from multiple classifications"""
        if not classifications:
            return None
        
        # Apply selected aggregation method
        aggregator = self.aggregation_methods.get(method, self._confidence_weighted)
        return aggregator(classifications, weights)
    
    def _confidence_weighted(self, classifications: Dict[str, Dict],
                           weights: Dict[str, float]) -> Optional[ConsensusRegime]:
        """Weighted by both classifier weight and confidence"""
        regime_scores = defaultdict(float)
        total_weight = 0
        
        for classifier_id, classification in classifications.items():
            regime = classification['regime']
            confidence = classification['confidence']
            weight = weights.get(classifier_id, 1.0)
            
            # Combined weight = classifier weight × confidence
            combined_weight = weight * confidence
            regime_scores[regime] += combined_weight
            total_weight += combined_weight
        
        if total_weight == 0:
            return None
        
        # Find highest scoring regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        consensus_confidence = best_regime[1] / total_weight
        
        # Check if consensus meets threshold
        if consensus_confidence < self.confidence_threshold:
            return None
        
        # Calculate agreement level
        agreement = sum(
            1 for c in classifications.values() 
            if c['regime'] == best_regime[0]
        ) / len(classifications)
        
        if agreement < self.min_agreement:
            return None
        
        return ConsensusRegime(
            regime=best_regime[0],
            confidence=consensus_confidence,
            agreement=agreement,
            contributing_classifiers=list(classifications.keys()),
            method=method
        )
```

### 3. Classifier Performance Tracking

```python
# src/strategy/classifiers/performance_tracking.py
class ClassifierPerformanceTracker:
    """
    Tracks individual classifier performance for dynamic weighting.
    Implements various scoring methods.
    """
    
    def __init__(self, evaluation_window: int = 100):
        self.evaluation_window = evaluation_window
        self.regime_history: deque = deque(maxlen=evaluation_window)
        self.classifier_predictions: Dict[str, deque] = {}
        self.performance_metrics: Dict[str, ClassifierMetrics] = {}
    
    def record_prediction(self, classifier_id: str, regime: MarketRegime,
                         confidence: float, timestamp: datetime) -> None:
        """Record classifier prediction"""
        if classifier_id not in self.classifier_predictions:
            self.classifier_predictions[classifier_id] = deque(
                maxlen=self.evaluation_window
            )
        
        self.classifier_predictions[classifier_id].append({
            'regime': regime,
            'confidence': confidence,
            'timestamp': timestamp
        })
    
    def update_actual_regime(self, regime: MarketRegime, 
                           timestamp: datetime) -> None:
        """Update with actual regime (from returns or external validation)"""
        self.regime_history.append({
            'regime': regime,
            'timestamp': timestamp
        })
        
        # Update performance metrics
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate accuracy, precision, recall for each classifier"""
        for classifier_id, predictions in self.classifier_predictions.items():
            if len(predictions) < 10:  # Need minimum history
                continue
            
            # Match predictions with actual regimes
            correct_predictions = 0
            confidence_weighted_score = 0
            regime_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            for pred in predictions:
                # Find corresponding actual regime
                actual = self._find_actual_regime(pred['timestamp'])
                if actual:
                    regime_accuracy[pred['regime']]['total'] += 1
                    if pred['regime'] == actual['regime']:
                        correct_predictions += 1
                        regime_accuracy[pred['regime']]['correct'] += 1
                        confidence_weighted_score += pred['confidence']
            
            # Calculate metrics
            total_predictions = len(predictions)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate per-regime precision
            regime_precision = {}
            for regime, stats in regime_accuracy.items():
                if stats['total'] > 0:
                    regime_precision[regime] = stats['correct'] / stats['total']
            
            # Update metrics
            self.performance_metrics[classifier_id] = ClassifierMetrics(
                accuracy=accuracy,
                confidence_weighted_score=confidence_weighted_score / total_predictions,
                regime_precision=regime_precision,
                total_predictions=total_predictions
            )
    
    def get_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        if not self.performance_metrics:
            return {}
        
        # Use confidence-weighted accuracy for weighting
        scores = {
            cid: metrics.confidence_weighted_score 
            for cid, metrics in self.performance_metrics.items()
        }
        
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal weights if no performance data
            return {cid: 1.0 / len(scores) for cid in scores}
        
        # Normalize scores to weights
        return {cid: score / total_score for cid, score in scores.items()}
```

### 4. A/B Testing Framework

```python
# src/strategy/classifiers/ab_testing.py
class ClassifierABTester:
    """
    A/B testing framework for comparing classifier performance.
    Supports statistical significance testing.
    """
    
    def __init__(self, test_config: ABTestConfig):
        self.test_config = test_config
        self.control_group = test_config.control_classifier
        self.treatment_groups = test_config.treatment_classifiers
        self.test_results: Dict[str, ABTestResult] = {}
        self.test_start = datetime.now()
    
    def run_test(self, market_data: pd.DataFrame) -> ABTestReport:
        """Run A/B test on historical data"""
        # Initialize classifiers
        classifiers = self._initialize_test_classifiers()
        
        # Run classifiers on same data
        results = {}
        for classifier_id, classifier in classifiers.items():
            regime_changes = []
            
            for _, bar in market_data.iterrows():
                classifier.on_bar(bar)
                if classifier.current_regime != classifier.previous_regime:
                    regime_changes.append({
                        'timestamp': bar.name,
                        'regime': classifier.current_regime,
                        'confidence': classifier.confidence
                    })
            
            results[classifier_id] = regime_changes
        
        # Analyze results
        return self._analyze_results(results, market_data)
    
    def _analyze_results(self, results: Dict, market_data: pd.DataFrame) -> ABTestReport:
        """Analyze A/B test results with statistical tests"""
        # Calculate performance metrics
        metrics = {}
        for classifier_id, regime_changes in results.items():
            metrics[classifier_id] = self._calculate_classifier_metrics(
                regime_changes, market_data
            )
        
        # Statistical significance tests
        significance_tests = {}
        control_metrics = metrics[self.control_group]
        
        for treatment_id in self.treatment_groups:
            treatment_metrics = metrics[treatment_id]
            
            # Perform t-test on accuracy
            t_stat, p_value = stats.ttest_ind(
                control_metrics['accuracy_samples'],
                treatment_metrics['accuracy_samples']
            )
            
            significance_tests[treatment_id] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < self.test_config.significance_level,
                'performance_delta': (
                    treatment_metrics['overall_accuracy'] - 
                    control_metrics['overall_accuracy']
                )
            }
        
        return ABTestReport(
            control_performance=control_metrics,
            treatment_performance={tid: metrics[tid] for tid in self.treatment_groups},
            significance_tests=significance_tests,
            recommendation=self._make_recommendation(significance_tests)
        )
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step6_ensemble_classifier.py`:

```python
class TestRegimeConsensus:
    """Test consensus building mechanisms"""
    
    def test_confidence_weighted_consensus(self):
        """Test confidence-weighted voting"""
        builder = RegimeConsensusBuilder()
        
        classifications = {
            'hmm': {'regime': MarketRegime.TRENDING, 'confidence': 0.9},
            'pattern': {'regime': MarketRegime.TRENDING, 'confidence': 0.8},
            'ml': {'regime': MarketRegime.VOLATILE, 'confidence': 0.6},
            'statistical': {'regime': MarketRegime.TRENDING, 'confidence': 0.7}
        }
        
        weights = {
            'hmm': 0.3,
            'pattern': 0.2,
            'ml': 0.3,
            'statistical': 0.2
        }
        
        consensus = builder.build_consensus(classifications, weights)
        
        assert consensus is not None
        assert consensus.regime == MarketRegime.TRENDING
        assert consensus.confidence > 0.7
        assert consensus.agreement == 0.75  # 3 out of 4

class TestPerformanceTracking:
    """Test classifier performance tracking"""
    
    def test_dynamic_weight_calculation(self):
        """Test dynamic weight updates based on performance"""
        tracker = ClassifierPerformanceTracker()
        
        # Simulate predictions and actuals
        for i in range(50):
            # HMM performs well
            tracker.record_prediction(
                'hmm', MarketRegime.TRENDING, 0.9, datetime.now()
            )
            # Pattern performs poorly
            tracker.record_prediction(
                'pattern', MarketRegime.VOLATILE, 0.8, datetime.now()
            )
            
            # Actual was trending
            tracker.update_actual_regime(MarketRegime.TRENDING, datetime.now())
        
        weights = tracker.get_dynamic_weights()
        
        # HMM should have higher weight
        assert weights['hmm'] > weights['pattern']
```

### Integration Tests

Create `tests/integration/test_step6_ensemble_integration.py`:

```python
def test_ensemble_isolation():
    """Test ensemble classifiers maintain isolation"""
    ensemble = ClassifierEnsemble("test_ensemble", EnsembleConfig())
    
    # Verify each classifier has isolated event bus
    buses = [
        classifier.event_bus 
        for classifier in ensemble.classifiers.values()
    ]
    
    # All buses should be different
    assert len(set(id(bus) for bus in buses)) == len(buses)
    
    # Test no cross-contamination
    test_events = []
    ensemble.classifiers['hmm'].event_bus.subscribe(
        "TEST", lambda e: test_events.append(e)
    )
    
    # Emit from different classifier
    ensemble.classifiers['pattern'].event_bus.publish("TEST", {"data": "test"})
    
    # Should not receive
    assert len(test_events) == 0

def test_consensus_under_disagreement():
    """Test consensus when classifiers disagree"""
    ensemble = create_test_ensemble()
    
    # Create scenario with disagreement
    conflicting_data = create_regime_transition_data()
    
    consensus_events = []
    ensemble.event_bus.subscribe(
        "CONSENSUS_REGIME", lambda e: consensus_events.append(e)
    )
    
    # Process data
    for bar in conflicting_data:
        ensemble.on_bar(bar)
    
    # Should still reach consensus on clear regimes
    assert len(consensus_events) > 0
    
    # But not during transitions
    transition_consensuses = [
        e for e in consensus_events 
        if e.agreement < 0.6
    ]
    assert len(transition_consensuses) == 0  # No weak consensus emitted
```

### System Tests

Create `tests/system/test_step6_ensemble_system.py`:

```python
def test_ensemble_ab_testing():
    """Test A/B testing framework end-to-end"""
    # Configure A/B test
    test_config = ABTestConfig(
        control_classifier='hmm',
        treatment_classifiers=['ml_ensemble', 'hybrid'],
        test_duration_days=30,
        significance_level=0.05
    )
    
    # Run test on historical data
    historical_data = load_test_market_data(days=30)
    ab_tester = ClassifierABTester(test_config)
    report = ab_tester.run_test(historical_data)
    
    # Verify test completeness
    assert report.control_performance is not None
    assert len(report.treatment_performance) == 2
    assert all(test['p_value'] is not None 
              for test in report.significance_tests.values())
    
    # Check recommendation is made
    assert report.recommendation in ['keep_control', 'switch_to_ml_ensemble', 
                                   'switch_to_hybrid', 'insufficient_data']

def test_ensemble_regime_detection_accuracy():
    """Test ensemble improves regime detection"""
    # Create data with known regimes
    synthetic_data = create_known_regime_data()
    expected_regimes = synthetic_data['actual_regime']
    
    # Test single classifier
    single_classifier = ClassifierContainer("single", HMMConfig())
    single_predictions = run_classifier(single_classifier, synthetic_data)
    
    # Test ensemble
    ensemble = ClassifierEnsemble("ensemble", EnsembleConfig())
    ensemble_predictions = run_classifier(ensemble, synthetic_data)
    
    # Calculate accuracies
    single_accuracy = calculate_accuracy(single_predictions, expected_regimes)
    ensemble_accuracy = calculate_accuracy(ensemble_predictions, expected_regimes)
    
    # Ensemble should outperform
    assert ensemble_accuracy > single_accuracy
    assert ensemble_accuracy > 0.75  # Reasonable threshold
```

## ✅ Validation Checklist

### Ensemble Functionality
- [ ] Multiple classifiers run in parallel
- [ ] Each classifier properly isolated
- [ ] Consensus mechanism working
- [ ] Confidence weighting applied
- [ ] Regime changes properly aggregated

### Performance Tracking
- [ ] Individual classifier metrics tracked
- [ ] Dynamic weight adjustment working
- [ ] Performance history maintained
- [ ] Statistical significance calculated

### A/B Testing
- [ ] Control/treatment groups compared
- [ ] Statistical tests implemented
- [ ] Recommendations generated
- [ ] Results reproducible

## 📊 Memory & Performance

### Memory Management
```python
class MemoryEfficientEnsemble:
    """Optimized ensemble for memory efficiency"""
    
    def __init__(self, config):
        # Limit history sizes
        self.max_history = 1000
        self.regime_history = deque(maxlen=self.max_history)
        
        # Use weak references for old predictions
        self.prediction_cache = weakref.WeakValueDictionary()
```

### Performance Targets
- Process all classifiers: < 10ms per bar
- Consensus building: < 2ms
- Memory per classifier: < 30MB
- Total ensemble memory: < 200MB

## 🐛 Common Issues

1. **Classifier Synchronization**
   - Ensure all classifiers process same bar
   - Handle timing differences
   - Maintain order consistency

2. **Consensus Deadlock**
   - Set minimum classifier count
   - Handle tie-breaking rules
   - Timeout for missing classifiers

3. **Performance Degradation**
   - Monitor individual classifier latency
   - Implement circuit breakers
   - Cache consensus results

## 🎯 Success Criteria

Step 6 is complete when:
1. ✅ Ensemble runs multiple classifiers
2. ✅ Consensus mechanism robust
3. ✅ Performance tracking implemented
4. ✅ A/B testing framework functional
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 7: Signal Capture](../03-signal-capture-replay/step-07-signal-capture.md)

## 📚 Additional Resources

- [Ensemble Methods Theory](../references/ensemble-methods.md)
- [Statistical Testing Guide](../references/statistical-testing.md)
- [Classifier Comparison Studies](../references/classifier-comparison.md)