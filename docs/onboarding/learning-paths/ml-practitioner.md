# Learning Path: ML Practitioner

A comprehensive guide for machine learning practitioners integrating ML models with ADMF-PC for algorithmic trading.

## Overview

As an ML Practitioner, you'll learn how to:
- Integrate ML models as trading strategies
- Use ADMF-PC for feature engineering
- Optimize model hyperparameters
- Deploy ensemble models
- Validate ML strategies properly

## Prerequisites

- Machine learning fundamentals
- Python programming skills
- Understanding of financial markets
- Familiarity with scikit-learn/TensorFlow/PyTorch

## Learning Path

### Phase 1: ML Integration Basics (Week 1)

#### Day 1-2: Understanding ADMF-PC for ML
- [ ] Complete [Quick Start Guide](../QUICK_START.md)
- [ ] Study [Protocol + Composition](../../architecture/03-PROTOCOL-COMPOSITION.md)
- [ ] Learn how ML models fit as components

#### Day 3-4: ML as Trading Strategies
```python
# Any ML model can be a strategy!
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Use directly in ADMF-PC
strategies = [
    model.predict,  # Just use the predict method!
    neural_net.forward,
    xgboost_model.predict_proba
]
```

#### Day 5-7: Feature Engineering Pipeline
```yaml
# ml_features.yaml
indicators:
  # Technical indicators as features
  - type: "sma"
    periods: [5, 10, 20, 50]
  - type: "rsi"
    period: 14
  - type: "atr"
    period: 14
  - type: "volume_ratio"
    
  # Custom features
  - type: "price_acceleration"
  - type: "microstructure_features"
  
feature_engineering:
  # Automated feature creation
  rolling_statistics:
    windows: [5, 10, 20]
    functions: ["mean", "std", "skew", "kurt"]
    
  # Interaction features
  create_interactions: true
  polynomial_degree: 2
```

### Phase 2: ML Strategy Development (Week 2)

#### Classification-Based Trading
```python
class MLTradingStrategy:
    """ML model as ADMF-PC strategy"""
    
    def __init__(self, model, feature_config):
        self.model = model
        self.feature_config = feature_config
        self.scaler = StandardScaler()
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract features
        features = self.extract_features(data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict_proba(features_scaled)
        
        # Convert to trading signal
        if prediction[0][1] > 0.6:  # Bullish probability
            return {"action": "BUY", "strength": prediction[0][1]}
        elif prediction[0][1] < 0.4:  # Bearish probability
            return {"action": "SELL", "strength": 1 - prediction[0][1]}
        else:
            return {"action": "HOLD", "strength": 0}
```

#### Regression-Based Trading
```python
class PriceTargetStrategy:
    """Regression model for price targets"""
    
    def __init__(self, model, horizon=5):
        self.model = model
        self.horizon = horizon
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.extract_features(data)
        
        # Predict future price
        predicted_price = self.model.predict(features)[0]
        current_price = data['close'][-1]
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Generate signal based on threshold
        if expected_return > 0.02:  # 2% expected gain
            return {
                "action": "BUY",
                "strength": min(expected_return * 10, 1.0),
                "metadata": {"target": predicted_price}
            }
        elif expected_return < -0.02:
            return {
                "action": "SELL",
                "strength": min(abs(expected_return) * 10, 1.0),
                "metadata": {"target": predicted_price}
            }
        else:
            return {"action": "HOLD", "strength": 0}
```

### Phase 3: Advanced ML Patterns (Week 3)

#### Ensemble Models
```yaml
# ml_ensemble.yaml
strategies:
  # Different ML models as ensemble members
  - name: "random_forest"
    type: "ml_model"
    model_path: "models/rf_classifier.pkl"
    weight: 0.3
    
  - name: "gradient_boost"
    type: "ml_model"
    model_path: "models/xgb_model.pkl"
    weight: 0.3
    
  - name: "neural_network"
    type: "ml_model"
    model_path: "models/lstm_model.h5"
    weight: 0.2
    
  - name: "svm_classifier"
    type: "ml_model"
    model_path: "models/svm_model.pkl"
    weight: 0.2

ensemble:
  method: "weighted_voting"
  require_agreement: 0.6  # 60% must agree
```

#### Reinforcement Learning Integration
```python
class RLTradingAgent:
    """RL agent as ADMF-PC strategy"""
    
    def __init__(self, model_path: str):
        self.agent = load_rl_agent(model_path)
        self.state_builder = StateBuilder()
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Build state representation
        state = self.state_builder.build_state(
            prices=data['close'],
            volume=data['volume'],
            indicators=data['indicators'],
            portfolio_state=data.get('portfolio', {})
        )
        
        # Get action from RL agent
        action = self.agent.act(state)
        
        # Map RL action to trading signal
        action_map = {
            0: {"action": "HOLD", "strength": 0},
            1: {"action": "BUY", "strength": 0.5},
            2: {"action": "BUY", "strength": 1.0},
            3: {"action": "SELL", "strength": 0.5},
            4: {"action": "SELL", "strength": 1.0}
        }
        
        return action_map[action]
```

#### Online Learning
```python
class OnlineLearningStrategy:
    """Model that updates during trading"""
    
    def __init__(self, base_model, update_frequency=100):
        self.model = base_model
        self.update_frequency = update_frequency
        self.sample_buffer = []
        self.trade_count = 0
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.extract_features(data)
        signal = self._get_model_signal(features)
        
        # Store sample for later training
        self.sample_buffer.append({
            'features': features,
            'timestamp': data['timestamp'],
            'signal': signal
        })
        
        # Periodic model updates
        self.trade_count += 1
        if self.trade_count % self.update_frequency == 0:
            self._update_model()
        
        return signal
    
    def _update_model(self):
        """Incrementally update model with recent data"""
        if len(self.sample_buffer) < 50:
            return
            
        # Calculate labels based on forward returns
        X, y = self._prepare_training_data(self.sample_buffer[-200:])
        
        # Incremental learning
        self.model.partial_fit(X, y)
        
        # Clear old samples
        self.sample_buffer = self.sample_buffer[-100:]
```

### Phase 4: ML-Specific Optimizations (Week 4)

#### Hyperparameter Optimization
```yaml
# ml_hyperopt.yaml
workflow:
  type: "ml_optimization"
  
ml_model:
  type: "random_forest"
  
hyperparameter_search:
  method: "bayesian"  # or "grid", "random"
  
  parameter_space:
    n_estimators: [50, 100, 200, 500]
    max_depth: [3, 5, 10, 20, null]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    
  cross_validation:
    method: "time_series_split"
    n_splits: 5
    gap: 20  # Gap between train and test
    
  scoring:
    primary_metric: "sharpe_ratio"
    secondary_metrics: ["accuracy", "precision", "recall"]
```

#### Feature Selection
```python
class FeatureSelectionStrategy:
    """Auto feature selection for ML models"""
    
    def __init__(self, base_model, n_features=20):
        self.base_model = base_model
        self.n_features = n_features
        self.selected_features = None
        self.selector = None
    
    def select_features(self, X, y):
        """Select best features using various methods"""
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_classif(X, y)
        
        # Method 2: Feature Importance from Trees
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        # Method 3: L1 Regularization
        lasso = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear')
        )
        lasso.fit(X, y)
        
        # Combine rankings
        combined_scores = self._combine_rankings(
            mi_scores, importances, lasso
        )
        
        # Select top features
        self.selected_features = combined_scores.argsort()[-self.n_features:]
```

### Phase 5: Production ML Deployment

#### Model Versioning
```yaml
# ml_model_registry.yaml
model_registry:
  models:
    - name: "rf_v1"
      version: "1.0.0"
      path: "models/rf_v1.pkl"
      trained_date: "2024-01-15"
      metrics:
        validation_sharpe: 1.85
        validation_accuracy: 0.68
      
    - name: "rf_v2"
      version: "2.0.0"
      path: "models/rf_v2.pkl"
      trained_date: "2024-02-01"
      metrics:
        validation_sharpe: 2.10
        validation_accuracy: 0.72
        
  active_model: "rf_v2"
  fallback_model: "rf_v1"
```

#### A/B Testing Models
```python
class ABTestingStrategy:
    """A/B test different models in production"""
    
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.performance_tracker = PerformanceTracker()
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Randomly assign to model
        use_model_a = random.random() < self.split_ratio
        
        if use_model_a:
            signal = self.model_a.generate_signal(data)
            self.performance_tracker.record("model_a", signal)
        else:
            signal = self.model_b.generate_signal(data)
            self.performance_tracker.record("model_b", signal)
        
        # Add metadata for tracking
        signal['metadata'] = {
            'model': 'a' if use_model_a else 'b',
            'experiment_id': self.experiment_id
        }
        
        return signal
```

## ML Best Practices for Trading

### 1. Proper Train/Test Splits
```python
# NEVER use random splits for time series!
# Bad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Good - Temporal splits
train_end = int(len(data) * 0.7)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
```

### 2. Feature Engineering
```python
def create_trading_features(df):
    """Create ML features from market data"""
    
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['dollar_volume'] = df['close'] * df['volume']
    
    # Technical indicators
    features['rsi'] = ta.RSI(df['close'])
    features['macd'] = ta.MACD(df['close'])['macd']
    features['atr'] = ta.ATR(df['high'], df['low'], df['close'])
    
    # Rolling statistics
    for window in [5, 10, 20]:
        features[f'return_mean_{window}'] = features['returns'].rolling(window).mean()
        features[f'return_std_{window}'] = features['returns'].rolling(window).std()
        features[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
    
    return features
```

### 3. Avoiding Lookahead Bias
```python
class NoLookaheadPipeline:
    """Ensure no future information leaks"""
    
    def __init__(self):
        self.scaler = None
        self.feature_stats = None
    
    def fit_transform(self, X_train):
        """Fit only on training data"""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Store feature statistics from training only
        self.feature_stats = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'min': X_train.min(),
            'max': X_train.max()
        }
        
        return X_scaled
    
    def transform(self, X_new):
        """Apply training statistics to new data"""
        return self.scaler.transform(X_new)
```

### 4. Model Validation
```yaml
# ml_validation.yaml
validation:
  # Walk-forward validation
  walk_forward:
    train_periods: 252  # 1 year
    test_periods: 63    # 3 months
    step_size: 21       # 1 month
    
  # Purged cross-validation
  cross_validation:
    method: "purged_kfold"
    n_splits: 5
    purge_gap: 10  # Days between train/test
    
  # Statistical tests
  statistical_validation:
    - permutation_importance
    - partial_dependence
    - model_stability_test
```

## Common ML Trading Patterns

### 1. Market Regime Classification
```python
class RegimeClassifier:
    """Identify market regimes for adaptive trading"""
    
    def __init__(self):
        self.model = GaussianMixture(n_components=3)
        self.regime_names = ['bull', 'bear', 'sideways']
    
    def fit(self, returns, volatility):
        features = np.column_stack([returns, volatility])
        self.model.fit(features)
    
    def predict_regime(self, current_returns, current_vol):
        features = np.array([[current_returns, current_vol]])
        regime = self.model.predict(features)[0]
        return self.regime_names[regime]
```

### 2. Ensemble Voting
```python
class EnsembleVoting:
    """Combine multiple ML models"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X)[:, 1]  # Probability of positive class
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
```

### 3. Feature Importance Trading
```python
class FeatureImportanceStrategy:
    """Trade based on feature importance changes"""
    
    def __init__(self, base_model, lookback=20):
        self.base_model = base_model
        self.lookback = lookback
        self.importance_history = []
    
    def generate_signal(self, data):
        # Train model on recent data
        X, y = self.prepare_recent_data(data, self.lookback)
        self.base_model.fit(X, y)
        
        # Get current feature importances
        current_importance = self.base_model.feature_importances_
        self.importance_history.append(current_importance)
        
        if len(self.importance_history) > 2:
            # Detect significant changes
            importance_change = (
                current_importance - 
                np.mean(self.importance_history[-5:], axis=0)
            )
            
            # Trade if momentum features become important
            momentum_idx = self.get_momentum_feature_indices()
            if importance_change[momentum_idx].mean() > 0.1:
                return {"action": "BUY", "strength": 0.8}
                
        return {"action": "HOLD", "strength": 0}
```

## Resources

- [ML Strategy Examples](../../strategy/strategies/)
- [Feature Engineering Guide](../../complexity-guide/05-intermediate-complexity/step-10.6-custom-indicators.md)
- [Model Validation](../../complexity-guide/validation-framework/README.md)
- [Production ML](../../complexity-guide/06-going-beyond/step-14-ml-models.md)

## Next Steps

After completing this learning path:
1. Implement your own ML trading strategies
2. Build automated feature engineering pipelines
3. Create model monitoring dashboards
4. Contribute ML components to the community

---

*Remember: ML models are powerful but can overfit easily. Always validate properly, use appropriate train/test splits, and consider transaction costs. The market is adversarial - what worked yesterday might not work tomorrow.*