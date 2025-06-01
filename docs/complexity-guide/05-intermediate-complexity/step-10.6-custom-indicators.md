# Step 10.6: Custom Indicators

**Status**: Intermediate Complexity Step
**Complexity**: High
**Prerequisites**: [Step 10.5: Regime Adaptation](step-10.5-regime-adaptation.md) completed
**Architecture Ref**: [Indicator Framework Architecture](../architecture/indicator-framework.md)

## 🎯 Objective

Implement extensible custom indicator framework:
- Plugin-based indicator architecture
- Performance-optimized calculation engines
- Real-time indicator updates and streaming
- Machine learning-based indicators
- Composite indicator combinations
- Indicator validation and testing framework

## 📋 Required Reading

Before starting:
1. [Technical Analysis Theory](../references/technical-analysis.md)
2. [Signal Processing for Finance](../references/signal-processing.md)
3. [Plugin Architecture Patterns](../references/plugin-patterns.md)

## 🏗️ Implementation Tasks

### 1. Core Indicator Framework

```python
# src/indicators/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import inspect
from functools import wraps

class IndicatorType(Enum):
    """Types of indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    COMPOSITE = "composite"
    ML_BASED = "ml_based"
    CUSTOM = "custom"

@dataclass
class IndicatorMetadata:
    """Metadata for indicators"""
    name: str
    indicator_type: IndicatorType
    description: str
    
    # Input requirements
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    min_periods: int = 1
    
    # Output specification
    output_columns: List[str] = field(default_factory=list)
    output_type: str = "float"  # float, int, bool, categorical
    
    # Performance characteristics
    computational_complexity: str = "O(n)"  # Big O notation
    memory_complexity: str = "O(1)"
    supports_streaming: bool = True
    
    # Validation
    valid_range: Optional[tuple] = None
    expected_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndicatorResult:
    """Result from indicator calculation"""
    values: Union[pd.Series, pd.DataFrame, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    # Quality metrics
    completeness: float  # Fraction of non-NaN values
    stability: float  # Measure of result stability
    confidence: float  # Confidence in result quality

class BaseIndicator(ABC):
    """Base class for all indicators"""
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.metadata = self._create_metadata()
        
        # State for streaming calculations
        self.state = {}
        self.history = []
        self.max_history = kwargs.get('max_history', 1000)
        
        # Performance tracking
        self.calculation_times = []
        self.error_count = 0
        
        # Validation
        self.validator = IndicatorValidator(self.metadata)
        
        self.logger = ComponentLogger(f"Indicator_{self.metadata.name}", "indicators")
    
    @abstractmethod
    def _create_metadata(self) -> IndicatorMetadata:
        """Create indicator metadata"""
        pass
    
    @abstractmethod
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Core calculation logic"""
        pass
    
    def calculate(self, data: pd.DataFrame, 
                 validate: bool = True,
                 **kwargs) -> IndicatorResult:
        """Main calculation interface with validation and error handling"""
        
        start_time = datetime.now()
        
        try:
            # Input validation
            if validate:
                self.validator.validate_input(data)
            
            # Core calculation
            result = self._calculate(data, **kwargs)
            
            # Output validation
            if validate:
                self.validator.validate_output(result)
            
            # Update state for streaming
            self._update_state(data, result)
            
            # Performance tracking
            calc_time = (datetime.now() - start_time).total_seconds()
            self.calculation_times.append(calc_time)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Calculation failed: {e}")
            
            # Return empty result
            return IndicatorResult(
                values=pd.Series(dtype=float),
                metadata={'error': str(e)},
                timestamp=datetime.now(),
                completeness=0.0,
                stability=0.0,
                confidence=0.0
            )
    
    def calculate_streaming(self, new_data: pd.Series) -> IndicatorResult:
        """Calculate indicator incrementally for streaming data"""
        
        if not self.metadata.supports_streaming:
            raise NotImplementedError(f"{self.metadata.name} does not support streaming")
        
        # Update history
        self.history.append(new_data)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Convert to DataFrame for calculation
        df = pd.DataFrame(self.history)
        
        # Calculate and return only the latest value
        full_result = self.calculate(df, validate=False)
        
        # Extract latest value
        if isinstance(full_result.values, pd.Series):
            latest_value = full_result.values.iloc[-1] if not full_result.values.empty else np.nan
        else:
            latest_value = full_result.values.iloc[-1] if not full_result.values.empty else np.nan
        
        return IndicatorResult(
            values=pd.Series([latest_value]),
            metadata=full_result.metadata,
            timestamp=datetime.now(),
            completeness=1.0,
            stability=full_result.stability,
            confidence=full_result.confidence
        )
    
    def _update_state(self, data: pd.DataFrame, result: IndicatorResult) -> None:
        """Update internal state for streaming calculations"""
        # Default implementation - can be overridden
        self.state['last_calculation'] = datetime.now()
        self.state['last_data_length'] = len(data)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get indicator performance statistics"""
        
        if not self.calculation_times:
            return {}
        
        return {
            'avg_calc_time_ms': np.mean(self.calculation_times) * 1000,
            'max_calc_time_ms': np.max(self.calculation_times) * 1000,
            'total_calculations': len(self.calculation_times),
            'error_rate': self.error_count / len(self.calculation_times),
            'throughput_per_sec': 1.0 / np.mean(self.calculation_times) if self.calculation_times else 0
        }

class IndicatorValidator:
    """Validates indicator inputs and outputs"""
    
    def __init__(self, metadata: IndicatorMetadata):
        self.metadata = metadata
    
    def validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        
        # Check required columns
        missing_cols = set(self.metadata.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check minimum periods
        if len(data) < self.metadata.min_periods:
            raise ValueError(f"Insufficient data: need {self.metadata.min_periods}, got {len(data)}")
        
        # Check for valid data types and ranges
        for col in self.metadata.required_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column {col} must be numeric")
                
                if data[col].isna().all():
                    raise ValueError(f"Column {col} is all NaN")
    
    def validate_output(self, result: IndicatorResult) -> None:
        """Validate output result"""
        
        # Check valid range if specified
        if self.metadata.valid_range:
            min_val, max_val = self.metadata.valid_range
            
            if isinstance(result.values, pd.Series):
                values = result.values.dropna()
            else:
                values = result.values.values.flatten()
                values = values[~np.isnan(values)]
            
            if len(values) > 0:
                if values.min() < min_val or values.max() > max_val:
                    raise ValueError(f"Output values outside valid range {self.metadata.valid_range}")
        
        # Check expected properties
        for prop, expected in self.metadata.expected_properties.items():
            if prop == 'monotonic' and expected:
                if isinstance(result.values, pd.Series):
                    if not result.values.dropna().is_monotonic_increasing:
                        raise ValueError("Expected monotonic increasing values")

def indicator_plugin(indicator_type: IndicatorType = IndicatorType.CUSTOM):
    """Decorator to register indicator as plugin"""
    
    def decorator(cls):
        # Register in plugin registry
        IndicatorRegistry.register(cls, indicator_type)
        
        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)
        
        return wrapper
    
    return decorator

class IndicatorRegistry:
    """Registry for indicator plugins"""
    
    _indicators: Dict[str, type] = {}
    _metadata: Dict[str, IndicatorMetadata] = {}
    
    @classmethod
    def register(cls, indicator_class: type, indicator_type: IndicatorType) -> None:
        """Register an indicator class"""
        
        # Create instance to get metadata
        instance = indicator_class()
        metadata = instance.metadata
        
        cls._indicators[metadata.name] = indicator_class
        cls._metadata[metadata.name] = metadata
        
        print(f"Registered indicator: {metadata.name} ({indicator_type.value})")
    
    @classmethod
    def get_indicator(cls, name: str, **kwargs) -> BaseIndicator:
        """Create indicator instance by name"""
        
        if name not in cls._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        
        return cls._indicators[name](**kwargs)
    
    @classmethod
    def list_indicators(cls, indicator_type: Optional[IndicatorType] = None) -> List[str]:
        """List available indicators"""
        
        if indicator_type is None:
            return list(cls._indicators.keys())
        
        return [
            name for name, metadata in cls._metadata.items()
            if metadata.indicator_type == indicator_type
        ]
    
    @classmethod
    def get_metadata(cls, name: str) -> IndicatorMetadata:
        """Get indicator metadata"""
        
        if name not in cls._metadata:
            raise ValueError(f"Unknown indicator: {name}")
        
        return cls._metadata[name]
```

### 2. Performance-Optimized Indicators

```python
# src/indicators/optimized.py
import numba
from numba import jit, prange
import talib
from scipy import signal

@indicator_plugin(IndicatorType.TREND)
class OptimizedMovingAverage(BaseIndicator):
    """High-performance moving average with multiple algorithms"""
    
    def __init__(self, period: int = 20, algorithm: str = 'sma', **kwargs):
        self.period = period
        self.algorithm = algorithm
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="optimized_ma",
            indicator_type=IndicatorType.TREND,
            description=f"Optimized {self.algorithm.upper()} with period {self.period}",
            required_columns=['close'],
            min_periods=self.period,
            output_columns=['ma'],
            computational_complexity="O(n)",
            memory_complexity="O(1)",
            supports_streaming=True
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate moving average using optimized algorithms"""
        
        close_prices = data['close'].values
        
        if self.algorithm == 'sma':
            ma_values = self._fast_sma(close_prices, self.period)
        elif self.algorithm == 'ema':
            ma_values = self._fast_ema(close_prices, self.period)
        elif self.algorithm == 'wma':
            ma_values = self._fast_wma(close_prices, self.period)
        elif self.algorithm == 'hull':
            ma_values = self._hull_ma(close_prices, self.period)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        result_series = pd.Series(ma_values, index=data.index, name='ma')
        
        # Calculate quality metrics
        completeness = 1.0 - (np.isnan(ma_values).sum() / len(ma_values))
        stability = self._calculate_stability(ma_values)
        
        return IndicatorResult(
            values=result_series,
            metadata={
                'algorithm': self.algorithm,
                'period': self.period,
                'calculation_method': 'optimized_numba'
            },
            timestamp=datetime.now(),
            completeness=completeness,
            stability=stability,
            confidence=min(completeness, stability)
        )
    
    @staticmethod
    @jit(nopython=True)
    def _fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized simple moving average using Numba"""
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        # Calculate first SMA
        sum_val = np.sum(prices[:period])
        result[period - 1] = sum_val / period
        
        # Rolling calculation
        for i in range(period, n):
            sum_val += prices[i] - prices[i - period]
            result[i] = sum_val / period
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized exponential moving average using Numba"""
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n == 0:
            return result
        
        alpha = 2.0 / (period + 1)
        result[0] = prices[0]
        
        for i in range(1, n):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def _fast_wma(prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized weighted moving average using Numba"""
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        # Create weights
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        
        for i in range(period - 1, n):
            window = prices[i - period + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        
        return result
    
    def _hull_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Hull Moving Average calculation"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA with half period
        wma_half = self._fast_wma(prices, half_period)
        
        # WMA with full period
        wma_full = self._fast_wma(prices, period)
        
        # 2 * WMA(n/2) - WMA(n)
        diff = 2 * wma_half - wma_full
        
        # WMA of the difference with sqrt(n) period
        hull_ma = self._fast_wma(diff, sqrt_period)
        
        return hull_ma
    
    def _calculate_stability(self, values: np.ndarray) -> float:
        """Calculate stability measure for the indicator"""
        
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 2:
            return 0.0
        
        # Use coefficient of variation as stability measure
        cv = np.std(clean_values) / (np.mean(np.abs(clean_values)) + 1e-8)
        
        # Convert to stability score (lower CV = higher stability)
        stability = 1.0 / (1.0 + cv)
        
        return stability

@indicator_plugin(IndicatorType.MOMENTUM)
class OptimizedRSI(BaseIndicator):
    """High-performance RSI calculation"""
    
    def __init__(self, period: int = 14, **kwargs):
        self.period = period
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="optimized_rsi",
            indicator_type=IndicatorType.MOMENTUM,
            description=f"Optimized RSI with period {self.period}",
            required_columns=['close'],
            min_periods=self.period + 1,
            output_columns=['rsi'],
            valid_range=(0, 100),
            computational_complexity="O(n)",
            memory_complexity="O(1)",
            supports_streaming=True
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate RSI using optimized algorithm"""
        
        close_prices = data['close'].values
        rsi_values = self._fast_rsi(close_prices, self.period)
        
        result_series = pd.Series(rsi_values, index=data.index, name='rsi')
        
        completeness = 1.0 - (np.isnan(rsi_values).sum() / len(rsi_values))
        
        return IndicatorResult(
            values=result_series,
            metadata={
                'period': self.period,
                'overbought_level': 70,
                'oversold_level': 30
            },
            timestamp=datetime.now(),
            completeness=completeness,
            stability=0.8,  # RSI is generally stable
            confidence=completeness * 0.8
        )
    
    @staticmethod
    @jit(nopython=True)
    def _fast_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """Optimized RSI calculation using Numba"""
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n <= period:
            return result
        
        # Calculate price changes
        changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Avoid division by zero
        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Rolling calculation using Wilder's smoothing
        for i in range(period + 1, n):
            gain = gains[i - 1]
            loss = losses[i - 1]
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return result

@indicator_plugin(IndicatorType.VOLATILITY)
class OptimizedBollingerBands(BaseIndicator):
    """High-performance Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, **kwargs):
        self.period = period
        self.std_dev = std_dev
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="optimized_bollinger",
            indicator_type=IndicatorType.VOLATILITY,
            description=f"Bollinger Bands ({self.period}, {self.std_dev})",
            required_columns=['close'],
            min_periods=self.period,
            output_columns=['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent'],
            computational_complexity="O(n)",
            memory_complexity="O(1)",
            supports_streaming=True
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate Bollinger Bands"""
        
        close_prices = data['close'].values
        
        # Calculate components
        middle = self._fast_sma(close_prices, self.period)
        std = self._fast_rolling_std(close_prices, self.period)
        
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        # Additional metrics
        width = (upper - lower) / middle
        percent = (close_prices - lower) / (upper - lower)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': width,
            'bb_percent': percent
        }, index=data.index)
        
        completeness = 1.0 - (np.isnan(middle).sum() / len(middle))
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'period': self.period,
                'std_dev': self.std_dev
            },
            timestamp=datetime.now(),
            completeness=completeness,
            stability=0.7,
            confidence=completeness * 0.7
        )
    
    @staticmethod
    @jit(nopython=True)
    def _fast_rolling_std(prices: np.ndarray, period: int) -> np.ndarray:
        """Fast rolling standard deviation using Numba"""
        n = len(prices)
        result = np.full(n, np.nan)
        
        if n < period:
            return result
        
        for i in range(period - 1, n):
            window = prices[i - period + 1:i + 1]
            result[i] = np.std(window)
        
        return result
```

### 3. Machine Learning Indicators

```python
# src/indicators/ml_indicators.py
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf

@indicator_plugin(IndicatorType.ML_BASED)
class MLTrendPredictor(BaseIndicator):
    """Machine learning-based trend prediction indicator"""
    
    def __init__(self, lookback: int = 50, prediction_horizon: int = 5, 
                 model_type: str = 'random_forest', **kwargs):
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Training state
        self.is_trained = False
        self.last_training_time = None
        self.training_frequency = timedelta(days=7)  # Retrain weekly
        
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ml_trend_predictor",
            indicator_type=IndicatorType.ML_BASED,
            description=f"ML trend prediction ({self.model_type})",
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.lookback + self.prediction_horizon + 10,
            output_columns=['trend_prediction', 'trend_confidence', 'trend_strength'],
            valid_range=(-1, 1),
            computational_complexity="O(n log n)",
            memory_complexity="O(n)",
            supports_streaming=False  # Requires retraining
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate ML-based trend prediction"""
        
        # Check if model needs training/retraining
        if self._should_retrain():
            self._train_model(data)
        
        if not self.is_trained:
            # Return empty result if model not trained
            empty_result = pd.DataFrame({
                'trend_prediction': np.nan,
                'trend_confidence': np.nan,
                'trend_strength': np.nan
            }, index=data.index)
            
            return IndicatorResult(
                values=empty_result,
                metadata={'model_status': 'not_trained'},
                timestamp=datetime.now(),
                completeness=0.0,
                stability=0.0,
                confidence=0.0
            )
        
        # Generate features
        features = self._generate_features(data)
        
        # Make predictions
        predictions = self._make_predictions(features)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'trend_prediction': predictions['prediction'],
            'trend_confidence': predictions['confidence'],
            'trend_strength': predictions['strength']
        }, index=data.index)
        
        completeness = 1.0 - (np.isnan(predictions['prediction']).sum() / len(predictions['prediction']))
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'model_type': self.model_type,
                'features_used': len(self.feature_names),
                'last_training': self.last_training_time
            },
            timestamp=datetime.now(),
            completeness=completeness,
            stability=0.6,  # ML models can be less stable
            confidence=completeness * 0.6
        )
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        return datetime.now() - self.last_training_time > self.training_frequency
    
    def _train_model(self, data: pd.DataFrame) -> None:
        """Train the ML model"""
        
        self.logger.info(f"Training {self.model_type} model...")
        
        # Generate features and targets
        features = self._generate_features(data)
        targets = self._generate_targets(data)
        
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index]
        y = targets.loc[common_index]
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) < 50:
            self.logger.warning("Insufficient training data")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        self.feature_names = list(X_clean.columns)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'neural_network':
            self.model = self._create_neural_network(X_scaled.shape[1])
        
        try:
            if self.model_type == 'neural_network':
                self.model.fit(
                    X_scaled, y_clean,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2
                )
            else:
                self.model.fit(X_scaled, y_clean)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Model training completed. Features: {len(self.feature_names)}")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.is_trained = False
    
    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for ML model"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['return_1'] = data['close'].pct_change()
        features['return_5'] = data['close'].pct_change(5)
        features['return_20'] = data['close'].pct_change(20)
        
        # Volatility features
        features['volatility_5'] = features['return_1'].rolling(5).std()
        features['volatility_20'] = features['return_1'].rolling(20).std()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].pct_change(5)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'], 14)
        features['bb_position'] = self._calculate_bb_position(data['close'], 20)
        
        # Price position features
        features['price_position_20'] = (data['close'] - data['close'].rolling(20).min()) / \
                                       (data['close'].rolling(20).max() - data['close'].rolling(20).min())
        
        # Momentum features
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        return features.dropna()
    
    def _generate_targets(self, data: pd.DataFrame) -> pd.Series:
        """Generate target variable (future returns)"""
        
        # Future return over prediction horizon
        future_return = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        
        return future_return.dropna()
    
    def _make_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using trained model"""
        
        if not self.is_trained:
            n = len(features)
            return {
                'prediction': np.full(n, np.nan),
                'confidence': np.full(n, np.nan),
                'strength': np.full(n, np.nan)
            }
        
        # Scale features
        features_clean = features.fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(features_clean)
        
        # Make predictions
        if self.model_type == 'neural_network':
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        # Calculate confidence (for tree-based models)
        if hasattr(self.model, 'predict_proba'):
            # Use prediction variance as confidence measure
            confidence = 1.0 - np.std([tree.predict(X_scaled) for tree in self.model.estimators_], axis=0)
        else:
            confidence = np.full(len(predictions), 0.5)  # Default confidence
        
        # Calculate strength (absolute value of prediction)
        strength = np.abs(predictions)
        
        return {
            'prediction': predictions,
            'confidence': confidence,
            'strength': strength
        }
    
    def _create_neural_network(self, input_dim: int) -> tf.keras.Model:
        """Create neural network for trend prediction"""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')  # Output between -1 and 1
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model

@indicator_plugin(IndicatorType.ML_BASED)
class AnomalyDetectionIndicator(BaseIndicator):
    """ML-based anomaly detection for market data"""
    
    def __init__(self, contamination: float = 0.1, **kwargs):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="anomaly_detection",
            indicator_type=IndicatorType.ML_BASED,
            description="ML-based anomaly detection",
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            output_columns=['anomaly_score', 'is_anomaly'],
            computational_complexity="O(n log n)",
            supports_streaming=False
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate anomaly scores"""
        
        # Prepare features
        features = self._prepare_anomaly_features(data)
        
        # Fit model if not fitted
        if not self.is_fitted:
            self.model.fit(features)
            self.is_fitted = True
        
        # Calculate anomaly scores
        anomaly_scores = self.model.decision_function(features)
        is_anomaly = self.model.predict(features) == -1
        
        result_df = pd.DataFrame({
            'anomaly_score': anomaly_scores,
            'is_anomaly': is_anomaly
        }, index=data.index)
        
        return IndicatorResult(
            values=result_df,
            metadata={'contamination': self.contamination},
            timestamp=datetime.now(),
            completeness=1.0,
            stability=0.7,
            confidence=0.7
        )
    
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        
        features = []
        
        # Price features
        returns = data['close'].pct_change()
        features.append(returns.values)
        
        # Volume features
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        features.append(volume_ratio.values)
        
        # Volatility features
        volatility = returns.rolling(10).std()
        features.append(volatility.values)
        
        # Range features
        price_range = (data['high'] - data['low']) / data['close']
        features.append(price_range.values)
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        return feature_matrix
```

### 4. Composite Indicators

```python
# src/indicators/composite.py
@indicator_plugin(IndicatorType.COMPOSITE)
class TrendStrengthComposite(BaseIndicator):
    """Composite indicator combining multiple trend indicators"""
    
    def __init__(self, **kwargs):
        # Component indicators
        self.sma_fast = OptimizedMovingAverage(period=10, algorithm='sma')
        self.sma_slow = OptimizedMovingAverage(period=30, algorithm='sma')
        self.rsi = OptimizedRSI(period=14)
        self.bb = OptimizedBollingerBands(period=20, std_dev=2.0)
        
        # Weights for combination
        self.weights = kwargs.get('weights', [0.3, 0.3, 0.2, 0.2])
        
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="trend_strength_composite",
            indicator_type=IndicatorType.COMPOSITE,
            description="Composite trend strength indicator",
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=30,
            output_columns=['trend_strength', 'trend_direction', 'confidence'],
            valid_range=(-1, 1),
            computational_complexity="O(n)",
            supports_streaming=True
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate composite trend strength"""
        
        # Calculate component indicators
        sma_fast_result = self.sma_fast.calculate(data, validate=False)
        sma_slow_result = self.sma_slow.calculate(data, validate=False)
        rsi_result = self.rsi.calculate(data, validate=False)
        bb_result = self.bb.calculate(data, validate=False)
        
        # Extract signals
        signals = self._extract_signals(data, sma_fast_result, sma_slow_result, rsi_result, bb_result)
        
        # Combine signals
        trend_strength = self._combine_signals(signals)
        
        # Calculate trend direction
        trend_direction = np.sign(trend_strength)
        
        # Calculate confidence
        confidence = self._calculate_confidence(signals)
        
        result_df = pd.DataFrame({
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'confidence': confidence
        }, index=data.index)
        
        completeness = 1.0 - (np.isnan(trend_strength).sum() / len(trend_strength))
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'component_weights': self.weights,
                'components': ['sma_cross', 'rsi_momentum', 'bb_position', 'price_momentum']
            },
            timestamp=datetime.now(),
            completeness=completeness,
            stability=0.8,
            confidence=completeness * 0.8
        )
    
    def _extract_signals(self, data: pd.DataFrame, 
                        sma_fast_result: IndicatorResult,
                        sma_slow_result: IndicatorResult,
                        rsi_result: IndicatorResult,
                        bb_result: IndicatorResult) -> Dict[str, np.ndarray]:
        """Extract normalized signals from component indicators"""
        
        signals = {}
        
        # SMA crossover signal
        sma_fast = sma_fast_result.values.values
        sma_slow = sma_slow_result.values.values
        sma_signal = np.where(sma_fast > sma_slow, 1, -1)
        signals['sma_cross'] = sma_signal
        
        # RSI momentum signal
        rsi_values = rsi_result.values.values
        rsi_signal = (rsi_values - 50) / 50  # Normalize to [-1, 1]
        signals['rsi_momentum'] = rsi_signal
        
        # Bollinger Bands position signal
        bb_percent = bb_result.values['bb_percent'].values
        bb_signal = (bb_percent - 0.5) * 2  # Normalize to [-1, 1]
        signals['bb_position'] = bb_signal
        
        # Price momentum signal
        close_prices = data['close']
        price_momentum = close_prices.pct_change(10)
        price_signal = np.tanh(price_momentum * 100)  # Normalize with tanh
        signals['price_momentum'] = price_signal.values
        
        return signals
    
    def _combine_signals(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine signals using weighted average"""
        
        signal_names = ['sma_cross', 'rsi_momentum', 'bb_position', 'price_momentum']
        combined_signal = np.zeros_like(signals[signal_names[0]])
        
        for i, signal_name in enumerate(signal_names):
            if i < len(self.weights):
                weight = self.weights[i]
                signal_values = signals[signal_name]
                
                # Handle NaN values
                signal_values = np.nan_to_num(signal_values)
                
                combined_signal += weight * signal_values
        
        # Normalize to [-1, 1]
        combined_signal = np.tanh(combined_signal)
        
        return combined_signal
    
    def _calculate_confidence(self, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate confidence based on signal agreement"""
        
        signal_values = list(signals.values())
        signal_matrix = np.column_stack(signal_values)
        
        # Replace NaN with 0
        signal_matrix = np.nan_to_num(signal_matrix)
        
        # Calculate agreement (correlation between signals)
        confidence = np.zeros(signal_matrix.shape[0])
        
        for i in range(signal_matrix.shape[0]):
            row = signal_matrix[i]
            
            # Calculate pairwise agreement
            agreements = []
            for j in range(len(row)):
                for k in range(j + 1, len(row)):
                    agreement = 1 - abs(row[j] - row[k]) / 2  # Agreement measure
                    agreements.append(agreement)
            
            confidence[i] = np.mean(agreements) if agreements else 0.5
        
        return confidence

@indicator_plugin(IndicatorType.COMPOSITE)
class MarketRegimeIndicator(BaseIndicator):
    """Composite indicator for market regime identification"""
    
    def __init__(self, **kwargs):
        self.volatility_window = kwargs.get('volatility_window', 20)
        self.trend_window = kwargs.get('trend_window', 50)
        super().__init__(**kwargs)
    
    def _create_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="market_regime",
            indicator_type=IndicatorType.COMPOSITE,
            description="Market regime identification indicator",
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.trend_window,
            output_columns=['regime', 'volatility_regime', 'trend_regime', 'regime_strength'],
            computational_complexity="O(n)",
            supports_streaming=True
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate market regime"""
        
        # Calculate regime components
        volatility_regime = self._calculate_volatility_regime(data)
        trend_regime = self._calculate_trend_regime(data)
        volume_regime = self._calculate_volume_regime(data)
        
        # Combine into overall regime
        overall_regime = self._combine_regimes(volatility_regime, trend_regime, volume_regime)
        
        # Calculate regime strength
        regime_strength = self._calculate_regime_strength(data)
        
        result_df = pd.DataFrame({
            'regime': overall_regime,
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'regime_strength': regime_strength
        }, index=data.index)
        
        return IndicatorResult(
            values=result_df,
            metadata={
                'volatility_window': self.volatility_window,
                'trend_window': self.trend_window
            },
            timestamp=datetime.now(),
            completeness=0.9,
            stability=0.7,
            confidence=0.8
        )
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate volatility regime"""
        
        returns = data['close'].pct_change()
        rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # Define volatility thresholds
        vol_low_threshold = rolling_vol.quantile(0.33)
        vol_high_threshold = rolling_vol.quantile(0.67)
        
        regime = np.where(
            rolling_vol < vol_low_threshold, 'low_vol',
            np.where(rolling_vol > vol_high_threshold, 'high_vol', 'medium_vol')
        )
        
        return regime
    
    def _calculate_trend_regime(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate trend regime"""
        
        close_prices = data['close']
        sma_short = close_prices.rolling(self.trend_window // 2).mean()
        sma_long = close_prices.rolling(self.trend_window).mean()
        
        trend_signal = (sma_short - sma_long) / sma_long
        
        trend_up_threshold = 0.02
        trend_down_threshold = -0.02
        
        regime = np.where(
            trend_signal > trend_up_threshold, 'uptrend',
            np.where(trend_signal < trend_down_threshold, 'downtrend', 'sideways')
        )
        
        return regime
    
    def _calculate_volume_regime(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate volume regime"""
        
        volume = data['volume']
        volume_sma = volume.rolling(self.volatility_window).mean()
        volume_ratio = volume / volume_sma
        
        volume_threshold = 1.5
        
        regime = np.where(volume_ratio > volume_threshold, 'high_volume', 'normal_volume')
        
        return regime
    
    def _combine_regimes(self, vol_regime: np.ndarray, 
                        trend_regime: np.ndarray,
                        volume_regime: np.ndarray) -> np.ndarray:
        """Combine individual regimes into overall regime"""
        
        combined = []
        
        for v, t, vol in zip(vol_regime, trend_regime, volume_regime):
            if t == 'uptrend':
                if v == 'low_vol':
                    regime = 'bull_low_vol'
                else:
                    regime = 'bull_high_vol'
            elif t == 'downtrend':
                if v == 'low_vol':
                    regime = 'bear_low_vol'
                else:
                    regime = 'bear_high_vol'
            else:  # sideways
                if v == 'high_vol':
                    regime = 'volatile_range'
                else:
                    regime = 'quiet_range'
            
            combined.append(regime)
        
        return np.array(combined)
    
    def _calculate_regime_strength(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate strength/confidence of regime classification"""
        
        # Use multiple measures to assess regime strength
        returns = data['close'].pct_change()
        
        # Trend strength
        trend_strength = abs(returns.rolling(self.trend_window).mean())
        
        # Volatility consistency
        vol_consistency = 1 / (1 + returns.rolling(self.volatility_window).std().rolling(10).std())
        
        # Combine measures
        regime_strength = (trend_strength + vol_consistency) / 2
        
        return regime_strength.fillna(0.5).values
```

### 5. Indicator Performance Engine

```python
# src/indicators/performance_engine.py
class IndicatorPerformanceEngine:
    """
    High-performance engine for calculating multiple indicators.
    Optimizes computation and memory usage.
    """
    
    def __init__(self):
        self.indicators: Dict[str, BaseIndicator] = {}
        self.calculation_graph = {}
        self.cache = {}
        self.performance_stats = {}
        
        self.logger = ComponentLogger("IndicatorPerformanceEngine", "indicators")
    
    def register_indicator(self, name: str, indicator: BaseIndicator) -> None:
        """Register an indicator with the engine"""
        self.indicators[name] = indicator
        self.performance_stats[name] = {
            'calculations': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def calculate_indicators(self, data: pd.DataFrame,
                           indicator_names: List[str],
                           use_cache: bool = True,
                           parallel: bool = True) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators efficiently"""
        
        results = {}
        start_time = datetime.now()
        
        # Create dependency graph
        dependency_graph = self._create_dependency_graph(indicator_names)
        
        # Calculate in topological order
        calculation_order = self._topological_sort(dependency_graph)
        
        if parallel and len(calculation_order) > 1:
            # Parallel calculation for independent indicators
            results = self._calculate_parallel(data, calculation_order, use_cache)
        else:
            # Sequential calculation
            results = self._calculate_sequential(data, calculation_order, use_cache)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(
            f"Calculated {len(indicator_names)} indicators in {total_time:.3f}s"
        )
        
        return results
    
    def _calculate_sequential(self, data: pd.DataFrame,
                            calculation_order: List[str],
                            use_cache: bool) -> Dict[str, IndicatorResult]:
        """Calculate indicators sequentially"""
        
        results = {}
        
        for indicator_name in calculation_order:
            if indicator_name not in self.indicators:
                continue
            
            # Check cache first
            cache_key = self._generate_cache_key(indicator_name, data)
            
            if use_cache and cache_key in self.cache:
                results[indicator_name] = self.cache[cache_key]
                self.performance_stats[indicator_name]['cache_hits'] += 1
                continue
            
            # Calculate indicator
            indicator = self.indicators[indicator_name]
            
            calc_start = datetime.now()
            result = indicator.calculate(data)
            calc_time = (datetime.now() - calc_start).total_seconds()
            
            # Update statistics
            stats = self.performance_stats[indicator_name]
            stats['calculations'] += 1
            stats['total_time'] += calc_time
            stats['cache_misses'] += 1
            
            # Cache result
            if use_cache:
                self.cache[cache_key] = result
            
            results[indicator_name] = result
        
        return results
    
    def _calculate_parallel(self, data: pd.DataFrame,
                          calculation_order: List[str],
                          use_cache: bool) -> Dict[str, IndicatorResult]:
        """Calculate indicators in parallel where possible"""
        
        import concurrent.futures
        
        results = {}
        
        # Group indicators by dependency level
        dependency_levels = self._group_by_dependency_level(calculation_order)
        
        for level_indicators in dependency_levels:
            if len(level_indicators) == 1:
                # Single indicator - calculate directly
                indicator_name = level_indicators[0]
                indicator = self.indicators[indicator_name]
                result = indicator.calculate(data)
                results[indicator_name] = result
            else:
                # Multiple indicators - calculate in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    
                    for indicator_name in level_indicators:
                        if indicator_name in self.indicators:
                            indicator = self.indicators[indicator_name]
                            future = executor.submit(indicator.calculate, data)
                            futures[future] = indicator_name
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        indicator_name = futures[future]
                        try:
                            result = future.result()
                            results[indicator_name] = result
                        except Exception as e:
                            self.logger.error(f"Parallel calculation failed for {indicator_name}: {e}")
        
        return results
    
    def _create_dependency_graph(self, indicator_names: List[str]) -> Dict[str, List[str]]:
        """Create dependency graph for indicators"""
        
        # For now, assume all indicators are independent
        # In a more sophisticated system, indicators could depend on each other
        
        graph = {}
        for name in indicator_names:
            graph[name] = []  # No dependencies
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Topological sort of dependency graph"""
        
        # Simple implementation since we don't have dependencies yet
        return list(graph.keys())
    
    def _group_by_dependency_level(self, calculation_order: List[str]) -> List[List[str]]:
        """Group indicators by dependency level for parallel execution"""
        
        # For now, put all indicators at the same level since no dependencies
        return [calculation_order]
    
    def _generate_cache_key(self, indicator_name: str, data: pd.DataFrame) -> str:
        """Generate cache key for indicator calculation"""
        
        # Use data hash and indicator name
        data_hash = hash(tuple(data.index.tolist() + data.iloc[-1].tolist()))
        
        return f"{indicator_name}_{data_hash}"
    
    def clear_cache(self) -> None:
        """Clear indicator cache"""
        self.cache.clear()
        self.logger.info("Indicator cache cleared")
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all indicators"""
        
        summary = {}
        
        for name, stats in self.performance_stats.items():
            if stats['calculations'] > 0:
                avg_time = stats['total_time'] / stats['calculations']
                cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
                
                summary[name] = {
                    'total_calculations': stats['calculations'],
                    'avg_calc_time_ms': avg_time * 1000,
                    'total_time_s': stats['total_time'],
                    'cache_hit_rate': cache_hit_rate,
                    'throughput_per_sec': 1.0 / avg_time if avg_time > 0 else 0
                }
        
        return summary
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_6_custom_indicators.py`:

```python
class TestIndicatorFramework:
    """Test core indicator framework"""
    
    def test_indicator_registration(self):
        """Test indicator plugin registration"""
        
        @indicator_plugin(IndicatorType.CUSTOM)
        class TestIndicator(BaseIndicator):
            def _create_metadata(self):
                return IndicatorMetadata(
                    name="test_indicator",
                    indicator_type=IndicatorType.CUSTOM,
                    description="Test indicator",
                    required_columns=['close']
                )
            
            def _calculate(self, data, **kwargs):
                return IndicatorResult(
                    values=data['close'] * 2,
                    metadata={},
                    timestamp=datetime.now(),
                    completeness=1.0,
                    stability=1.0,
                    confidence=1.0
                )
        
        # Should be registered
        assert "test_indicator" in IndicatorRegistry.list_indicators()
        
        # Should be creatable
        indicator = IndicatorRegistry.get_indicator("test_indicator")
        assert isinstance(indicator, TestIndicator)

class TestOptimizedIndicators:
    """Test performance-optimized indicators"""
    
    def test_optimized_sma_performance(self):
        """Test optimized SMA calculation"""
        # Generate test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(10000) * 0.01)
        data = pd.DataFrame({'close': prices})
        
        # Test optimized calculation
        indicator = OptimizedMovingAverage(period=20)
        
        start_time = time.time()
        result = indicator.calculate(data)
        calc_time = time.time() - start_time
        
        # Should be fast
        assert calc_time < 0.1  # Less than 100ms for 10k points
        
        # Should be accurate
        expected_sma = data['close'].rolling(20).mean()
        np.testing.assert_array_almost_equal(
            result.values.dropna().values,
            expected_sma.dropna().values,
            decimal=6
        )
    
    def test_fast_rsi_calculation(self):
        """Test optimized RSI calculation"""
        # Create known test case
        prices = np.array([44, 44.34, 44.09, 44.15, 44.25, 43.23, 42.55, 42.30, 42.66, 43.13])
        
        rsi_values = OptimizedRSI._fast_rsi(prices, 9)
        
        # Should not be all NaN
        assert not np.isnan(rsi_values[-1])
        
        # Should be in valid range
        valid_values = rsi_values[~np.isnan(rsi_values)]
        assert np.all((valid_values >= 0) & (valid_values <= 100))

class TestMLIndicators:
    """Test machine learning indicators"""
    
    def test_ml_trend_predictor_training(self):
        """Test ML trend predictor training"""
        # Generate synthetic data with trend
        np.random.seed(42)
        trend = np.linspace(0, 0.1, 1000)
        noise = np.random.randn(1000) * 0.01
        prices = 100 * np.exp(np.cumsum(trend + noise))
        
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Create and train indicator
        indicator = MLTrendPredictor(lookback=50, model_type='random_forest')
        result = indicator.calculate(data)
        
        # Should have predictions
        assert not result.values['trend_prediction'].isna().all()
        
        # Should be in valid range
        predictions = result.values['trend_prediction'].dropna()
        assert np.all((predictions >= -1) & (predictions <= 1))

class TestCompositeIndicators:
    """Test composite indicators"""
    
    def test_trend_strength_composite(self):
        """Test trend strength composite indicator"""
        # Generate test data
        data = generate_test_ohlcv_data(periods=200)
        
        indicator = TrendStrengthComposite()
        result = indicator.calculate(data)
        
        # Should have all output columns
        expected_columns = ['trend_strength', 'trend_direction', 'confidence']
        for col in expected_columns:
            assert col in result.values.columns
        
        # Values should be in valid ranges
        trend_strength = result.values['trend_strength'].dropna()
        assert np.all((trend_strength >= -1) & (trend_strength <= 1))
        
        trend_direction = result.values['trend_direction'].dropna()
        assert np.all(np.isin(trend_direction, [-1, 0, 1]))
```

### Integration Tests

Create `tests/integration/test_step10_6_indicator_integration.py`:

```python
def test_indicator_performance_engine():
    """Test performance engine with multiple indicators"""
    # Setup engine
    engine = IndicatorPerformanceEngine()
    
    # Register indicators
    indicators = {
        'sma_20': OptimizedMovingAverage(period=20),
        'rsi_14': OptimizedRSI(period=14),
        'bb_20': OptimizedBollingerBands(period=20),
        'trend_composite': TrendStrengthComposite()
    }
    
    for name, indicator in indicators.items():
        engine.register_indicator(name, indicator)
    
    # Generate test data
    data = generate_test_ohlcv_data(periods=1000)
    
    # Calculate all indicators
    results = engine.calculate_indicators(
        data, 
        list(indicators.keys()),
        parallel=True
    )
    
    # Should have all results
    assert len(results) == len(indicators)
    
    # All results should be valid
    for name, result in results.items():
        assert isinstance(result, IndicatorResult)
        assert result.completeness > 0

def test_streaming_indicator_updates():
    """Test real-time streaming indicator updates"""
    # Setup streaming indicators
    sma = OptimizedMovingAverage(period=10)
    rsi = OptimizedRSI(period=14)
    
    # Initial data
    initial_data = generate_test_ohlcv_data(periods=50)
    
    # Calculate initial values
    sma_result = sma.calculate(initial_data)
    rsi_result = rsi.calculate(initial_data)
    
    # Simulate streaming updates
    for i in range(10):
        # Generate new data point
        new_point = pd.Series({
            'open': 100 + np.random.randn() * 0.5,
            'high': 101 + np.random.randn() * 0.5,
            'low': 99 + np.random.randn() * 0.5,
            'close': 100 + np.random.randn() * 0.5,
            'volume': np.random.randint(1000, 5000)
        })
        
        # Update indicators
        sma_update = sma.calculate_streaming(new_point)
        rsi_update = rsi.calculate_streaming(new_point)
        
        # Should have valid updates
        assert not np.isnan(sma_update.values.iloc[0])
        assert 0 <= rsi_update.values.iloc[0] <= 100

def test_indicator_validation_framework():
    """Test indicator validation"""
    # Create indicator with validation
    rsi = OptimizedRSI(period=14)
    
    # Test with valid data
    valid_data = generate_test_ohlcv_data(periods=50)
    result = rsi.calculate(valid_data)
    assert result.completeness > 0
    
    # Test with insufficient data
    insufficient_data = generate_test_ohlcv_data(periods=5)
    result = rsi.calculate(insufficient_data)
    assert result.completeness == 0  # Should fail validation
    
    # Test with missing columns
    invalid_data = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
    result = rsi.calculate(invalid_data)
    assert result.completeness == 0  # Should fail validation
```

### System Tests

Create `tests/system/test_step10_6_production_indicators.py`:

```python
def test_high_frequency_indicator_calculations():
    """Test indicators under high-frequency conditions"""
    # Setup multiple indicators
    indicators = {
        'sma_fast': OptimizedMovingAverage(period=5),
        'sma_slow': OptimizedMovingAverage(period=20),
        'rsi': OptimizedRSI(period=14),
        'bb': OptimizedBollingerBands(period=20),
        'trend': TrendStrengthComposite()
    }
    
    engine = IndicatorPerformanceEngine()
    for name, indicator in indicators.items():
        engine.register_indicator(name, indicator)
    
    # Test with high-frequency updates
    performance_stats = []
    
    for i in range(1000):  # 1000 updates
        # Generate market data update
        data = generate_market_update()
        
        start_time = time.time()
        
        # Calculate all indicators
        results = engine.calculate_indicators(
            data,
            list(indicators.keys()),
            use_cache=True,
            parallel=True
        )
        
        calc_time = time.time() - start_time
        performance_stats.append(calc_time)
    
    # Performance requirements
    avg_calc_time = np.mean(performance_stats)
    p95_calc_time = np.percentile(performance_stats, 95)
    
    assert avg_calc_time < 0.050  # 50ms average
    assert p95_calc_time < 0.100  # 100ms p95
    
    # Cache should be helping
    perf_summary = engine.get_performance_summary()
    for indicator_stats in perf_summary.values():
        assert indicator_stats['cache_hit_rate'] > 0.5  # At least 50% cache hits

def test_memory_efficiency():
    """Test memory efficiency of indicator calculations"""
    import psutil
    import gc
    
    # Get baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create many indicators
    indicators = []
    for i in range(100):
        indicators.extend([
            OptimizedMovingAverage(period=10 + i),
            OptimizedRSI(period=14),
            OptimizedBollingerBands(period=20 + i)
        ])
    
    # Generate large dataset
    large_data = generate_test_ohlcv_data(periods=10000)
    
    # Calculate all indicators
    for indicator in indicators:
        result = indicator.calculate(large_data)
        # Don't store results to test cleanup
        del result
    
    # Force garbage collection
    gc.collect()
    
    # Check memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - baseline_memory
    
    # Should not use excessive memory
    assert memory_increase < 500  # Less than 500MB increase

def test_indicator_accuracy_benchmark():
    """Test indicator accuracy against known implementations"""
    # Generate test data
    np.random.seed(42)
    data = generate_test_ohlcv_data(periods=1000)
    
    # Test SMA accuracy against pandas
    sma_optimized = OptimizedMovingAverage(period=20)
    sma_result = sma_optimized.calculate(data)
    
    pandas_sma = data['close'].rolling(20).mean()
    
    # Should match pandas implementation
    np.testing.assert_array_almost_equal(
        sma_result.values.dropna().values,
        pandas_sma.dropna().values,
        decimal=8
    )
    
    # Test RSI accuracy against TA-Lib if available
    try:
        import talib
        
        rsi_optimized = OptimizedRSI(period=14)
        rsi_result = rsi_optimized.calculate(data)
        
        talib_rsi = talib.RSI(data['close'].values, timeperiod=14)
        
        # Should be close to TA-Lib implementation
        optimized_values = rsi_result.values.dropna().values
        talib_values = talib_rsi[~np.isnan(talib_rsi)]
        
        # Allow small differences due to implementation details
        max_diff = np.max(np.abs(optimized_values - talib_values))
        assert max_diff < 0.1  # Less than 0.1 RSI point difference
        
    except ImportError:
        pass  # Skip if TA-Lib not available

def test_ml_indicator_robustness():
    """Test ML indicators under various market conditions"""
    scenarios = ['trending_up', 'trending_down', 'sideways', 'volatile']
    
    ml_indicator = MLTrendPredictor(lookback=50)
    
    for scenario in scenarios:
        # Generate scenario data
        data = generate_market_scenario(scenario, periods=500)
        
        # Calculate indicator
        result = ml_indicator.calculate(data)
        
        # Should handle all scenarios
        assert result.completeness > 0.5
        
        # Predictions should be reasonable
        predictions = result.values['trend_prediction'].dropna()
        assert len(predictions) > 0
        assert np.all((predictions >= -1) & (predictions <= 1))
        
        # Confidence should be reasonable
        confidence = result.values['trend_confidence'].dropna()
        assert np.all((confidence >= 0) & (confidence <= 1))
        assert np.mean(confidence) > 0.3  # Average confidence > 30%
```

## ✅ Validation Checklist

### Core Framework
- [ ] Plugin architecture working
- [ ] Indicator registration functional
- [ ] Metadata system complete
- [ ] Validation framework operational
- [ ] Performance tracking active

### Optimized Indicators
- [ ] Numba optimization working
- [ ] Performance benchmarks met
- [ ] Accuracy verified
- [ ] Streaming support functional
- [ ] Memory efficiency confirmed

### ML Indicators
- [ ] Model training successful
- [ ] Prediction accuracy reasonable
- [ ] Feature engineering robust
- [ ] Online learning functional
- [ ] Error handling comprehensive

### Composite Indicators
- [ ] Signal combination working
- [ ] Confidence calculation accurate
- [ ] Component weighting effective
- [ ] Output validation passing
- [ ] Performance acceptable

### Performance Engine
- [ ] Multi-indicator calculation
- [ ] Caching system working
- [ ] Parallel processing functional
- [ ] Memory management effective
- [ ] Performance monitoring active

## 📊 Performance Benchmarks

### Calculation Performance
- Simple indicators: < 1ms per 1000 points
- Complex indicators: < 10ms per 1000 points
- ML indicators: < 100ms per training
- Composite indicators: < 5ms per calculation

### Memory Efficiency
- Memory usage: < 100MB per indicator
- Cache hit rate: > 80%
- Memory growth: < 1MB per hour
- Garbage collection: < 50ms impact

### Accuracy Requirements
- SMA/EMA accuracy: 8 decimal places
- RSI accuracy: < 0.1 point difference
- ML prediction stability: > 70%
- Composite confidence: > 60%

## 🐛 Common Issues

1. **Performance Bottlenecks**
   - Use Numba for hot loops
   - Implement proper caching
   - Optimize memory access patterns
   - Profile regularly

2. **ML Model Stability**
   - Regular retraining
   - Feature scaling
   - Robust error handling
   - Validation frameworks

3. **Memory Leaks**
   - Clear unused data
   - Limit cache size
   - Monitor memory growth
   - Implement cleanup

## 🎯 Success Criteria

Step 10.6 is complete when:
1. ✅ Plugin framework operational
2. ✅ Optimized indicators functional
3. ✅ ML indicators working
4. ✅ Composite indicators effective
5. ✅ Performance benchmarks met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.7: Advanced Visualization](step-10.7-visualization.md)

## 📚 Additional Resources

- [Technical Analysis Programming](../references/ta-programming.md)
- [High-Performance Computing](../references/hpc-patterns.md)
- [ML for Finance](../references/ml-finance.md)
- [Plugin Architecture Design](../references/plugin-architecture.md)