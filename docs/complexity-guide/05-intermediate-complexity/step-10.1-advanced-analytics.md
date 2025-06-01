# Step 10.1: Advanced Analytics

**Status**: Intermediate Complexity Step
**Complexity**: High
**Prerequisites**: [Step 10.8: Memory & Batch Processing](../04-multi-phase-integration/step-10.8-memory-batch.md) completed
**Architecture Ref**: [Analytics Architecture](../architecture/analytics-architecture.md)

## 🎯 Objective

Implement advanced analytics capabilities:
- Market microstructure analysis
- Regime detection and analysis
- Pattern recognition and clustering
- Feature engineering automation
- Multi-factor attribution analysis

## 📋 Required Reading

Before starting:
1. [Market Microstructure Theory](../references/market-microstructure.md)
2. [Regime Detection Methods](../references/regime-detection.md)
3. [Factor Analysis in Trading](../references/factor-analysis.md)

## 🏗️ Implementation Tasks

### 1. Market Microstructure Analysis

```python
# src/analytics/microstructure.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statsmodels.api as sm

@dataclass
class MicrostructureMetrics:
    """Market microstructure analysis results"""
    bid_ask_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    
    # Liquidity measures
    depth_imbalance: float
    order_flow_toxicity: float
    volume_clock_variance: float
    
    # Information measures
    pin_score: float  # Probability of informed trading
    vpin: float  # Volume-synchronized PIN
    kyle_lambda: float  # Price impact coefficient
    
    # Execution quality
    implementation_shortfall: float
    market_impact_cost: float
    timing_cost: float

class MicrostructureAnalyzer:
    """
    Analyzes market microstructure for better execution.
    Implements advanced liquidity and information metrics.
    """
    
    def __init__(self):
        self.metrics_history: List[MicrostructureMetrics] = []
        self.logger = ComponentLogger("MicrostructureAnalyzer", "analytics")
    
    def analyze_market_quality(self, 
                             order_book_data: pd.DataFrame,
                             trade_data: pd.DataFrame,
                             time_window: str = '5min') -> MicrostructureMetrics:
        """Comprehensive market quality analysis"""
        
        # Calculate spreads
        spreads = self._calculate_spreads(order_book_data, trade_data)
        
        # Analyze liquidity
        liquidity_metrics = self._analyze_liquidity(order_book_data, trade_data)
        
        # Information content
        information_metrics = self._analyze_information_content(
            order_book_data, trade_data, time_window
        )
        
        # Execution quality
        execution_metrics = self._analyze_execution_quality(trade_data)
        
        metrics = MicrostructureMetrics(
            **spreads,
            **liquidity_metrics,
            **information_metrics,
            **execution_metrics
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_spreads(self, order_book: pd.DataFrame, 
                         trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate various spread measures"""
        # Quoted spread
        bid_ask_spread = (order_book['ask'] - order_book['bid']).mean()
        
        # Effective spread (using trade prices)
        midpoint = (order_book['ask'] + order_book['bid']) / 2
        trade_side = self._classify_trade_side(trades, midpoint)
        effective_spread = 2 * abs(trades['price'] - midpoint).mean()
        
        # Realized spread (5-min price change)
        future_midpoint = midpoint.shift(-100)  # 5-min ahead
        realized_spread = 2 * trade_side * (trades['price'] - future_midpoint)
        realized_spread = realized_spread.dropna().mean()
        
        # Price impact
        price_impact = effective_spread - realized_spread
        
        return {
            'bid_ask_spread': bid_ask_spread,
            'effective_spread': effective_spread,
            'realized_spread': realized_spread,
            'price_impact': price_impact
        }
    
    def _analyze_liquidity(self, order_book: pd.DataFrame,
                         trades: pd.DataFrame) -> Dict[str, float]:
        """Analyze market liquidity metrics"""
        # Depth imbalance
        bid_depth = order_book['bid_size']
        ask_depth = order_book['ask_size']
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        # Order flow toxicity (simplified VPIN)
        volume_buckets = self._create_volume_buckets(trades, bucket_size=50)
        buy_volume = volume_buckets['buy_volume']
        sell_volume = volume_buckets['sell_volume']
        order_flow_toxicity = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # Volume clock variance
        volume_clock_variance = self._calculate_volume_clock_variance(trades)
        
        return {
            'depth_imbalance': depth_imbalance.mean(),
            'order_flow_toxicity': order_flow_toxicity.mean(),
            'volume_clock_variance': volume_clock_variance
        }
    
    def _analyze_information_content(self, order_book: pd.DataFrame,
                                   trades: pd.DataFrame,
                                   time_window: str) -> Dict[str, float]:
        """Analyze information content in order flow"""
        # PIN (Probability of Informed Trading)
        pin_score = self._calculate_pin(trades)
        
        # VPIN (Volume-synchronized PIN)
        vpin = self._calculate_vpin(trades)
        
        # Kyle's Lambda (price impact of order flow)
        kyle_lambda = self._estimate_kyle_lambda(trades, order_book)
        
        return {
            'pin_score': pin_score,
            'vpin': vpin,
            'kyle_lambda': kyle_lambda
        }
    
    def _calculate_pin(self, trades: pd.DataFrame) -> float:
        """
        Calculate Probability of Informed Trading (PIN).
        Uses the Easley-O'Hara model.
        """
        # Group trades by day
        daily_trades = trades.groupby(trades.index.date)
        
        # Count buys and sells
        buy_counts = []
        sell_counts = []
        
        for date, day_trades in daily_trades:
            buys = (day_trades['side'] == 'buy').sum()
            sells = (day_trades['side'] == 'sell').sum()
            buy_counts.append(buys)
            sell_counts.append(sells)
        
        # Maximum likelihood estimation of PIN parameters
        # Simplified version - real implementation would use EM algorithm
        total_days = len(buy_counts)
        informed_days = sum(1 for b, s in zip(buy_counts, sell_counts) 
                          if abs(b - s) > np.std(buy_counts) + np.std(sell_counts))
        
        alpha = informed_days / total_days  # Probability of information event
        
        # Estimate order arrival rates
        epsilon_b = np.mean(buy_counts)  # Uninformed buy rate
        epsilon_s = np.mean(sell_counts)  # Uninformed sell rate
        mu = np.mean([abs(b - s) for b, s in zip(buy_counts, sell_counts)])  # Informed rate
        
        # PIN = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
        pin = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s) if (epsilon_b + epsilon_s) > 0 else 0
        
        return pin
    
    def _calculate_vpin(self, trades: pd.DataFrame, 
                       bucket_size: int = 50) -> float:
        """
        Calculate Volume-Synchronized PIN (VPIN).
        More robust than traditional PIN.
        """
        # Create volume buckets
        volume_buckets = self._create_volume_buckets(trades, bucket_size)
        
        # Calculate order imbalance for each bucket
        imbalances = []
        for _, bucket in volume_buckets.iterrows():
            buy_vol = bucket['buy_volume']
            sell_vol = bucket['sell_volume']
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                imbalance = abs(buy_vol - sell_vol) / total_vol
                imbalances.append(imbalance)
        
        # VPIN is the average imbalance
        vpin = np.mean(imbalances) if imbalances else 0
        
        return vpin
    
    def _estimate_kyle_lambda(self, trades: pd.DataFrame,
                            order_book: pd.DataFrame) -> float:
        """
        Estimate Kyle's lambda (price impact coefficient).
        Measures how much prices move per unit of net order flow.
        """
        # Calculate price changes
        midpoint = (order_book['ask'] + order_book['bid']) / 2
        price_changes = midpoint.diff()
        
        # Calculate net order flow
        signed_volume = trades['volume'].values
        signed_volume[trades['side'] == 'sell'] *= -1
        net_order_flow = pd.Series(signed_volume).rolling(window=10).sum()
        
        # Align timestamps
        price_changes = price_changes.reindex(net_order_flow.index, method='ffill')
        
        # Regression: price_change = lambda * net_order_flow
        valid_idx = ~(price_changes.isna() | net_order_flow.isna())
        
        if valid_idx.sum() > 10:
            X = sm.add_constant(net_order_flow[valid_idx])
            y = price_changes[valid_idx]
            model = sm.OLS(y, X).fit()
            kyle_lambda = model.params[1]  # Coefficient on order flow
        else:
            kyle_lambda = 0
        
        return abs(kyle_lambda)  # Lambda should be positive
```

### 2. Regime Detection

```python
# src/analytics/regime_detection.py
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import ruptures as rpt

class RegimeDetector:
    """
    Detects market regimes using multiple methods.
    Combines HMM, change point detection, and clustering.
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.models = {}
        self.current_regime = None
        self.regime_history = []
        self.logger = ComponentLogger("RegimeDetector", "analytics")
    
    def detect_regimes(self, market_data: pd.DataFrame,
                      features: Optional[List[str]] = None) -> RegimeAnalysis:
        """Comprehensive regime detection"""
        
        if features is None:
            features = self._extract_regime_features(market_data)
        else:
            features = market_data[features]
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Method 1: Hidden Markov Model
        hmm_regimes = self._detect_hmm_regimes(features_scaled)
        
        # Method 2: Gaussian Mixture Model
        gmm_regimes = self._detect_gmm_regimes(features_scaled)
        
        # Method 3: Change Point Detection
        changepoints = self._detect_changepoints(features_scaled)
        
        # Method 4: Rolling Statistics
        rolling_regimes = self._detect_rolling_regimes(market_data)
        
        # Combine methods
        consensus_regimes = self._combine_regime_detections(
            hmm_regimes, gmm_regimes, changepoints, rolling_regimes
        )
        
        # Analyze regime characteristics
        regime_stats = self._analyze_regime_characteristics(
            market_data, consensus_regimes
        )
        
        return RegimeAnalysis(
            regimes=consensus_regimes,
            regime_stats=regime_stats,
            changepoints=changepoints,
            transition_matrix=self._calculate_transition_matrix(consensus_regimes),
            regime_durations=self._calculate_regime_durations(consensus_regimes)
        )
    
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Returns and volatility
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_change'] = features['volatility'].pct_change()
        
        # Volume patterns
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_volatility'] = data['volume'].rolling(20).std()
        
        # Price patterns
        features['price_momentum'] = data['close'].pct_change(20)
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Market microstructure
        if 'spread' in data.columns:
            features['spread_mean'] = data['spread'].rolling(20).mean()
            features['spread_vol'] = data['spread'].rolling(20).std()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd_signal'] = self._calculate_macd_signal(data['close'])
        
        return features.dropna()
    
    def _detect_hmm_regimes(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using Hidden Markov Model"""
        # Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        model.fit(features)
        regimes = model.predict(features)
        
        # Store model for future predictions
        self.models['hmm'] = model
        
        return regimes
    
    def _detect_gmm_regimes(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using Gaussian Mixture Model"""
        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        
        gmm.fit(features)
        regimes = gmm.predict(features)
        
        # Store model
        self.models['gmm'] = gmm
        
        return regimes
    
    def _detect_changepoints(self, features: np.ndarray) -> List[int]:
        """Detect change points in the data"""
        # Use multiple change point detection methods
        changepoints = []
        
        # Method 1: Pelt (Pruned Exact Linear Time)
        algo_pelt = rpt.Pelt(model="rbf").fit(features)
        changepoints_pelt = algo_pelt.predict(pen=10)
        
        # Method 2: Binary Segmentation
        algo_binseg = rpt.Binseg(model="rbf").fit(features)
        changepoints_binseg = algo_binseg.predict(n_bkps=self.n_regimes-1)
        
        # Method 3: Window sliding
        algo_window = rpt.Window(width=50, model="rbf").fit(features)
        changepoints_window = algo_window.predict(n_bkps=self.n_regimes-1)
        
        # Combine and deduplicate
        all_changepoints = sorted(set(
            changepoints_pelt[:-1] + 
            changepoints_binseg[:-1] + 
            changepoints_window[:-1]
        ))
        
        # Filter changepoints that are too close
        filtered_changepoints = []
        for cp in all_changepoints:
            if not filtered_changepoints or cp - filtered_changepoints[-1] > 50:
                filtered_changepoints.append(cp)
        
        return filtered_changepoints
    
    def _detect_rolling_regimes(self, data: pd.DataFrame, 
                              window: int = 60) -> np.ndarray:
        """Detect regimes using rolling statistics"""
        # Calculate rolling metrics
        returns = data['close'].pct_change()
        
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_skew = returns.rolling(window).skew()
        rolling_kurt = returns.rolling(window).kurt()
        
        # Create regime indicators
        regimes = np.zeros(len(data))
        
        # Bull market: positive returns, low volatility
        bull_condition = (rolling_mean > 0.0005) & (rolling_std < rolling_std.median())
        
        # Bear market: negative returns
        bear_condition = rolling_mean < -0.0005
        
        # High volatility regime
        high_vol_condition = rolling_std > rolling_std.quantile(0.75)
        
        # Assign regimes
        regimes[bull_condition] = 0
        regimes[bear_condition] = 1
        regimes[high_vol_condition] = 2
        
        return regimes
    
    def _combine_regime_detections(self, *regime_arrays) -> np.ndarray:
        """Combine multiple regime detection methods"""
        # Stack all regime predictions
        all_regimes = np.column_stack(regime_arrays)
        
        # Use majority voting
        from scipy.stats import mode
        consensus_regimes = mode(all_regimes, axis=1)[0].flatten()
        
        # Smooth regimes to avoid rapid switching
        smoothed_regimes = self._smooth_regime_transitions(consensus_regimes)
        
        return smoothed_regimes
    
    def _smooth_regime_transitions(self, regimes: np.ndarray, 
                                  min_duration: int = 20) -> np.ndarray:
        """Smooth regime transitions to avoid noise"""
        smoothed = regimes.copy()
        
        # Find regime changes
        changes = np.where(np.diff(regimes) != 0)[0] + 1
        changes = np.concatenate([[0], changes, [len(regimes)]])
        
        # Remove short regimes
        for i in range(len(changes) - 1):
            start, end = changes[i], changes[i + 1]
            if end - start < min_duration and i > 0:
                # Replace with previous regime
                smoothed[start:end] = smoothed[start - 1]
        
        return smoothed
```

### 3. Pattern Recognition

```python
# src/analytics/pattern_recognition.py
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from dtaidistance import dtw
import numpy as np

class PatternRecognizer:
    """
    Recognizes complex patterns in market data.
    Uses clustering and similarity measures.
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.pattern_models = {}
        self.logger = ComponentLogger("PatternRecognizer", "analytics")
    
    def find_patterns(self, price_data: pd.DataFrame,
                     pattern_length: int = 50,
                     min_similarity: float = 0.8) -> PatternAnalysisResult:
        """Find recurring patterns in price data"""
        
        # Extract candidate patterns
        patterns = self._extract_pattern_candidates(price_data, pattern_length)
        
        # Normalize patterns
        normalized_patterns = self._normalize_patterns(patterns)
        
        # Cluster similar patterns
        clusters = self._cluster_patterns(normalized_patterns)
        
        # Identify significant patterns
        significant_patterns = self._identify_significant_patterns(
            clusters, normalized_patterns, price_data
        )
        
        # Analyze pattern performance
        pattern_performance = self._analyze_pattern_performance(
            significant_patterns, price_data
        )
        
        return PatternAnalysisResult(
            patterns=significant_patterns,
            clusters=clusters,
            performance=pattern_performance,
            pattern_probabilities=self._calculate_pattern_probabilities(significant_patterns)
        )
    
    def _extract_pattern_candidates(self, data: pd.DataFrame, 
                                  pattern_length: int) -> List[np.ndarray]:
        """Extract overlapping windows as pattern candidates"""
        patterns = []
        prices = data['close'].values
        
        for i in range(len(prices) - pattern_length):
            pattern = prices[i:i + pattern_length]
            patterns.append(pattern)
        
        return patterns
    
    def _normalize_patterns(self, patterns: List[np.ndarray]) -> np.ndarray:
        """Normalize patterns for comparison"""
        normalized = []
        
        for pattern in patterns:
            # Z-score normalization
            pattern_norm = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
            
            # Also store percentage changes
            pct_changes = np.diff(pattern) / pattern[:-1]
            
            # Combine both representations
            combined = np.concatenate([pattern_norm, pct_changes])
            normalized.append(combined)
        
        return np.array(normalized)
    
    def _cluster_patterns(self, patterns: np.ndarray) -> Dict[int, List[int]]:
        """Cluster similar patterns"""
        # Reduce dimensionality
        pca = PCA(n_components=min(10, patterns.shape[1]))
        patterns_reduced = pca.fit_transform(patterns)
        
        # DBSCAN clustering for arbitrary shapes
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(patterns_reduced)
        
        # Also try K-means for comparison
        n_clusters = min(20, len(patterns) // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(patterns_reduced)
        
        # Combine clustering results
        clusters = {}
        for i, (db_label, km_label) in enumerate(zip(cluster_labels, kmeans_labels)):
            # Use DBSCAN as primary, K-means as secondary
            if db_label != -1:  # Not noise
                cluster_id = db_label
            else:
                cluster_id = n_clusters + km_label
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        return clusters
    
    def _calculate_pattern_similarity(self, pattern1: np.ndarray, 
                                    pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns using DTW"""
        distance = dtw.distance(pattern1, pattern2)
        
        # Convert distance to similarity score (0-1)
        max_distance = np.sqrt(len(pattern1) * 4)  # Theoretical maximum
        similarity = 1 - (distance / max_distance)
        
        return similarity
    
    def _identify_chart_patterns(self, price_data: pd.DataFrame) -> Dict[str, List[ChartPattern]]:
        """Identify classical chart patterns"""
        patterns_found = {
            'head_and_shoulders': [],
            'double_top': [],
            'double_bottom': [],
            'triangle': [],
            'flag': [],
            'wedge': []
        }
        
        # Head and Shoulders
        h_s_patterns = self._find_head_and_shoulders(price_data)
        patterns_found['head_and_shoulders'].extend(h_s_patterns)
        
        # Double Top/Bottom
        double_patterns = self._find_double_patterns(price_data)
        patterns_found['double_top'].extend(double_patterns['tops'])
        patterns_found['double_bottom'].extend(double_patterns['bottoms'])
        
        # Triangles
        triangles = self._find_triangle_patterns(price_data)
        patterns_found['triangle'].extend(triangles)
        
        return patterns_found
    
    def _find_head_and_shoulders(self, data: pd.DataFrame, 
                                window: int = 20) -> List[ChartPattern]:
        """Find head and shoulders patterns"""
        patterns = []
        prices = data['close'].values
        
        # Find local peaks and troughs
        peaks = self._find_peaks(prices, window)
        troughs = self._find_troughs(prices, window)
        
        # Look for H&S pattern: trough-peak-trough-peak-trough-peak-trough
        for i in range(len(peaks) - 2):
            if i < len(troughs) - 3:
                # Check pattern structure
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Shoulders should be roughly equal height
                shoulder_ratio = prices[right_shoulder] / prices[left_shoulder]
                if 0.9 < shoulder_ratio < 1.1:
                    # Head should be higher
                    if prices[head] > prices[left_shoulder] * 1.02:
                        pattern = ChartPattern(
                            pattern_type='head_and_shoulders',
                            start_idx=left_shoulder,
                            end_idx=right_shoulder,
                            confidence=self._calculate_pattern_confidence(
                                prices[left_shoulder:right_shoulder+1]
                            ),
                            key_points={
                                'left_shoulder': left_shoulder,
                                'head': head,
                                'right_shoulder': right_shoulder
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
```

### 4. Feature Engineering Automation

```python
# src/analytics/feature_engineering.py
from typing import List, Dict, Callable, Union
import talib
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class AutomatedFeatureEngineer:
    """
    Automatically generates and selects features.
    Uses domain knowledge and statistical methods.
    """
    
    def __init__(self):
        self.feature_generators = self._initialize_feature_generators()
        self.selected_features = []
        self.feature_importance = {}
        self.logger = ComponentLogger("AutomatedFeatureEngineer", "analytics")
    
    def engineer_features(self, data: pd.DataFrame,
                        target: Optional[pd.Series] = None,
                        feature_types: List[str] = None) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        
        if feature_types is None:
            feature_types = ['price', 'volume', 'technical', 'microstructure', 'statistical']
        
        features = pd.DataFrame(index=data.index)
        
        # Generate features by type
        for ftype in feature_types:
            if ftype in self.feature_generators:
                type_features = self.feature_generators[ftype](data)
                features = pd.concat([features, type_features], axis=1)
        
        # Generate interaction features
        if len(features.columns) > 1:
            interaction_features = self._generate_interaction_features(features)
            features = pd.concat([features, interaction_features], axis=1)
        
        # Generate lagged features
        lagged_features = self._generate_lagged_features(features)
        features = pd.concat([features, lagged_features], axis=1)
        
        # Feature selection if target provided
        if target is not None:
            features = self._select_features(features, target)
        
        return features
    
    def _initialize_feature_generators(self) -> Dict[str, Callable]:
        """Initialize feature generation functions"""
        return {
            'price': self._generate_price_features,
            'volume': self._generate_volume_features,
            'technical': self._generate_technical_features,
            'microstructure': self._generate_microstructure_features,
            'statistical': self._generate_statistical_features
        }
    
    def _generate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features"""
        features = pd.DataFrame(index=data.index)
        
        # Returns at multiple scales
        for period in [1, 5, 10, 20, 60]:
            features[f'return_{period}'] = data['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(data['close']).diff(period)
        
        # Price position
        for period in [20, 50, 200]:
            features[f'price_position_{period}'] = (
                (data['close'] - data['close'].rolling(period).min()) /
                (data['close'].rolling(period).max() - data['close'].rolling(period).min())
            )
        
        # Price patterns
        features['higher_high'] = (
            (data['high'] > data['high'].shift(1)) & 
            (data['high'].shift(1) > data['high'].shift(2))
        ).astype(int)
        
        features['lower_low'] = (
            (data['low'] < data['low'].shift(1)) & 
            (data['low'].shift(1) < data['low'].shift(2))
        ).astype(int)
        
        # Volatility features
        for period in [10, 20, 60]:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}'] / 
                features[f'volatility_{period}'].rolling(period).mean()
            )
        
        # Price acceleration
        features['price_acceleration'] = data['close'].pct_change().diff()
        
        return features
    
    def _generate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators"""
        features = pd.DataFrame(index=data.index)
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = talib.SMA(data['close'], period)
            features[f'ema_{period}'] = talib.EMA(data['close'], period)
            features[f'sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
        
        # MACD
        macd, signal, hist = talib.MACD(data['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # RSI
        for period in [14, 28]:
            features[f'rsi_{period}'] = talib.RSI(data['close'], period)
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(data['close'], period)
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = upper - lower
            features[f'bb_position_{period}'] = (data['close'] - lower) / (upper - lower)
        
        # Stochastic
        slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'])
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        
        # ATR
        for period in [14, 28]:
            features[f'atr_{period}'] = talib.ATR(data['high'], data['low'], data['close'], period)
        
        # ADX
        features['adx'] = talib.ADX(data['high'], data['low'], data['close'])
        
        return features
    
    def _generate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features"""
        features = pd.DataFrame(index=data.index)
        
        if 'bid' in data.columns and 'ask' in data.columns:
            # Spread features
            features['spread'] = data['ask'] - data['bid']
            features['spread_pct'] = features['spread'] / data['close']
            features['spread_ma'] = features['spread'].rolling(20).mean()
            features['spread_std'] = features['spread'].rolling(20).std()
            
            # Mid-price
            features['mid_price'] = (data['ask'] + data['bid']) / 2
            features['price_to_mid'] = data['close'] / features['mid_price']
        
        if 'volume' in data.columns:
            # Volume features
            features['volume_ma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']
            features['volume_trend'] = features['volume_ma'].pct_change(20)
            
            # Dollar volume
            features['dollar_volume'] = data['close'] * data['volume']
            features['dollar_volume_ma'] = features['dollar_volume'].rolling(20).mean()
        
        # Tick features
        features['tick_direction'] = np.sign(data['close'].diff())
        features['tick_count_up'] = (features['tick_direction'] > 0).rolling(20).sum()
        features['tick_count_down'] = (features['tick_direction'] < 0).rolling(20).sum()
        features['tick_ratio'] = features['tick_count_up'] / (features['tick_count_down'] + 1)
        
        return features
    
    def _select_features(self, features: pd.DataFrame, 
                        target: pd.Series,
                        max_features: int = 50) -> pd.DataFrame:
        """Select most important features"""
        # Remove features with too many NaN values
        valid_features = features.dropna(thresh=len(features) * 0.8, axis=1)
        
        # Fill remaining NaN values
        valid_features = valid_features.fillna(method='ffill').fillna(0)
        
        # Align target
        common_index = valid_features.index.intersection(target.index)
        valid_features = valid_features.loc[common_index]
        aligned_target = target.loc[common_index]
        
        # Calculate feature importance using multiple methods
        importance_scores = {}
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(valid_features, aligned_target)
        for i, col in enumerate(valid_features.columns):
            importance_scores[col] = mi_scores[i]
        
        # Method 2: Random Forest Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(valid_features, aligned_target)
        
        for i, col in enumerate(valid_features.columns):
            if col in importance_scores:
                importance_scores[col] = (importance_scores[col] + rf.feature_importances_[i]) / 2
            else:
                importance_scores[col] = rf.feature_importances_[i]
        
        # Select top features
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        selected_cols = [col for col, _ in sorted_features[:max_features]]
        
        self.selected_features = selected_cols
        self.feature_importance = importance_scores
        
        return valid_features[selected_cols]
```

### 5. Multi-Factor Attribution

```python
# src/analytics/factor_attribution.py
class FactorAttributionAnalyzer:
    """
    Performs multi-factor attribution analysis.
    Decomposes returns into factor contributions.
    """
    
    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_models = {}
        self.attribution_history = []
        self.logger = ComponentLogger("FactorAttributionAnalyzer", "analytics")
    
    def analyze_attribution(self, returns: pd.Series,
                          factor_data: pd.DataFrame,
                          method: str = 'regression') -> AttributionResult:
        """Perform factor attribution analysis"""
        
        # Align data
        common_index = returns.index.intersection(factor_data.index)
        returns_aligned = returns.loc[common_index]
        factors_aligned = factor_data.loc[common_index, self.factors]
        
        # Perform attribution based on method
        if method == 'regression':
            attribution = self._regression_attribution(returns_aligned, factors_aligned)
        elif method == 'brinson':
            attribution = self._brinson_attribution(returns_aligned, factors_aligned)
        elif method == 'risk_parity':
            attribution = self._risk_parity_attribution(returns_aligned, factors_aligned)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Calculate factor statistics
        factor_stats = self._calculate_factor_statistics(
            returns_aligned, factors_aligned, attribution
        )
        
        # Residual analysis
        residual_analysis = self._analyze_residuals(
            returns_aligned, factors_aligned, attribution
        )
        
        result = AttributionResult(
            factor_returns=attribution['factor_returns'],
            factor_exposures=attribution['exposures'],
            r_squared=attribution['r_squared'],
            factor_statistics=factor_stats,
            residual_analysis=residual_analysis
        )
        
        self.attribution_history.append(result)
        return result
    
    def _regression_attribution(self, returns: pd.Series,
                              factors: pd.DataFrame) -> Dict:
        """Factor attribution using regression"""
        # Add constant for intercept
        X = sm.add_constant(factors)
        
        # Run regression
        model = sm.OLS(returns, X).fit()
        
        # Calculate factor returns
        factor_returns = {}
        for i, factor in enumerate(self.factors):
            factor_returns[factor] = model.params[i+1] * factors[factor]
        
        # Add alpha
        factor_returns['alpha'] = model.params[0]
        
        # Calculate total explained return
        explained_return = sum(factor_returns.values())
        
        return {
            'factor_returns': pd.DataFrame(factor_returns),
            'exposures': model.params[1:].to_dict(),
            'r_squared': model.rsquared,
            'model': model
        }
    
    def _calculate_factor_statistics(self, returns: pd.Series,
                                   factors: pd.DataFrame,
                                   attribution: Dict) -> Dict:
        """Calculate detailed factor statistics"""
        stats = {}
        
        factor_returns = attribution['factor_returns']
        
        for factor in self.factors:
            if factor in factor_returns.columns:
                factor_contribution = factor_returns[factor]
                
                stats[factor] = {
                    'total_contribution': factor_contribution.sum(),
                    'avg_contribution': factor_contribution.mean(),
                    'contribution_vol': factor_contribution.std(),
                    'contribution_pct': factor_contribution.sum() / returns.sum() * 100,
                    'information_ratio': factor_contribution.mean() / factor_contribution.std() * np.sqrt(252),
                    'correlation_with_total': factor_contribution.corr(returns)
                }
        
        return stats
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_1_analytics.py`:

```python
class TestMicrostructureAnalysis:
    """Test market microstructure analytics"""
    
    def test_spread_calculations(self):
        """Test various spread measures"""
        # Create test order book data
        order_book = pd.DataFrame({
            'bid': [99.95, 99.96, 99.94],
            'ask': [100.05, 100.04, 100.06],
            'bid_size': [100, 150, 200],
            'ask_size': [120, 100, 180]
        })
        
        trades = pd.DataFrame({
            'price': [100.00, 99.98, 100.02],
            'volume': [50, 75, 100],
            'side': ['buy', 'sell', 'buy']
        })
        
        analyzer = MicrostructureAnalyzer()
        metrics = analyzer._calculate_spreads(order_book, trades)
        
        assert metrics['bid_ask_spread'] > 0
        assert metrics['effective_spread'] >= metrics['bid_ask_spread']
        assert 'price_impact' in metrics

class TestRegimeDetection:
    """Test regime detection methods"""
    
    def test_hmm_regime_detection(self):
        """Test HMM-based regime detection"""
        # Generate synthetic data with regimes
        np.random.seed(42)
        n_samples = 1000
        
        # Create data with 3 distinct regimes
        regime1 = np.random.normal(0.001, 0.01, 300)
        regime2 = np.random.normal(-0.001, 0.02, 400)
        regime3 = np.random.normal(0.002, 0.005, 300)
        
        returns = np.concatenate([regime1, regime2, regime3])
        
        data = pd.DataFrame({
            'close': (1 + returns).cumprod() * 100,
            'volume': np.random.uniform(1000, 2000, n_samples)
        })
        
        detector = RegimeDetector(n_regimes=3)
        features = detector._extract_regime_features(data)
        regimes = detector._detect_hmm_regimes(features.values)
        
        # Should detect 3 unique regimes
        assert len(np.unique(regimes)) == 3
        
        # Regimes should be relatively stable
        regime_changes = np.sum(np.diff(regimes) != 0)
        assert regime_changes < 50  # Not too many transitions
```

### Integration Tests

Create `tests/integration/test_step10_1_analytics_integration.py`:

```python
def test_complete_analytics_pipeline():
    """Test full analytics pipeline integration"""
    # Load test data
    market_data = load_test_market_data()
    
    # 1. Microstructure analysis
    microstructure = MicrostructureAnalyzer()
    micro_metrics = microstructure.analyze_market_quality(
        market_data['order_book'],
        market_data['trades']
    )
    
    # 2. Regime detection
    regime_detector = RegimeDetector(n_regimes=4)
    regime_analysis = regime_detector.detect_regimes(market_data['ohlcv'])
    
    # 3. Pattern recognition
    pattern_recognizer = PatternRecognizer()
    patterns = pattern_recognizer.find_patterns(market_data['ohlcv'])
    
    # 4. Feature engineering
    feature_engineer = AutomatedFeatureEngineer()
    features = feature_engineer.engineer_features(
        market_data['ohlcv'],
        target=market_data['forward_returns']
    )
    
    # 5. Factor attribution
    factor_analyzer = FactorAttributionAnalyzer(['momentum', 'value', 'quality'])
    attribution = factor_analyzer.analyze_attribution(
        market_data['strategy_returns'],
        market_data['factor_data']
    )
    
    # Verify all components produce valid results
    assert micro_metrics.pin_score >= 0
    assert len(regime_analysis.regimes) == len(market_data['ohlcv'])
    assert len(patterns.patterns) > 0
    assert len(features.columns) > 10
    assert attribution.r_squared > 0

def test_adaptive_analytics():
    """Test analytics adaptation to market conditions"""
    # Simulate different market conditions
    market_conditions = ['bull', 'bear', 'volatile', 'ranging']
    
    analytics_suite = AdvancedAnalyticsSuite()
    
    for condition in market_conditions:
        # Generate appropriate test data
        test_data = generate_market_data(condition)
        
        # Run analytics
        analysis = analytics_suite.analyze(test_data)
        
        # Verify adaptation
        if condition == 'volatile':
            assert analysis.microstructure.spread > analysis.historical_average_spread
            assert analysis.regime_detector.current_regime == 'high_volatility'
        
        elif condition == 'bull':
            assert analysis.patterns.bullish_patterns > analysis.patterns.bearish_patterns
            assert analysis.factor_attribution.momentum_contribution > 0
```

### System Tests

Create `tests/system/test_step10_1_production_analytics.py`:

```python
def test_real_time_analytics_performance():
    """Test analytics performance with real-time data"""
    # Setup real-time data feed
    data_feed = create_realtime_feed()
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Process time windows
    processing_times = []
    results = []
    
    for i in range(100):  # 100 time windows
        window_data = data_feed.get_next_window()
        
        start_time = time.time()
        result = analytics_engine.process(window_data)
        processing_time = time.time() - start_time
        
        processing_times.append(processing_time)
        results.append(result)
    
    # Performance requirements
    assert np.mean(processing_times) < 0.1  # Average under 100ms
    assert np.percentile(processing_times, 99) < 0.5  # 99th percentile under 500ms
    assert all(r.is_valid for r in results)

def test_multi_asset_analytics():
    """Test analytics across multiple assets"""
    assets = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT']
    
    multi_asset_analyzer = MultiAssetAnalyzer()
    
    # Analyze each asset
    asset_analyses = {}
    for asset in assets:
        data = load_asset_data(asset)
        analysis = multi_asset_analyzer.analyze_asset(asset, data)
        asset_analyses[asset] = analysis
    
    # Cross-asset analysis
    correlation_matrix = multi_asset_analyzer.calculate_correlations(asset_analyses)
    regime_synchronization = multi_asset_analyzer.analyze_regime_sync(asset_analyses)
    factor_exposures = multi_asset_analyzer.aggregate_factor_exposures(asset_analyses)
    
    # Verify multi-asset insights
    assert correlation_matrix.shape == (len(assets), len(assets))
    assert 0 <= regime_synchronization <= 1
    assert all(factor in factor_exposures for factor in ['market', 'size', 'value'])
```

## ✅ Validation Checklist

### Microstructure Analysis
- [ ] All spread measures calculated correctly
- [ ] PIN and VPIN scores valid (0-1 range)
- [ ] Kyle's lambda estimation working
- [ ] Execution quality metrics accurate

### Regime Detection
- [ ] Multiple detection methods integrated
- [ ] Regime transitions smooth
- [ ] Change points identified correctly
- [ ] Regime characteristics analyzed

### Pattern Recognition
- [ ] Pattern clustering working
- [ ] Classical chart patterns detected
- [ ] Pattern performance tracked
- [ ] Similarity measures accurate

### Feature Engineering
- [ ] All feature types generated
- [ ] Feature selection working
- [ ] Interaction features created
- [ ] Importance scores calculated

### Factor Attribution
- [ ] Attribution sums to total return
- [ ] R-squared values reasonable
- [ ] Factor statistics accurate
- [ ] Residual analysis complete

## 📊 Performance Benchmarks

### Real-time Processing
- Single asset analytics: < 100ms
- Multi-asset analytics: < 500ms
- Pattern recognition: < 200ms
- Feature generation: < 50ms

### Accuracy Metrics
- Regime detection accuracy: > 85%
- Pattern recognition precision: > 70%
- Factor attribution R²: > 0.8
- Feature selection stability: > 0.9

## 🐛 Common Issues

1. **Data Quality**
   - Handle missing data gracefully
   - Validate input ranges
   - Check for outliers
   - Ensure time alignment

2. **Computational Efficiency**
   - Cache intermediate results
   - Use vectorized operations
   - Implement lazy evaluation
   - Consider approximations

3. **Model Stability**
   - Regularize regime detection
   - Validate pattern significance
   - Cross-validate features
   - Test attribution robustness

## 🎯 Success Criteria

Step 10.1 is complete when:
1. ✅ All analytics modules implemented
2. ✅ Real-time performance achieved
3. ✅ Multi-asset support working
4. ✅ Integration tests passing
5. ✅ Production-ready performance

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.2: Multi-Asset Support](step-10.2-multi-asset.md)

## 📚 Additional Resources

- [Market Microstructure Theory](../references/market-microstructure-theory.md)
- [Advanced Pattern Recognition](../references/advanced-patterns.md)
- [Factor Model Construction](../references/factor-models.md)
- [Time Series Analytics](../references/time-series-advanced.md)