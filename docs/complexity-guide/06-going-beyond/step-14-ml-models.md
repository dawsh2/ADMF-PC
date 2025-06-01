# Step 14: Machine Learning Models

**Status**: Extreme Complexity Step  
**Complexity**: Extreme  
**Prerequisites**: [Step 13: Cross-Exchange Arbitrage](step-13-cross-exchange-arbitrage.md) completed  
**Architecture Ref**: [ML Architecture](../architecture/ml-architecture.md)

## 🎯 Objective

Deploy advanced machine learning and AI models for trading:
- Deep learning for price prediction
- Reinforcement learning for strategy optimization
- Natural language processing for sentiment analysis
- Computer vision for chart pattern recognition
- Ensemble methods for robust predictions
- Online learning for adaptive strategies
- Explainable AI for model interpretability

## 📋 Required Reading

Before starting:
1. [Financial Machine Learning](../references/financial-ml.md)
2. [Deep Learning for Trading](../references/deep-learning-trading.md)
3. [Reinforcement Learning in Finance](../references/rl-finance.md)
4. [MLOps Best Practices](../references/mlops-practices.md)

## 🏗️ Implementation Tasks

### 1. ML Framework Core

```python
# src/ml/framework.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import mlflow
import optuna
from ray import tune

class ModelType(Enum):
    """Types of ML models"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    REINFORCEMENT = "reinforcement"

class PredictionTarget(Enum):
    """What the model predicts"""
    PRICE = "price"
    RETURNS = "returns"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    REGIME = "regime"
    SIGNALS = "signals"

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: ModelType
    target: PredictionTarget
    
    # Model architecture
    input_features: List[str]
    hidden_layers: List[int]
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Feature engineering
    lookback_window: int = 60
    prediction_horizon: int = 1
    use_technical_indicators: bool = True
    use_market_microstructure: bool = True
    
    # Hyperparameter optimization
    enable_hpo: bool = True
    hpo_trials: int = 100
    
    # Model management
    model_name: str = ""
    experiment_name: str = ""
    tracking_uri: str = "mlflow"

@dataclass
class ModelPrediction:
    """Model prediction output"""
    model_id: str
    timestamp: datetime
    
    # Predictions
    point_estimate: float
    confidence_interval: Tuple[float, float]
    prediction_distribution: Optional[np.ndarray] = None
    
    # Metadata
    features_used: Dict[str, float] = field(default_factory=dict)
    model_confidence: float = 0.0
    prediction_latency_ms: float = 0.0

class BaseMLModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.training_history = {}
        self.is_trained = False
        
        # MLflow tracking
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name or "trading_models")
        
        self.logger = ComponentLogger(f"MLModel_{config.model_type.value}", "ml")
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make predictions"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for training"""
        
        # Feature engineering
        features = self._engineer_features(data)
        
        # Create sequences for time series
        X, y = self._create_sequences(features)
        
        # Scale features
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return X_scaled, y
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        features = data.copy()
        
        if self.config.use_technical_indicators:
            # Price-based features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'ma_{period}'] = features['close'].rolling(period).mean()
                features[f'ma_ratio_{period}'] = features['close'] / features[f'ma_{period}']
            
            # Volatility
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['volatility_60'] = features['returns'].rolling(60).std()
            
            # RSI
            features['rsi'] = self._calculate_rsi(features['close'])
            
            # Bollinger Bands
            features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(features['close'])
            features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        if self.config.use_market_microstructure:
            # Volume features
            features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
            features['dollar_volume'] = features['close'] * features['volume']
            
            # Spread features (if available)
            if 'bid' in features.columns and 'ask' in features.columns:
                features['spread'] = features['ask'] - features['bid']
                features['spread_pct'] = features['spread'] / features['close']
            
            # Order imbalance (if available)
            if 'bid_volume' in features.columns and 'ask_volume' in features.columns:
                features['order_imbalance'] = (features['bid_volume'] - features['ask_volume']) / (features['bid_volume'] + features['ask_volume'])
        
        # Time-based features
        features['hour'] = pd.to_datetime(features.index).hour
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek
        features['month'] = pd.to_datetime(features.index).month
        
        # Drop NaN values
        features = features.dropna()
        
        # Select only configured features
        if self.config.input_features:
            features = features[self.config.input_features]
        
        return features
    
    def _create_sequences(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        
        n_features = len(features.columns)
        n_samples = len(features) - self.config.lookback_window - self.config.prediction_horizon + 1
        
        X = np.zeros((n_samples, self.config.lookback_window, n_features))
        
        # Target based on prediction type
        if self.config.target == PredictionTarget.PRICE:
            y = features['close'].values
        elif self.config.target == PredictionTarget.RETURNS:
            y = features['returns'].values
        elif self.config.target == PredictionTarget.DIRECTION:
            y = (features['returns'] > 0).astype(int).values
        elif self.config.target == PredictionTarget.VOLATILITY:
            y = features['volatility_20'].values
        else:
            y = features['close'].values  # Default
        
        # Create sequences
        for i in range(n_samples):
            X[i] = features.iloc[i:i+self.config.lookback_window].values
        
        # Adjust y for prediction horizon
        y = y[self.config.lookback_window + self.config.prediction_horizon - 1:]
        
        return X, y
    
    def save_model(self, path: str):
        """Save trained model"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        
        joblib.dump(model_artifacts, path)
        
        # Log to MLflow
        if self.config.tracking_uri:
            mlflow.sklearn.log_model(model_artifacts, "model")
    
    def load_model(self, path: str):
        """Load trained model"""
        model_artifacts = joblib.load(path)
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.config = model_artifacts['config']
        self.feature_importance = model_artifacts['feature_importance']
        self.training_history = model_artifacts['training_history']
        self.is_trained = True
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        return upper_band, lower_band

class LSTMPricePredictor(BaseMLModel):
    """LSTM model for price prediction"""
    
    def build_model(self):
        """Build LSTM model"""
        n_features = len(self.config.input_features)
        
        model = nn.Sequential()
        
        # LSTM layers
        model.add_module('lstm1', nn.LSTM(
            input_size=n_features,
            hidden_size=self.config.hidden_layers[0],
            num_layers=1,
            batch_first=True,
            dropout=self.config.dropout_rate
        ))
        
        # Additional LSTM layers
        for i in range(1, len(self.config.hidden_layers)):
            model.add_module(f'lstm{i+1}', nn.LSTM(
                input_size=self.config.hidden_layers[i-1],
                hidden_size=self.config.hidden_layers[i],
                num_layers=1,
                batch_first=True,
                dropout=self.config.dropout_rate
            ))
        
        # Dense layers
        model.add_module('flatten', nn.Flatten())
        model.add_module('dense1', nn.Linear(
            self.config.hidden_layers[-1] * self.config.lookback_window,
            64
        ))
        model.add_module('relu', nn.ReLU())
        model.add_module('dropout', nn.Dropout(self.config.dropout_rate))
        model.add_module('output', nn.Linear(64, 1))
        
        self.model = model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None):
        """Train LSTM model"""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_losses.append(val_loss.item())
                
                # Early stopping
                if len(val_losses) > self.config.early_stopping_patience:
                    if all(val_losses[-self.config.early_stopping_patience:][i] >= 
                          val_losses[-self.config.early_stopping_patience:][i-1] 
                          for i in range(1, self.config.early_stopping_patience)):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Train Loss: {loss.item():.6f}")
        
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make predictions with uncertainty estimation"""
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        start_time = datetime.now()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        # Multiple forward passes for uncertainty
        n_predictions = 100
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_predictions):
                pred = self.model(X_tensor).numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        point_estimate = np.mean(predictions)
        std_dev = np.std(predictions)
        confidence_interval = (
            point_estimate - 2 * std_dev,
            point_estimate + 2 * std_dev
        )
        
        # Model confidence based on prediction variance
        model_confidence = 1.0 / (1.0 + std_dev)
        
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ModelPrediction(
            model_id=f"{self.config.model_type.value}_{self.config.model_name}",
            timestamp=datetime.now(),
            point_estimate=float(point_estimate),
            confidence_interval=confidence_interval,
            prediction_distribution=predictions.flatten(),
            model_confidence=float(model_confidence),
            prediction_latency_ms=prediction_time
        )
```

### 2. Transformer Models

```python
# src/ml/transformer_models.py
class TransformerPredictor(BaseMLModel):
    """Transformer model for financial time series"""
    
    def build_model(self):
        """Build Transformer model"""
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-np.log(10000.0) / d_model))
                
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                return x + self.pe[:x.size(0), :]
        
        class TransformerModel(nn.Module):
            def __init__(self, n_features, d_model=512, nhead=8, 
                        num_layers=6, dropout=0.1):
                super().__init__()
                
                self.input_projection = nn.Linear(n_features, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=2048, 
                    dropout=dropout, activation='gelu'
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers, num_layers
                )
                
                self.decoder = nn.Linear(d_model, 1)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, src):
                # src shape: (batch, seq_len, features)
                src = src.transpose(0, 1)  # (seq_len, batch, features)
                
                src = self.input_projection(src)
                src = self.pos_encoder(src)
                src = self.dropout(src)
                
                output = self.transformer_encoder(src)
                output = output[-1]  # Take last time step
                output = self.decoder(output)
                
                return output
        
        n_features = len(self.config.input_features)
        self.model = TransformerModel(
            n_features=n_features,
            d_model=512,
            nhead=8,
            num_layers=6,
            dropout=self.config.dropout_rate
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None):
        """Train Transformer with advanced techniques"""
        
        # Similar to LSTM but with learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        # Mixed precision training for efficiency
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop with mixed precision
        for epoch in range(self.config.epochs):
            with torch.cuda.amp.autocast():
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

class AttentionAnalyzer:
    """Analyze attention weights for interpretability"""
    
    def __init__(self, model: TransformerModel):
        self.model = model
        self.attention_weights = {}
    
    def extract_attention_weights(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract attention weights from transformer"""
        
        attention_maps = []
        
        def hook_fn(module, input, output):
            attention_maps.append(output[1].detach().cpu().numpy())
        
        # Register hooks
        handles = []
        for layer in self.model.transformer_encoder.layers:
            handle = layer.self_attn.register_forward_hook(hook_fn)
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(X)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return {
            f'layer_{i}': attention_map 
            for i, attention_map in enumerate(attention_maps)
        }
    
    def visualize_attention(self, attention_weights: Dict[str, np.ndarray],
                          feature_names: List[str]):
        """Visualize attention patterns"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        for layer_name, weights in attention_weights.items():
            # Average over heads and batch
            avg_attention = weights.mean(axis=(0, 1))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attention, xticklabels=feature_names,
                       yticklabels=feature_names, cmap='Blues')
            plt.title(f'Attention Weights - {layer_name}')
            plt.tight_layout()
            plt.show()
```

### 3. Reinforcement Learning

```python
# src/ml/reinforcement_learning.py
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        
        # Action space: [hold, buy, sell] with continuous position sizing
        self.action_space = spaces.Box(
            low=np.array([-1, 0]),  # action, size
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # Observation space: market features
        n_features = 20  # Price, volume, indicators, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one time step"""
        
        # Parse action
        action_type = action[0]  # -1 to 1
        position_size = action[1]  # 0 to 1
        
        # Current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute trade
        if action_type > 0.5:  # Buy
            max_shares = self.balance / current_price
            shares_to_buy = int(max_shares * position_size)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += shares_to_buy
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price
                    })
        
        elif action_type < -0.5:  # Sell
            shares_to_sell = int(self.position * position_size)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.position -= shares_to_sell
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        portfolio_value = self.balance + self.position * current_price
        returns = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Reward function
        reward = returns * 100  # Scale returns
        
        # Sharpe ratio component
        if len(self.trades) > 1:
            trade_returns = self._calculate_trade_returns()
            if len(trade_returns) > 0 and np.std(trade_returns) > 0:
                sharpe = np.mean(trade_returns) / np.std(trade_returns)
                reward += sharpe * 10
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get current market state"""
        
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        # Get current and historical data
        lookback = 20
        start_idx = max(0, self.current_step - lookback)
        market_data = self.data.iloc[start_idx:self.current_step+1]
        
        if len(market_data) < 2:
            return np.zeros(self.observation_space.shape)
        
        # Calculate features
        features = []
        
        # Price features
        current_price = market_data['close'].iloc[-1]
        features.append(current_price / market_data['close'].iloc[0] - 1)  # Return
        
        # Technical indicators
        features.append(market_data['close'].pct_change().mean())  # Avg return
        features.append(market_data['close'].pct_change().std())   # Volatility
        
        # Moving averages
        for period in [5, 10, 20]:
            if len(market_data) >= period:
                ma = market_data['close'].rolling(period).mean().iloc[-1]
                features.append(current_price / ma - 1)
            else:
                features.append(0)
        
        # Volume
        features.append(market_data['volume'].iloc[-1] / market_data['volume'].mean() - 1)
        
        # Position info
        features.append(self.position * current_price / self.initial_balance)  # Position ratio
        features.append(self.balance / self.initial_balance)  # Cash ratio
        
        # Pad to match observation space
        while len(features) < self.observation_space.shape[0]:
            features.append(0)
        
        return np.array(features[:self.observation_space.shape[0]], dtype=np.float32)
    
    def _calculate_trade_returns(self) -> List[float]:
        """Calculate returns for executed trades"""
        returns = []
        
        for i in range(1, len(self.trades)):
            if self.trades[i]['action'] == 'sell' and self.trades[i-1]['action'] == 'buy':
                buy_price = self.trades[i-1]['price']
                sell_price = self.trades[i]['price']
                trade_return = (sell_price - buy_price) / buy_price
                returns.append(trade_return)
        
        return returns

class RLTradingAgent:
    """Reinforcement Learning trading agent"""
    
    def __init__(self, algorithm: str = 'PPO'):
        self.algorithm = algorithm
        self.model = None
        self.env = None
        self.logger = ComponentLogger("RLAgent", "ml")
    
    def create_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment"""
        self.env = DummyVecEnv([lambda: TradingEnvironment(data)])
        return self.env
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
             total_timesteps: int = 100000):
        """Train RL agent"""
        
        # Create environments
        train_env = self.create_environment(train_data)
        eval_env = self.create_environment(val_data)
        
        # Select algorithm
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./rl_tensorboard/"
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                train_env,
                learning_rate=3e-4,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./rl_models/',
            log_path='./rl_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make trading decision"""
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Calculate confidence based on action certainty
        if hasattr(self.model, 'policy'):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                distribution = self.model.policy.get_distribution(obs_tensor)
                entropy = distribution.entropy().item()
                
                # Lower entropy = higher confidence
                confidence = 1.0 / (1.0 + entropy)
        else:
            confidence = 0.5
        
        return action, confidence
    
    def backtest(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Backtest RL agent"""
        
        env = TradingEnvironment(test_data)
        obs = env.reset()
        
        done = False
        total_reward = 0
        
        while not done:
            action, _ = self.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        
        # Calculate metrics
        final_value = env.balance + env.position * test_data.iloc[-1]['close']
        total_return = (final_value - env.initial_balance) / env.initial_balance
        
        # Calculate Sharpe ratio
        trade_returns = env._calculate_trade_returns()
        if len(trade_returns) > 0:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(env.trades),
            'final_value': final_value,
            'total_reward': total_reward
        }
```

### 4. Ensemble Methods

```python
# src/ml/ensemble_models.py
class EnsemblePredictor:
    """Ensemble of multiple ML models"""
    
    def __init__(self, models: List[BaseMLModel]):
        self.models = models
        self.weights = None
        self.meta_learner = None
        self.logger = ComponentLogger("EnsemblePredictor", "ml")
    
    def train_stacking(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Train ensemble with stacking"""
        
        # Get predictions from base models
        train_predictions = []
        val_predictions = []
        
        for model in self.models:
            # Train on subset to avoid overfitting
            train_idx = np.random.choice(len(X_train), size=int(0.8*len(X_train)), replace=False)
            
            model.train(X_train[train_idx], y_train[train_idx])
            
            # Get predictions
            train_pred = np.array([model.predict(X_train[i:i+1]).point_estimate 
                                  for i in range(len(X_train))])
            val_pred = np.array([model.predict(X_val[i:i+1]).point_estimate 
                                for i in range(len(X_val))])
            
            train_predictions.append(train_pred)
            val_predictions.append(val_pred)
        
        # Stack predictions
        X_train_meta = np.column_stack(train_predictions)
        X_val_meta = np.column_stack(val_predictions)
        
        # Train meta-learner
        from sklearn.linear_model import Ridge
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(X_train_meta, y_train)
        
        # Evaluate
        val_score = self.meta_learner.score(X_val_meta, y_val)
        self.logger.info(f"Meta-learner validation R²: {val_score:.4f}")
    
    def train_voting(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble with weighted voting"""
        
        # Train all models
        for model in self.models:
            model.train(X_train, y_train)
        
        # Calculate weights based on validation performance
        # In production, use proper cross-validation
        self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make ensemble prediction"""
        
        predictions = []
        confidences = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred.point_estimate)
            confidences.append(pred.model_confidence)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Combine predictions
        if self.meta_learner is not None:
            # Stacking
            meta_features = predictions.reshape(1, -1)
            final_prediction = self.meta_learner.predict(meta_features)[0]
        else:
            # Weighted voting
            weights = self.weights if self.weights is not None else confidences / confidences.sum()
            final_prediction = np.average(predictions, weights=weights)
        
        # Calculate uncertainty
        prediction_std = np.std(predictions)
        confidence_interval = (
            final_prediction - 2 * prediction_std,
            final_prediction + 2 * prediction_std
        )
        
        # Ensemble confidence
        ensemble_confidence = np.mean(confidences) * (1 - prediction_std / (abs(final_prediction) + 1e-6))
        
        return ModelPrediction(
            model_id="ensemble",
            timestamp=datetime.now(),
            point_estimate=float(final_prediction),
            confidence_interval=confidence_interval,
            model_confidence=float(ensemble_confidence),
            prediction_latency_ms=0  # Would measure actual latency
        )

class OnlineLearningEnsemble:
    """Ensemble that adapts weights online"""
    
    def __init__(self, models: List[BaseMLModel], learning_rate: float = 0.01):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)
        self.learning_rate = learning_rate
        self.performance_history = defaultdict(list)
    
    def predict_and_update(self, X: np.ndarray, y_true: Optional[float] = None) -> ModelPrediction:
        """Predict and optionally update weights"""
        
        # Get predictions
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred.point_estimate)
        
        predictions = np.array(predictions)
        
        # Make ensemble prediction
        ensemble_pred = np.average(predictions, weights=self.weights)
        
        # Update weights if true value provided
        if y_true is not None:
            errors = np.abs(predictions - y_true)
            
            # Update weights using exponential weighting
            self.weights *= np.exp(-self.learning_rate * errors)
            self.weights /= self.weights.sum()
            
            # Track performance
            for i, error in enumerate(errors):
                self.performance_history[f'model_{i}'].append(error)
        
        return ModelPrediction(
            model_id="online_ensemble",
            timestamp=datetime.now(),
            point_estimate=float(ensemble_pred),
            confidence_interval=(float(ensemble_pred - np.std(predictions)), 
                               float(ensemble_pred + np.std(predictions))),
            model_confidence=float(1.0 - np.std(predictions) / (abs(ensemble_pred) + 1e-6)),
            prediction_latency_ms=0
        )
```

### 5. Model Management and MLOps

```python
# src/ml/model_management.py
class ModelRegistry:
    """Central registry for all ML models"""
    
    def __init__(self, backend_store_uri: str = "sqlite:///mlflow.db",
                 artifact_location: str = "./mlartifacts"):
        self.backend_store_uri = backend_store_uri
        self.artifact_location = artifact_location
        
        # Initialize MLflow
        mlflow.set_tracking_uri(backend_store_uri)
        
        self.models = {}
        self.active_models = {}
        self.model_metrics = defaultdict(dict)
        
        self.logger = ComponentLogger("ModelRegistry", "ml")
    
    def register_model(self, model: BaseMLModel, name: str, 
                      tags: Dict[str, str] = None):
        """Register a new model"""
        
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log parameters
            mlflow.log_params({
                'model_type': model.config.model_type.value,
                'target': model.config.target.value,
                'lookback_window': model.config.lookback_window,
                'prediction_horizon': model.config.prediction_horizon
            })
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, name)
        
        self.models[name] = model
        self.logger.info(f"Model '{name}' registered successfully")
    
    def load_model(self, name: str, version: Optional[int] = None) -> BaseMLModel:
        """Load model from registry"""
        
        if version:
            model_uri = f"models:/{name}/{version}"
        else:
            model_uri = f"models:/{name}/latest"
        
        model = mlflow.sklearn.load_model(model_uri)
        
        self.models[name] = model
        return model
    
    def deploy_model(self, name: str, version: Optional[int] = None):
        """Deploy model for production use"""
        
        model = self.load_model(name, version)
        self.active_models[name] = model
        
        self.logger.info(f"Model '{name}' deployed to production")
    
    def predict(self, model_name: str, X: np.ndarray) -> ModelPrediction:
        """Make prediction using deployed model"""
        
        if model_name not in self.active_models:
            raise ValueError(f"Model '{model_name}' not deployed")
        
        model = self.active_models[model_name]
        prediction = model.predict(X)
        
        # Track prediction
        self._track_prediction(model_name, prediction)
        
        return prediction
    
    def _track_prediction(self, model_name: str, prediction: ModelPrediction):
        """Track model predictions for monitoring"""
        
        metrics = self.model_metrics[model_name]
        
        if 'predictions' not in metrics:
            metrics['predictions'] = []
        
        metrics['predictions'].append({
            'timestamp': prediction.timestamp,
            'value': prediction.point_estimate,
            'confidence': prediction.model_confidence,
            'latency_ms': prediction.prediction_latency_ms
        })
        
        # Keep only recent predictions
        if len(metrics['predictions']) > 10000:
            metrics['predictions'] = metrics['predictions'][-10000:]
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        if model_name not in self.model_metrics:
            return {}
        
        metrics = self.model_metrics[model_name]
        predictions = metrics.get('predictions', [])
        
        if not predictions:
            return {}
        
        latencies = [p['latency_ms'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        return {
            'num_predictions': len(predictions),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'avg_confidence': np.mean(confidences),
            'last_prediction': predictions[-1]['timestamp']
        }

class ModelMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.drift_detector = DataDriftDetector()
        self.performance_tracker = {}
        self.alerts = []
        
    def monitor_predictions(self, model_name: str, X: np.ndarray, 
                          y_true: Optional[float] = None):
        """Monitor model predictions and detect issues"""
        
        # Check for data drift
        drift_score = self.drift_detector.detect_drift(X)
        
        if drift_score > 0.7:
            self.alerts.append({
                'model': model_name,
                'type': 'data_drift',
                'severity': 'high',
                'timestamp': datetime.now(),
                'details': f'Drift score: {drift_score:.2f}'
            })
        
        # Get prediction
        prediction = self.registry.predict(model_name, X)
        
        # Track performance if ground truth available
        if y_true is not None:
            error = abs(prediction.point_estimate - y_true)
            
            if model_name not in self.performance_tracker:
                self.performance_tracker[model_name] = []
            
            self.performance_tracker[model_name].append({
                'timestamp': datetime.now(),
                'error': error,
                'confidence': prediction.model_confidence
            })
            
            # Check for performance degradation
            recent_errors = [p['error'] for p in self.performance_tracker[model_name][-100:]]
            if len(recent_errors) >= 100:
                recent_avg_error = np.mean(recent_errors[-50:])
                baseline_avg_error = np.mean(recent_errors[:50])
                
                if recent_avg_error > baseline_avg_error * 1.5:
                    self.alerts.append({
                        'model': model_name,
                        'type': 'performance_degradation',
                        'severity': 'medium',
                        'timestamp': datetime.now(),
                        'details': f'Error increased by {(recent_avg_error/baseline_avg_error - 1)*100:.1f}%'
                    })
        
        return prediction

class DataDriftDetector:
    """Detect data drift in production"""
    
    def __init__(self):
        self.reference_data = None
        self.drift_threshold = 0.1
    
    def set_reference_data(self, X: np.ndarray):
        """Set reference data distribution"""
        self.reference_data = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }
    
    def detect_drift(self, X: np.ndarray) -> float:
        """Detect drift using statistical tests"""
        
        if self.reference_data is None:
            return 0.0
        
        # Calculate current statistics
        current_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }
        
        # Calculate drift scores
        drift_scores = []
        
        # Mean drift
        mean_drift = np.abs(current_stats['mean'] - self.reference_data['mean']) / (self.reference_data['std'] + 1e-6)
        drift_scores.extend(mean_drift)
        
        # Variance drift
        var_drift = np.abs(current_stats['std'] - self.reference_data['std']) / (self.reference_data['std'] + 1e-6)
        drift_scores.extend(var_drift)
        
        # Calculate overall drift score
        drift_score = np.mean(drift_scores)
        
        return float(drift_score)
```

### 6. Testing Framework

```python
# tests/unit/test_step14_ml_models.py
class TestMLFramework:
    """Test ML framework components"""
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation"""
        config = ModelConfig(
            model_type=ModelType.LSTM,
            target=PredictionTarget.PRICE,
            input_features=['close', 'volume', 'rsi'],
            hidden_layers=[128, 64],
            lookback_window=30,
            prediction_horizon=1
        )
        
        model = LSTMPricePredictor(config)
        model.build_model()
        
        assert model.model is not None
        assert isinstance(model.model, nn.Module)
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.uniform(1000, 5000, 100),
            'high': 101 + np.cumsum(np.random.randn(100) * 0.01),
            'low': 99 + np.cumsum(np.random.randn(100) * 0.01)
        }, index=dates)
        
        config = ModelConfig(
            model_type=ModelType.LSTM,
            target=PredictionTarget.RETURNS,
            input_features=[],
            hidden_layers=[64],
            use_technical_indicators=True
        )
        
        model = LSTMPricePredictor(config)
        features = model._engineer_features(data)
        
        # Check features created
        assert 'returns' in features.columns
        assert 'rsi' in features.columns
        assert 'volatility_20' in features.columns
        assert len(features) < len(data)  # Some rows dropped due to NaN
    
    def test_sequence_creation(self):
        """Test sequence creation for time series"""
        model = LSTMPricePredictor(ModelConfig(
            model_type=ModelType.LSTM,
            target=PredictionTarget.PRICE,
            input_features=['close', 'volume'],
            hidden_layers=[64],
            lookback_window=10,
            prediction_horizon=1
        ))
        
        # Create test data
        data = pd.DataFrame({
            'close': np.arange(100),
            'volume': np.arange(100) * 10
        })
        
        X, y = model._create_sequences(data)
        
        assert X.shape[0] == len(data) - model.config.lookback_window - model.config.prediction_horizon + 1
        assert X.shape[1] == model.config.lookback_window
        assert X.shape[2] == 2  # Two features
        assert len(y) == X.shape[0]

class TestTransformerModel:
    """Test Transformer implementation"""
    
    def test_transformer_architecture(self):
        """Test Transformer model architecture"""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            target=PredictionTarget.DIRECTION,
            input_features=['close', 'volume', 'rsi'],
            hidden_layers=[512],
            lookback_window=60
        )
        
        model = TransformerPredictor(config)
        model.build_model()
        
        # Test forward pass
        batch_size = 16
        seq_len = config.lookback_window
        n_features = len(config.input_features)
        
        test_input = torch.randn(batch_size, seq_len, n_features)
        output = model.model(test_input)
        
        assert output.shape == (batch_size, 1)
    
    def test_attention_extraction(self):
        """Test attention weight extraction"""
        # Create simple transformer
        model = TransformerModel(n_features=5, d_model=64, nhead=4, num_layers=2)
        analyzer = AttentionAnalyzer(model)
        
        # Test input
        X = torch.randn(1, 10, 5)  # batch=1, seq=10, features=5
        
        attention_weights = analyzer.extract_attention_weights(X)
        
        assert len(attention_weights) == 2  # Two layers
        for layer, weights in attention_weights.items():
            assert weights.shape[1] == 4  # Number of heads

class TestReinforcementLearning:
    """Test RL components"""
    
    def test_trading_environment(self):
        """Test trading environment"""
        # Create test data
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        env = TradingEnvironment(data)
        
        # Test reset
        obs = env.reset()
        assert obs.shape == env.observation_space.shape
        
        # Test step
        action = np.array([0.8, 0.5])  # Buy with 50% position
        obs, reward, done, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    
    def test_rl_agent_training(self):
        """Test RL agent training"""
        # Create simple data
        train_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.uniform(1000, 5000, 1000)
        })
        
        val_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(200) * 0.01),
            'volume': np.random.uniform(1000, 5000, 200)
        })
        
        agent = RLTradingAgent(algorithm='PPO')
        
        # Train for small number of steps
        agent.train(train_data, val_data, total_timesteps=1000)
        
        assert agent.model is not None
        
        # Test prediction
        env = TradingEnvironment(val_data)
        obs = env.reset()
        action, confidence = agent.predict(obs)
        
        assert action.shape == (2,)  # Action and size
        assert 0 <= confidence <= 1

class TestEnsembleMethods:
    """Test ensemble models"""
    
    def test_ensemble_voting(self):
        """Test ensemble with voting"""
        # Create mock models
        models = []
        for i in range(3):
            config = ModelConfig(
                model_type=ModelType.LSTM,
                target=PredictionTarget.PRICE,
                input_features=['close'],
                hidden_layers=[32]
            )
            models.append(MockMLModel(config, base_prediction=100 + i))
        
        ensemble = EnsemblePredictor(models)
        ensemble.train_voting(np.random.randn(100, 10, 1), np.random.randn(100))
        
        # Test prediction
        X = np.random.randn(1, 10, 1)
        prediction = ensemble.predict(X)
        
        # Should be average of base predictions
        assert 100 <= prediction.point_estimate <= 102
        assert prediction.model_confidence > 0
    
    def test_online_learning_ensemble(self):
        """Test online learning ensemble"""
        models = [MockMLModel(ModelConfig(
            model_type=ModelType.LSTM,
            target=PredictionTarget.PRICE,
            input_features=['close'],
            hidden_layers=[32]
        ), base_prediction=100 + i) for i in range(3)]
        
        ensemble = OnlineLearningEnsemble(models)
        
        # Make predictions and update
        X = np.random.randn(1, 10, 1)
        
        # Initial prediction
        pred1 = ensemble.predict_and_update(X)
        initial_weights = ensemble.weights.copy()
        
        # Update with true value
        y_true = 101.5
        pred2 = ensemble.predict_and_update(X, y_true)
        
        # Weights should have changed
        assert not np.array_equal(initial_weights, ensemble.weights)
        
        # Model closest to true value should have higher weight
        assert ensemble.weights[1] > ensemble.weights[0]
        assert ensemble.weights[1] > ensemble.weights[2]
```

### 7. Integration Tests

```python
# tests/integration/test_step14_ml_integration.py
async def test_complete_ml_pipeline():
    """Test complete ML pipeline from data to prediction"""
    
    # Load historical data
    data = load_test_market_data()
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # Create and train models
    models = []
    
    # LSTM model
    lstm_config = ModelConfig(
        model_type=ModelType.LSTM,
        target=PredictionTarget.RETURNS,
        input_features=['close', 'volume', 'rsi', 'ma_ratio_20'],
        hidden_layers=[128, 64],
        epochs=50
    )
    lstm_model = LSTMPricePredictor(lstm_config)
    
    # Preprocess data
    X_train, y_train = lstm_model.preprocess_data(train_data)
    X_val, y_val = lstm_model.preprocess_data(val_data)
    
    # Train
    lstm_model.train(X_train, y_train, X_val, y_val)
    models.append(lstm_model)
    
    # Create ensemble
    ensemble = EnsemblePredictor(models)
    ensemble.train_voting(X_train, y_train)
    
    # Test predictions
    X_test, y_test = lstm_model.preprocess_data(test_data)
    
    predictions = []
    for i in range(len(X_test)):
        pred = ensemble.predict(X_test[i:i+1])
        predictions.append(pred.point_estimate)
    
    # Calculate metrics
    predictions = np.array(predictions)
    mse = np.mean((predictions - y_test)**2)
    mae = np.mean(np.abs(predictions - y_test))
    
    assert mse < 0.01  # Reasonable error
    assert mae < 0.005

async def test_model_deployment_workflow():
    """Test model deployment and monitoring"""
    
    # Create registry
    registry = ModelRegistry()
    
    # Train model
    config = ModelConfig(
        model_type=ModelType.LSTM,
        target=PredictionTarget.DIRECTION,
        input_features=['close', 'volume'],
        hidden_layers=[64]
    )
    
    model = LSTMPricePredictor(config)
    
    # Mock training
    model.is_trained = True
    model.model = MockNeuralNetwork()
    
    # Register model
    registry.register_model(model, "price_predictor_v1", tags={'environment': 'test'})
    
    # Deploy model
    registry.deploy_model("price_predictor_v1")
    
    # Create monitor
    monitor = ModelMonitor(registry)
    
    # Make predictions with monitoring
    test_features = np.random.randn(1, 30, 2)
    
    for i in range(100):
        # Add some drift
        if i > 50:
            test_features += np.random.randn(1, 30, 2) * 0.1
        
        prediction = monitor.monitor_predictions(
            "price_predictor_v1", 
            test_features,
            y_true=0.5 if i < 50 else 0.7
        )
        
        assert prediction is not None
    
    # Check for alerts
    assert len(monitor.alerts) > 0
    
    # Should detect drift
    drift_alerts = [a for a in monitor.alerts if a['type'] == 'data_drift']
    assert len(drift_alerts) > 0
```

### 8. System Tests

```python
# tests/system/test_step14_production_ml.py
async def test_ml_system_performance():
    """Test ML system under production load"""
    
    # Create multiple models
    models = []
    
    for model_type in [ModelType.LSTM, ModelType.GRU, ModelType.TRANSFORMER]:
        config = ModelConfig(
            model_type=model_type,
            target=PredictionTarget.DIRECTION,
            input_features=['close', 'volume', 'rsi'],
            hidden_layers=[64] if model_type != ModelType.TRANSFORMER else [256]
        )
        
        model = create_model(config)
        models.append(model)
    
    # Create ensemble
    ensemble = EnsemblePredictor(models)
    
    # Performance test
    n_predictions = 10000
    batch_size = 32
    
    start_time = time.time()
    latencies = []
    
    for i in range(0, n_predictions, batch_size):
        batch_data = np.random.randn(batch_size, 60, 3)
        
        batch_start = time.time()
        predictions = [ensemble.predict(batch_data[j:j+1]) for j in range(batch_size)]
        batch_latency = (time.time() - batch_start) / batch_size * 1000
        
        latencies.extend([batch_latency] * batch_size)
    
    total_time = time.time() - start_time
    
    # Performance assertions
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = n_predictions / total_time
    
    assert avg_latency < 10  # < 10ms average
    assert p95_latency < 20  # < 20ms p95
    assert throughput > 100  # > 100 predictions/second

async def test_ml_fault_tolerance():
    """Test ML system resilience"""
    
    registry = ModelRegistry()
    
    # Deploy multiple model versions
    for version in range(3):
        model = create_test_model(f"model_v{version}")
        registry.register_model(model, f"predictor", tags={'version': str(version)})
    
    # Deploy latest
    registry.deploy_model("predictor")
    
    # Simulate model failure
    test_data = np.random.randn(1, 60, 5)
    
    # Force error in active model
    registry.active_models["predictor"] = FailingModel()
    
    try:
        prediction = registry.predict("predictor", test_data)
    except:
        # Should fallback to previous version
        registry.deploy_model("predictor", version=2)
        prediction = registry.predict("predictor", test_data)
    
    assert prediction is not None
    assert prediction.model_confidence > 0
```

## ✅ Validation Checklist

### Model Development
- [ ] LSTM models training successfully
- [ ] Transformer architecture working
- [ ] Feature engineering comprehensive
- [ ] Data preprocessing correct
- [ ] Model serialization functional

### Advanced Models
- [ ] Reinforcement learning agents training
- [ ] Trading environment realistic
- [ ] Ensemble methods working
- [ ] Online learning adapting
- [ ] Attention mechanisms interpretable

### MLOps Infrastructure
- [ ] Model registry operational
- [ ] Experiment tracking working
- [ ] Model versioning functional
- [ ] Deployment pipeline smooth
- [ ] Monitoring active

### Performance
- [ ] Prediction latency < 10ms
- [ ] Throughput > 100 pred/sec
- [ ] Model accuracy acceptable
- [ ] Memory usage controlled
- [ ] GPU utilization efficient

### Production Readiness
- [ ] Drift detection working
- [ ] Performance monitoring active
- [ ] Alert system functional
- [ ] Fallback mechanisms ready
- [ ] A/B testing supported

## 📊 Performance Benchmarks

### Model Performance
- Training time: < 1 hour for 1M samples
- Prediction latency: < 10ms average
- Model accuracy: > 60% directional
- Sharpe ratio improvement: > 0.5

### System Performance
- Throughput: 1000+ predictions/second
- Memory per model: < 500MB
- GPU memory: < 4GB per model
- CPU usage: < 80% sustained

### MLOps Metrics
- Model deployment: < 5 minutes
- Experiment tracking: Real-time
- Model rollback: < 1 minute
- Monitoring lag: < 1 second

## 🐛 Common Issues

1. **Overfitting**
   - Use dropout and regularization
   - Implement early stopping
   - Cross-validate properly
   - Monitor validation metrics

2. **Data Leakage**
   - Careful feature engineering
   - Proper train/test splits
   - No future information
   - Walk-forward validation

3. **Model Drift**
   - Continuous monitoring
   - Regular retraining
   - Feature importance tracking
   - Data quality checks

## 🎯 Success Criteria

Step 14 is complete when:
1. ✅ Multiple ML models implemented
2. ✅ Ensemble methods working
3. ✅ RL agents training successfully
4. ✅ MLOps pipeline operational
5. ✅ Production monitoring active

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 15: Advanced Alternative Data](step-15-advanced-alt-data.md)

## 📚 Additional Resources

- [Financial ML Best Practices](../references/financial-ml-practices.md)
- [Deep Learning Architecture Guide](../references/dl-architectures.md)
- [Reinforcement Learning Tutorial](../references/rl-tutorial.md)
- [MLOps Maturity Model](../references/mlops-maturity.md)