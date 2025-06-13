# Hybrid Multi-Process Implementation Plan for ADMF-PC

## Executive Summary

This plan combines the architectural excellence of Document 2 (Protocol + Composition) with the ML-focused scaling strategy of Document 3, while incorporating the practical implementation details from Document 1.

## Architecture Principles

### 1. Protocol + Composition Foundation (From Doc 2)
```python
@runtime_checkable
class WorkerProtocol(Protocol):
    """Base protocol for all worker processes"""
    @property
    def worker_id(self) -> str: ...
    def setup(self) -> None: ...
    def process_messages(self) -> None: ...
    def cleanup(self) -> None: ...

@runtime_checkable
class MLWorkerProtocol(WorkerProtocol):
    """Protocol for ML strategy workers"""
    def load_model(self, model_path: str) -> None: ...
    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, float]: ...
    def get_model_info(self) -> Dict[str, Any]: ...

@runtime_checkable
class BatchWorkerProtocol(WorkerProtocol):
    """Protocol for batch portfolio processing"""
    def add_portfolio(self, portfolio_config: Dict[str, Any]) -> str: ...
    def process_signal_batch(self, signals: List[Event]) -> List[Event]: ...
    def get_batch_metrics(self) -> Dict[str, Any]: ...
```

### 2. ML-Aware Process Architecture (From Doc 3)
```
Producer Process (Core 1)     Strategy Workers (Cores 2-4)     Portfolio Batches (Cores 5-7)
├── Data Streamers           ├── ML Model A (LSTM)            ├── Batch 1 (500 portfolios)
├── Shared Features          ├── ML Model B (XGBoost)         ├── Batch 2 (500 portfolios)  
└── Feature Cache            └── Traditional Strategies       └── Batch 3 (500 portfolios)
```

## Phase 1: Foundation with ML Focus (Weeks 1-2)

### 1.1 ML-Aware IPC System
```python
# src/core/multiprocess/ml_ipc.py
from typing import Dict, Any, Optional
import numpy as np
import pickle
import msgpack

class MLEventSerializer:
    """Specialized serializer for ML model events"""
    
    @staticmethod
    def serialize_features(features: Dict[str, np.ndarray]) -> bytes:
        """Efficiently serialize numpy feature arrays"""
        # Use compression for large feature matrices
        return msgpack.packb({
            'features': {k: v.tobytes() for k, v in features.items()},
            'shapes': {k: v.shape for k, v in features.items()},
            'dtypes': {k: str(v.dtype) for k, v in features.items()}
        }, use_bin_type=True)
    
    @staticmethod
    def deserialize_features(data: bytes) -> Dict[str, np.ndarray]:
        """Restore numpy arrays from serialized data"""
        unpacked = msgpack.unpackb(data, raw=False)
        features = {}
        for key in unpacked['features']:
            features[key] = np.frombuffer(
                unpacked['features'][key],
                dtype=unpacked['dtypes'][key]
            ).reshape(unpacked['shapes'][key])
        return features

class FeatureCache:
    """Shared feature cache for ML models"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.features: Dict[str, Dict[str, np.ndarray]] = {}
        self.timestamps: Dict[str, float] = {}
    
    def store_features(self, symbol: str, timestamp: float, 
                      features: Dict[str, np.ndarray]) -> None:
        """Store computed features for symbol"""
        key = f"{symbol}_{timestamp}"
        self.features[key] = features
        self.timestamps[key] = timestamp
        
        # Cleanup old entries
        if len(self.features) > self.cache_size:
            oldest_key = min(self.timestamps.keys(), 
                           key=lambda k: self.timestamps[k])
            del self.features[oldest_key]
            del self.timestamps[oldest_key]
    
    def get_features(self, symbol: str, timestamp: float) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve cached features"""
        key = f"{symbol}_{timestamp}"
        return self.features.get(key)
```

### 1.2 Process-Safe Event Tracing
```python
# src/core/events/distributed_tracing.py
from typing import Dict, Any, Optional
import uuid
from dataclasses import dataclass

@dataclass
class MLTraceContext:
    """Enhanced trace context for ML operations"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    feature_hash: Optional[str] = None
    prediction_metadata: Optional[Dict[str, Any]] = None

class MLDistributedTracer:
    """Distributed tracer optimized for ML workflows"""
    
    def trace_model_prediction(self, model_id: str, features: Dict[str, np.ndarray],
                              prediction: Dict[str, float]) -> MLTraceContext:
        """Create trace span for ML model prediction"""
        return MLTraceContext(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            model_info={'model_id': model_id, 'feature_count': len(features)},
            feature_hash=self._hash_features(features),
            prediction_metadata={'prediction_keys': list(prediction.keys())}
        )
    
    def _hash_features(self, features: Dict[str, np.ndarray]) -> str:
        """Create hash of feature data for tracing"""
        import hashlib
        combined = b""
        for key in sorted(features.keys()):
            combined += features[key].tobytes()
        return hashlib.md5(combined).hexdigest()[:8]
```

## Phase 2: ML Strategy Workers (Weeks 3-4)

### 2.1 Composable ML Worker
```python
# src/core/multiprocess/workers/ml_strategy_worker.py
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    model_id: str
    model_type: str  # 'pytorch', 'xgboost', 'tensorflow'
    model_path: str
    feature_names: List[str]
    requires_gpu: bool = False
    batch_size: int = 32
    inference_timeout: float = 1.0

class MLModelManager:
    """Manages ML model lifecycle within worker"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, MLModelConfig] = {}
        
    def load_model(self, config: MLModelConfig) -> None:
        """Load ML model based on type"""
        if config.model_type == 'pytorch':
            model = torch.load(config.model_path)
            if config.requires_gpu and torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            self.models[config.model_id] = model
            
        elif config.model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(config.model_path)
            self.models[config.model_id] = model
            
        # Add other model types as needed
        self.model_configs[config.model_id] = config
    
    def predict(self, model_id: str, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Generate predictions using specified model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model = self.models[model_id]
        config = self.model_configs[model_id]
        
        # Prepare features in correct order
        feature_array = np.column_stack([
            features[name] for name in config.feature_names
        ])
        
        if config.model_type == 'pytorch':
            with torch.no_grad():
                tensor = torch.FloatTensor(feature_array)
                if config.requires_gpu:
                    tensor = tensor.cuda()
                output = model(tensor).cpu().numpy()
                return {'prediction': float(output[0])}
                
        elif config.model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(feature_array)
            output = model.predict(dmatrix)
            return {'prediction': float(output[0])}

class MLStrategyWorker:
    """ML Strategy worker using composition"""
    
    def __init__(self, worker_id: str, model_configs: List[MLModelConfig],
                 ipc_config: IPCConfig):
        # Compose base worker capabilities
        self.base_worker = BaseWorker(WorkerConfig(
            worker_id=worker_id,
            worker_type='ml_strategy',
            subscriptions=['features.*', 'market_data.*']
        ))
        
        # Add ML-specific components
        self.model_manager = MLModelManager()
        self.feature_cache = FeatureCache()
        self.ml_tracer = MLDistributedTracer()
        
        # Load models
        for config in model_configs:
            self.model_manager.load_model(config)
    
    def setup(self) -> None:
        """Setup ML worker"""
        self.base_worker.setup()
        
        # Subscribe to feature events
        self.base_worker.event_bus.subscribe(
            'FEATURES',
            self._handle_features
        )
    
    def _handle_features(self, event: Event) -> None:
        """Process feature event and generate predictions"""
        features = MLEventSerializer.deserialize_features(
            event.payload['feature_data']
        )
        
        # Generate predictions from all models
        signals = []
        for model_id in self.model_manager.models:
            try:
                # Trace prediction
                trace_ctx = self.ml_tracer.trace_model_prediction(
                    model_id, features, {}
                )
                
                # Generate prediction
                prediction = self.model_manager.predict(model_id, features)
                
                # Convert to signal if threshold met
                if abs(prediction['prediction']) > 0.6:  # Configurable threshold
                    signal_event = Event(
                        event_type='SIGNAL',
                        source_id=f"ml_model_{model_id}",
                        payload={
                            'symbol': event.payload['symbol'],
                            'direction': 'BUY' if prediction['prediction'] > 0 else 'SELL',
                            'strength': abs(prediction['prediction']),
                            'model_id': model_id,
                            'trace_context': trace_ctx.__dict__
                        }
                    )
                    signals.append(signal_event)
                    
            except Exception as e:
                logger.error(f"Model {model_id} prediction failed: {e}")
        
        # Publish signals
        for signal in signals:
            self.base_worker.event_bus.publish(signal)
```

## Phase 3: Batch Portfolio Processing (Weeks 5-6)

### 3.1 Vectorized Portfolio Batch Worker
```python
# src/core/multiprocess/workers/portfolio_batch_worker.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class PortfolioBatchState:
    """Vectorized state for portfolio batch"""
    portfolio_ids: List[str]
    portfolio_values: np.ndarray  # [n_portfolios]
    position_counts: np.ndarray   # [n_portfolios]
    risk_scores: np.ndarray       # [n_portfolios]
    strategy_assignments: np.ndarray  # [n_portfolios, n_strategies] boolean mask
    
    def __post_init__(self):
        self.batch_size = len(self.portfolio_ids)

class VectorizedRiskManager:
    """Vectorized risk calculations for portfolio batches"""
    
    def __init__(self, risk_config: Dict[str, Any]):
        self.max_position_size = risk_config.get('max_position_size', 0.05)
        self.max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.15)
        self.max_positions = risk_config.get('max_positions', 10)
    
    def calculate_risk_batch(self, batch_state: PortfolioBatchState,
                            signal: Dict[str, Any]) -> np.ndarray:
        """Calculate risk scores for entire batch"""
        # Position count risk
        position_risk = batch_state.position_counts / self.max_positions
        
        # Portfolio utilization risk  
        utilization_risk = batch_state.risk_scores / self.max_portfolio_risk
        
        # Signal-specific risk (volatility, correlation, etc.)
        signal_risk = np.full(batch_state.batch_size, 
                             signal.get('risk_factor', 0.1))
        
        # Combined risk score
        total_risk = (position_risk * 0.4 + 
                     utilization_risk * 0.4 + 
                     signal_risk * 0.2)
        
        return total_risk
    
    def filter_allowed_portfolios(self, batch_state: PortfolioBatchState,
                                 risk_scores: np.ndarray) -> np.ndarray:
        """Return mask of portfolios allowed to trade"""
        return risk_scores < 0.95  # Risk threshold

class VectorizedPositionSizer:
    """Vectorized position sizing for portfolio batches"""
    
    def calculate_position_sizes(self, batch_state: PortfolioBatchState,
                               signal: Dict[str, Any],
                               allowed_mask: np.ndarray) -> np.ndarray:
        """Calculate position sizes for allowed portfolios"""
        # Base position size as percentage of portfolio value
        base_sizes = batch_state.portfolio_values * self.max_position_size
        
        # Adjust by signal strength
        signal_strength = signal.get('strength', 1.0)
        adjusted_sizes = base_sizes * signal_strength
        
        # Apply allowed mask
        final_sizes = np.where(allowed_mask, adjusted_sizes, 0.0)
        
        return final_sizes

class PortfolioBatchWorker:
    """Vectorized portfolio batch processing"""
    
    def __init__(self, batch_id: int, portfolio_configs: List[Dict[str, Any]],
                 ipc_config: IPCConfig):
        
        # Initialize batch state
        self.batch_state = self._init_batch_state(portfolio_configs)
        
        # Compose worker components
        self.base_worker = BaseWorker(WorkerConfig(
            worker_id=f"portfolio_batch_{batch_id}",
            worker_type='portfolio_batch',
            subscriptions=self._build_subscriptions()
        ))
        
        # Vectorized components
        self.risk_manager = VectorizedRiskManager(
            portfolio_configs[0].get('risk_config', {})
        )
        self.position_sizer = VectorizedPositionSizer()
        
    def setup(self) -> None:
        """Setup batch worker"""
        self.base_worker.setup()
        
        # Subscribe to signals with batch processing
        self.base_worker.event_bus.subscribe(
            'SIGNAL',
            self._handle_signal_batch
        )
    
    def _handle_signal_batch(self, event: Event) -> None:
        """Process signal across entire portfolio batch"""
        signal = event.payload
        strategy_id = signal.get('model_id') or signal.get('strategy_id')
        
        # Find portfolios interested in this strategy
        strategy_mask = self._get_strategy_mask(strategy_id)
        interested_portfolios = np.where(strategy_mask)[0]
        
        if len(interested_portfolios) == 0:
            return
        
        # Vectorized risk assessment
        risk_scores = self.risk_manager.calculate_risk_batch(
            self.batch_state, signal
        )
        allowed_mask = self.risk_manager.filter_allowed_portfolios(
            self.batch_state, risk_scores
        )
        
        # Apply strategy interest mask
        final_mask = strategy_mask & allowed_mask
        final_portfolios = np.where(final_mask)[0]
        
        if len(final_portfolios) == 0:
            return
        
        # Vectorized position sizing
        position_sizes = self.position_sizer.calculate_position_sizes(
            self.batch_state, signal, final_mask
        )
        
        # Generate orders for portfolios with non-zero sizes
        self._generate_orders_batch(final_portfolios, signal, position_sizes[final_portfolios])
    
    def _generate_orders_batch(self, portfolio_indices: np.ndarray,
                              signal: Dict[str, Any], sizes: np.ndarray) -> None:
        """Generate orders for multiple portfolios efficiently"""
        
        # Batch order creation
        orders = []
        for idx, size in zip(portfolio_indices, sizes):
            if size > 0:
                portfolio_id = self.batch_state.portfolio_ids[idx]
                
                order = {
                    'symbol': signal['symbol'],
                    'side': signal['direction'],
                    'quantity': int(size / signal.get('price', 100)),
                    'portfolio_id': portfolio_id,
                    'order_type': 'MARKET'
                }
                orders.append(order)
        
        # Batch publish orders
        if orders:
            batch_event = Event(
                event_type='ORDER_BATCH',
                source_id=self.base_worker.worker_id,
                payload={'orders': orders}
            )
            self.base_worker.event_bus.publish(batch_event)
    
    def _init_batch_state(self, configs: List[Dict[str, Any]]) -> PortfolioBatchState:
        """Initialize vectorized batch state"""
        n_portfolios = len(configs)
        
        return PortfolioBatchState(
            portfolio_ids=[f"portfolio_{i}" for i in range(n_portfolios)],
            portfolio_values=np.array([
                config.get('initial_capital', 100000) for config in configs
            ]),
            position_counts=np.zeros(n_portfolios),
            risk_scores=np.zeros(n_portfolios),
            strategy_assignments=self._build_strategy_assignments(configs)
        )
```

## Phase 4: Integration & Scaling (Weeks 7-8)

### 4.1 Multi-Process Topology Runner
```python
# src/core/coordinator/ml_multiprocess_runner.py
class MLMultiProcessTopologyRunner(TopologyRunner):
    """ML-optimized multi-process topology runner"""
    
    def _determine_process_allocation(self, config: Dict[str, Any]) -> Dict[str, int]:
        """Intelligently allocate processes based on workload"""
        total_cores = config.get('multiprocess', {}).get('max_cores', 8)
        
        # ML model requirements
        ml_models = config.get('ml_models', [])
        gpu_models = [m for m in ml_models if m.get('requires_gpu', False)]
        cpu_models = [m for m in ml_models if not m.get('requires_gpu', False)]
        
        # Portfolio batch sizing
        portfolios = config.get('portfolios', [])
        batch_size = min(500, max(100, len(portfolios) // (total_cores - 3)))
        
        allocation = {
            'producer': 1,  # Data + shared features
            'ml_gpu_workers': min(len(gpu_models), 2),  # GPU-bound models
            'ml_cpu_workers': min(len(cpu_models), 2),  # CPU-bound models
            'portfolio_batches': max(1, (total_cores - 3) // 2),
            'execution': 1
        }
        
        # Adjust if over-allocated
        total_allocated = sum(allocation.values())
        if total_allocated > total_cores:
            # Reduce portfolio batches first
            allocation['portfolio_batches'] = max(1, total_cores - 4)
        
        return allocation
    
    def _start_ml_workers(self, config: Dict[str, Any],
                         allocation: Dict[str, int]) -> List[str]:
        """Start ML strategy workers with proper resource allocation"""
        workers = []
        ml_models = config.get('ml_models', [])
        
        # Group models by resource requirements
        gpu_models = [m for m in ml_models if m.get('requires_gpu', False)]
        cpu_models = [m for m in ml_models if not m.get('requires_gpu', False)]
        
        # Start GPU workers
        for i in range(allocation['ml_gpu_workers']):
            models_for_worker = gpu_models[i::allocation['ml_gpu_workers']]
            if models_for_worker:
                worker_config = {
                    'worker_id': f'ml_gpu_worker_{i}',
                    'worker_type': 'ml_strategy',
                    'model_configs': models_for_worker,
                    'gpu_id': i % torch.cuda.device_count() if torch.cuda.is_available() else None
                }
                worker_id = self.process_manager.start_worker(worker_config)
                workers.append(worker_id)
        
        # Start CPU workers
        for i in range(allocation['ml_cpu_workers']):
            models_for_worker = cpu_models[i::allocation['ml_cpu_workers']]
            if models_for_worker:
                worker_config = {
                    'worker_id': f'ml_cpu_worker_{i}',
                    'worker_type': 'ml_strategy', 
                    'model_configs': models_for_worker
                }
                worker_id = self.process_manager.start_worker(worker_config)
                workers.append(worker_id)
        
        return workers
```

## Performance Characteristics

### **Expected Performance (8-Core Machine)**

| Scale | Approach | Cores Used | Throughput | Memory |
|-------|----------|------------|------------|--------|
| 10 portfolios | Individual processes | 8 | 100% | Low |
| 100 portfolios | Individual processes | 8 (limited) | 60% | Medium |
| 1,000 portfolios | Batch processing | 8 | 95% | Medium |
| 10,000 portfolios | Optimized batching | 8 | 90% | High |

### **Scaling Benefits**
- **ML Model Isolation**: Each model type runs in dedicated process
- **Shared Feature Computation**: 90% reduction in redundant calculations
- **Vectorized Portfolio Processing**: 10-100x faster than individual processing
- **Resource Optimization**: Near 100% CPU utilization across all cores

## Implementation Priority

### **Phase 1 (Immediate Value)**
Focus on ML strategy workers with basic portfolio batching. This provides immediate parallelization benefits while maintaining your Protocol + Composition architecture.

### **Phase 2 (Scale Optimization)** 
Add advanced vectorized batch processing for handling thousands of portfolios efficiently.

### **Phase 3 (Production Hardening)**
Add comprehensive monitoring, error handling, and process management for production deployment.

This hybrid approach gives you the architectural cleanliness of Document 2, the ML-focused scaling of Document 3, and the practical implementation guidance of Document 1.
