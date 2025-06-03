# ADMF-PC Container Extension Protocol

## Overview

This guide explains how to extend ADMF-PC with new container types and leverage the powerful signal generation and analysis workflow. The modular architecture enables sophisticated research workflows where signal generation, analysis, and processing can be developed and tested independently before deployment.

## Signal Generation for Analysis Pattern

One of ADMF-PC's key architectural features is the ability to separate signal generation from execution. This enables a powerful research workflow where you can:

1. Generate raw signals without execution overhead
2. Analyze and transform signals offline
3. Test multiple processing hypotheses in parallel
4. Deploy only the best approaches to live trading

### Workflow Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                  SIGNAL RESEARCH WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Signal Generation                                     │
│  ┌─────────────────────────┐                                   │
│  │ Market Data             │                                   │
│  │     ↓                   │                                   │
│  │ Indicators              │                                   │
│  │     ↓                   │                                   │
│  │ Strategies              │                                   │
│  │     ↓                   │                                   │
│  │ Signal Capture ────────────→ Raw Signals Database          │
│  │ (No Execution)          │                                   │
│  └─────────────────────────┘                                   │
│                                                                 │
│  Phase 2: Offline Analysis                                      │
│  ┌─────────────────────────┐                                   │
│  │ Load Raw Signals        │                                   │
│  │     ↓                   │                                   │
│  │ Apply Transformations   │                                   │
│  │ • Filter by confidence  │                                   │
│  │ • Regime adjustments    │                                   │
│  │ • ML scoring           │                                   │
│  │ • Rate limiting        │                                   │
│  │     ↓                   │                                   │
│  │ Analyze & Visualize    │                                   │
│  │ • Signal distribution  │                                   │
│  │ • Quality metrics      │                                   │
│  │ • Pattern detection    │                                   │
│  └─────────────────────────┘                                   │
│                                                                 │
│  Phase 3: Implementation & Testing                              │
│  ┌─────────────────────────┐                                   │
│  │ Best Transformation     │                                   │
│  │     ↓                   │                                   │
│  │ Signal Processor        │                                   │
│  │ Container              │                                   │
│  │     ↓                   │                                   │
│  │ Full Backtest          │                                   │
│  │ with Execution         │                                   │
│  └─────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Signal Generation Configuration

```yaml
# config/signal_generation_analysis.yaml
workflow_type: "signal_generation"  # No execution, just signals

containers:
  - type: "data"
    name: "market_data"
    
  - type: "indicator"
    name: "indicators"
    
  - type: "strategy"
    name: "momentum_strategy"
    parameters:
      fast_period: 10
      slow_period: 30
    
  # Signal capture container instead of execution
  - type: "signal_logger"
    name: "signal_capture"
    output_path: "signals/momentum_raw.parquet"
    capture_fields: ["timestamp", "symbol", "action", "strength", 
                     "confidence", "regime_context", "indicators"]

# No risk/execution containers needed!
```

### Phase 2: Offline Signal Analysis

```python
# analyze_signals.py
import pandas as pd
import numpy as np

# Load raw signals
signals = pd.read_parquet("signals/momentum_raw.parquet")

# Apply hypothetical transformations
def hypothetical_signal_processor(signal_df):
    """Test signal processing ideas without running backtest"""
    
    # Hypothesis 1: Filter low confidence signals
    filtered = signal_df[signal_df.confidence > 0.7].copy()
    
    # Hypothesis 2: Boost signals in trending regimes
    filtered.loc[filtered.regime_context == 'TRENDING', 'strength'] *= 1.5
    
    # Hypothesis 3: Rate limit signals
    filtered = filtered.groupby('symbol').resample('1H').first()
    
    # Hypothesis 4: Apply ML-based quality scoring
    filtered['ml_score'] = ml_model.predict(filtered[feature_cols])
    filtered = filtered[filtered.ml_score > 0.6]
    
    return filtered

# Test multiple processing hypotheses
processed_v1 = hypothetical_signal_processor(signals)
processed_v2 = another_processor(signals)

# Analyze and visualize
plot_signal_distribution(signals, processed_v1, processed_v2)
calculate_signal_metrics(processed_v1)
```

### Phase 3: Deploy Best Approach

```yaml
# config/backtest_with_signal_processor.yaml
workflow_type: "full_backtest"

containers:
  # ... same data/indicator/strategy ...
  
  # Implement the processor you designed
  - type: "signal_processor"
    name: "optimized_processor"
    min_confidence: 0.7  # From analysis
    regime_boost:
      TRENDING: 1.5      # From analysis
    rate_limit: "1H"     # From analysis
    ml_model: "models/signal_quality.pkl"
    
  - type: "risk"
  - type: "execution"
```

## Advanced Signal Processing Pipeline

### Multi-Stage Signal Processing with Regime Integration

```
┌─────────────────────────────────────────────────────────────────┐
│              ADVANCED SIGNAL PROCESSING PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Signals                                                    │
│      ↓                                                          │
│  ┌─────────────────┐                                           │
│  │ Regime Filter   │ ← Regime signals processed in chain!      │
│  │                 │                                           │
│  │ if regime==BULL │                                           │
│  │   boost *= 1.5  │                                           │
│  │ elif BEAR       │                                           │
│  │   reduce *= 0.7 │                                           │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │ ML Quality      │                                           │
│  │ Scorer          │                                           │
│  │                 │                                           │
│  │ score > 0.8?    │                                           │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │ Correlation     │                                           │
│  │ Filter          │                                           │
│  │                 │                                           │
│  │ max 3 corr      │                                           │
│  │ positions       │                                           │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  ┌─────────────────┐                                           │
│  │ Rate Limiter    │                                           │
│  │                 │                                           │
│  │ 1 signal/hour   │                                           │
│  │ per symbol      │                                           │
│  └────────┬────────┘                                           │
│           ↓                                                     │
│  Processed Signals → Risk Management → Execution               │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Laboratory Implementation

```python
class SignalLab:
    """Experiment with signal transformations offline"""
    
    def __init__(self, signal_data_path):
        self.signals = pd.read_parquet(signal_data_path)
        self.transformations = []
        
    def add_transformation(self, name, func):
        """Register a transformation to test"""
        self.transformations.append((name, func))
        
    def compare_transformations(self):
        """Apply all transformations and compare"""
        results = {}
        
        for name, transform in self.transformations:
            # Apply transformation
            transformed = transform(self.signals.copy())
            
            # Calculate metrics
            results[name] = {
                'signal_count': len(transformed),
                'avg_confidence': transformed.confidence.mean(),
                'signal_quality': self.calculate_quality_score(transformed),
                'regime_distribution': transformed.regime_context.value_counts(),
                'expected_sharpe': self.estimate_sharpe(transformed)
            }
            
        return pd.DataFrame(results).T
    
    def visualize_transformation_impact(self):
        """Visualize how transformations change signal distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original distribution
        self.signals.confidence.hist(ax=axes[0,0], bins=50, alpha=0.7)
        axes[0,0].set_title('Original Signal Confidence')
        
        # After transformation
        for name, transform in self.transformations:
            transformed = transform(self.signals.copy())
            transformed.confidence.hist(ax=axes[0,1], bins=50, alpha=0.5, label=name)
        axes[0,1].legend()
        axes[0,1].set_title('Transformed Signal Confidence')
        
        # Signal frequency
        signal_freq = self.signals.groupby(pd.Grouper(key='timestamp', freq='D')).size()
        signal_freq.plot(ax=axes[1,0])
        axes[1,0].set_title('Signal Frequency Over Time')
        
        # Regime distribution
        regime_counts = self.signals.regime_context.value_counts()
        regime_counts.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Signals by Regime')
        
        plt.tight_layout()
        return fig
```

## Implementing Signal Processing Containers

### Signal Logger Container (For Capture)

```python
# src/containers/signal_logger_container.py
import pandas as pd
from typing import Dict, Any, List
import pyarrow.parquet as pq

class SignalLoggerContainer:
    """Captures signals for offline analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get('name', 'signal_logger')
        self.output_path = config['output_path']
        self.capture_fields = config.get('capture_fields', [
            'timestamp', 'symbol', 'action', 'strength', 'confidence'
        ])
        self.buffer = []
        self.buffer_size = config.get('buffer_size', 1000)
        
    def setup(self, event_bus):
        self.event_bus = event_bus
        self.event_bus.subscribe('SIGNAL', self.on_signal)
        self.event_bus.subscribe('REGIME', self.on_regime)  # Capture regime context
        self.current_regime = None
        
    def on_signal(self, event: SignalEvent):
        """Capture signal with full context"""
        signal_data = {
            field: getattr(event, field, None) 
            for field in self.capture_fields
        }
        
        # Add regime context
        signal_data['regime_context'] = self.current_regime
        
        # Add any indicators that were used
        if hasattr(event, 'indicator_values'):
            signal_data['indicators'] = event.indicator_values
            
        self.buffer.append(signal_data)
        
        # Flush to disk periodically
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def on_regime(self, event: RegimeEvent):
        """Track regime changes"""
        self.current_regime = event.regime
        
    def _flush_buffer(self):
        """Write buffer to parquet file"""
        if self.buffer:
            df = pd.DataFrame(self.buffer)
            
            # Append to existing file or create new
            if os.path.exists(self.output_path):
                existing = pd.read_parquet(self.output_path)
                df = pd.concat([existing, df], ignore_index=True)
                
            df.to_parquet(self.output_path, index=False)
            self.buffer = []
    
    def teardown(self):
        """Ensure all signals are saved"""
        self._flush_buffer()
```

### Signal Processor Container (For Deployment)

```python
# src/containers/signal_processor_container.py
class SignalProcessorContainer:
    """Processes signals based on learned transformations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get('name', 'signal_processor')
        self.config = config
        
        # Processing parameters learned from analysis
        self.min_confidence = config.get('min_confidence', 0.5)
        self.regime_boost = config.get('regime_boost', {})
        self.rate_limit = config.get('rate_limit', None)
        self.ml_model_path = config.get('ml_model')
        
        # State
        self.current_regime = None
        self.signal_history = {}
        self.ml_model = None
        
        if self.ml_model_path:
            self.ml_model = joblib.load(self.ml_model_path)
            
    def setup(self, event_bus):
        self.event_bus = event_bus
        self.event_bus.subscribe('SIGNAL', self.on_signal)
        self.event_bus.subscribe('REGIME', self.on_regime)
        
    def on_signal(self, event: SignalEvent):
        """Process signal through pipeline"""
        
        # Step 1: Confidence filter
        if event.confidence < self.min_confidence:
            self._emit_rejection(event, "Low confidence")
            return
            
        # Step 2: Regime adjustment
        if self.current_regime and self.current_regime in self.regime_boost:
            event.strength *= self.regime_boost[self.current_regime]
            
        # Step 3: ML scoring (if available)
        if self.ml_model:
            features = self._extract_features(event)
            ml_score = self.ml_model.predict_proba([features])[0, 1]
            if ml_score < 0.5:
                self._emit_rejection(event, "Low ML score")
                return
            event.ml_score = ml_score
            
        # Step 4: Rate limiting
        if self.rate_limit and not self._check_rate_limit(event):
            self._emit_rejection(event, "Rate limited")
            return
            
        # Emit processed signal
        self.event_bus.publish('SIGNAL', event)
        
    def _check_rate_limit(self, signal: SignalEvent) -> bool:
        """Check if signal passes rate limiting"""
        symbol = signal.symbol
        current_time = signal.timestamp
        
        if symbol in self.signal_history:
            last_signal_time = self.signal_history[symbol]
            time_diff = current_time - last_signal_time
            
            if self.rate_limit == "1H" and time_diff.total_seconds() < 3600:
                return False
            elif self.rate_limit == "30M" and time_diff.total_seconds() < 1800:
                return False
                
        self.signal_history[symbol] = current_time
        return True
```

## Multi-Phase Optimization with Signal Analysis

```yaml
# config/multi_phase_signal_optimization.yaml
phases:
  - name: "signal_generation"
    type: "signal_generation"
    strategies: 
      - {type: "momentum", params: {fast: 10, slow: 30}}
      - {type: "mean_reversion", params: {lookback: 20}}
      - {type: "ml_based", model: "rf_v1.pkl"}
    classifiers:
      - {type: "hmm", states: 3}
      - {type: "volatility_regime"}
    output: "signals/raw/"
    
  - name: "signal_analysis" 
    type: "offline_analysis"
    input: "signals/raw/"
    notebooks: 
      - "analyze_signal_quality.ipynb"
      - "design_transformations.ipynb"
      - "regime_impact_analysis.ipynb"
    output: "signals/insights/"
    
  - name: "transformation_testing"
    type: "transformation_lab"
    input: "signals/raw/"
    transformations:
      - name: "conservative"
        filters: {min_confidence: 0.8, max_per_hour: 2}
      - name: "regime_aware"
        regime_adjustments: {BULL: 1.5, BEAR: 0.7, NEUTRAL: 1.0}
      - name: "ml_enhanced"
        ml_model: "models/signal_quality_rf.pkl"
        min_ml_score: 0.6
    output: "signals/transformed/"
    
  - name: "backtest_comparison"
    type: "signal_replay"
    signal_sources: "signals/transformed/*.parquet"
    parallel: true
    metrics: ["sharpe", "max_drawdown", "win_rate"]
    
  - name: "final_implementation"
    type: "full_backtest"
    signal_processor: 
      config: "configs/best_processor.yaml"  # From phase 4
    validation_period: "2023-01-01:2023-12-31"
```

## Container Chaining for Complex Pipelines

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONTAINER CHAINING EXAMPLE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Market Data                                                    │
│      ↓                                                          │
│  Indicators                                                     │
│      ↓                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Momentum    │  │ Mean Rev    │  │ ML Strategy │            │
│  │ Strategy    │  │ Strategy    │  │             │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         └─────────────────┴─────────────────┘                  │
│                           ↓                                     │
│                  Signal Aggregator                              │
│                  (Weight by quality)                            │
│                           ↓                                     │
│         ┌─────────────────────────────────┐                    │
│         │     Signal Processing Chain     │                    │
│         │  ┌───────────────────────────┐  │                    │
│         │  │ 1. Regime Context Filter  │  │                    │
│         │  └────────────┬──────────────┘  │                    │
│         │               ↓                  │                    │
│         │  ┌───────────────────────────┐  │                    │
│         │  │ 2. ML Quality Scorer      │  │                    │
│         │  └────────────┬──────────────┘  │                    │
│         │               ↓                  │                    │
│         │  ┌───────────────────────────┐  │                    │
│         │  │ 3. Correlation Filter     │  │                    │
│         │  └────────────┬──────────────┘  │                    │
│         │               ↓                  │                    │
│         │  ┌───────────────────────────┐  │                    │
│         │  │ 4. Rate Limiter           │  │                    │
│         │  └────────────┬──────────────┘  │                    │
│         └───────────────┴─────────────────┘                    │
│                           ↓                                     │
│                    Risk Management                              │
│                           ↓                                     │
│                      Execution                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation of Chained Processors

```yaml
# config/chained_signal_processing.yaml
containers:
  # Signal generation
  - type: "strategy_ensemble"
    name: "signal_generator"
    strategies: ["momentum", "mean_reversion", "ml_based"]
    
  # Processing chain
  - type: "signal_aggregator"
    name: "weighted_aggregator"
    weighting_method: "sharpe_weighted"
    
  - type: "regime_context_filter"
    name: "regime_filter"
    rules:
      BULL: {min_strength: 0.3, boost: 1.2}
      BEAR: {min_strength: 0.7, boost: 0.8}
      NEUTRAL: {min_strength: 0.5, boost: 1.0}
      
  - type: "ml_quality_scorer"
    name: "quality_scorer"
    model: "models/signal_quality_v2.pkl"
    threshold: 0.6
    
  - type: "correlation_filter"
    name: "correlation_filter"
    max_correlated: 3
    lookback_window: 20
    correlation_threshold: 0.7
    
  - type: "rate_limiter"
    name: "rate_limiter"
    limits:
      per_symbol: "1H"
      total_per_hour: 10
      
adapters:
  - type: "pipeline"
    containers: ["signal_generator", "weighted_aggregator", "regime_filter",
                "quality_scorer", "correlation_filter", "rate_limiter",
                "risk_management", "execution"]
```

## Benefits of This Architecture

1. **Research Velocity**: Test signal processing ideas without running full backtests
2. **Hypothesis Testing**: Compare multiple approaches on the same signal dataset
3. **Performance Optimization**: Identify bottlenecks before adding execution overhead
4. **Modular Development**: Each processor can be developed and tested independently
5. **Reusability**: Processors work across different strategies and markets

## Summary

The signal generation and analysis pattern is a key architectural feature of ADMF-PC that enables:

- **Separation of Concerns**: Signal generation, processing, and execution are independent
- **Rapid Experimentation**: Test ideas on historical signals without full backtests
- **Sophisticated Pipelines**: Chain multiple processors for complex transformations
- **Regime Integration**: Process regime signals alongside trading signals
- **Performance Analysis**: Understand signal characteristics before deployment

This modular approach transforms ADMF-PC from a simple backtesting framework into a comprehensive research platform for systematic trading strategy development!
