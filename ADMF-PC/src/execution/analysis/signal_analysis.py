"""
Signal Analysis Engine for pure signal generation and analysis.

This module implements Pattern #3 from BACKTEST.MD - signal generation
without execution for analysis, MAE/MFE optimization, and classifier comparison.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import json
import logging

from ...core.events import Event, EventType, EventBus
from ...risk.protocols import Signal
from ...data.models import MarketData


logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of signal analysis."""
    MAE_MFE = "mae_mfe"  # Maximum Adverse/Favorable Excursion
    SIGNAL_QUALITY = "signal_quality"  # Win rate, expectancy
    CORRELATION = "correlation"  # Signal correlation across strategies
    REGIME_ANALYSIS = "regime_analysis"  # Signal performance by regime
    FORWARD_RETURNS = "forward_returns"  # N-bar forward returns


@dataclass
class SignalAnalysisResult:
    """Results from signal analysis."""
    signal_id: str
    timestamp: datetime
    strategy_id: str
    symbol: str
    direction: str  # BUY/SELL
    strength: float
    price_at_signal: float
    
    # Forward returns
    forward_returns: Dict[str, float] = field(default_factory=dict)  # {"1_bar": 0.002, "5_bar": 0.015}
    
    # MAE/MFE
    mae: Optional[float] = None  # Maximum Adverse Excursion
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    mae_bars: Optional[int] = None  # Bars to MAE
    mfe_bars: Optional[int] = None  # Bars to MFE
    
    # Signal quality metrics
    signal_quality: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "price_at_signal": self.price_at_signal,
            "forward_returns": self.forward_returns,
            "mae": self.mae,
            "mfe": self.mfe,
            "mae_bars": self.mae_bars,
            "mfe_bars": self.mfe_bars,
            "signal_quality": self.signal_quality,
            "metadata": self.metadata
        }


class SignalAnalysisEngine:
    """
    Engine for analyzing signals without execution.
    
    This engine:
    - Captures all signals with metadata
    - Calculates signal quality metrics (win rate, MAE/MFE)
    - Analyzes signal correlation across strategies
    - Stores signals for later replay
    - NO execution, NO portfolio tracking
    """
    
    def __init__(
        self,
        lookback_bars: int = 20,
        forward_bars: List[int] = None,
        analysis_types: List[AnalysisType] = None
    ):
        """
        Initialize the signal analysis engine.
        
        Args:
            lookback_bars: Number of bars to look back for context
            forward_bars: List of forward bar counts for return calculation [1, 5, 10, 20]
            analysis_types: Types of analysis to perform
        """
        self.lookback_bars = lookback_bars
        self.forward_bars = forward_bars or [1, 5, 10, 20]
        self.analysis_types = analysis_types or list(AnalysisType)
        
        # Storage
        self._signals: List[SignalAnalysisResult] = []
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._signal_buffer: Dict[str, List[Signal]] = {}  # Buffer signals until we have forward data
        
        # Analysis results
        self._strategy_metrics: Dict[str, Dict[str, Any]] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        
        # Event handling
        self._event_bus: Optional[EventBus] = None
        
    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus for publishing analysis events."""
        self._event_bus = event_bus
        
    def process_market_data(self, timestamp: datetime, market_data: Dict[str, MarketData]) -> None:
        """
        Process market data to update price history.
        
        Args:
            timestamp: Current timestamp
            market_data: Market data by symbol
        """
        # Update price history
        for symbol, data in market_data.items():
            if symbol not in self._price_history:
                self._price_history[symbol] = []
                
            # Use close price for analysis
            price = data.close if hasattr(data, 'close') else data.price
            self._price_history[symbol].append((timestamp, price))
            
            # Limit history size
            if len(self._price_history[symbol]) > self.lookback_bars + max(self.forward_bars):
                self._price_history[symbol].pop(0)
                
        # Process any buffered signals that now have enough forward data
        self._process_buffered_signals(timestamp)
        
    def capture_signal(self, signal: Signal, market_data: Dict[str, MarketData]) -> None:
        """
        Capture a signal for analysis.
        
        Args:
            signal: The signal to analyze
            market_data: Current market data
        """
        # Get current price
        if signal.symbol not in market_data:
            logger.warning(f"No market data for symbol {signal.symbol}")
            return
            
        current_data = market_data[signal.symbol]
        price = current_data.close if hasattr(current_data, 'close') else current_data.price
        
        # Create analysis result
        result = SignalAnalysisResult(
            signal_id=signal.signal_id,
            timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            direction="BUY" if signal.side.value == "BUY" else "SELL",
            strength=signal.strength,
            price_at_signal=price,
            metadata=signal.metadata or {}
        )
        
        # Buffer the signal for forward analysis
        if signal.symbol not in self._signal_buffer:
            self._signal_buffer[signal.symbol] = []
        self._signal_buffer[signal.symbol].append((signal, result))
        
        # Emit signal captured event
        if self._event_bus:
            self._event_bus.publish(Event(
                event_type=EventType.INFO,
                payload={
                    'type': 'signal.captured',
                    'signal_id': signal.signal_id,
                    'strategy_id': signal.strategy_id,
                    'symbol': signal.symbol
                }
            ))
            
    def _process_buffered_signals(self, current_timestamp: datetime) -> None:
        """Process buffered signals that now have enough forward data."""
        for symbol, buffered_signals in list(self._signal_buffer.items()):
            if symbol not in self._price_history:
                continue
                
            price_history = self._price_history[symbol]
            processed = []
            
            for signal, result in buffered_signals:
                # Find signal timestamp in price history
                signal_idx = None
                for i, (ts, _) in enumerate(price_history):
                    if ts >= signal.timestamp:
                        signal_idx = i
                        break
                        
                if signal_idx is None:
                    continue
                    
                # Check if we have enough forward data
                max_forward = max(self.forward_bars)
                if signal_idx + max_forward >= len(price_history):
                    continue  # Not enough forward data yet
                    
                # Calculate forward returns
                if AnalysisType.FORWARD_RETURNS in self.analysis_types:
                    self._calculate_forward_returns(result, price_history, signal_idx)
                    
                # Calculate MAE/MFE
                if AnalysisType.MAE_MFE in self.analysis_types:
                    self._calculate_mae_mfe(result, price_history, signal_idx)
                    
                # Signal is fully analyzed
                self._signals.append(result)
                processed.append((signal, result))
                
                # Emit analysis complete event
                if self._event_bus:
                    self._event_bus.publish(Event(
                        event_type=EventType.INFO,
                        payload={
                            'type': 'signal.analyzed',
                            'result': result.to_dict()
                        }
                    ))
                    
            # Remove processed signals
            for item in processed:
                buffered_signals.remove(item)
                
            if not buffered_signals:
                del self._signal_buffer[symbol]
                
    def _calculate_forward_returns(
        self,
        result: SignalAnalysisResult,
        price_history: List[Tuple[datetime, float]],
        signal_idx: int
    ) -> None:
        """Calculate forward returns for various horizons."""
        signal_price = price_history[signal_idx][1]
        
        for n_bars in self.forward_bars:
            if signal_idx + n_bars < len(price_history):
                future_price = price_history[signal_idx + n_bars][1]
                return_pct = (future_price - signal_price) / signal_price
                
                # Adjust for direction
                if result.direction == "SELL":
                    return_pct = -return_pct
                    
                result.forward_returns[f"{n_bars}_bar"] = return_pct
                
    def _calculate_mae_mfe(
        self,
        result: SignalAnalysisResult,
        price_history: List[Tuple[datetime, float]],
        signal_idx: int
    ) -> None:
        """Calculate Maximum Adverse and Favorable Excursion."""
        signal_price = price_history[signal_idx][1]
        
        # Look at next N bars (use max forward bars)
        max_forward = max(self.forward_bars)
        
        if result.direction == "BUY":
            # For long positions
            worst_price = signal_price
            best_price = signal_price
            worst_idx = signal_idx
            best_idx = signal_idx
            
            for i in range(signal_idx + 1, min(signal_idx + max_forward + 1, len(price_history))):
                price = price_history[i][1]
                if price < worst_price:
                    worst_price = price
                    worst_idx = i
                if price > best_price:
                    best_price = price
                    best_idx = i
                    
            result.mae = (worst_price - signal_price) / signal_price
            result.mfe = (best_price - signal_price) / signal_price
            
        else:  # SELL
            # For short positions
            worst_price = signal_price
            best_price = signal_price
            worst_idx = signal_idx
            best_idx = signal_idx
            
            for i in range(signal_idx + 1, min(signal_idx + max_forward + 1, len(price_history))):
                price = price_history[i][1]
                if price > worst_price:
                    worst_price = price
                    worst_idx = i
                if price < best_price:
                    best_price = price
                    best_idx = i
                    
            result.mae = (signal_price - worst_price) / signal_price
            result.mfe = (signal_price - best_price) / signal_price
            
        result.mae_bars = worst_idx - signal_idx
        result.mfe_bars = best_idx - signal_idx
        
    def calculate_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for each strategy."""
        strategy_signals = {}
        
        # Group signals by strategy
        for signal in self._signals:
            if signal.strategy_id not in strategy_signals:
                strategy_signals[signal.strategy_id] = []
            strategy_signals[signal.strategy_id].append(signal)
            
        # Calculate metrics for each strategy
        for strategy_id, signals in strategy_signals.items():
            metrics = self._calculate_signal_metrics(signals)
            self._strategy_metrics[strategy_id] = metrics
            
        return self._strategy_metrics
        
    def _calculate_signal_metrics(self, signals: List[SignalAnalysisResult]) -> Dict[str, Any]:
        """Calculate quality metrics for a set of signals."""
        if not signals:
            return {}
            
        # Win rate (using 1-bar returns as example)
        wins = sum(1 for s in signals if s.forward_returns.get("1_bar", 0) > 0)
        win_rate = wins / len(signals) if signals else 0
        
        # Average returns
        avg_returns = {}
        for key in ["1_bar", "5_bar", "10_bar", "20_bar"]:
            returns = [s.forward_returns.get(key, 0) for s in signals if key in s.forward_returns]
            avg_returns[f"avg_{key}_return"] = np.mean(returns) if returns else 0
            
        # MAE/MFE statistics
        maes = [s.mae for s in signals if s.mae is not None]
        mfes = [s.mfe for s in signals if s.mfe is not None]
        
        # Calculate expectancy
        winning_returns = [s.forward_returns.get("1_bar", 0) for s in signals 
                          if s.forward_returns.get("1_bar", 0) > 0]
        losing_returns = [s.forward_returns.get("1_bar", 0) for s in signals 
                         if s.forward_returns.get("1_bar", 0) <= 0]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = np.mean(losing_returns) if losing_returns else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            "total_signals": len(signals),
            "win_rate": win_rate,
            "expectancy": expectancy,
            **avg_returns,
            "avg_mae": np.mean(maes) if maes else None,
            "avg_mfe": np.mean(mfes) if mfes else None,
            "mae_mfe_ratio": np.mean(mfes) / abs(np.mean(maes)) if maes and mfes and np.mean(maes) != 0 else None,
            "signal_strength_avg": np.mean([s.strength for s in signals]),
            "signal_strength_std": np.std([s.strength for s in signals])
        }
        
    def calculate_signal_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategies."""
        if AnalysisType.CORRELATION not in self.analysis_types:
            return pd.DataFrame()
            
        # Create signal strength time series for each strategy
        strategy_series = {}
        
        # Get all unique timestamps
        timestamps = sorted(set(s.timestamp for s in self._signals))
        
        for signal in self._signals:
            if signal.strategy_id not in strategy_series:
                strategy_series[signal.strategy_id] = pd.Series(
                    index=timestamps,
                    data=0.0,
                    name=signal.strategy_id
                )
                
            # Use signed strength (negative for SELL)
            strength = signal.strength if signal.direction == "BUY" else -signal.strength
            strategy_series[signal.strategy_id][signal.timestamp] = strength
            
        # Calculate correlation matrix
        if strategy_series:
            df = pd.DataFrame(strategy_series)
            self._correlation_matrix = df.corr()
            
        return self._correlation_matrix
        
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get comprehensive analysis results."""
        return {
            "total_signals_analyzed": len(self._signals),
            "signals_by_strategy": self._get_signal_counts_by_strategy(),
            "strategy_metrics": self.calculate_strategy_metrics(),
            "correlation_matrix": self.calculate_signal_correlations().to_dict() if self._correlation_matrix is not None else None,
            "mae_mfe_summary": self._get_mae_mfe_summary(),
            "forward_return_summary": self._get_forward_return_summary()
        }
        
    def _get_signal_counts_by_strategy(self) -> Dict[str, int]:
        """Get signal counts by strategy."""
        counts = {}
        for signal in self._signals:
            counts[signal.strategy_id] = counts.get(signal.strategy_id, 0) + 1
        return counts
        
    def _get_mae_mfe_summary(self) -> Dict[str, Any]:
        """Get MAE/MFE summary statistics."""
        if not self._signals:
            return {}
            
        maes = [s.mae for s in self._signals if s.mae is not None]
        mfes = [s.mfe for s in self._signals if s.mfe is not None]
        
        if not maes or not mfes:
            return {}
            
        return {
            "mae": {
                "mean": np.mean(maes),
                "std": np.std(maes),
                "min": np.min(maes),
                "max": np.max(maes),
                "percentiles": {
                    "25": np.percentile(maes, 25),
                    "50": np.percentile(maes, 50),
                    "75": np.percentile(maes, 75)
                }
            },
            "mfe": {
                "mean": np.mean(mfes),
                "std": np.std(mfes),
                "min": np.min(mfes),
                "max": np.max(mfes),
                "percentiles": {
                    "25": np.percentile(mfes, 25),
                    "50": np.percentile(mfes, 50),
                    "75": np.percentile(mfes, 75)
                }
            },
            "optimal_stop_loss": abs(np.percentile(maes, 75)),  # 75th percentile of MAE
            "optimal_take_profit": np.percentile(mfes, 75)  # 75th percentile of MFE
        }
        
    def _get_forward_return_summary(self) -> Dict[str, Any]:
        """Get forward return summary statistics."""
        summary = {}
        
        for horizon in ["1_bar", "5_bar", "10_bar", "20_bar"]:
            returns = [s.forward_returns.get(horizon, 0) for s in self._signals 
                      if horizon in s.forward_returns]
                      
            if returns:
                summary[horizon] = {
                    "mean": np.mean(returns),
                    "std": np.std(returns),
                    "sharpe": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                    "positive_rate": sum(1 for r in returns if r > 0) / len(returns),
                    "skew": self._calculate_skew(returns),
                    "kurtosis": self._calculate_kurtosis(returns)
                }
                
        return summary
        
    def _calculate_skew(self, data: List[float]) -> float:
        """Calculate skewness of returns."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of returns."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        
    def export_signals(self, filepath: str) -> None:
        """Export analyzed signals to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(
                [s.to_dict() for s in self._signals],
                f,
                indent=2,
                default=str
            )
            
    def clear(self) -> None:
        """Clear all stored signals and analysis results."""
        self._signals.clear()
        self._price_history.clear()
        self._signal_buffer.clear()
        self._strategy_metrics.clear()
        self._correlation_matrix = None