"""
Comprehensive Backtesting Framework with Look-Ahead Bias Prevention.

This framework provides:
1. Proper train/test/validation splits
2. Look-ahead bias detection and prevention
3. Bar index alignment verification
4. Statistical significance testing
5. Out-of-sample validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    train_ratio: float = 0.6  # 60% for training
    test_ratio: float = 0.2   # 20% for testing
    validation_ratio: float = 0.2  # 20% for validation
    
    min_bars_warmup: int = 200  # Minimum bars for indicator warmup
    min_trades_significance: int = 30  # Minimum trades for statistical significance
    
    # Look-ahead bias prevention
    feature_lag: int = 1  # Lag features by N bars to prevent look-ahead
    signal_delay: int = 1  # Delay signal execution by N bars
    
    # Validation parameters
    bootstrap_samples: int = 1000  # Number of bootstrap samples
    confidence_level: float = 0.95  # Confidence level for intervals
    
    # Risk parameters
    max_position_size: float = 1.0  # Maximum position size
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Transaction costs
    commission_bps: float = 10  # 10 basis points commission
    slippage_bps: float = 5  # 5 basis points slippage


@dataclass
class TradeResult:
    """Individual trade result."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    direction: int  # 1 for long, -1 for short
    pnl: float
    pnl_pct: float
    bars_held: int
    features_at_entry: Dict[str, float] = field(default_factory=dict)
    signal_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Trade results
    trades: List[TradeResult]
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return_per_trade: float
    
    # Statistical tests
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    
    # Data splits info
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    validation_period: Optional[Tuple[datetime, datetime]]
    
    # Look-ahead bias checks
    alignment_issues: List[Dict[str, Any]]
    feature_future_leakage: Dict[str, float]
    
    # Additional metrics
    metrics: Dict[str, float] = field(default_factory=dict)


class LookAheadDetector:
    """Detects potential look-ahead bias in features and signals."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.correlation_threshold = 0.95  # High correlation might indicate leakage
        
    def check_feature_alignment(self, 
                               features_df: pd.DataFrame,
                               prices_df: pd.DataFrame) -> Dict[str, float]:
        """
        Check if features are improperly aligned with future prices.
        
        Returns correlation of each feature with future returns.
        """
        future_leakage = {}
        
        # Calculate future returns
        future_returns = prices_df['close'].pct_change().shift(-1)
        
        for feature in self.feature_names:
            if feature not in features_df.columns:
                continue
                
            # Check correlation with future returns (should be low)
            corr = features_df[feature].corr(future_returns)
            
            # Also check if feature values appear before they should
            # by comparing with lagged versions
            lagged_feature = features_df[feature].shift(1)
            autocorr = features_df[feature].corr(lagged_feature)
            
            if abs(corr) > self.correlation_threshold:
                logger.warning(f"Feature '{feature}' has high correlation "
                             f"({corr:.3f}) with future returns - possible look-ahead bias!")
            
            future_leakage[feature] = abs(corr)
            
        return future_leakage
    
    def check_signal_timing(self,
                           signals: pd.Series,
                           prices: pd.Series,
                           expected_lag: int = 1) -> List[Dict[str, Any]]:
        """
        Check if signals are properly lagged relative to price data.
        """
        issues = []
        
        # Check if signals change exactly at price extremes (suspicious)
        price_peaks = (prices > prices.shift(1)) & (prices > prices.shift(-1))
        price_troughs = (prices < prices.shift(1)) & (prices < prices.shift(-1))
        
        signal_changes = signals != signals.shift(1)
        
        # Signals at exact extremes are suspicious
        suspicious_peaks = (signal_changes & price_peaks).sum()
        suspicious_troughs = (signal_changes & price_troughs).sum()
        
        if suspicious_peaks > len(signals) * 0.01:  # More than 1%
            issues.append({
                'type': 'signal_at_peaks',
                'count': suspicious_peaks,
                'percentage': suspicious_peaks / len(signals) * 100
            })
            
        if suspicious_troughs > len(signals) * 0.01:
            issues.append({
                'type': 'signal_at_troughs', 
                'count': suspicious_troughs,
                'percentage': suspicious_troughs / len(signals) * 100
            })
            
        return issues


class RobustBacktester:
    """
    Robust backtesting engine with look-ahead bias prevention.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.look_ahead_detector = None
        
    def prepare_data_splits(self, 
                           data: pd.DataFrame,
                           target_col: str = 'signal') -> Dict[str, pd.DataFrame]:
        """
        Split data into train/test/validation sets.
        
        Args:
            data: DataFrame with OHLCV + signals
            target_col: Column containing trading signals
            
        Returns:
            Dictionary with 'train', 'test', 'validation' DataFrames
        """
        n_samples = len(data)
        
        # Calculate split indices
        train_end = int(n_samples * self.config.train_ratio)
        test_end = train_end + int(n_samples * self.config.test_ratio)
        
        # Add warmup period to avoid initialization issues
        warmup_adjusted_train_start = self.config.min_bars_warmup
        
        splits = {
            'train': data.iloc[warmup_adjusted_train_start:train_end].copy(),
            'test': data.iloc[train_end:test_end].copy(),
            'validation': data.iloc[test_end:].copy()
        }
        
        # Store period information
        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                logger.info(f"{split_name.capitalize()} period: "
                           f"{split_data.index[0]} to {split_data.index[-1]} "
                           f"({len(split_data)} bars)")
                
        return splits
    
    def apply_realistic_lags(self,
                           data: pd.DataFrame,
                           feature_cols: List[str],
                           signal_col: str) -> pd.DataFrame:
        """
        Apply realistic lags to prevent look-ahead bias.
        
        Features are lagged by feature_lag bars.
        Signals are lagged by signal_delay bars.
        """
        data_lagged = data.copy()
        
        # Lag features
        for col in feature_cols:
            if col in data_lagged.columns:
                data_lagged[f'{col}_lagged'] = data_lagged[col].shift(self.config.feature_lag)
                # Keep original for comparison
                data_lagged[f'{col}_original'] = data_lagged[col]
                
        # Lag signals
        if signal_col in data_lagged.columns:
            data_lagged[f'{signal_col}_lagged'] = data_lagged[signal_col].shift(self.config.signal_delay)
            data_lagged[f'{signal_col}_original'] = data_lagged[signal_col]
            
        return data_lagged
    
    def simulate_trades(self,
                       data: pd.DataFrame,
                       signal_col: str,
                       price_col: str = 'close') -> List[TradeResult]:
        """
        Simulate realistic trade execution with proper timing.
        """
        trades = []
        position = 0
        entry_price = None
        entry_time = None
        entry_bar = None
        entry_features = {}
        
        for i in range(len(data)):
            current_bar = data.iloc[i]
            signal = current_bar[signal_col]
            
            # Skip NaN signals
            if pd.isna(signal):
                continue
                
            # Entry logic
            if position == 0 and signal != 0:
                # Enter position at next bar's open (realistic execution)
                if i < len(data) - 1:
                    next_bar = data.iloc[i + 1]
                    entry_price = next_bar['open']
                    entry_time = next_bar.name
                    entry_bar = i + 1
                    position = signal
                    
                    # Store features at entry
                    entry_features = {
                        col: current_bar[col] 
                        for col in data.columns 
                        if col.startswith(('sma', 'ema', 'rsi', 'macd'))
                        and not col.endswith('_lagged')
                    }
                    
            # Exit logic
            elif position != 0 and (signal == 0 or signal != position):
                # Exit position at next bar's open
                if i < len(data) - 1:
                    next_bar = data.iloc[i + 1]
                    exit_price = next_bar['open']
                    exit_time = next_bar.name
                    
                    # Apply transaction costs
                    total_cost_bps = self.config.commission_bps + self.config.slippage_bps
                    cost_multiplier = 1 - (total_cost_bps / 10000)
                    
                    # Calculate P&L
                    if position > 0:  # Long
                        pnl_pct = (exit_price / entry_price - 1) * cost_multiplier
                    else:  # Short
                        pnl_pct = (1 - exit_price / entry_price) * cost_multiplier
                        
                    pnl = pnl_pct * self.config.max_position_size
                    
                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=self.config.max_position_size,
                        direction=position,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        bars_held=i - entry_bar,
                        features_at_entry=entry_features,
                        signal_metadata={'signal_value': signal}
                    ))
                    
                    position = 0
                    entry_price = None
                    
        return trades
    
    def calculate_performance_metrics(self, 
                                    trades: List[TradeResult],
                                    data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        """
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'num_trades': 0
            }
            
        # Basic metrics
        returns = [t.pnl_pct for t in trades]
        wins = [r for r in returns if r > 0]
        
        total_return = np.prod([1 + r for r in returns]) - 1
        win_rate = len(wins) / len(trades) if trades else 0
        avg_return = np.mean(returns) if returns else 0
        
        # Sharpe ratio (annualized)
        if len(returns) > 1:
            # Estimate bars per year (assuming 5min bars, 252 trading days)
            bars_per_day = 78  # 6.5 hours * 12 bars/hour
            bars_per_year = bars_per_day * 252
            
            # Calculate annualized metrics
            avg_bars_held = np.mean([t.bars_held for t in trades])
            trades_per_year = bars_per_year / avg_bars_held if avg_bars_held > 0 else 0
            
            annual_return = (1 + avg_return) ** trades_per_year - 1
            annual_vol = np.std(returns) * np.sqrt(trades_per_year)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'num_trades': len(trades),
            'avg_bars_held': np.mean([t.bars_held for t in trades]) if trades else 0,
            'profit_factor': sum(wins) / abs(sum([r for r in returns if r < 0])) 
                           if any(r < 0 for r in returns) else np.inf
        }
    
    def perform_statistical_tests(self, 
                                returns: List[float]) -> Tuple[float, float, Tuple[float, float]]:
        """
        Perform statistical significance tests on returns.
        """
        if len(returns) < self.config.min_trades_significance:
            logger.warning(f"Only {len(returns)} trades - insufficient for statistical tests")
            return 0, 1, (0, 0)
            
        # T-test against zero mean
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(self.config.bootstrap_samples):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        confidence_interval = np.percentile(bootstrap_means, 
                                          [(1 - self.config.confidence_level) / 2 * 100,
                                           (1 + self.config.confidence_level) / 2 * 100])
        
        return t_stat, p_value, tuple(confidence_interval)
    
    def run_backtest(self,
                    data: pd.DataFrame,
                    signal_col: str,
                    feature_cols: List[str],
                    split: str = 'test') -> BacktestResult:
        """
        Run complete backtest with all checks.
        """
        # Initialize look-ahead detector
        self.look_ahead_detector = LookAheadDetector(feature_cols)
        
        # Split data
        splits = self.prepare_data_splits(data, signal_col)
        test_data = splits[split]
        
        # Apply realistic lags
        test_data_lagged = self.apply_realistic_lags(
            test_data, feature_cols, signal_col
        )
        
        # Check for look-ahead bias
        future_leakage = self.look_ahead_detector.check_feature_alignment(
            test_data[feature_cols], 
            test_data[['close']]
        )
        
        alignment_issues = self.look_ahead_detector.check_signal_timing(
            test_data[signal_col],
            test_data['close']
        )
        
        # Simulate trades using lagged signals
        trades = self.simulate_trades(
            test_data_lagged,
            f'{signal_col}_lagged',
            'close'
        )
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(trades, test_data)
        
        # Statistical tests
        returns = [t.pnl_pct for t in trades]
        t_stat, p_value, conf_interval = self.perform_statistical_tests(returns)
        
        # Compile results
        return BacktestResult(
            trades=trades,
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            avg_return_per_trade=metrics['avg_return_per_trade'],
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval=conf_interval,
            train_period=(splits['train'].index[0], splits['train'].index[-1]),
            test_period=(splits['test'].index[0], splits['test'].index[-1]),
            validation_period=(splits['validation'].index[0], splits['validation'].index[-1]) 
                            if len(splits['validation']) > 0 else None,
            alignment_issues=alignment_issues,
            feature_future_leakage=future_leakage,
            metrics=metrics
        )
    
    def validate_out_of_sample(self,
                             data: pd.DataFrame,
                             signal_col: str,
                             feature_cols: List[str]) -> Dict[str, BacktestResult]:
        """
        Run complete validation including out-of-sample test.
        """
        results = {}
        
        # Run on each split
        for split in ['train', 'test', 'validation']:
            logger.info(f"Running backtest on {split} split...")
            result = self.run_backtest(data, signal_col, feature_cols, split)
            results[split] = result
            
            logger.info(f"{split} results: "
                       f"Return={result.total_return:.2%}, "
                       f"Sharpe={result.sharpe_ratio:.2f}, "
                       f"p-value={result.p_value:.4f}")
            
        return results


def create_validation_report(results: Dict[str, BacktestResult],
                           output_path: str) -> None:
    """
    Create comprehensive validation report.
    """
    report = {
        'summary': {
            'train_performance': {
                'total_return': results['train'].total_return,
                'sharpe_ratio': results['train'].sharpe_ratio,
                'num_trades': len(results['train'].trades)
            },
            'test_performance': {
                'total_return': results['test'].total_return,
                'sharpe_ratio': results['test'].sharpe_ratio,
                'num_trades': len(results['test'].trades)
            },
            'validation_performance': {
                'total_return': results['validation'].total_return,
                'sharpe_ratio': results['validation'].sharpe_ratio,
                'num_trades': len(results['validation'].trades)
            }
        },
        'statistical_significance': {
            split: {
                't_statistic': result.t_statistic,
                'p_value': result.p_value,
                'confidence_interval': result.confidence_interval,
                'significant': result.p_value < 0.05
            }
            for split, result in results.items()
        },
        'look_ahead_bias_checks': {
            split: {
                'alignment_issues': result.alignment_issues,
                'feature_future_leakage': result.feature_future_leakage,
                'max_leakage': max(result.feature_future_leakage.values()) 
                              if result.feature_future_leakage else 0
            }
            for split, result in results.items()
        },
        'consistency_check': {
            'performance_degradation': (results['test'].total_return - results['train'].total_return) 
                                     / abs(results['train'].total_return) 
                                     if results['train'].total_return != 0 else 0,
            'sharpe_degradation': (results['test'].sharpe_ratio - results['train'].sharpe_ratio) 
                                / abs(results['train'].sharpe_ratio)
                                if results['train'].sharpe_ratio != 0 else 0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    logger.info(f"Validation report saved to {output_path}")