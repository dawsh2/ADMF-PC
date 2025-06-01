# Step 8.5: Statistical Validation (Monte Carlo & Bootstrap)

**Status**: Signal Capture & Replay Step
**Complexity**: High
**Prerequisites**: [Step 8: Signal Replay](step-08-signal-replay.md) completed
**Architecture Ref**: [Statistical Testing Guide](../references/statistical-testing.md)

## 🎯 Objective

Implement statistical validation methods:
- Monte Carlo simulation for strategy robustness
- Bootstrap methods for confidence intervals
- Statistical significance testing
- Performance distribution analysis
- Overfitting detection and prevention

## 📋 Required Reading

Before starting:
1. [Monte Carlo Methods in Finance](../references/monte-carlo-finance.md)
2. [Bootstrap Techniques](../references/bootstrap-methods.md)
3. [Statistical Significance in Trading](../references/statistical-significance.md)

## 🏗️ Implementation Tasks

### 1. Monte Carlo Simulation Engine

```python
# src/analysis/monte_carlo.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple, Callable
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 1000
    confidence_levels: List[float] = None
    random_seed: Optional[int] = None
    
    # Simulation methods
    resample_returns: bool = True
    shuffle_signals: bool = False
    randomize_parameters: bool = False
    synthetic_paths: bool = False
    
    # Parameter ranges for randomization
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Performance
    use_parallel: bool = True
    n_workers: Optional[int] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]
        if self.use_parallel and self.n_workers is None:
            self.n_workers = mp.cpu_count() - 1

class MonteCarloValidator:
    """
    Validates strategy robustness using Monte Carlo methods.
    Tests multiple sources of randomness.
    """
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.simulation_results: List[SimulationResult] = []
        self.logger = ComponentLogger("MonteCarloValidator", "validation")
        
        if config.random_seed:
            np.random.seed(config.random_seed)
    
    def validate_strategy(self, backtest_results: BacktestResults,
                        captured_signals: Optional[pd.DataFrame] = None) -> ValidationReport:
        """Run comprehensive Monte Carlo validation"""
        self.logger.info(
            f"Starting Monte Carlo validation with {self.config.n_simulations} simulations"
        )
        
        # Extract base data
        returns = self._extract_returns(backtest_results)
        trades = self._extract_trades(backtest_results)
        
        # Run different validation methods
        validation_results = {}
        
        if self.config.resample_returns:
            validation_results['return_resampling'] = \
                self._validate_return_resampling(returns)
        
        if self.config.shuffle_signals and captured_signals is not None:
            validation_results['signal_shuffling'] = \
                self._validate_signal_shuffling(captured_signals)
        
        if self.config.randomize_parameters:
            validation_results['parameter_randomization'] = \
                self._validate_parameter_stability(backtest_results)
        
        if self.config.synthetic_paths:
            validation_results['synthetic_paths'] = \
                self._validate_synthetic_paths(returns)
        
        # Aggregate results
        report = self._create_validation_report(validation_results, backtest_results)
        
        return report
    
    def _validate_return_resampling(self, returns: pd.Series) -> Dict:
        """Resample returns to test consistency"""
        simulation_results = []
        
        # Parallel or sequential execution
        if self.config.use_parallel:
            with mp.Pool(self.config.n_workers) as pool:
                tasks = [
                    (returns, self.config.random_seed + i if self.config.random_seed else None)
                    for i in range(self.config.n_simulations)
                ]
                simulation_results = pool.starmap(self._single_return_resample, tasks)
        else:
            for i in range(self.config.n_simulations):
                result = self._single_return_resample(
                    returns, 
                    self.config.random_seed + i if self.config.random_seed else None
                )
                simulation_results.append(result)
        
        # Calculate statistics
        sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
        total_returns = [r['total_return'] for r in simulation_results]
        max_drawdowns = [r['max_drawdown'] for r in simulation_results]
        
        return {
            'sharpe_distribution': self._calculate_distribution_stats(sharpe_ratios),
            'return_distribution': self._calculate_distribution_stats(total_returns),
            'drawdown_distribution': self._calculate_distribution_stats(max_drawdowns),
            'confidence_intervals': self._calculate_confidence_intervals({
                'sharpe_ratio': sharpe_ratios,
                'total_return': total_returns,
                'max_drawdown': max_drawdowns
            })
        }
    
    def _single_return_resample(self, returns: pd.Series, 
                              random_seed: Optional[int]) -> Dict:
        """Single return resampling simulation"""
        if random_seed:
            np.random.seed(random_seed)
        
        # Resample returns with replacement
        resampled_indices = np.random.choice(
            len(returns), size=len(returns), replace=True
        )
        resampled_returns = returns.iloc[resampled_indices].reset_index(drop=True)
        
        # Calculate metrics
        cumulative_returns = (1 + resampled_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Sharpe ratio
        sharpe_ratio = np.sqrt(252) * resampled_returns.mean() / resampled_returns.std() \
                      if resampled_returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }
    
    def _validate_signal_shuffling(self, signals: pd.DataFrame) -> Dict:
        """Test randomness of signal timing"""
        replay_engine = SignalReplayEngine(MemoryStorageBackend())
        replay_engine.loaded_signals = signals
        
        simulation_results = []
        
        for i in range(self.config.n_simulations):
            # Shuffle signal order while preserving dates
            shuffled_signals = self._shuffle_signals_preserving_dates(signals, i)
            
            # Replay with shuffled signals
            replay_engine.loaded_signals = shuffled_signals
            result = replay_engine.replay(ReplayConfig())
            
            simulation_results.append({
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'win_rate': result.win_rate
            })
        
        # Compare to original
        original_result = replay_engine.replay(ReplayConfig())
        
        return {
            'original_sharpe': original_result.sharpe_ratio,
            'shuffled_sharpe_mean': np.mean([r['sharpe_ratio'] for r in simulation_results]),
            'p_value': self._calculate_p_value(simulation_results, original_result),
            'is_random': self._test_randomness(simulation_results, original_result)
        }
    
    def _validate_parameter_stability(self, backtest_results: BacktestResults) -> Dict:
        """Test sensitivity to parameter variations"""
        if not self.config.param_ranges:
            raise ValueError("param_ranges required for parameter randomization")
        
        simulation_results = []
        base_params = backtest_results.strategy_params
        
        for i in range(self.config.n_simulations):
            # Randomize parameters within ranges
            randomized_params = self._randomize_parameters(
                base_params, self.config.param_ranges, i
            )
            
            # Run backtest with new parameters
            # (This requires access to original data and strategy)
            # For demonstration, we'll simulate the results
            simulated_result = self._simulate_parameter_result(
                randomized_params, backtest_results
            )
            
            simulation_results.append(simulated_result)
        
        # Analyze stability
        return self._analyze_parameter_stability(simulation_results, backtest_results)
```

### 2. Bootstrap Methods

```python
# src/analysis/bootstrap.py
class BootstrapValidator:
    """
    Implements bootstrap methods for confidence interval estimation.
    Provides non-parametric statistical inference.
    """
    
    def __init__(self, n_bootstrap: int = 10000, 
                 confidence_levels: List[float] = None,
                 random_seed: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.random_seed = random_seed
        self.logger = ComponentLogger("BootstrapValidator", "validation")
        
        if random_seed:
            np.random.seed(random_seed)
    
    def calculate_confidence_intervals(self, 
                                     data: Union[pd.Series, np.ndarray],
                                     statistic: Callable,
                                     method: str = 'percentile') -> Dict[float, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for any statistic"""
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        
        # Bootstrap resampling
        bootstrap_statistics = []
        n = len(data)
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            resample_indices = np.random.choice(n, size=n, replace=True)
            resampled_data = data[resample_indices]
            
            # Calculate statistic
            stat_value = statistic(resampled_data)
            bootstrap_statistics.append(stat_value)
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for conf_level in self.confidence_levels:
            if method == 'percentile':
                ci = self._percentile_method(bootstrap_statistics, conf_level)
            elif method == 'bca':  # Bias-corrected and accelerated
                ci = self._bca_method(data, bootstrap_statistics, statistic, conf_level)
            elif method == 'studentized':
                ci = self._studentized_method(data, bootstrap_statistics, statistic, conf_level)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            confidence_intervals[conf_level] = ci
        
        return confidence_intervals
    
    def _percentile_method(self, bootstrap_stats: np.ndarray, 
                         conf_level: float) -> Tuple[float, float]:
        """Simple percentile method"""
        alpha = 1 - conf_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return (
            np.percentile(bootstrap_stats, lower_percentile),
            np.percentile(bootstrap_stats, upper_percentile)
        )
    
    def _bca_method(self, original_data: np.ndarray,
                   bootstrap_stats: np.ndarray,
                   statistic: Callable,
                   conf_level: float) -> Tuple[float, float]:
        """Bias-corrected and accelerated bootstrap"""
        # Calculate original statistic
        theta_hat = statistic(original_data)
        
        # Calculate bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))
        
        # Calculate acceleration
        jackknife_stats = self._jackknife(original_data, statistic)
        jackknife_mean = np.mean(jackknife_stats)
        
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # Calculate adjusted percentiles
        alpha = 1 - conf_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
        
        alpha_lower = stats.norm.cdf(
            z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower))
        )
        alpha_upper = stats.norm.cdf(
            z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper))
        )
        
        return (
            np.percentile(bootstrap_stats, alpha_lower * 100),
            np.percentile(bootstrap_stats, alpha_upper * 100)
        )
    
    def _jackknife(self, data: np.ndarray, statistic: Callable) -> np.ndarray:
        """Jackknife resampling for bias and variance estimation"""
        n = len(data)
        jackknife_stats = np.zeros(n)
        
        for i in range(n):
            # Leave one out
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)
        
        return jackknife_stats
    
    def validate_strategy_metrics(self, trades: List[Trade]) -> BootstrapReport:
        """Comprehensive bootstrap validation of strategy metrics"""
        # Extract trade returns
        trade_returns = np.array([trade.pnl_percentage for trade in trades])
        
        # Define statistics of interest
        def sharpe_ratio(returns):
            return np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        def win_rate(returns):
            return np.mean(returns > 0)
        
        def profit_factor(returns):
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            return np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else np.inf
        
        def max_drawdown(returns):
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
        
        # Calculate confidence intervals
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }
        
        confidence_intervals = {}
        point_estimates = {}
        
        for metric_name, metric_func in metrics.items():
            # Point estimate
            point_estimates[metric_name] = metric_func(trade_returns)
            
            # Confidence intervals
            confidence_intervals[metric_name] = self.calculate_confidence_intervals(
                trade_returns, metric_func, method='bca'
            )
        
        return BootstrapReport(
            point_estimates=point_estimates,
            confidence_intervals=confidence_intervals,
            n_bootstrap=self.n_bootstrap,
            n_trades=len(trades)
        )
```

### 3. Statistical Significance Testing

```python
# src/analysis/significance_testing.py
class SignificanceTester:
    """
    Tests statistical significance of trading strategies.
    Implements multiple hypothesis testing corrections.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.test_results: List[TestResult] = []
        self.logger = ComponentLogger("SignificanceTester", "validation")
    
    def test_strategy_significance(self, 
                                 strategy_returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None) -> SignificanceResult:
        """Test if strategy returns are statistically significant"""
        tests_performed = {}
        
        # Test 1: Returns different from zero
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
        tests_performed['returns_nonzero'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'mean_return': strategy_returns.mean(),
            'std_return': strategy_returns.std()
        }
        
        # Test 2: Sharpe ratio significance
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / len(strategy_returns))
        z_stat = sharpe / sharpe_se
        p_value_sharpe = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        tests_performed['sharpe_significant'] = {
            'sharpe_ratio': sharpe,
            'standard_error': sharpe_se,
            'z_statistic': z_stat,
            'p_value': p_value_sharpe,
            'significant': p_value_sharpe < self.significance_level
        }
        
        # Test 3: Compare to benchmark if provided
        if benchmark_returns is not None:
            excess_returns = strategy_returns - benchmark_returns
            t_stat_excess, p_value_excess = stats.ttest_1samp(excess_returns, 0)
            
            tests_performed['outperforms_benchmark'] = {
                't_statistic': t_stat_excess,
                'p_value': p_value_excess,
                'significant': p_value_excess < self.significance_level,
                'mean_excess': excess_returns.mean()
            }
        
        # Test 4: Randomness test (runs test)
        runs_p_value = self._runs_test(strategy_returns > 0)
        tests_performed['returns_random'] = {
            'p_value': runs_p_value,
            'random': runs_p_value > self.significance_level
        }
        
        # Apply multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(tests_performed)
        
        return SignificanceResult(
            tests_performed=tests_performed,
            corrected_results=corrected_results,
            overall_significant=self._determine_overall_significance(corrected_results)
        )
    
    def _runs_test(self, binary_sequence: pd.Series) -> float:
        """Wald-Wolfowitz runs test for randomness"""
        n1 = sum(binary_sequence)
        n2 = len(binary_sequence) - n1
        
        if n1 == 0 or n2 == 0:
            return 1.0  # Can't test
        
        # Count runs
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence.iloc[i] != binary_sequence.iloc[i-1]:
                runs += 1
        
        # Expected runs and variance
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                   ((n1 + n2)**2 * (n1 + n2 - 1))
        
        if variance == 0:
            return 1.0
        
        # Z-statistic
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return p_value
    
    def _apply_multiple_testing_correction(self, test_results: Dict) -> Dict:
        """Apply Bonferroni or FDR correction"""
        p_values = [test['p_value'] for test in test_results.values()]
        
        # Bonferroni correction
        bonferroni_alpha = self.significance_level / len(p_values)
        
        # Benjamini-Hochberg FDR
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
        fdr_significant = []
        
        for i, (original_idx, p_val) in enumerate(sorted_p):
            threshold = (i + 1) * self.significance_level / len(p_values)
            if p_val <= threshold:
                fdr_significant.append(original_idx)
        
        # Apply corrections
        corrected_results = {}
        for i, (test_name, test_result) in enumerate(test_results.items()):
            corrected_results[test_name] = {
                **test_result,
                'bonferroni_significant': test_result['p_value'] < bonferroni_alpha,
                'fdr_significant': i in fdr_significant
            }
        
        return corrected_results
```

### 4. Overfitting Detection

```python
# src/analysis/overfitting_detection.py
class OverfittingDetector:
    """
    Detects potential overfitting in strategy development.
    Implements multiple detection methods.
    """
    
    def __init__(self):
        self.detection_results: List[DetectionResult] = []
        self.logger = ComponentLogger("OverfittingDetector", "validation")
    
    def detect_overfitting(self, 
                         in_sample_results: BacktestResults,
                         out_sample_results: BacktestResults,
                         strategy_complexity: int) -> OverfittingReport:
        """Comprehensive overfitting detection"""
        detections = {}
        
        # Method 1: Performance degradation
        degradation = self._calculate_performance_degradation(
            in_sample_results, out_sample_results
        )
        detections['performance_degradation'] = degradation
        
        # Method 2: Complexity penalty (AIC/BIC)
        complexity_scores = self._calculate_complexity_scores(
            in_sample_results, strategy_complexity
        )
        detections['complexity_penalty'] = complexity_scores
        
        # Method 3: Variance ratio test
        variance_test = self._variance_ratio_test(
            in_sample_results, out_sample_results
        )
        detections['variance_ratio'] = variance_test
        
        # Method 4: Cross-validation stability
        cv_stability = self._cross_validation_stability(
            in_sample_results, out_sample_results
        )
        detections['cv_stability'] = cv_stability
        
        # Aggregate detection
        overfitting_score = self._calculate_overfitting_score(detections)
        
        return OverfittingReport(
            detections=detections,
            overfitting_score=overfitting_score,
            is_overfit=overfitting_score > 0.7,
            recommendations=self._generate_recommendations(detections)
        )
    
    def _calculate_performance_degradation(self, 
                                         in_sample: BacktestResults,
                                         out_sample: BacktestResults) -> Dict:
        """Calculate performance degradation metrics"""
        metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']
        degradation = {}
        
        for metric in metrics:
            in_value = getattr(in_sample, metric)
            out_value = getattr(out_sample, metric)
            
            if in_value != 0:
                pct_degradation = (in_value - out_value) / abs(in_value) * 100
            else:
                pct_degradation = -100 if out_value < 0 else 0
            
            degradation[metric] = {
                'in_sample': in_value,
                'out_sample': out_value,
                'degradation_pct': pct_degradation,
                'severe_degradation': pct_degradation > 50
            }
        
        # Overall degradation score
        avg_degradation = np.mean([d['degradation_pct'] for d in degradation.values()])
        degradation['overall_score'] = avg_degradation
        degradation['likely_overfit'] = avg_degradation > 30
        
        return degradation
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step8_5_statistical_validation.py`:

```python
class TestMonteCarloValidation:
    """Test Monte Carlo simulation methods"""
    
    def test_return_resampling(self):
        """Test return resampling preserves properties"""
        # Create test returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Run Monte Carlo
        validator = MonteCarloValidator(MonteCarloConfig(
            n_simulations=100,
            random_seed=42
        ))
        
        result = validator._validate_return_resampling(returns)
        
        # Check distribution properties
        sharpe_dist = result['sharpe_distribution']
        assert 0.5 < sharpe_dist['mean'] < 1.5  # Reasonable Sharpe
        assert sharpe_dist['std'] > 0  # Some variation
        assert len(result['confidence_intervals']['sharpe_ratio']) == 2

class TestBootstrapMethods:
    """Test bootstrap confidence intervals"""
    
    def test_percentile_method(self):
        """Test basic percentile bootstrap"""
        data = np.random.normal(0, 1, 1000)
        bootstrap = BootstrapValidator(n_bootstrap=1000, random_seed=42)
        
        # Calculate CI for mean
        ci = bootstrap.calculate_confidence_intervals(
            data, 
            lambda x: np.mean(x),
            method='percentile'
        )
        
        # Check 95% CI contains true mean (0)
        assert ci[0.95][0] < 0 < ci[0.95][1]
        
    def test_bca_method(self):
        """Test bias-corrected accelerated bootstrap"""
        # Create skewed data
        data = np.random.exponential(1, 1000) - 1
        bootstrap = BootstrapValidator(n_bootstrap=1000)
        
        # BCA should handle skewness better
        ci_percentile = bootstrap.calculate_confidence_intervals(
            data, np.mean, method='percentile'
        )
        ci_bca = bootstrap.calculate_confidence_intervals(
            data, np.mean, method='bca'
        )
        
        # BCA intervals should be different (adjusted for bias)
        assert ci_bca[0.95] != ci_percentile[0.95]
```

### Integration Tests

Create `tests/integration/test_step8_5_validation_integration.py`:

```python
def test_complete_statistical_validation():
    """Test full statistical validation pipeline"""
    # Run backtest
    strategy = create_test_strategy()
    backtest_results = run_backtest(strategy, test_data)
    
    # Capture signals
    capture_engine = SignalCaptureEngine(test_config())
    captured_signals = capture_signals(strategy, test_data)
    
    # Monte Carlo validation
    mc_validator = MonteCarloValidator(MonteCarloConfig(
        n_simulations=100,
        resample_returns=True,
        shuffle_signals=True
    ))
    
    mc_report = mc_validator.validate_strategy(
        backtest_results, captured_signals
    )
    
    # Bootstrap validation
    bootstrap_validator = BootstrapValidator(n_bootstrap=1000)
    bootstrap_report = bootstrap_validator.validate_strategy_metrics(
        backtest_results.trades
    )
    
    # Significance testing
    sig_tester = SignificanceTester()
    sig_result = sig_tester.test_strategy_significance(
        backtest_results.returns
    )
    
    # Verify all validations complete
    assert mc_report.is_robust is not None
    assert bootstrap_report.confidence_intervals is not None
    assert sig_result.overall_significant is not None

def test_overfitting_detection_workflow():
    """Test overfitting detection with in/out sample"""
    # Split data
    full_data = load_test_data()
    split_point = len(full_data) // 2
    in_sample = full_data[:split_point]
    out_sample = full_data[split_point:]
    
    # Optimize on in-sample
    optimizer = create_optimizer()
    optimized_params = optimizer.optimize(in_sample)
    
    # Test on both samples
    strategy_optimized = create_strategy(optimized_params)
    in_sample_results = run_backtest(strategy_optimized, in_sample)
    out_sample_results = run_backtest(strategy_optimized, out_sample)
    
    # Detect overfitting
    detector = OverfittingDetector()
    overfitting_report = detector.detect_overfitting(
        in_sample_results,
        out_sample_results,
        strategy_complexity=len(optimized_params)
    )
    
    # Verify detection works
    assert overfitting_report.overfitting_score >= 0
    assert overfitting_report.recommendations is not None
```

### System Tests

Create `tests/system/test_step8_5_statistical_system.py`:

```python
def test_multi_strategy_statistical_validation():
    """Test statistical validation across multiple strategies"""
    strategies = ['momentum', 'mean_reversion', 'trend_following']
    validation_results = {}
    
    for strategy_name in strategies:
        # Run strategy
        strategy = create_strategy(strategy_name)
        results = run_full_backtest(strategy, historical_data)
        
        # Complete validation suite
        validation_suite = StatisticalValidationSuite()
        validation_results[strategy_name] = validation_suite.validate(
            results,
            methods=['monte_carlo', 'bootstrap', 'significance', 'overfitting']
        )
    
    # Compare strategies statistically
    comparison = compare_strategies_statistically(validation_results)
    
    # Verify statistical differentiation
    assert comparison.best_strategy is not None
    assert comparison.confidence_in_ranking > 0.8
    assert all(v.is_statistically_valid for v in validation_results.values())

def test_walk_forward_statistical_consistency():
    """Test statistical properties across walk-forward windows"""
    # Run walk-forward analysis
    wf_results = run_walk_forward_analysis(
        strategy=create_test_strategy(),
        data=load_long_history(),
        window_size=252,
        step_size=63
    )
    
    # Validate each window
    window_validations = []
    for window_result in wf_results.windows:
        validator = MonteCarloValidator(MonteCarloConfig(n_simulations=100))
        validation = validator.validate_strategy(window_result)
        window_validations.append(validation)
    
    # Check consistency across windows
    sharpe_variations = [
        v.metrics['sharpe_distribution']['std'] 
        for v in window_validations
    ]
    
    # Sharpe shouldn't vary too much across windows
    assert np.std(sharpe_variations) < 0.5
    
    # Most windows should be statistically significant
    significant_windows = sum(
        1 for v in window_validations if v.is_statistically_significant
    )
    assert significant_windows / len(window_validations) > 0.7
```

## ✅ Validation Checklist

### Monte Carlo Validation
- [ ] Return resampling implemented
- [ ] Signal shuffling working
- [ ] Parameter randomization tested
- [ ] Synthetic paths generated
- [ ] Results reproducible with seed

### Bootstrap Methods
- [ ] Percentile method accurate
- [ ] BCa method implemented
- [ ] Confidence intervals reasonable
- [ ] Multiple statistics supported

### Statistical Significance
- [ ] Hypothesis tests correct
- [ ] Multiple testing corrected
- [ ] Sharpe ratio significance tested
- [ ] Benchmark comparison working

### Overfitting Detection
- [ ] Performance degradation measured
- [ ] Complexity penalties applied
- [ ] Cross-validation stable
- [ ] Recommendations generated

## 📊 Performance Optimization

### Parallel Processing
```python
class ParallelMonteCarloValidator:
    """Optimized parallel Monte Carlo implementation"""
    
    def parallel_validation(self, n_simulations: int, n_workers: int):
        with mp.Pool(n_workers) as pool:
            # Distribute simulations
            chunk_size = n_simulations // n_workers
            tasks = [
                (chunk_size, seed) 
                for seed in range(n_workers)
            ]
            
            # Run parallel
            results = pool.starmap(self._run_simulation_chunk, tasks)
            
            # Aggregate results
            return self._aggregate_results(results)
```

### Memory Efficiency
- Stream large datasets
- Use generators for simulations
- Compress intermediate results
- Clean up after each simulation

## 🐛 Common Issues

1. **Random Seed Management**
   - Always set seeds for reproducibility
   - Use different seeds for each simulation
   - Document seed usage

2. **Computational Intensity**
   - Start with fewer simulations for testing
   - Use parallel processing
   - Consider cloud computing for large runs

3. **Statistical Misinterpretation**
   - Understand confidence interval meaning
   - Don't overinterpret p-values
   - Consider practical significance

## 🎯 Success Criteria

Step 8.5 is complete when:
1. ✅ Monte Carlo validation functional
2. ✅ Bootstrap methods implemented
3. ✅ Statistical significance tested
4. ✅ Overfitting detection working
5. ✅ All test tiers pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 9: Parameter Expansion](../04-multi-phase-integration/step-09-parameter-expansion.md)

## 📚 Additional Resources

- [Monte Carlo Methods in Finance](../references/monte-carlo-finance.md)
- [Bootstrap Theory](../references/bootstrap-theory.md)
- [Multiple Testing Corrections](../references/multiple-testing.md)
- [Overfitting in Trading Systems](../references/overfitting-prevention.md)