# Step 2.5: Walk-Forward Foundation

**Status**: Critical Foundation Step
**Complexity**: Medium
**Prerequisites**: [Step 2: Risk Container](step-02-risk-container.md) completed
**Architecture Ref**: [MULTIPHASE_OPTIMIZATION.md](../../MULTIPHASE_OPTIMIZATION.md)

## 🎯 Objective

Implement the critical walk-forward data splitting infrastructure that:
- Splits data into train/validation/test periods
- Enables proper out-of-sample validation
- Prevents look-ahead bias
- Supports rolling window optimization
- Forms the foundation for all optimization workflows

## ⚠️ Critical Importance

**This step is ESSENTIAL** - Without proper walk-forward splitting:
- Optimization results will be overfit
- Backtest results will be unrealistic
- Strategy performance will degrade in production
- Risk estimates will be incorrect

## 📋 Required Reading

Before starting:
1. [Walk-Forward Analysis](../../strategy/optimization/walk_forward.py)
2. [MULTIPHASE_OPTIMIZATION.md](../../MULTIPHASE_OPTIMIZATION.md#data-splitting)
3. [Optimization Reproducibility](../validation-framework/optimization-reproducibility.md)

## 🏗️ Implementation Tasks

### 1. Data Splitting Infrastructure

```python
# src/data/splitting.py
class WalkForwardSplitter:
    """
    Splits data into train/validation/test periods for walk-forward analysis.
    Ensures no look-ahead bias and proper out-of-sample testing.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.train_period = config.train_period
        self.validation_period = config.validation_period
        self.test_period = config.test_period
        self.step_size = config.step_size
        self.min_train_size = config.min_train_size
        self.logger = ComponentLogger("WalkForwardSplitter", "global")
    
    def split_data(self, data: pd.DataFrame) -> List[DataSplit]:
        """Split data into walk-forward windows"""
        splits = []
        
        # Calculate total periods
        total_days = len(data)
        window_size = self.train_period + self.validation_period + self.test_period
        
        # Ensure we have enough data
        if total_days < window_size:
            raise ValueError(
                f"Insufficient data: {total_days} days < "
                f"{window_size} days required"
            )
        
        # Generate splits
        start_idx = 0
        split_num = 0
        
        while start_idx + window_size <= total_days:
            # Define period boundaries
            train_end = start_idx + self.train_period
            val_end = train_end + self.validation_period
            test_end = val_end + self.test_period
            
            # Create split
            split = DataSplit(
                split_id=f"split_{split_num}",
                train_data=data.iloc[start_idx:train_end],
                validation_data=data.iloc[train_end:val_end],
                test_data=data.iloc[val_end:test_end],
                train_start=data.index[start_idx],
                test_end=data.index[test_end - 1]
            )
            
            splits.append(split)
            
            # Log split info
            self.logger.info(
                f"Created split {split_num}: "
                f"train={split.train_start} to {split.train_end}, "
                f"test={split.test_start} to {split.test_end}"
            )
            
            # Move window forward
            start_idx += self.step_size
            split_num += 1
        
        return splits
    
    def validate_no_overlap(self, splits: List[DataSplit]) -> bool:
        """Ensure no data leakage between splits"""
        for i, split in enumerate(splits):
            # Check train doesn't overlap with future test
            for future_split in splits[i+1:]:
                if split.train_end >= future_split.test_start:
                    raise ValueError(
                        f"Data leakage detected: Split {i} train overlaps "
                        f"with future test data"
                    )
        return True
```

### 2. Walk-Forward Coordinator

```python
# src/core/coordinator/walk_forward_coordinator.py
class WalkForwardCoordinator:
    """
    Coordinates walk-forward optimization workflow.
    Manages data splits and optimization phases.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.splitter = None
        self.current_split = None
        self.results = []
        self.logger = ComponentLogger("WalkForwardCoordinator", container_id)
    
    def initialize(self, data: pd.DataFrame, config: Dict) -> None:
        """Initialize walk-forward analysis"""
        # Create splitter
        wf_config = WalkForwardConfig(**config['walk_forward'])
        self.splitter = WalkForwardSplitter(wf_config)
        
        # Generate splits
        self.splits = self.splitter.split_data(data)
        self.splitter.validate_no_overlap(self.splits)
        
        self.logger.info(
            f"Initialized walk-forward with {len(self.splits)} splits"
        )
    
    def run_optimization(self, optimizer_factory: Callable) -> WalkForwardResults:
        """Run optimization across all splits"""
        all_results = []
        
        for split in self.splits:
            self.logger.info(f"Processing {split.split_id}")
            
            # Create optimizer for this split
            optimizer = optimizer_factory(split.split_id)
            
            # Phase 1: Optimize on training data
            train_results = optimizer.optimize(
                split.train_data,
                phase="train"
            )
            
            # Phase 2: Validate on validation data
            val_results = optimizer.validate(
                split.validation_data,
                train_results.best_params,
                phase="validation"
            )
            
            # Phase 3: Test on out-of-sample data
            test_results = optimizer.test(
                split.test_data,
                train_results.best_params,
                phase="test"
            )
            
            # Store results
            split_results = SplitResults(
                split_id=split.split_id,
                train_results=train_results,
                validation_results=val_results,
                test_results=test_results,
                params_used=train_results.best_params
            )
            
            all_results.append(split_results)
            
            # Log performance degradation
            self._log_performance_degradation(split_results)
        
        return WalkForwardResults(
            splits=self.splits,
            results=all_results,
            summary=self._calculate_summary(all_results)
        )
    
    def _log_performance_degradation(self, results: SplitResults) -> None:
        """Log train vs test performance to detect overfitting"""
        train_sharpe = results.train_results.sharpe_ratio
        test_sharpe = results.test_results.sharpe_ratio
        degradation = (train_sharpe - test_sharpe) / train_sharpe * 100
        
        self.logger.warning(
            f"Performance degradation in {results.split_id}: "
            f"Train Sharpe={train_sharpe:.2f}, Test Sharpe={test_sharpe:.2f}, "
            f"Degradation={degradation:.1f}%"
        )
```

### 3. Data Integrity Validation

```python
# src/data/integrity.py
class DataIntegrityValidator:
    """Ensures data integrity for walk-forward analysis"""
    
    @staticmethod
    def validate_chronological(data: pd.DataFrame) -> None:
        """Ensure data is properly sorted by time"""
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically")
    
    @staticmethod
    def validate_no_gaps(data: pd.DataFrame, frequency: str) -> None:
        """Check for missing time periods"""
        expected_periods = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq=frequency
        )
        
        missing = expected_periods.difference(data.index)
        if len(missing) > 0:
            raise ValueError(
                f"Missing {len(missing)} periods in data: "
                f"First missing: {missing[0]}"
            )
    
    @staticmethod
    def validate_sufficient_data(data: pd.DataFrame, 
                               min_periods: int) -> None:
        """Ensure enough data for analysis"""
        if len(data) < min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} < {min_periods} required"
            )
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step2_5_walk_forward.py`:

```python
class TestWalkForwardSplitter:
    """Test data splitting logic"""
    
    def test_basic_split(self):
        """Test basic walk-forward split"""
        # Create 365 days of data
        data = create_daily_data(365)
        
        config = WalkForwardConfig(
            train_period=200,
            validation_period=50,
            test_period=50,
            step_size=50
        )
        
        splitter = WalkForwardSplitter(config)
        splits = splitter.split_data(data)
        
        # Should have 2 splits
        assert len(splits) == 2
        
        # Check first split sizes
        assert len(splits[0].train_data) == 200
        assert len(splits[0].validation_data) == 50
        assert len(splits[0].test_data) == 50
    
    def test_no_look_ahead_bias(self):
        """Ensure no future data in training"""
        data = create_daily_data(400)
        config = WalkForwardConfig(100, 50, 50, 50)
        
        splitter = WalkForwardSplitter(config)
        splits = splitter.split_data(data)
        
        # Verify no overlap
        assert splitter.validate_no_overlap(splits)
        
        # Check specific dates
        for split in splits:
            assert split.train_end < split.validation_start
            assert split.validation_end < split.test_start
```

### Integration Tests

Create `tests/integration/test_step2_5_walk_forward_integration.py`:

```python
def test_walk_forward_with_optimization():
    """Test complete walk-forward optimization workflow"""
    # Setup
    data = SyntheticDataGenerator.create_trending_with_regime_changes()
    
    config = {
        'walk_forward': {
            'train_period': 250,
            'validation_period': 50,
            'test_period': 50,
            'step_size': 50
        },
        'optimization': {
            'param_grid': {
                'fast_period': [10, 20, 30],
                'slow_period': [40, 50, 60]
            }
        }
    }
    
    # Create coordinator
    coordinator = WalkForwardCoordinator("test_wf")
    coordinator.initialize(data, config)
    
    # Run optimization
    results = coordinator.run_optimization(
        lambda split_id: GridSearchOptimizer(config['optimization'])
    )
    
    # Verify results
    assert len(results.results) > 0
    
    # Check performance degradation
    for split_result in results.results:
        train_sharpe = split_result.train_results.sharpe_ratio
        test_sharpe = split_result.test_results.sharpe_ratio
        # Test performance should be lower but not terrible
        assert test_sharpe > train_sharpe * 0.5
```

### System Tests

Create `tests/system/test_step2_5_complete_walk_forward.py`:

```python
def test_multi_year_walk_forward():
    """Test walk-forward over multiple years"""
    # Create 5 years of data
    data = create_multi_year_data(years=5)
    
    # Configure walk-forward
    config = {
        'walk_forward': {
            'train_period': 504,  # 2 years
            'validation_period': 126,  # 6 months
            'test_period': 126,  # 6 months  
            'step_size': 63  # 3 months
        }
    }
    
    # Run complete system
    system = create_complete_system_with_walk_forward(config)
    results = system.run(data)
    
    # Verify consistency
    sharpe_ratios = [r.test_results.sharpe_ratio for r in results.results]
    assert np.std(sharpe_ratios) < 0.5  # Reasonable consistency
    
    # Verify no data leakage
    for result in results.results:
        assert result.test_results.num_trades > 0
        assert result.test_results.is_out_of_sample
```

## ✅ Validation Checklist

### Data Splitting Validation
- [ ] Splits have correct sizes
- [ ] No overlap between train/test
- [ ] Chronological order maintained
- [ ] No look-ahead bias possible

### Walk-Forward Validation
- [ ] All splits processed correctly
- [ ] Parameters optimized on train only
- [ ] Validation used for selection
- [ ] Test truly out-of-sample

### Performance Validation
- [ ] Reasonable train/test degradation
- [ ] Consistent results across splits
- [ ] No catastrophic overfitting
- [ ] Reproducible results

### Testing Validation
- [ ] Unit tests for splitting logic
- [ ] Integration tests for workflow
- [ ] System tests for multi-year
- [ ] Edge cases handled

## 📊 Memory & Performance

### Memory Optimization
```python
class MemoryEfficientSplitter:
    """Memory-efficient data splitting using views"""
    
    def create_split_view(self, data: pd.DataFrame, 
                         start: int, end: int) -> pd.DataFrame:
        """Create view instead of copy to save memory"""
        return data.iloc[start:end]  # View, not copy
```

### Performance Benchmarks
- Split generation: < 100ms for 10 years data
- Memory overhead: < 10% of data size
- Parallel split processing supported

## 🐛 Common Issues

1. **Insufficient Data**
   - Ensure enough data for all splits
   - Adjust periods if needed
   - Consider shorter validation/test

2. **Overfitting Detection**
   - Monitor train/test gap
   - Use validation for early stopping
   - Consider parameter stability

3. **Memory with Large Datasets**
   - Use data views not copies
   - Process splits sequentially
   - Clear results after saving

## 🎯 Success Criteria

Step 2.5 is complete when:
1. ✅ Walk-forward splitter implemented
2. ✅ No data leakage possible
3. ✅ Coordinator manages workflow
4. ✅ All test tiers pass
5. ✅ Overfitting detection working

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 3: Classifier Container](../02-container-architecture/step-03-classifier-container.md)

## 📚 Additional Resources

- [Walk-Forward Theory](https://en.wikipedia.org/wiki/Walk_forward_optimization)
- [Preventing Overfitting](../best-practices/overfitting-prevention.md)
- [Time Series Cross-Validation](../references/time-series-cv.md)