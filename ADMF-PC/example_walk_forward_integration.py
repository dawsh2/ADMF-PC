"""
Example showing how refactored walk-forward validation integrates with ADMF-PC architecture.
Demonstrates the complete workflow with proper separation of concerns.
"""

import sys
sys.path.append('/Users/daws/ADMF-PC/ADMF-PC')

from typing import Dict, Any, List
import asyncio
from dataclasses import dataclass
import numpy as np

# Import core ADMF-PC components (simulated for example)
from src.strategy.optimization.walk_forward_refactored import (
    create_walk_forward_validator,
    DataProvider,
    BacktestExecutor
)

# Simulated ADMF-PC components
@dataclass
class MarketData:
    """Market data container."""
    symbols: List[str]
    prices: Dict[str, np.ndarray]
    timestamps: np.ndarray
    
    def get_length(self) -> int:
        return len(self.timestamps)


class ADMFDataProvider:
    """ADMF-PC compatible data provider."""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
    
    def get_slice(self, start: int, end: int) -> MarketData:
        """Get market data slice."""
        return MarketData(
            symbols=self.market_data.symbols,
            prices={
                symbol: prices[start:end] 
                for symbol, prices in self.market_data.prices.items()
            },
            timestamps=self.market_data.timestamps[start:end]
        )
    
    def get_length(self) -> int:
        """Get total data length."""
        return self.market_data.get_length()


class BacktestContainerFactory:
    """ADMF-PC Backtest Container Factory."""
    
    def __init__(self):
        self.containers_created = []
    
    def create_instance(self, config: Dict[str, Any]):
        """Create standardized backtest container."""
        container_id = config['container_id']
        
        print(f"\nCreating Backtest Container: {container_id}")
        print("  Following standardized pattern:")
        print("  1. Create Data Streamer")
        print("  2. Create Indicator Hub")
        print("  3. Create Classifier")
        print("  4. Create Risk & Portfolio Container")
        print("  5. Create Strategy")
        print("  6. Create Backtest Engine")
        print("  7. Wire Event Buses")
        
        # Return mock container for example
        return BacktestContainer(container_id, config)


class BacktestContainer:
    """Standardized backtest container."""
    
    def __init__(self, container_id: str, config: Dict[str, Any]):
        self.container_id = container_id
        self.config = config
        self.components = {}
        
        # Initialize components in standardized order
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components in standardized order."""
        # 1. Data layer
        self.components['data_streamer'] = f"DataStreamer_{self.container_id}"
        self.components['indicator_hub'] = f"IndicatorHub_{self.container_id}"
        
        # 2. Classifier layer
        self.components['classifier'] = f"Classifier_{self.container_id}"
        
        # 3. Risk & Portfolio container
        self.components['risk_portfolio'] = f"RiskPortfolio_{self.container_id}"
        
        # 4. Strategy
        strategy_config = self.config.get('strategy_config', {})
        self.components['strategy'] = f"Strategy_{strategy_config.get('class', 'Unknown')}"
        
        # 5. Execution layer
        self.components['backtest_engine'] = f"BacktestEngine_{self.container_id}"
    
    def dispose(self):
        """Clean disposal of all components."""
        print(f"  Disposing container {self.container_id}")
        self.components.clear()


class ADMFBacktestEngine:
    """ADMF-PC Backtest Engine."""
    
    def run(self, container: BacktestContainer) -> Dict[str, Any]:
        """Execute backtest in container."""
        print(f"  Executing backtest in {container.container_id}")
        
        # Extract strategy parameters
        strategy_params = container.config['strategy_config']['params']
        
        # Simulate backtest execution
        # In real implementation, this would:
        # 1. Stream data through indicator hub
        # 2. Generate classifier signals
        # 3. Apply risk & portfolio constraints
        # 4. Execute trades
        # 5. Calculate performance metrics
        
        # Mock results based on parameters
        lookback = strategy_params.get('lookback_period', 20)
        threshold = strategy_params.get('momentum_threshold', 0.02)
        
        # Simulate performance correlation with parameters
        base_sharpe = 1.2
        param_factor = (30 - lookback) * 0.02 + (0.025 - threshold) * 20
        
        return {
            'metrics': {
                'sharpe_ratio': base_sharpe + param_factor,
                'total_return': 0.15 + param_factor * 0.05,
                'max_drawdown': 0.12 - param_factor * 0.02,
                'win_rate': 0.52 + param_factor * 0.03,
                'calmar_ratio': (0.15 + param_factor * 0.05) / (0.12 - param_factor * 0.02)
            },
            'returns': np.random.normal(0.0005, 0.01, 252).tolist(),
            'trades': [{'id': i, 'symbol': 'AAPL'} for i in range(50)],
            'positions': []
        }


class ADMFOptimizer:
    """ADMF-PC compatible optimizer."""
    
    def __init__(self, method: str = 'grid'):
        self.method = method
        self.optimization_history = []
    
    def optimize(self, objective_func, parameter_space: Dict[str, List[Any]]):
        """Run optimization."""
        print("\n  Running Grid Search Optimization")
        
        best_score = -float('inf')
        best_params = None
        
        # Grid search
        for lookback in parameter_space['lookback_period']:
            for threshold in parameter_space['momentum_threshold']:
                params = {
                    'lookback_period': lookback,
                    'momentum_threshold': threshold
                }
                
                score = objective_func(params)
                print(f"    Testing params {params}: score = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def get_best_score(self):
        return self.best_score
    
    def get_optimization_history(self):
        return self.optimization_history


class CompositeObjective:
    """Composite objective combining multiple metrics."""
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
    
    def calculate(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted objective score."""
        score = 0.0
        for metric, weight in self.weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        return score


class ADMFCoordinator:
    """ADMF-PC Coordinator with all required components."""
    
    def __init__(self):
        self.checkpointing = CheckpointManager()
        self.container_naming = ContainerNaming()
        self.phase_manager = PhaseManager()


class CheckpointManager:
    """Manages checkpointing for walk-forward."""
    
    def __init__(self):
        self.checkpoints = {}
    
    def save_checkpoint(self, name: str, data: Any):
        """Save checkpoint data."""
        self.checkpoints[name] = data
        print(f"  ✓ Checkpoint saved: {name}")


class ContainerNaming:
    """Standardized container naming."""
    
    def generate_container_id(self, **kwargs) -> str:
        """Generate unique container ID."""
        # Format: {container_type}_{phase}_{classifier}_{risk_profile}_{timestamp}
        phase = kwargs.get('phase', 'unknown')
        period = kwargs.get('period', '')
        strategy = kwargs.get('strategy', 'unknown')
        
        # In real implementation, would include timestamp
        return f"backtest_{phase}_{period}_{strategy}"


class PhaseManager:
    """Manages optimization phases."""
    
    def transition_phase(self, from_phase: str, to_phase: str):
        """Handle phase transitions."""
        print(f"\nPhase transition: {from_phase} → {to_phase}")


async def run_integrated_walk_forward():
    """Run complete walk-forward validation with ADMF-PC architecture."""
    
    print("=" * 80)
    print("ADMF-PC Integrated Walk-Forward Validation Example")
    print("=" * 80)
    
    # Step 1: Create market data
    print("\nStep 1: Preparing Market Data")
    print("-" * 40)
    
    # Generate realistic market data
    np.random.seed(42)
    timestamps = np.arange(504)  # 2 years of trading days
    
    # Multi-symbol data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    prices = {}
    
    for symbol in symbols:
        # Generate correlated price series
        returns = np.random.normal(0.0005, 0.015, 504)
        price_series = 100 * np.exp(np.cumsum(returns))
        prices[symbol] = price_series
    
    market_data = MarketData(symbols=symbols, prices=prices, timestamps=timestamps)
    print(f"  Created market data: {len(symbols)} symbols, {len(timestamps)} days")
    
    # Step 2: Create ADMF-PC components
    print("\nStep 2: Creating ADMF-PC Components")
    print("-" * 40)
    
    # Data provider
    data_provider = ADMFDataProvider(market_data)
    
    # Coordinator
    coordinator = ADMFCoordinator()
    
    # Container factory
    container_factory = BacktestContainerFactory()
    
    # Backtest engine
    backtest_engine = ADMFBacktestEngine()
    
    # Optimizer with composite objective
    optimizer = ADMFOptimizer(method='grid')
    objective = CompositeObjective(weights={
        'sharpe_ratio': 0.6,
        'calmar_ratio': 0.3,
        'win_rate': 0.1
    })
    
    print("  ✓ Created all ADMF-PC components")
    
    # Step 3: Configure walk-forward validation
    print("\nStep 3: Configuring Walk-Forward Validation")
    print("-" * 40)
    
    train_size = 252  # 1 year
    test_size = 63    # 3 months
    step_size = 63    # Quarterly rebalancing
    
    print(f"  Training window: {train_size} days (1 year)")
    print(f"  Test window: {test_size} days (3 months)")
    print(f"  Step size: {step_size} days (quarterly)")
    
    # Create walk-forward coordinator using factory
    wf_coordinator = create_walk_forward_validator(
        coordinator=coordinator,
        data_provider=data_provider,
        optimizer=optimizer,
        objective=objective,
        container_factory=container_factory,
        backtest_engine=backtest_engine,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=False  # Rolling window
    )
    
    print("  ✓ Walk-forward validator created")
    
    # Step 4: Define strategy and parameters
    print("\nStep 4: Strategy Configuration")
    print("-" * 40)
    
    strategy_class = 'MomentumStrategy'
    base_params = {
        'signal_cooldown': 3600,
        'position_size': 0.1,
        'stop_loss': 0.02
    }
    
    parameter_space = {
        'lookback_period': [10, 20, 30],
        'momentum_threshold': [0.015, 0.02, 0.025]
    }
    
    print(f"  Strategy: {strategy_class}")
    print(f"  Base parameters: {base_params}")
    print(f"  Search space: {len(parameter_space['lookback_period'])} x "
          f"{len(parameter_space['momentum_threshold'])} = "
          f"{len(parameter_space['lookback_period']) * len(parameter_space['momentum_threshold'])} combinations")
    
    # Step 5: Run walk-forward validation
    print("\nStep 5: Running Walk-Forward Validation")
    print("-" * 40)
    
    # Show phase transitions
    coordinator.phase_manager.transition_phase('INITIALIZATION', 'WALK_FORWARD')
    
    results = await wf_coordinator.run_walk_forward(
        strategy_class=strategy_class,
        base_params=base_params,
        parameter_space=parameter_space
    )
    
    # Step 6: Analyze results
    print("\n\nStep 6: Walk-Forward Results")
    print("=" * 80)
    
    summary = results['summary']
    print(f"\nSummary Statistics:")
    print(f"  Periods analyzed: {summary['num_periods']}")
    print(f"  Average train score: {summary['train_mean']:.3f} ± {summary['train_std']:.3f}")
    print(f"  Average test score: {summary['test_mean']:.3f} ± {summary['test_std']:.3f}")
    print(f"  Overfitting ratio: {summary['overfitting_ratio']:.3f}")
    print(f"  Strategy robust: {'YES' if summary['robust'] else 'NO'}")
    
    # Show period details
    print(f"\nPeriod-by-Period Results:")
    for i, period_result in enumerate(results['periods']):
        period = period_result['period']
        opt_result = period_result['optimization']
        test_result = period_result['test']
        
        print(f"\nPeriod {i+1} (Q{i+1}):")
        print(f"  Training: days {period.train_start}-{period.train_end}")
        print(f"  Testing: days {period.test_start}-{period.test_end}")
        print(f"  Optimal params: {opt_result['best_params']}")
        print(f"  Train score: {opt_result['best_score']:.3f}")
        print(f"  Test score: {test_result['objective_score']:.3f}")
        print(f"  Test Sharpe: {test_result['metrics']['sharpe_ratio']:.3f}")
    
    # Step 7: Architecture benefits
    print("\n\nArchitecture Benefits Demonstrated:")
    print("=" * 80)
    
    print("\n1. Separation of Concerns:")
    print("   - Period Manager: Handled data slicing for each period")
    print("   - Optimizer: Focused only on parameter search")
    print("   - Executor: Managed container lifecycle")
    print("   - Coordinator: Orchestrated the entire workflow")
    
    print("\n2. Container Standardization:")
    print(f"   - Created {len(container_factory.containers_created)} identical containers")
    print("   - Each followed exact same initialization pattern")
    print("   - No state leakage between periods")
    
    print("\n3. Checkpointing:")
    print(f"   - Saved {len(coordinator.checkpointing.checkpoints)} checkpoints")
    print("   - Can resume from any period if interrupted")
    
    print("\n4. Scalability:")
    print("   - Each period can run in parallel")
    print("   - Containers provide resource isolation")
    print("   - Results streamed to disk (in production)")
    
    print("\n" + "=" * 80)
    print("Walk-Forward Validation Complete! 🎯")
    print("=" * 80)


if __name__ == "__main__":
    # Run the integrated example
    asyncio.run(run_integrated_walk_forward())