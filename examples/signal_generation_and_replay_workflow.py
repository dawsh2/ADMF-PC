"""
Complete example demonstrating grid search signal generation and regime-filtered replay.

This example shows the step-by-step mechanics of:
1. Running a grid search to generate signals with multiple strategy parameters
2. Storing signals with classifier state changes
3. Replaying signals filtered by regime for efficient backtesting
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: Define Stateless Classifier and Strategy Functions
# ============================================================================

def trend_classifier(features: Dict[str, Any]) -> str:
    """
    Stateless classifier that determines market regime.
    
    Returns: 'TRENDING', 'CHOPPY', or 'VOLATILE'
    """
    # Example logic using features
    sma_20 = features.get('SPY', {}).get('sma_20', 0)
    sma_50 = features.get('SPY', {}).get('sma_50', 0)
    volatility = features.get('SPY', {}).get('volatility', 0)
    
    if volatility > 0.02:
        return 'VOLATILE'
    elif sma_20 > sma_50 * 1.01:  # 1% above
        return 'TRENDING'
    else:
        return 'CHOPPY'


def momentum_strategy(features: Dict[str, Any], classifier_states: Dict[str, str], 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless momentum strategy with configurable parameters.
    
    Only generates signals in TRENDING regime.
    """
    # Check regime
    regime = classifier_states.get('trend_classifier', 'UNKNOWN')
    if regime != 'TRENDING':
        return {'value': 0}  # No signal in non-trending markets
    
    # Get parameters
    fast_period = parameters.get('fast_period', 10)
    slow_period = parameters.get('slow_period', 20)
    signal_threshold = parameters.get('signal_threshold', 0.01)
    
    # Get features for primary symbol
    symbol_features = features.get('SPY', {})
    fast_ma = symbol_features.get(f'sma_{fast_period}', 0)
    slow_ma = symbol_features.get(f'sma_{slow_period}', 0)
    
    # Generate signal
    if slow_ma > 0:
        momentum = (fast_ma - slow_ma) / slow_ma
        
        if momentum > signal_threshold:
            return {
                'symbol': 'SPY',
                'value': 1.0,  # Buy signal
                'metadata': {
                    'momentum': momentum,
                    'regime': regime
                }
            }
        elif momentum < -signal_threshold:
            return {
                'symbol': 'SPY',
                'value': -1.0,  # Sell signal
                'metadata': {
                    'momentum': momentum,
                    'regime': regime
                }
            }
    
    return {'value': 0}  # No signal


def pairs_strategy(features: Dict[str, Any], classifier_states: Dict[str, str],
                  parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless pairs trading strategy for SPY/QQQ.
    
    Works best in CHOPPY markets.
    """
    # Check regime
    regime = classifier_states.get('trend_classifier', 'UNKNOWN')
    if regime != 'CHOPPY':
        return {'value': 0}
    
    # Get parameters
    zscore_threshold = parameters.get('zscore_threshold', 2.0)
    lookback = parameters.get('lookback', 20)
    
    # Get price ratios from features
    spy_price = features.get('SPY', {}).get('close', 0)
    qqq_price = features.get('QQQ', {}).get('close', 0)
    
    if spy_price > 0 and qqq_price > 0:
        ratio = spy_price / qqq_price
        ratio_mean = features.get('cross_symbol', {}).get(f'ratio_mean_{lookback}', ratio)
        ratio_std = features.get('cross_symbol', {}).get(f'ratio_std_{lookback}', 1)
        
        if ratio_std > 0:
            zscore = (ratio - ratio_mean) / ratio_std
            
            if zscore > zscore_threshold:
                # Ratio too high - sell SPY, buy QQQ
                return {
                    'strategy_type': 'pairs',
                    'orders': [
                        {'symbol': 'SPY', 'value': -1.0},
                        {'symbol': 'QQQ', 'value': 1.0}
                    ],
                    'metadata': {
                        'zscore': zscore,
                        'regime': regime
                    }
                }
            elif zscore < -zscore_threshold:
                # Ratio too low - buy SPY, sell QQQ
                return {
                    'strategy_type': 'pairs',
                    'orders': [
                        {'symbol': 'SPY', 'value': 1.0},
                        {'symbol': 'QQQ', 'value': -1.0}
                    ],
                    'metadata': {
                        'zscore': zscore,
                        'regime': regime
                    }
                }
    
    return {'value': 0}


# ============================================================================
# STEP 2: Signal Generation Configuration with Grid Search
# ============================================================================

def create_signal_generation_config() -> Dict[str, Any]:
    """Create configuration for grid search signal generation."""
    
    config = {
        'workflow_id': 'grid_search_2024',
        'signal_output_dir': './signals/grid_search_2024',
        
        # Data sources
        'data_sources': [
            ('SPY', '1m'),
            ('QQQ', '1m'),
            ('NVDA', '5m')
        ],
        'data_files': {
            'SPY': './data/SPY_1m.csv',
            'QQQ': './data/QQQ_1m.csv',
            'NVDA': './data/NVDA_5m.csv'
        },
        
        # Classifiers (no parameters to search)
        'classifiers': [
            {
                'name': 'trend_classifier',
                'function': trend_classifier
            }
        ],
        
        # Strategies with parameter grids
        'strategies': [
            {
                'name': 'momentum',
                'function': momentum_strategy,
                'required_data': [('SPY', '1m')],
                'classifier_id': 'trend_classifier',
                'base_parameters': {
                    'position_size': 0.1
                },
                'parameter_grid': {
                    'fast_period': [5, 10, 20],
                    'slow_period': [20, 50, 100],
                    'signal_threshold': [0.005, 0.01, 0.02]
                }
            },
            {
                'name': 'pairs',
                'function': pairs_strategy,
                'required_data': [('SPY', '1m'), ('QQQ', '1m')],
                'classifier_id': 'trend_classifier',
                'base_parameters': {
                    'position_size': 0.05
                },
                'parameter_grid': {
                    'zscore_threshold': [1.5, 2.0, 2.5],
                    'lookback': [10, 20, 30]
                }
            }
        ],
        
        # Features to calculate
        'indicators': ['sma', 'rsi', 'volatility', 'volume_profile'],
        
        # Event tracing for signal capture
        'enable_event_tracing': True
    }
    
    return config


# ============================================================================
# STEP 3: Run Signal Generation
# ============================================================================

def run_signal_generation():
    """Execute the signal generation workflow."""
    from src.core.coordinator.topologies.signal_generation import (
        build_signal_generation_topology, calculate_grid_search_size
    )
    from src.core.coordinator import Coordinator
    
    # Create configuration
    config = create_signal_generation_config()
    
    # Calculate total combinations
    total_combinations = calculate_grid_search_size(config)
    logger.info(f"Grid search will generate {total_combinations} strategy variants")
    
    # Build topology
    topology = build_signal_generation_topology(config)
    
    # Create coordinator and run
    coordinator = Coordinator()
    
    # Initialize containers
    coordinator.initialize_topology(topology)
    
    # Run signal generation
    logger.info("Starting signal generation...")
    coordinator.run()
    
    logger.info(f"Signal generation complete. Signals saved to {config['signal_output_dir']}")
    
    # Show what was generated
    show_generated_signals(config['signal_output_dir'])


def show_generated_signals(signal_dir: str):
    """Display summary of generated signals."""
    signal_path = Path(signal_dir)
    
    if not signal_path.exists():
        logger.warning(f"Signal directory {signal_dir} not found")
        return
    
    # List strategy variants
    signal_files = list((signal_path / 'signals').glob('*.parquet'))
    logger.info(f"\nGenerated {len(signal_files)} strategy variants:")
    
    for signal_file in signal_files[:5]:  # Show first 5
        # Load metadata
        meta_file = signal_file.with_suffix('.meta.json')
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            logger.info(f"  - {meta['strategy_id']}: {meta['signal_count']} signals")
            logger.info(f"    Parameters: {meta['parameters']}")
    
    # Show classifier changes
    classifier_files = list((signal_path / 'classifier_changes').glob('*.parquet'))
    logger.info(f"\nClassifier state changes:")
    
    for clf_file in classifier_files:
        df = pd.read_parquet(clf_file)
        logger.info(f"  - {clf_file.stem}: {len(df)} regime changes")
        if len(df) > 0:
            logger.info(f"    First change: {df.iloc[0]['old']} -> {df.iloc[0]['new']}")


# ============================================================================
# STEP 4: Signal Replay Configuration
# ============================================================================

def create_replay_config_for_regime(regime: str) -> Dict[str, Any]:
    """Create replay configuration filtered by regime."""
    
    config = {
        'signal_storage_path': './signals',
        'workflow_id': 'grid_search_2024',
        
        # Filter settings
        'regime_filter': regime,
        'sparse_replay': True,  # Only replay bars with signals
        
        # Only replay momentum strategies in TRENDING regime
        # Only replay pairs strategies in CHOPPY regime
        'strategy_filter': None,  # Will be set based on regime
        
        # Portfolio configurations to test
        'portfolios': [
            {
                'id': 'conservative',
                'initial_capital': 100000,
                'risk_params': {
                    'max_position_size': 0.05,
                    'max_portfolio_risk': 0.01
                },
                'strategy_assignments': []  # Will be filled based on regime
            },
            {
                'id': 'moderate',
                'initial_capital': 100000,
                'risk_params': {
                    'max_position_size': 0.10,
                    'max_portfolio_risk': 0.02
                },
                'strategy_assignments': []
            },
            {
                'id': 'aggressive',
                'initial_capital': 100000,
                'risk_params': {
                    'max_position_size': 0.20,
                    'max_portfolio_risk': 0.05
                },
                'strategy_assignments': []
            }
        ],
        
        # Execution settings
        'execution_mode': 'simulated',
        'slippage': {
            'model': 'linear',
            'bps': 5  # 5 basis points
        },
        'commission': {
            'model': 'per_share',
            'rate': 0.005
        }
    }
    
    # Set strategy filter based on regime
    if regime == 'TRENDING':
        # Only momentum strategies work in trending markets
        strategy_prefix = 'momentum_'
    elif regime == 'CHOPPY':
        # Only pairs strategies work in choppy markets
        strategy_prefix = 'pairs_'
    else:
        # In volatile markets, maybe use both but with reduced size
        strategy_prefix = None
    
    # Assign strategies to portfolios
    # In practice, would load actual strategy IDs from signal storage
    if strategy_prefix:
        for portfolio in config['portfolios']:
            # Assign subset of strategies based on risk level
            if portfolio['id'] == 'conservative':
                # Conservative portfolio uses only best performing parameter sets
                portfolio['strategy_assignments'] = [
                    f'{strategy_prefix}fast_period_10_slow_period_50_signal_threshold_0.01'
                ]
            elif portfolio['id'] == 'moderate':
                # Moderate uses top 3 parameter sets
                portfolio['strategy_assignments'] = [
                    f'{strategy_prefix}fast_period_10_slow_period_50_signal_threshold_0.01',
                    f'{strategy_prefix}fast_period_20_slow_period_50_signal_threshold_0.01',
                    f'{strategy_prefix}fast_period_10_slow_period_100_signal_threshold_0.02'
                ]
            else:
                # Aggressive uses more parameter sets
                portfolio['strategy_assignments'] = [
                    f'{strategy_prefix}fast_period_5_slow_period_20_signal_threshold_0.005',
                    f'{strategy_prefix}fast_period_10_slow_period_50_signal_threshold_0.01',
                    f'{strategy_prefix}fast_period_20_slow_period_100_signal_threshold_0.02'
                ]
    
    return config


# ============================================================================
# STEP 5: Run Regime-Filtered Replay
# ============================================================================

def run_regime_filtered_replay(regime: str):
    """Run signal replay for a specific regime."""
    from src.core.coordinator.topologies.signal_replay import build_signal_replay_topology
    from src.core.coordinator import Coordinator
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running signal replay for {regime} regime")
    logger.info(f"{'='*60}")
    
    # Create configuration
    config = create_replay_config_for_regime(regime)
    
    # Build topology
    topology = build_signal_replay_topology(config)
    
    # Create coordinator and run
    coordinator = Coordinator()
    coordinator.initialize_topology(topology)
    
    # Run replay
    logger.info(f"Starting {regime} regime replay...")
    results = coordinator.run()
    
    # Display results
    display_replay_results(regime, results)
    
    return results


def display_replay_results(regime: str, results: Dict[str, Any]):
    """Display results from regime-filtered replay."""
    logger.info(f"\nResults for {regime} regime:")
    
    for portfolio_id, portfolio_results in results.get('portfolios', {}).items():
        metrics = portfolio_results.get('metrics', {})
        logger.info(f"\n  {portfolio_id.upper()} Portfolio:")
        logger.info(f"    - Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"    - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"    - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"    - Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"    - Total Trades: {metrics.get('total_trades', 0)}")


# ============================================================================
# STEP 6: Complete Workflow
# ============================================================================

def main():
    """Run the complete signal generation and replay workflow."""
    
    # Step 1: Generate signals with grid search
    logger.info("="*80)
    logger.info("STEP 1: SIGNAL GENERATION WITH GRID SEARCH")
    logger.info("="*80)
    
    run_signal_generation()
    
    # Step 2: Run regime-filtered replays
    logger.info("\n" + "="*80)
    logger.info("STEP 2: REGIME-FILTERED SIGNAL REPLAY")
    logger.info("="*80)
    
    # Test each regime
    all_results = {}
    for regime in ['TRENDING', 'CHOPPY', 'VOLATILE']:
        results = run_regime_filtered_replay(regime)
        all_results[regime] = results
    
    # Step 3: Compare results across regimes
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CROSS-REGIME COMPARISON")
    logger.info("="*80)
    
    compare_regime_results(all_results)


def compare_regime_results(all_results: Dict[str, Dict[str, Any]]):
    """Compare performance across different regimes."""
    
    # Create comparison table
    comparison_data = []
    
    for regime, results in all_results.items():
        for portfolio_id, portfolio_results in results.get('portfolios', {}).items():
            metrics = portfolio_results.get('metrics', {})
            comparison_data.append({
                'Regime': regime,
                'Portfolio': portfolio_id,
                'Return': metrics.get('total_return', 0),
                'Sharpe': metrics.get('sharpe_ratio', 0),
                'Drawdown': metrics.get('max_drawdown', 0),
                'Trades': metrics.get('total_trades', 0)
            })
    
    # Display as table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        logger.info("\nPerformance Summary:")
        logger.info(df.to_string(index=False))
        
        # Find best configurations
        best_sharpe = df.loc[df['Sharpe'].idxmax()]
        logger.info(f"\nBest Sharpe Ratio: {best_sharpe['Sharpe']:.2f}")
        logger.info(f"  - Regime: {best_sharpe['Regime']}")
        logger.info(f"  - Portfolio: {best_sharpe['Portfolio']}")


# ============================================================================
# Event Tracing Analysis (Bonus)
# ============================================================================

def analyze_event_traces(trace_dir: str):
    """Analyze event traces to understand signal flow."""
    from src.core.events.tracing import EventTracer
    
    logger.info("\n" + "="*80)
    logger.info("EVENT TRACE ANALYSIS")
    logger.info("="*80)
    
    # Load traces
    tracer = EventTracer()
    traces = tracer.load_traces(trace_dir)
    
    # Analyze signal generation flow
    signal_events = [e for e in traces if e.event_type == 'SIGNAL']
    logger.info(f"\nTotal signals generated: {len(signal_events)}")
    
    # Group by strategy
    signals_by_strategy = {}
    for event in signal_events:
        strategy_id = event.payload.get('strategy_id', 'unknown')
        if strategy_id not in signals_by_strategy:
            signals_by_strategy[strategy_id] = []
        signals_by_strategy[strategy_id].append(event)
    
    logger.info("\nSignals by strategy:")
    for strategy_id, signals in list(signals_by_strategy.items())[:5]:
        logger.info(f"  - {strategy_id}: {len(signals)} signals")
    
    # Analyze classifier changes
    classifier_events = [e for e in traces if e.event_type == 'CLASSIFICATION_CHANGE']
    logger.info(f"\nClassifier state changes: {len(classifier_events)}")
    
    if classifier_events:
        # Show regime distribution
        regime_counts = {}
        for event in classifier_events:
            new_regime = event.payload.get('new_state', 'unknown')
            regime_counts[new_regime] = regime_counts.get(new_regime, 0) + 1
        
        logger.info("\nRegime distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  - {regime}: {count} transitions")


if __name__ == "__main__":
    main()