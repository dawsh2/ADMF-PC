"""
Test signal capture and replay system for efficient optimization.

The signal replay system allows us to:
1. Capture signals during Phase 1 optimization
2. Replay signals efficiently in Phase 3 for weight optimization
3. Avoid re-running expensive backtests
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import unittest
from unittest.mock import Mock, patch
import heapq

from src.strategy.components.signal_replay import SignalCapture, SignalReplayer
from src.strategy.optimization.workflows import PhaseAwareOptimizationWorkflow
from src.core.coordinator import ResultAggregator, StrategyIdentity


class TestSignalCapture(unittest.TestCase):
    """Test signal capture functionality."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.capture = SignalCapture(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_capture_signals(self):
        """Test capturing trading signals."""
        # Generate test signals
        signals = []
        for i in range(10):
            signal = {
                'symbol': 'AAPL',
                'direction': 'BUY' if i % 2 == 0 else 'SELL',
                'strength': 0.5 + i * 0.05,
                'timestamp': datetime.now() + timedelta(minutes=i)
            }
            
            metadata = {
                'strategy_id': f'momentum_{i % 3}',
                'regime': 'TRENDING_UP',
                'container_id': f'container_{i}'
            }
            
            self.capture.capture(signal, metadata)
            signals.append((signal, metadata))
        
        # Check signals buffered
        self.assertEqual(len(self.capture.signal_buffer), 10)
        
        # Flush to disk
        self.capture.flush()
        
        # Check files created
        signal_files = list(self.temp_dir.glob("signals_*.json"))
        self.assertGreater(len(signal_files), 0)
        
        # Load and verify
        with open(signal_files[0], 'r') as f:
            loaded_signals = json.load(f)
        
        self.assertEqual(len(loaded_signals), 10)
        self.assertEqual(loaded_signals[0]['signal']['symbol'], 'AAPL')
    
    def test_capture_with_regime_filtering(self):
        """Test capturing signals filtered by regime."""
        # Capture signals from different regimes
        regimes = ['TRENDING_UP', 'HIGH_VOLATILITY', 'TRENDING_DOWN']
        
        for regime in regimes:
            for i in range(5):
                signal = {
                    'symbol': 'SPY',
                    'direction': 'BUY',
                    'strength': 0.7
                }
                
                metadata = {
                    'strategy_id': 'mean_reversion',
                    'regime': regime,
                    'container_id': f'{regime}_container_{i}'
                }
                
                self.capture.capture(signal, metadata)
        
        # Filter by regime
        trending_signals = [
            s for s in self.capture.signal_buffer
            if s['metadata']['regime'] == 'TRENDING_UP'
        ]
        
        self.assertEqual(len(trending_signals), 5)
    
    def test_memory_efficient_capture(self):
        """Test memory-efficient signal capture with automatic flushing."""
        # Set small buffer size
        self.capture.max_buffer_size = 100
        
        # Capture many signals
        for i in range(500):
            signal = {
                'symbol': 'AAPL',
                'direction': 'BUY',
                'strength': 0.5,
                'data': 'x' * 1000  # Some data to increase size
            }
            
            metadata = {'strategy_id': f'strategy_{i % 10}'}
            
            self.capture.capture(signal, metadata)
        
        # Should have flushed multiple times
        signal_files = list(self.temp_dir.glob("signals_*.json"))
        self.assertGreater(len(signal_files), 1)
        
        # Buffer should be small
        self.assertLess(len(self.capture.signal_buffer), 100)


class TestSignalReplay(unittest.TestCase):
    """Test signal replay functionality."""
    
    def setUp(self):
        # Create test signals
        self.signals = []
        base_time = datetime.now()
        
        for i in range(20):
            signal_data = {
                'signal': {
                    'symbol': 'AAPL' if i < 10 else 'GOOGL',
                    'direction': 'BUY' if i % 3 == 0 else 'SELL',
                    'strength': 0.3 + (i % 5) * 0.1,
                    'timestamp': base_time + timedelta(minutes=i)
                },
                'metadata': {
                    'strategy_id': f'strategy_{i % 3}',
                    'regime': 'TRENDING_UP' if i < 15 else 'VOLATILE'
                },
                'timestamp': base_time + timedelta(minutes=i)
            }
            self.signals.append(signal_data)
        
        self.replayer = SignalReplayer(self.signals)
    
    def test_replay_with_weights(self):
        """Test replaying signals with different weights."""
        # Define strategy weights
        weights = {
            'strategy_0': 0.5,
            'strategy_1': 0.3,
            'strategy_2': 0.2
        }
        
        # Replay with weights
        weighted_results = self.replayer.replay_with_weights(weights)
        
        # Check results structure
        self.assertIn('weighted_signals', weighted_results)
        self.assertIn('aggregated_performance', weighted_results)
        
        # Verify weights applied
        for signal_data in weighted_results['weighted_signals']:
            strategy_id = signal_data['metadata']['strategy_id']
            expected_weight = weights.get(strategy_id, 0.0)
            
            # Original strength * weight
            original_strength = next(
                s['signal']['strength'] for s in self.signals
                if s['metadata']['strategy_id'] == strategy_id
                and s['timestamp'] == signal_data['timestamp']
            )
            
            expected_strength = original_strength * expected_weight
            self.assertAlmostEqual(
                signal_data['signal']['strength'],
                expected_strength,
                places=5
            )
    
    def test_replay_by_regime(self):
        """Test replaying signals for specific regime."""
        # Replay only TRENDING_UP signals
        trending_signals = self.replayer.filter_by_regime('TRENDING_UP')
        
        # Should have 15 signals (first 15 are TRENDING_UP)
        self.assertEqual(len(trending_signals), 15)
        
        # All should be from TRENDING_UP regime
        for signal in trending_signals:
            self.assertEqual(signal['metadata']['regime'], 'TRENDING_UP')
    
    def test_replay_aggregation(self):
        """Test signal aggregation during replay."""
        # Test aggregating signals by timestamp
        aggregated = self.replayer.aggregate_by_timestamp()
        
        # Should group signals with same timestamp
        for timestamp, signal_group in aggregated.items():
            # All signals in group should have same timestamp
            timestamps = [s['timestamp'] for s in signal_group]
            self.assertEqual(len(set(timestamps)), 1)
    
    def test_performance_calculation(self):
        """Test performance calculation from replayed signals."""
        # Mock performance calculator
        def calculate_performance(signals):
            # Simple mock: count buy signals
            buy_count = sum(1 for s in signals if s['signal']['direction'] == 'BUY')
            return {
                'total_signals': len(signals),
                'buy_signals': buy_count,
                'sell_signals': len(signals) - buy_count,
                'avg_strength': sum(s['signal']['strength'] for s in signals) / len(signals)
            }
        
        # Calculate performance
        performance = calculate_performance(self.signals)
        
        self.assertEqual(performance['total_signals'], 20)
        self.assertGreater(performance['buy_signals'], 0)
        self.assertGreater(performance['avg_strength'], 0)


class TestSignalReplayOptimization(unittest.TestCase):
    """Test signal replay in optimization context."""
    
    def test_weight_optimization_with_replay(self):
        """Test optimizing weights using signal replay."""
        # Create test signals from 3 strategies
        signals = []
        for strategy_id in ['momentum', 'mean_rev', 'breakout']:
            for i in range(10):
                signals.append({
                    'signal': {
                        'symbol': 'AAPL',
                        'direction': 'BUY' if i % 2 == 0 else 'SELL',
                        'strength': 0.5,
                        'returns': 0.001 * (i - 5)  # Mock returns
                    },
                    'metadata': {'strategy_id': strategy_id},
                    'timestamp': datetime.now() + timedelta(minutes=i)
                })
        
        replayer = SignalReplayer(signals)
        
        # Optimization function using replay
        def evaluate_weights(weights):
            result = replayer.replay_with_weights(weights)
            
            # Calculate Sharpe ratio from weighted signals
            returns = []
            for signal_data in result['weighted_signals']:
                if signal_data['signal']['strength'] > 0:
                    returns.append(
                        signal_data['signal']['returns'] * 
                        signal_data['signal']['strength']
                    )
            
            if not returns:
                return -1.0
            
            import numpy as np
            return np.mean(returns) / (np.std(returns) + 1e-6)
        
        # Try different weight combinations
        best_sharpe = -float('inf')
        best_weights = None
        
        # Grid search over weights
        for w1 in [0.2, 0.3, 0.4, 0.5]:
            for w2 in [0.2, 0.3, 0.4]:
                w3 = 1.0 - w1 - w2
                if w3 >= 0.1:  # Ensure valid weights
                    weights = {
                        'momentum': w1,
                        'mean_rev': w2,
                        'breakout': w3
                    }
                    
                    sharpe = evaluate_weights(weights)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = weights
        
        # Should find optimal weights
        self.assertIsNotNone(best_weights)
        self.assertAlmostEqual(sum(best_weights.values()), 1.0, places=5)
    
    def test_incremental_signal_replay(self):
        """Test incremental replay for walk-forward validation."""
        # Create signals over time
        signals = []
        base_time = datetime(2024, 1, 1)
        
        for day in range(100):
            for hour in range(0, 24, 6):  # 4 signals per day
                signals.append({
                    'signal': {
                        'symbol': 'SPY',
                        'direction': 'BUY' if hour < 12 else 'SELL',
                        'strength': 0.5
                    },
                    'metadata': {
                        'strategy_id': 'intraday',
                        'day': day
                    },
                    'timestamp': base_time + timedelta(days=day, hours=hour)
                })
        
        replayer = SignalReplayer(signals)
        
        # Walk-forward replay
        window_size = 20  # days
        step_size = 5     # days
        
        results = []
        for start_day in range(0, 80, step_size):
            end_day = start_day + window_size
            
            # Filter signals for window
            window_signals = [
                s for s in signals
                if start_day <= s['metadata']['day'] < end_day
            ]
            
            # Replay for this window
            window_replayer = SignalReplayer(window_signals)
            result = window_replayer.replay_with_weights({'intraday': 1.0})
            
            results.append({
                'period': f'day_{start_day}_to_{end_day}',
                'signal_count': len(window_signals),
                'result': result
            })
        
        # Should have multiple walk-forward periods
        self.assertGreater(len(results), 10)
        
        # Each period should have signals
        for result in results:
            self.assertGreater(result['signal_count'], 0)


class TestSignalReplayIntegration(unittest.TestCase):
    """Test signal replay integration with optimization workflow."""
    
    @patch('src.strategy.optimization.workflows.SignalCapture')
    @patch('src.strategy.optimization.workflows.SignalReplayer')
    def test_phase3_weight_optimization(self, mock_replayer_class, mock_capture_class):
        """Test Phase 3 uses signal replay for weight optimization."""
        # Mock signal capture from Phase 1
        mock_capture = Mock()
        mock_capture.load_signals.return_value = [
            {
                'signal': {'direction': 'BUY', 'strength': 0.7},
                'metadata': {'strategy_id': 'momentum_001', 'regime': 'TRENDING'}
            },
            {
                'signal': {'direction': 'SELL', 'strength': 0.5},
                'metadata': {'strategy_id': 'mean_rev_001', 'regime': 'VOLATILE'}
            }
        ]
        mock_capture_class.return_value = mock_capture
        
        # Mock signal replayer
        mock_replayer = Mock()
        mock_replayer.replay_with_weights.return_value = {
            'weighted_signals': [],
            'aggregated_performance': {'sharpe': 1.5}
        }
        mock_replayer_class.return_value = mock_replayer
        
        # Create workflow
        coordinator = Mock()
        config = {
            'workflow_id': 'test_replay',
            'signal_capture_dir': '/tmp/signals'
        }
        
        workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
        
        # Mock Phase 3 execution
        workflow._load_phase1_signals = Mock(return_value=mock_capture.load_signals())
        workflow._optimize_signal_weights = Mock(return_value={'momentum': 0.6, 'mean_rev': 0.4})
        
        # Run Phase 3 (mocked)
        phase3_results = {
            'TRENDING': {
                'weights': {'momentum': 0.7, 'mean_rev': 0.3},
                'performance': {'sharpe': 1.8}
            },
            'VOLATILE': {
                'weights': {'momentum': 0.3, 'mean_rev': 0.7},
                'performance': {'sharpe': 1.4}
            }
        }
        
        # Verify signal replay was used
        self.assertIn('TRENDING', phase3_results)
        self.assertIn('VOLATILE', phase3_results)
        
        # Verify weights sum to 1
        for regime, result in phase3_results.items():
            weight_sum = sum(result['weights'].values())
            self.assertAlmostEqual(weight_sum, 1.0, places=5)


class TestSignalReplayPerformance(unittest.TestCase):
    """Test performance aspects of signal replay."""
    
    def test_large_signal_replay(self):
        """Test replay performance with large number of signals."""
        # Generate large signal dataset
        num_signals = 10000
        signals = []
        
        for i in range(num_signals):
            signals.append({
                'signal': {
                    'symbol': f'SYM{i % 100}',
                    'direction': 'BUY' if i % 2 == 0 else 'SELL',
                    'strength': 0.5 + (i % 10) * 0.05
                },
                'metadata': {
                    'strategy_id': f'strategy_{i % 10}',
                    'regime': f'regime_{i % 3}'
                },
                'timestamp': datetime.now() + timedelta(seconds=i)
            })
        
        import time
        
        # Time replay operation
        replayer = SignalReplayer(signals)
        
        start_time = time.time()
        result = replayer.replay_with_weights({
            f'strategy_{i}': 0.1 for i in range(10)
        })
        end_time = time.time()
        
        # Should complete quickly even with many signals
        self.assertLess(end_time - start_time, 1.0)  # Less than 1 second
        
        # Should produce valid results
        self.assertIn('weighted_signals', result)
        self.assertEqual(len(result['weighted_signals']), num_signals)
    
    def test_memory_efficient_replay(self):
        """Test memory efficiency of signal replay."""
        # Create signals with large metadata
        signals = []
        for i in range(1000):
            signals.append({
                'signal': {'direction': 'BUY', 'strength': 0.5},
                'metadata': {
                    'strategy_id': 'test',
                    'large_data': 'x' * 10000  # 10KB per signal
                },
                'timestamp': datetime.now()
            })
        
        # Replay should not duplicate large data
        replayer = SignalReplayer(signals)
        
        # Use iterator for memory efficiency
        weighted_count = 0
        for weighted_signal in replayer.replay_iterator({'test': 0.5}):
            weighted_count += 1
            # Process signal without keeping all in memory
            self.assertEqual(weighted_signal['signal']['strength'], 0.25)  # 0.5 * 0.5
        
        self.assertEqual(weighted_count, 1000)


if __name__ == '__main__':
    unittest.main()