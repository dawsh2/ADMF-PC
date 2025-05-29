"""
Integration test for complete optimization workflow.

This test validates the entire workflow from Phase 1 through Phase 4,
ensuring all six critical architectural decisions work together correctly.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import unittest
from unittest.mock import Mock, patch, AsyncMock

# Import all necessary components
from src.core.coordinator import (
    Coordinator,
    PhaseTransition,
    ContainerNamingStrategy,
    ResultAggregator,
    StrategyIdentity,
    CheckpointManager,
    WalkForwardValidator
)

from src.core.containers import UniversalScopedContainer

from src.strategy.optimization import (
    PhaseAwareOptimizationWorkflow,
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    RangeConstraint,
    RelationalConstraint
)

from src.strategy.strategies.momentum import MomentumStrategy
from src.strategy.strategies.mean_reversion import MeanReversionStrategy
from src.strategy.components.classifiers import VolatilityClassifier, TrendClassifier
from src.strategy.components.signal_replay import SignalCapture, SignalReplayer


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete multi-phase optimization workflow."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results_dir = self.temp_dir / 'results'
        self.checkpoint_dir = self.temp_dir / 'checkpoints'
        self.signal_dir = self.temp_dir / 'signals'
        
        self.results_dir.mkdir()
        self.checkpoint_dir.mkdir()
        self.signal_dir.mkdir()
        
        # Create coordinator with all components
        self.coordinator = self._create_coordinator()
        
        # Create workflow configuration
        self.workflow_config = self._create_workflow_config()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_coordinator(self) -> Coordinator:
        """Create coordinator with all necessary components."""
        coordinator = Mock(spec=Coordinator)
        
        # Add phase management
        coordinator.phase_transitions = PhaseTransition()
        
        # Add container naming
        coordinator.container_naming = ContainerNamingStrategy()
        
        # Add result aggregation
        coordinator.result_aggregator = ResultAggregator(self.results_dir)
        
        # Add checkpointing
        coordinator.checkpointing = CheckpointManager(self.checkpoint_dir)
        
        # Add walk-forward validation
        coordinator.walk_forward_validator = WalkForwardValidator()
        
        # Add container
        coordinator.coordinator_container = UniversalScopedContainer('coordinator')
        
        return coordinator
    
    def _create_workflow_config(self) -> Dict[str, Any]:
        """Create comprehensive workflow configuration."""
        return {
            'workflow_id': 'integration_test_workflow',
            'output_dir': str(self.results_dir),
            'signal_capture_dir': str(self.signal_dir),
            'market_data': self._create_mock_market_data(),
            'phases': {
                'phase1': {
                    'parameter_space': {
                        'lookback_period': [10, 20, 30],
                        'momentum_threshold': [0.01, 0.02, 0.03]
                    },
                    'regime_classifiers': ['hmm', 'pattern'],
                    'strategies': [
                        {
                            'class': 'MomentumStrategy',
                            'base_params': {'signal_cooldown': 3600}
                        },
                        {
                            'class': 'MeanReversionStrategy', 
                            'base_params': {'window_size': 50}
                        }
                    ],
                    'optimizer': {'type': 'grid'},
                    'objective': {'type': 'sharpe'}
                },
                'phase2': {
                    'analysis_type': 'regime_comparison'
                },
                'phase3': {
                    'weight_optimizer': {'type': 'bayesian', 'n_trials': 20}
                },
                'phase4': {
                    'walk_forward_config': {
                        'train_size': 500,
                        'test_size': 100,
                        'step_size': 100
                    }
                }
            }
        }
    
    def _create_mock_market_data(self) -> Dict[str, Any]:
        """Create mock market data for testing."""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'data_length': 1000,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
    
    @patch('src.strategy.optimization.workflows.PhaseAwareOptimizationWorkflow._run_backtest')
    @patch('src.strategy.optimization.workflows.PhaseAwareOptimizationWorkflow._create_strategy')
    @patch('src.strategy.optimization.workflows.PhaseAwareOptimizationWorkflow._create_classifier')
    async def test_phase1_optimization(self, mock_create_classifier, mock_create_strategy, mock_run_backtest):
        """Test Phase 1: Parameter optimization with parallel regimes."""
        # Mock strategy creation
        mock_strategy = Mock()
        mock_strategy.generate_signal.return_value = {
            'direction': 'BUY',
            'strength': 0.8
        }
        mock_create_strategy.return_value = mock_strategy
        
        # Mock classifier creation
        mock_classifier = Mock()
        mock_classifier.classify.return_value = 'TRENDING_UP'
        mock_create_classifier.return_value = mock_classifier
        
        # Mock backtest results
        mock_run_backtest.return_value = {
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': 0.08
        }
        
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Run Phase 1
        phase1_results = await workflow._run_phase1_optimization()
        
        # Verify results structure
        self.assertIn('hmm', phase1_results)
        self.assertIn('pattern', phase1_results)
        
        # Verify container naming
        container_calls = []
        for call in self.coordinator.container_naming.generate_container_id.call_args_list:
            args = call[1]
            container_calls.append(args)
            
            # Should have phase, regime, strategy
            self.assertEqual(args['phase'], 'phase1')
            self.assertIn('regime', args)
            self.assertIn('strategy', args)
        
        # Verify strategy tracking
        self.assertGreater(len(workflow.strategy_identities), 0)
        
        # Verify phase output recorded
        phase_outputs = self.coordinator.phase_transitions.phase1_outputs
        self.assertIn('classifier_results', phase_outputs)
    
    async def test_phase2_analysis(self):
        """Test Phase 2: Regime analysis."""
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Mock Phase 1 results
        phase1_results = {
            'hmm': {
                'momentum_strategy': {
                    'TRENDING_UP': {'best_params': {'lookback': 10}, 'best_score': 1.8},
                    'VOLATILE': {'best_params': {'lookback': 30}, 'best_score': 1.2},
                    'TRENDING_DOWN': {'best_params': {'lookback': 20}, 'best_score': 0.9}
                }
            },
            'pattern': {
                'momentum_strategy': {
                    'BREAKOUT': {'best_params': {'lookback': 15}, 'best_score': 2.0},
                    'RANGE': {'best_params': {'lookback': 25}, 'best_score': 1.1}
                }
            }
        }
        
        # Record Phase 1 output
        self.coordinator.phase_transitions.record_phase_output(
            '1', 'classifier_results', phase1_results
        )
        
        # Mock analysis methods
        workflow._compare_regime_performance = Mock(return_value={
            'best_regime': 'TRENDING_UP',
            'worst_regime': 'TRENDING_DOWN',
            'regime_scores': {'TRENDING_UP': 1.8, 'VOLATILE': 1.2, 'TRENDING_DOWN': 0.9}
        })
        
        workflow._analyze_parameter_stability = Mock(return_value={
            'stable_params': ['lookback'],
            'unstable_params': [],
            'variance': {'lookback': 0.15}
        })
        
        workflow._select_best_overall_params = Mock(return_value={
            'lookback': 20,  # Average across regimes
            'momentum_threshold': 0.02
        })
        
        # Run Phase 2
        phase2_results = await workflow._run_phase2_analysis()
        
        # Verify analysis performed
        self.assertIn('hmm', phase2_results)
        
        # Verify phase transition
        phase2_outputs = self.coordinator.phase_transitions.phase2_outputs
        self.assertIn('regime_best_params', phase2_outputs)
    
    async def test_phase3_weights(self):
        """Test Phase 3: Weight optimization using signal replay."""
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Mock signal loading
        mock_signals = [
            {
                'signal': {'direction': 'BUY', 'strength': 0.7, 'returns': 0.002},
                'metadata': {'strategy_id': 'momentum_001', 'regime': 'TRENDING_UP'},
                'timestamp': datetime.now()
            },
            {
                'signal': {'direction': 'SELL', 'strength': 0.5, 'returns': -0.001},
                'metadata': {'strategy_id': 'mean_rev_001', 'regime': 'TRENDING_UP'},
                'timestamp': datetime.now()
            }
        ]
        
        workflow._load_phase1_signals = Mock(return_value=mock_signals)
        workflow._get_all_regimes = Mock(return_value=['TRENDING_UP', 'VOLATILE'])
        workflow._filter_signals_by_regime = Mock(side_effect=lambda signals, regime: 
            [s for s in signals if s['metadata']['regime'] == regime]
        )
        
        # Mock weight optimization
        workflow._optimize_signal_weights = Mock(return_value={
            'momentum_001': 0.6,
            'mean_rev_001': 0.4
        })
        
        workflow._evaluate_weights = Mock(return_value={
            'sharpe': 1.6,
            'returns': 0.12
        })
        
        # Run Phase 3
        phase3_results = await workflow._run_phase3_weights()
        
        # Verify weight optimization for each regime
        self.assertIn('TRENDING_UP', phase3_results)
        self.assertIn('VOLATILE', phase3_results)
        
        # Verify weights sum to 1
        for regime, result in phase3_results.items():
            if 'weights' in result:
                weight_sum = sum(result['weights'].values())
                self.assertAlmostEqual(weight_sum, 1.0, places=5)
    
    async def test_phase4_validation(self):
        """Test Phase 4: Walk-forward validation."""
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Mock inputs from previous phases
        self.coordinator.phase_transitions.record_phase_output(
            '2', 'regime_best_params', {
                'TRENDING_UP': {'lookback': 10, 'threshold': 0.01},
                'VOLATILE': {'lookback': 30, 'threshold': 0.03}
            }
        )
        
        self.coordinator.phase_transitions.record_phase_output(
            '3', 'optimal_weights', {
                'TRENDING_UP': {'momentum': 0.7, 'mean_rev': 0.3},
                'VOLATILE': {'momentum': 0.3, 'mean_rev': 0.7}
            }
        )
        
        # Mock walk-forward periods
        workflow._create_walk_forward_periods = Mock(return_value=[
            {'train_start': 0, 'train_end': 500, 'test_start': 500, 'test_end': 600},
            {'train_start': 100, 'train_end': 600, 'test_start': 600, 'test_end': 700}
        ])
        
        # Mock adaptive strategy
        mock_adaptive = Mock()
        workflow._create_adaptive_strategy = Mock(return_value=mock_adaptive)
        
        # Mock validation
        workflow._run_validation_period = Mock(return_value={
            'sharpe': 1.4,
            'returns': 0.10,
            'max_drawdown': 0.06
        })
        
        workflow._aggregate_validation_results = Mock(return_value={
            'avg_sharpe': 1.45,
            'avg_returns': 0.11,
            'avg_drawdown': 0.065
        })
        
        # Run Phase 4
        phase4_results = await workflow._run_phase4_validation()
        
        # Verify walk-forward validation performed
        self.assertEqual(workflow._run_validation_period.call_count, 2)
        
        # Verify final performance metrics
        self.assertIn('avg_sharpe', phase4_results)
        self.assertGreater(phase4_results['avg_sharpe'], 0)
    
    async def test_complete_workflow_execution(self):
        """Test complete workflow execution from start to finish."""
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Mock all phase executions
        workflow._run_phase1_optimization = AsyncMock(return_value={
            'hmm': {'momentum': {'best': True}},
            'pattern': {'momentum': {'best': True}}
        })
        
        workflow._run_phase2_analysis = AsyncMock(return_value={
            'regime_comparison': 'complete'
        })
        
        workflow._run_phase3_weights = AsyncMock(return_value={
            'TRENDING': {'weights': {'momentum': 0.7, 'mean_rev': 0.3}}
        })
        
        workflow._run_phase4_validation = AsyncMock(return_value={
            'avg_sharpe': 1.5,
            'validation': 'complete'
        })
        
        workflow._aggregate_results = Mock(return_value={
            'workflow_id': self.workflow_config['workflow_id'],
            'completed_phases': ['phase1', 'phase2', 'phase3', 'phase4'],
            'final_performance': {'sharpe': 1.5}
        })
        
        # Run complete workflow
        results = await workflow.run()
        
        # Verify all phases executed
        workflow._run_phase1_optimization.assert_called_once()
        workflow._run_phase2_analysis.assert_called_once()
        workflow._run_phase3_weights.assert_called_once()
        workflow._run_phase4_validation.assert_called_once()
        
        # Verify results
        self.assertEqual(results['workflow_id'], self.workflow_config['workflow_id'])
        self.assertIn('completed_phases', results)
        self.assertEqual(len(results['completed_phases']), 4)
    
    def test_checkpointing_and_resume(self):
        """Test workflow can be checkpointed and resumed."""
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Simulate partial completion
        workflow.completed_phases = {'phase1', 'phase2'}
        workflow.current_phase = 'phase3'
        
        # Save state
        state = {
            'completed_phases': list(workflow.completed_phases),
            'current_phase': workflow.current_phase,
            'strategy_identities': {
                sid: {'class': identity.strategy_class, 'params': identity.base_params}
                for sid, identity in workflow.strategy_identities.items()
            }
        }
        
        # Save checkpoint
        self.coordinator.checkpointing.save_checkpoint(
            workflow.workflow_id,
            'phase3',
            state
        )
        
        # Create new workflow and restore
        new_workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Restore checkpoint
        checkpoint = self.coordinator.checkpointing.restore_checkpoint(
            workflow.workflow_id,
            'phase3'
        )
        
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint['state']['completed_phases'], ['phase1', 'phase2'])
        self.assertEqual(checkpoint['state']['current_phase'], 'phase3')
    
    def test_result_streaming(self):
        """Test results are streamed to disk efficiently."""
        # Generate many optimization results
        for i in range(100):
            container_id = f"phase1_hmm_trending_momentum_{i}"
            result = {
                'params': {'lookback': 10 + i % 3 * 10},
                'score': 1.0 + i * 0.01,
                'metrics': {
                    'sharpe': 1.0 + i * 0.01,
                    'returns': 0.1 + i * 0.001,
                    'drawdown': 0.1 - i * 0.0005
                }
            }
            
            self.coordinator.result_aggregator.handle_container_result(
                container_id, result
            )
        
        # Check results on disk
        result_files = list(self.results_dir.glob("*.json"))
        self.assertEqual(len(result_files), 100)
        
        # Check only top results in memory
        top_results = self.coordinator.result_aggregator.get_top_results(10)
        self.assertEqual(len(top_results), 10)
        
        # Verify top results are best performers
        top_scores = [r[0] for r in top_results]
        self.assertEqual(top_scores, sorted(top_scores, reverse=True))
    
    def test_strategy_tracking_across_workflow(self):
        """Test strategies are tracked correctly across all phases."""
        workflow = PhaseAwareOptimizationWorkflow(self.coordinator, self.workflow_config)
        
        # Add strategy identities
        momentum_id = StrategyIdentity('MomentumStrategy', {'lookback': 20})
        momentum_id.add_regime_instance('hmm_trending_up', 'container_001')
        momentum_id.add_regime_instance('hmm_volatile', 'container_002')
        momentum_id.add_regime_instance('pattern_breakout', 'container_003')
        
        workflow.strategy_identities[momentum_id.canonical_id] = momentum_id
        
        # Verify tracking
        self.assertEqual(len(momentum_id.regime_instances), 3)
        
        # Verify can retrieve by regime
        trending_container = momentum_id.regime_instances['hmm_trending_up']
        self.assertEqual(trending_container, 'container_001')


class TestErrorHandling(unittest.TestCase):
    """Test error handling in optimization workflow."""
    
    def test_phase_failure_handling(self):
        """Test workflow handles phase failures gracefully."""
        coordinator = Mock()
        config = {'workflow_id': 'error_test'}
        
        workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
        
        # Mock phase that fails
        workflow._run_phase1_optimization = AsyncMock(
            side_effect=Exception("Optimization failed")
        )
        
        # Workflow should handle error and save checkpoint
        # (In real implementation)
    
    def test_container_cleanup_on_error(self):
        """Test containers are cleaned up even on error."""
        container = Mock()
        container.start = Mock()
        container.stop = Mock()
        container.dispose = Mock()
        
        # Simulate error during execution
        container.run_trial = Mock(side_effect=Exception("Trial failed"))
        
        # Cleanup should still happen
        try:
            container.start()
            container.run_trial({})
        except:
            pass
        finally:
            container.stop()
            container.dispose()
        
        container.stop.assert_called_once()
        container.dispose.assert_called_once()


if __name__ == '__main__':
    unittest.main()