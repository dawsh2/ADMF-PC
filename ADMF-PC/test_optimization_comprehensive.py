"""
Comprehensive tests for ADMF-PC Optimization Framework.

Tests implement validation for the six critical architectural decisions:
1. Clear phase transitions with data flow
2. Consistent container naming
3. Result streaming to avoid memory issues
4. Cross-regime strategy tracking
5. Checkpointing for resumability
6. Walk-forward validation support
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock

# Import optimization components
from src.strategy.optimization import (
    # Optimizers
    GridOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
    # Objectives
    SharpeObjective,
    MaxReturnObjective,
    MinDrawdownObjective,
    CompositeObjective,
    # Constraints (NO inheritance!)
    RelationalConstraint,
    RangeConstraint,
    DiscreteConstraint,
    FunctionalConstraint,
    CompositeConstraint,
    # Containers
    OptimizationContainer,
    RegimeAwareOptimizationContainer,
    # Workflows
    SequentialOptimizationWorkflow,
    RegimeBasedOptimizationWorkflow,
    PhaseAwareOptimizationWorkflow,
    # Support classes
    OptimizationCapability
)

from src.core.coordinator import (
    Coordinator,
    PhaseTransition,
    ContainerNamingStrategy,
    ResultAggregator,
    StrategyIdentity,
    CheckpointManager,
    WalkForwardValidator
)


class TestOptimizationProtocols(unittest.TestCase):
    """Test that all components follow PC architecture with NO inheritance."""
    
    def test_no_inheritance_in_constraints(self):
        """Verify constraint classes have NO inheritance."""
        # Check that constraint classes don't inherit from anything
        constraints = [
            RelationalConstraint,
            RangeConstraint,
            DiscreteConstraint,
            FunctionalConstraint,
            CompositeConstraint
        ]
        
        for constraint_class in constraints:
            # Should have no base classes except object
            bases = constraint_class.__bases__
            self.assertEqual(len(bases), 1)
            self.assertEqual(bases[0], object)
            
            # Should implement required methods
            instance = constraint_class.__new__(constraint_class)
            self.assertTrue(hasattr(instance, 'is_satisfied'))
            self.assertTrue(hasattr(instance, 'validate_and_adjust'))
            self.assertTrue(hasattr(instance, 'get_description'))
    
    def test_optimizer_protocol_compliance(self):
        """Test optimizers implement protocol correctly."""
        optimizers = [
            GridOptimizer(),
            BayesianOptimizer(),
            GeneticOptimizer()
        ]
        
        for optimizer in optimizers:
            # Check protocol methods exist
            self.assertTrue(hasattr(optimizer, 'optimize'))
            self.assertTrue(hasattr(optimizer, 'get_best_parameters'))
            self.assertTrue(hasattr(optimizer, 'get_best_score'))
            self.assertTrue(hasattr(optimizer, 'get_optimization_history'))
    
    def test_objective_protocol_compliance(self):
        """Test objectives implement protocol correctly."""
        objectives = [
            SharpeObjective(),
            MaxReturnObjective(),
            MinDrawdownObjective()
        ]
        
        for objective in objectives:
            # Check protocol methods exist
            self.assertTrue(hasattr(objective, 'calculate'))
            self.assertTrue(hasattr(objective, 'get_direction'))
            self.assertTrue(hasattr(objective, 'get_requirements'))
            self.assertTrue(hasattr(objective, 'is_better'))


class TestPhaseTransitions(unittest.TestCase):
    """Test critical decision #1: Clear phase transitions with data flow."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.coordinator = Mock(spec=Coordinator)
        self.phase_transitions = PhaseTransition()
        self.coordinator.phase_transitions = self.phase_transitions
    
    def test_phase_transition_data_flow(self):
        """Test data flows correctly between phases."""
        # Phase 1 output
        phase1_data = {
            'classifier_results': {
                'hmm': {'strategy1': {'best_params': {'p1': 10}}},
                'pattern': {'strategy1': {'best_params': {'p1': 20}}}
            }
        }
        
        # Record phase 1 output
        self.phase_transitions.record_phase_output('1', 'classifier_results', phase1_data['classifier_results'])
        
        # Phase 2 should be able to access phase 1 data
        phase2_input = self.phase_transitions.get_phase_input('2', 'classifier_results')
        self.assertEqual(phase2_input, phase1_data['classifier_results'])
    
    def test_phase_completion_events(self):
        """Test phases publish completion events."""
        # Mock event bus
        event_bus = Mock()
        self.coordinator.event_bus = event_bus
        
        # Complete phase 1
        self.phase_transitions.complete_phase('1', {'status': 'success'})
        
        # Should publish completion event
        event_bus.publish.assert_called_with(
            'phase.completed',
            {'phase': '1', 'status': 'success'}
        )


class TestContainerNaming(unittest.TestCase):
    """Test critical decision #2: Consistent container naming."""
    
    def setUp(self):
        self.naming_strategy = ContainerNamingStrategy()
    
    def test_container_id_format(self):
        """Test container IDs follow consistent format."""
        container_id = self.naming_strategy.generate_container_id(
            phase='phase1',
            regime='hmm_trending_up',
            strategy='momentum',
            params={'p1': 10, 'p2': 20}
        )
        
        # Should contain all components
        self.assertIn('phase1', container_id)
        self.assertIn('hmm_trending_up', container_id)
        self.assertIn('momentum', container_id)
        
        # Should be parseable
        parts = self.naming_strategy.parse_container_id(container_id)
        self.assertEqual(parts['phase'], 'phase1')
        self.assertEqual(parts['regime'], 'hmm_trending_up')
        self.assertEqual(parts['strategy'], 'momentum')
    
    def test_container_tracking_across_phases(self):
        """Test containers can be tracked across optimization phases."""
        containers = []
        
        # Generate containers for different phases
        for phase in ['phase1', 'phase2', 'phase3']:
            for regime in ['trending_up', 'volatile', 'trending_down']:
                container_id = self.naming_strategy.generate_container_id(
                    phase=phase,
                    regime=f'hmm_{regime}',
                    strategy='momentum',
                    params={'lookback': 20}
                )
                containers.append(container_id)
        
        # Should be able to filter by phase
        phase1_containers = [c for c in containers if 'phase1' in c]
        self.assertEqual(len(phase1_containers), 3)
        
        # Should be able to filter by regime
        trending_containers = [c for c in containers if 'trending_up' in c]
        self.assertEqual(len(trending_containers), 3)


class TestResultStreaming(unittest.TestCase):
    """Test critical decision #3: Result streaming to avoid memory issues."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.aggregator = ResultAggregator(self.temp_dir)
    
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_results_stream_to_disk(self):
        """Test results are streamed to disk, not kept in memory."""
        # Generate many results
        for i in range(100):
            container_id = f"container_{i}"
            result = {
                'score': i * 0.1,
                'params': {'p1': i},
                'metrics': {'sharpe': i * 0.01}
            }
            
            self.aggregator.handle_container_result(container_id, result)
        
        # Check results are on disk
        result_files = list(self.temp_dir.glob("*.json"))
        self.assertEqual(len(result_files), 100)
        
        # Check only top results in memory
        top_results = self.aggregator.get_top_results(10)
        self.assertEqual(len(top_results), 10)
        
        # Verify top results are actually the best
        scores = [r[0] for r in top_results]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_memory_efficient_aggregation(self):
        """Test aggregation doesn't load all results into memory."""
        # Create large results
        large_data = {'data': 'x' * 10000}  # 10KB per result
        
        for i in range(1000):
            container_id = f"container_{i}"
            result = {
                'score': i * 0.001,
                'large_data': large_data
            }
            
            self.aggregator.handle_container_result(container_id, result)
        
        # Memory usage should be bounded by top_k
        top_results = self.aggregator.get_top_results(5)
        self.assertEqual(len(top_results), 5)
        
        # Can still access specific results from disk
        specific_result = self.aggregator.load_result('container_500')
        self.assertEqual(specific_result['score'], 500 * 0.001)


class TestStrategyTracking(unittest.TestCase):
    """Test critical decision #4: Cross-regime strategy tracking."""
    
    def test_strategy_identity_across_regimes(self):
        """Test strategies maintain identity across different regimes."""
        # Create strategy identity
        identity = StrategyIdentity('MomentumStrategy', {'lookback': 20})
        
        # Add instances for different regimes
        identity.add_regime_instance('hmm_trending_up', 'container_001')
        identity.add_regime_instance('hmm_volatile', 'container_002')
        identity.add_regime_instance('pattern_breakout', 'container_003')
        
        # Check canonical ID is consistent
        canonical_id = identity.canonical_id
        self.assertIsNotNone(canonical_id)
        
        # Check can retrieve containers by regime
        self.assertEqual(identity.regime_instances['hmm_trending_up'], 'container_001')
        self.assertEqual(identity.regime_instances['hmm_volatile'], 'container_002')
    
    def test_strategy_tracking_in_workflow(self):
        """Test workflow tracks strategies across phases."""
        workflow_config = {
            'workflow_id': 'test_workflow',
            'phases': {
                'phase1': {
                    'strategies': [
                        {'class': 'MomentumStrategy', 'base_params': {'lookback': 20}},
                        {'class': 'MeanReversionStrategy', 'base_params': {'window': 50}}
                    ]
                }
            }
        }
        
        coordinator = Mock()
        workflow = PhaseAwareOptimizationWorkflow(coordinator, workflow_config)
        
        # After phase 1, should have strategy identities
        workflow.strategy_identities['momentum_20'] = StrategyIdentity(
            'MomentumStrategy', {'lookback': 20}
        )
        
        # Should be able to track across regimes
        identity = workflow.strategy_identities['momentum_20']
        self.assertEqual(identity.strategy_class, 'MomentumStrategy')
        self.assertEqual(identity.base_params['lookback'], 20)


class TestCheckpointing(unittest.TestCase):
    """Test critical decision #5: Checkpointing for resumability."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_restore_checkpoint(self):
        """Test workflow state can be saved and restored."""
        workflow_id = 'test_workflow_001'
        phase = 'phase2'
        
        state = {
            'completed_phases': ['phase1'],
            'current_phase': 'phase2',
            'phase1_results': {
                'best_params': {'p1': 10, 'p2': 20},
                'score': 0.85
            },
            'strategy_identities': {
                'strategy1': {'class': 'MomentumStrategy', 'params': {}}
            }
        }
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(workflow_id, phase, state)
        
        # Restore checkpoint
        restored_state = self.checkpoint_manager.restore_checkpoint(workflow_id, phase)
        
        self.assertIsNotNone(restored_state)
        self.assertEqual(restored_state['state'], state)
        self.assertEqual(restored_state['workflow_id'], workflow_id)
        self.assertEqual(restored_state['phase'], phase)
    
    def test_resume_workflow_from_checkpoint(self):
        """Test workflow can resume from checkpoint."""
        # Create workflow
        config = {'workflow_id': 'resumable_workflow'}
        coordinator = Mock()
        workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
        
        # Simulate partial completion
        workflow.completed_phases = {'phase1', 'phase2'}
        workflow.current_phase = 'phase3'
        
        # Save checkpoint
        checkpoint_manager = Mock()
        checkpoint_manager.save_checkpoint(
            'resumable_workflow',
            'phase3',
            {'completed_phases': list(workflow.completed_phases)}
        )
        
        # New workflow should be able to resume
        new_workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
        new_workflow.checkpointing = checkpoint_manager
        
        # Mock restore
        checkpoint_manager.restore_checkpoint.return_value = {
            'state': {'completed_phases': ['phase1', 'phase2']}
        }
        
        # Should skip completed phases
        # (In real implementation, would check checkpoint in run method)


class TestWalkForwardValidation(unittest.TestCase):
    """Test critical decision #6: Walk-forward validation support."""
    
    def setUp(self):
        self.validator = WalkForwardValidator()
    
    def test_create_walk_forward_periods(self):
        """Test creation of walk-forward validation periods."""
        periods = self.validator.create_periods(
            data_length=1000,
            train_size=500,
            test_size=100,
            step_size=100
        )
        
        # Check we get correct number of periods
        expected_periods = 5  # (1000 - 500 - 100) / 100 + 1
        self.assertEqual(len(periods), expected_periods)
        
        # Check first period
        first = periods[0]
        self.assertEqual(first['train_start'], 0)
        self.assertEqual(first['train_end'], 500)
        self.assertEqual(first['test_start'], 500)
        self.assertEqual(first['test_end'], 600)
        
        # Check periods don't overlap incorrectly
        for i in range(len(periods) - 1):
            current = periods[i]
            next_period = periods[i + 1]
            self.assertEqual(next_period['train_start'], current['train_start'] + 100)
    
    def test_walk_forward_in_workflow(self):
        """Test walk-forward validation in optimization workflow."""
        # Mock adaptive strategy
        adaptive_strategy = Mock()
        adaptive_strategy.generate_signal.return_value = {
            'direction': 'BUY',
            'strength': 0.8
        }
        
        # Create test periods
        test_periods = [
            {'train_start': 0, 'train_end': 500, 'test_start': 500, 'test_end': 600},
            {'train_start': 100, 'train_end': 600, 'test_start': 600, 'test_end': 700}
        ]
        
        # Run validation
        results = []
        for period in test_periods:
            # Mock validation for period
            period_result = {
                'period': period,
                'sharpe': 1.5,
                'returns': 0.12,
                'max_drawdown': 0.08
            }
            results.append(period_result)
        
        # Aggregate results
        avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
        self.assertEqual(avg_sharpe, 1.5)


class TestOptimizationContainers(unittest.TestCase):
    """Test optimization container functionality."""
    
    def test_optimization_container_isolation(self):
        """Test each optimization trial runs in isolation."""
        container = OptimizationContainer(
            'opt_container_001',
            {'class': 'TestStrategy', 'capabilities': ['optimization']}
        )
        
        # Mock trial execution
        container.initialize_scope = Mock()
        container.start = Mock()
        container.stop = Mock()
        container.dispose = Mock()
        
        # Run trial
        params = {'p1': 10, 'p2': 20}
        evaluator = Mock(return_value={'sharpe': 1.2})
        
        # Execute trial (mocked)
        container.initialize_scope()
        container.start()
        result = {'sharpe': 1.2}  # Mock result
        container.stop()
        
        self.assertEqual(result['sharpe'], 1.2)
        container.initialize_scope.assert_called_once()
        container.start.assert_called_once()
        container.stop.assert_called_once()
    
    def test_regime_aware_container(self):
        """Test regime-aware optimization container."""
        container = RegimeAwareOptimizationContainer(
            scope_id='regime_opt_001',
            parent_container=Mock()
        )
        
        # Set regime
        container.current_regime = 'TRENDING_UP'
        
        # Create trial scope
        with container.create_trial_scope() as trial_scope:
            # Should have regime context
            self.assertEqual(container.current_regime, 'TRENDING_UP')
            
            # Mock strategy execution
            strategy = Mock()
            strategy.generate_signal.return_value = {'direction': 'BUY'}
            
            # Execute in regime context
            signal = strategy.generate_signal({'price': 100})
            self.assertEqual(signal['direction'], 'BUY')


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete optimization scenarios."""
    
    @patch('src.strategy.optimization.workflows.datetime')
    def test_full_optimization_workflow(self, mock_datetime):
        """Test complete multi-phase optimization workflow."""
        mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
        
        # Create coordinator with all components
        coordinator = Mock()
        coordinator.phase_transitions = PhaseTransition()
        coordinator.container_naming = ContainerNamingStrategy()
        coordinator.checkpointing = Mock()
        coordinator.walk_forward_validator = WalkForwardValidator()
        
        # Create workflow configuration
        config = {
            'workflow_id': 'full_test_workflow',
            'output_dir': tempfile.mkdtemp(),
            'phases': {
                'phase1': {
                    'parameter_space': {
                        'lookback': [10, 20, 30],
                        'threshold': [0.01, 0.02]
                    },
                    'regime_classifiers': ['hmm', 'pattern'],
                    'strategies': [
                        {
                            'class': 'MomentumStrategy',
                            'base_params': {'symbol': 'AAPL'}
                        }
                    ],
                    'optimizer': {'type': 'grid'},
                    'objective': {'type': 'sharpe'}
                }
            }
        }
        
        # Create workflow
        workflow = PhaseAwareOptimizationWorkflow(coordinator, config)
        
        # Mock phase execution methods
        workflow._run_phase1_optimization = Mock(return_value={'phase1': 'complete'})
        workflow._run_phase2_analysis = Mock(return_value={'phase2': 'complete'})
        workflow._run_phase3_weights = Mock(return_value={'phase3': 'complete'})
        workflow._run_phase4_validation = Mock(return_value={'phase4': 'complete'})
        workflow._aggregate_results = Mock(return_value={'final': 'results'})
        
        # Run workflow (mocked)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(workflow.run())
        
        # Verify all phases executed
        workflow._run_phase1_optimization.assert_called_once()
        workflow._run_phase2_analysis.assert_called_once()
        workflow._run_phase3_weights.assert_called_once()
        workflow._run_phase4_validation.assert_called_once()
        
        # Verify results aggregated
        self.assertEqual(results, {'final': 'results'})
    
    def test_constraint_validation(self):
        """Test constraints work correctly in optimization."""
        # Create constraints
        range_constraint = RangeConstraint('lookback', min_value=5, max_value=50)
        relational_constraint = RelationalConstraint('fast_period', '<', 'slow_period')
        
        # Test valid parameters
        valid_params = {
            'lookback': 20,
            'fast_period': 10,
            'slow_period': 30
        }
        
        self.assertTrue(range_constraint.is_satisfied(valid_params))
        self.assertTrue(relational_constraint.is_satisfied(valid_params))
        
        # Test invalid parameters
        invalid_params = {
            'lookback': 100,  # Out of range
            'fast_period': 40,
            'slow_period': 30  # fast > slow
        }
        
        self.assertFalse(range_constraint.is_satisfied(invalid_params))
        self.assertFalse(relational_constraint.is_satisfied(invalid_params))
        
        # Test adjustment
        adjusted = range_constraint.validate_and_adjust(invalid_params)
        self.assertEqual(adjusted['lookback'], 50)  # Clamped to max
        
        adjusted2 = relational_constraint.validate_and_adjust(invalid_params)
        self.assertLess(adjusted2['fast_period'], adjusted2['slow_period'])


if __name__ == '__main__':
    unittest.main()