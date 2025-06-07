"""
Comprehensive test suite for the refactored coordinator system.

Tests the clean architecture with:
- Result extraction from event streams
- Multi-phase workflow execution
- Decorator-based workflow discovery
- Enhanced result streaming
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from pathlib import Path
import tempfile
import json

from src.core.coordinator.coordinator_clean import Coordinator, WorkflowPattern
from src.core.coordinator.sequencer_clean import Sequencer
from src.core.coordinator.topology_clean import TopologyBuilder
from src.core.events.result_extraction import (
    PortfolioMetricsExtractor,
    SignalExtractor,
    FillExtractor
)
from src.core.events.enhanced_tracer import EnhancedEventTracer, StreamingResultProcessor
from src.core.events.types import Event, EventType
from src.core.components.discovery import workflow, get_component_registry


class TestResultExtraction:
    """Test result extraction from event streams."""
    
    def test_portfolio_metrics_extraction(self):
        """Test extracting portfolio metrics from events."""
        extractor = PortfolioMetricsExtractor()
        
        # Create portfolio update event
        event = Event(
            type=EventType.PORTFOLIO_UPDATE,
            source='test_portfolio',
            data={
                'timestamp': datetime.now(),
                'total_value': 100000,
                'cash': 50000,
                'positions_value': 50000,
                'pnl': 5000,
                'returns': 0.05
            }
        )
        
        # Test extraction
        assert extractor.can_extract(event)
        result = extractor.extract(event)
        
        assert result is not None
        assert result['total_value'] == 100000
        assert result['pnl'] == 5000
        assert result['returns'] == 0.05
        assert 'timestamp' in result
    
    def test_signal_extraction(self):
        """Test extracting signals from events."""
        extractor = SignalExtractor()
        
        # Create signal event
        event = Event(
            type=EventType.SIGNAL_GENERATED,
            source='test_strategy',
            data={
                'timestamp': datetime.now(),
                'symbol': 'AAPL',
                'signal_type': 'BUY',
                'confidence': 0.85,
                'strategy': 'momentum',
                'features': {
                    'sma_20': 150.5,
                    'rsi': 65.2
                }
            }
        )
        
        assert extractor.can_extract(event)
        result = extractor.extract(event)
        
        assert result['symbol'] == 'AAPL'
        assert result['signal_type'] == 'BUY'
        assert result['confidence'] == 0.85
        assert 'features' in result
    
    def test_enhanced_event_tracer(self):
        """Test enhanced event tracer with integrated extraction."""
        extractors = [
            PortfolioMetricsExtractor(),
            SignalExtractor(),
            FillExtractor()
        ]
        
        tracer = EnhancedEventTracer(
            trace_id='test_trace',
            result_extractors=extractors
        )
        
        # Trace various events
        events = [
            Event(
                type=EventType.SIGNAL_GENERATED,
                source='strategy1',
                data={'symbol': 'AAPL', 'signal_type': 'BUY'}
            ),
            Event(
                type=EventType.ORDER_FILLED,
                source='broker',
                data={'symbol': 'AAPL', 'quantity': 100, 'price': 150.0}
            ),
            Event(
                type=EventType.PORTFOLIO_UPDATE,
                source='portfolio',
                data={'total_value': 100000, 'pnl': 1000}
            )
        ]
        
        for event in events:
            tracer.trace_event(event)
        
        # Check extracted results
        results = tracer.get_extracted_results()
        
        assert 'SignalExtractor' in results
        assert 'FillExtractor' in results
        assert 'PortfolioMetricsExtractor' in results
        
        assert len(results['SignalExtractor']) == 1
        assert len(results['FillExtractor']) == 1
        assert len(results['PortfolioMetricsExtractor']) == 1
        
        # Check summary
        summary = tracer.get_summary()
        assert summary['event_count'] == 3
        assert summary['extraction_results']['SignalExtractor'] == 1


class TestStreamingResultProcessor:
    """Test streaming result processor with multiple output formats."""
    
    def test_streaming_processor_parquet(self):
        """Test streaming results to Parquet format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractors = [PortfolioMetricsExtractor()]
            
            processor = StreamingResultProcessor(
                extractors=extractors,
                output_config={
                    'format': 'parquet',
                    'directory': tmpdir,
                    'buffer_size': 2
                }
            )
            
            # Process events
            events = [
                Event(
                    type=EventType.PORTFOLIO_UPDATE,
                    source='portfolio',
                    data={'total_value': 100000 + i * 1000, 'pnl': i * 100}
                )
                for i in range(5)
            ]
            
            for event in events:
                processor.process_event(event)
            
            # Flush remaining
            processor.flush_all()
            
            # Check files created
            output_files = list(Path(tmpdir).glob('*.parquet'))
            assert len(output_files) > 0
            
            # Check statistics
            stats = processor.get_statistics()
            assert stats['events_processed'] == 5
            assert stats['results_extracted']['PortfolioMetricsExtractor'] == 5
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation in streaming processor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = StreamingResultProcessor(
                extractors=[PortfolioMetricsExtractor()],
                output_config={'directory': tmpdir}
            )
            
            # Process events with metrics
            events = [
                Event(
                    type=EventType.PORTFOLIO_UPDATE,
                    source=f'container_{i}',
                    data={
                        'container_id': f'container_{i}',
                        'sharpe_ratio': 1.5 + i * 0.1,
                        'total_value': 100000 + i * 10000,
                        'pnl': 1000 * i,
                        'max_drawdown': 0.1 - i * 0.01
                    }
                )
                for i in range(3)
            ]
            
            for event in events:
                processor.process_event(event)
            
            # Get aggregated results
            aggregated = processor.get_aggregated_results()
            
            assert 'metrics_summary' in aggregated
            assert 'best_sharpe' in aggregated['metrics_summary']
            assert 'best_container' in aggregated['metrics_summary']
            assert aggregated['metrics_summary']['best_container'] == 'container_2'
            
            # Check top performers
            assert 'top_performers' in aggregated
            assert len(aggregated['top_performers']) <= 3


class TestWorkflowDiscovery:
    """Test decorator-based workflow discovery."""
    
    def test_workflow_decorator(self):
        """Test workflow registration via decorator."""
        # Clear registry for test
        registry = get_component_registry()
        
        # Define test workflow
        @workflow(
            name='test_workflow',
            description='Test workflow for unit tests',
            tags=['test', 'example']
        )
        def test_workflow_def():
            return {
                'phases': [
                    {'name': 'phase1', 'topology': 'backtest'},
                    {'name': 'phase2', 'topology': 'analysis'}
                ]
            }
        
        # Check registration
        workflow_info = registry.get_component('test_workflow')
        assert workflow_info is not None
        assert workflow_info.component_type == 'workflow'
        assert 'test' in workflow_info.metadata['tags']
        
        # Check factory
        workflow_def = workflow_info.factory()
        assert len(workflow_def['phases']) == 2
    
    def test_coordinator_workflow_discovery(self):
        """Test coordinator discovers workflows from registry."""
        coordinator = Coordinator()
        
        # Should have discovered built-in workflows
        assert 'simple_backtest' in coordinator.workflow_patterns
        assert 'signal_generation' in coordinator.workflow_patterns
        assert 'signal_generation_replay' in coordinator.workflow_patterns
        
        # Check if discovered decorated workflows
        if 'test_workflow' in coordinator.workflow_patterns:
            pattern = coordinator.workflow_patterns['test_workflow']
            assert len(pattern.phases) == 2


class TestCleanSequencer:
    """Test clean sequencer without PhaseDataManager."""
    
    @pytest.mark.asyncio
    async def test_sequencer_with_extracted_results(self):
        """Test sequencer uses extracted results for dependencies."""
        # Mock components
        mock_topology_builder = MagicMock()
        mock_topology = {
            'containers': {'strategy': MagicMock()},
            'communication': {'event_bus': MagicMock()}
        }
        mock_topology_builder.build_topology.return_value = mock_topology
        
        # Create sequencer with mock
        sequencer = Sequencer()
        
        # Define multi-phase pattern
        pattern = {
            'phases': [
                {
                    'name': 'generate_signals',
                    'topology': 'signal_generation',
                    'dependencies': []
                },
                {
                    'name': 'replay_signals',
                    'topology': 'signal_replay',
                    'dependencies': [{
                        'phase': 'generate_signals',
                        'result_types': ['signals']
                    }]
                }
            ]
        }
        
        # Mock execution with extracted results
        with patch.object(sequencer, '_execute_phase') as mock_execute:
            # First phase returns signals
            mock_execute.side_effect = [
                {
                    'success': True,
                    'extracted_results': {
                        'SignalExtractor': [
                            {'symbol': 'AAPL', 'signal_type': 'BUY'},
                            {'symbol': 'GOOGL', 'signal_type': 'SELL'}
                        ]
                    }
                },
                {
                    'success': True,
                    'extracted_results': {}
                }
            ]
            
            result = await sequencer.execute_phases(
                pattern=pattern,
                config={},
                context={'workflow_id': 'test'},
                topology_builder=mock_topology_builder
            )
            
            assert result['success']
            assert result['completed_phases'] == ['generate_signals', 'replay_signals']
            
            # Check second phase received signals
            second_call = mock_execute.call_args_list[1]
            phase_config = second_call[1]['config']
            assert 'dependencies' in phase_config
            assert 'signals' in phase_config['dependencies']
            assert len(phase_config['dependencies']['signals']) == 2


class TestCoordinatorIntegration:
    """Test integrated coordinator with all components."""
    
    def test_coordinator_execution_flow(self):
        """Test complete execution flow through coordinator."""
        coordinator = Coordinator()
        
        # Execute simple workflow
        config = {
            'workflow': 'simple_backtest',
            'data': {'symbol': 'AAPL'},
            'strategy': {'type': 'momentum'}
        }
        
        with patch.object(coordinator.sequencer, 'execute_phases') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'completed_phases': ['backtest'],
                'extracted_results': {
                    'backtest': {
                        'PortfolioMetricsExtractor': [
                            {'total_value': 110000, 'returns': 0.1}
                        ]
                    }
                }
            }
            
            result = coordinator.execute_workflow(config)
            
            assert result['success']
            assert 'workflow_id' in result
            assert 'context' in result
            
            # Check sequencer was called correctly
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            pattern = call_args[0][0]
            assert pattern['name'] == 'simple_backtest'
            assert len(pattern['phases']) == 1
    
    def test_custom_workflow_pattern(self):
        """Test registering and executing custom workflow pattern."""
        coordinator = Coordinator()
        
        # Register custom pattern
        custom_pattern = WorkflowPattern(
            name='custom_test',
            phases=[
                {'name': 'phase1', 'topology': 'backtest'},
                {'name': 'phase2', 'topology': 'analysis', 
                 'dependencies': [{'phase': 'phase1', 'result_types': ['metrics']}]}
            ]
        )
        
        coordinator.register_workflow_pattern('custom_test', custom_pattern)
        
        # Execute custom workflow
        config = {'workflow': 'custom_test'}
        
        with patch.object(coordinator.sequencer, 'execute_phases') as mock_execute:
            mock_execute.return_value = {'success': True}
            
            result = coordinator.execute_workflow(config)
            
            # Check pattern was used
            call_args = mock_execute.call_args
            pattern_dict = call_args[0][0]
            assert len(pattern_dict['phases']) == 2
            assert pattern_dict['phases'][1]['dependencies'][0]['phase'] == 'phase1'


class TestTraceAnalysis:
    """Test trace analysis capabilities."""
    
    def test_trace_analyzer_pattern_detection(self):
        """Test pattern detection in trace analysis."""
        from src.core.analysis.trace_analyzer import TraceAnalyzer
        
        analyzer = TraceAnalyzer()
        
        # Create test trace
        events = [
            Event(
                type=EventType.SIGNAL_GENERATED,
                source='strategy1',
                timestamp=datetime.now(),
                data={'symbol': 'AAPL', 'signal_type': 'BUY'}
            ),
            Event(
                type=EventType.ORDER_SUBMITTED,
                source='risk_manager',
                timestamp=datetime.now(),
                data={'symbol': 'AAPL', 'quantity': 100}
            ),
            Event(
                type=EventType.ORDER_FILLED,
                source='broker',
                timestamp=datetime.now(),
                data={'symbol': 'AAPL', 'quantity': 100, 'price': 150}
            )
        ]
        
        # Analyze phase
        analysis = analyzer.analyze_phase('test_phase', events)
        
        assert analysis['event_count'] == 3
        assert 'event_patterns' in analysis
        assert 'common_sequences' in analysis
        
        # Should detect signal->order->fill pattern
        sequences = analysis['common_sequences']
        assert any('SIGNAL_GENERATED' in str(seq) and 'ORDER_FILLED' in str(seq) 
                  for seq in sequences)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])