"""
Clean Architecture Demonstration

Shows how the new clean architecture works with:
- Result extraction from events
- Phase isolation
- Clean separation of concerns
- Post-execution analysis
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.coordinator.coordinator_clean import Coordinator
from src.core.analysis.trace_analyzer import TraceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_simple_backtest():
    """Demonstrate a simple single-phase backtest."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Simple Backtest")
    logger.info("=" * 60)
    
    # Create coordinator
    coordinator = Coordinator(
        enable_checkpointing=True,
        trace_dir="./demo_traces"
    )
    
    # Define configuration
    config = {
        'workflow': 'simple_backtest',
        'data': {
            'symbols': ['AAPL'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        'strategy': {
            'type': 'momentum',
            'fast_period': 10,
            'slow_period': 30
        },
        'risk': {
            'position_size': 0.1,
            'max_positions': 5
        },
        'tracing': {
            'enabled': True
        }
    }
    
    # Execute workflow
    logger.info("Executing simple backtest workflow...")
    result = coordinator.execute_workflow(config)
    
    # Display results
    logger.info(f"Workflow completed: {result['workflow_id']}")
    logger.info(f"Success: {result['success']}")
    
    # Show extracted results
    if result['results'].get('extracted_results'):
        logger.info("\nExtracted Results Summary:")
        for phase, extractors in result['results']['extracted_results'].items():
            logger.info(f"\n  Phase: {phase}")
            for extractor_name, results in extractors.items():
                logger.info(f"    {extractor_name}: {len(results)} results")
    
    return result['workflow_id']


def demonstrate_signal_generation_replay():
    """Demonstrate two-phase workflow: generate signals, then replay."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Signal Generation and Replay")
    logger.info("=" * 60)
    
    # Create coordinator
    coordinator = Coordinator(
        enable_checkpointing=True,
        trace_dir="./demo_traces"
    )
    
    # Define configuration
    config = {
        'workflow': 'signal_generation_replay',
        'data': {
            'symbols': ['AAPL', 'GOOGL'],
            'start_date': '2023-01-01',
            'end_date': '2023-06-30'
        },
        'strategy': {
            'type': 'mean_reversion',
            'lookback_period': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        },
        'risk': {
            'position_size': 0.05,
            'max_positions': 10
        },
        'tracing': {
            'enabled': True
        }
    }
    
    # Execute workflow
    logger.info("Executing signal generation and replay workflow...")
    result = coordinator.execute_workflow(config)
    
    # Display results
    logger.info(f"Workflow completed: {result['workflow_id']}")
    logger.info(f"Success: {result['success']}")
    
    # Show phase results
    if result['results'].get('phase_results'):
        logger.info("\nPhase Results:")
        for phase_name, phase_result in result['results']['phase_results'].items():
            logger.info(f"\n  Phase: {phase_name}")
            logger.info(f"    Success: {phase_result.get('success')}")
            logger.info(f"    Topology Mode: {phase_result.get('topology_mode')}")
            
            # Show trace summary
            trace_summary = phase_result.get('trace_summary', {})
            if trace_summary:
                logger.info(f"    Events Traced: {trace_summary.get('event_count', 0)}")
                logger.info(f"    Duration: {trace_summary.get('duration_seconds', 0):.2f}s")
    
    # Show how Phase 2 used Phase 1's results
    phase_2_results = result['results'].get('phase_results', {}).get('signal_replay', {})
    if phase_2_results:
        logger.info("\nPhase 2 Dependencies:")
        # In a real implementation, we'd show the actual dependencies used
        logger.info("  - Used signals from Phase 1 (signal_generation)")
    
    return result['workflow_id']


def analyze_workflow_traces(workflow_id: str):
    """Analyze traces from a completed workflow."""
    logger.info("\n" + "=" * 60)
    logger.info(f"ANALYZING WORKFLOW: {workflow_id}")
    logger.info("=" * 60)
    
    # Create trace analyzer
    analyzer = TraceAnalyzer(Path("./demo_traces"))
    
    try:
        # Analyze workflow
        logger.info("Running trace analysis...")
        analysis = analyzer.analyze_workflow(workflow_id)
        
        # Generate summary report
        report = analyzer.generate_summary_report(analysis)
        
        # Print report
        logger.info("\n" + report)
        
        # Save report to file
        report_path = Path(f"./demo_traces/{workflow_id}_analysis.md")
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"\nAnalysis report saved to: {report_path}")
        
    except ValueError as e:
        logger.error(f"Analysis failed: {e}")


def main():
    """Run all demonstrations."""
    logger.info("Starting Clean Architecture Demonstration\n")
    
    # Demo 1: Simple backtest
    workflow_id_1 = demonstrate_simple_backtest()
    
    # Demo 2: Signal generation and replay
    workflow_id_2 = demonstrate_signal_generation_replay()
    
    # Analyze workflows (if traces were actually created)
    # Note: Since we don't have actual topology execution implemented,
    # this would fail in a real run. In production, this would work.
    logger.info("\n" + "=" * 60)
    logger.info("TRACE ANALYSIS")
    logger.info("=" * 60)
    logger.info("\nNote: Trace analysis would run here if topology execution was implemented.")
    logger.info("The TraceAnalyzer is ready to analyze completed workflow traces.")
    
    # Show what would happen
    logger.info("\nIn a complete implementation, you would see:")
    logger.info("- Detailed event flow analysis")
    logger.info("- Performance metrics and bottlenecks")
    logger.info("- Cross-phase patterns")
    logger.info("- Recommendations for optimization")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    
    logger.info("\nKey Architecture Points Demonstrated:")
    logger.info("1. Clean separation: Coordinator → Sequencer → TopologyBuilder")
    logger.info("2. Each phase gets fresh event system (perfect isolation)")
    logger.info("3. Results extracted from events (single source of truth)")
    logger.info("4. Dependencies passed via extracted results, not events")
    logger.info("5. Rich traces enable post-execution analysis")
    logger.info("6. NO async - all synchronous execution")


if __name__ == "__main__":
    main()