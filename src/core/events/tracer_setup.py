"""
Multi-Strategy Tracer Setup Module

This module provides functionality for setting up multi-strategy tracers
with proper workspace organization and event bus attachment.

Moved from core.coordinator.topology for better separation of concerns.
"""

import os
import uuid
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_multi_strategy_tracer(topology: Dict[str, Any], 
                               context: Dict[str, Any], 
                               tracing_config: Dict[str, Any]) -> None:
    """Set up unified multi-strategy tracer on the root event bus."""
    # Check if streaming tracer should be used for large runs
    max_bars = context['config'].get('max_bars', 0)
    use_streaming = max_bars > 2000 or context['config'].get('streaming_tracer', False)
    
    if use_streaming:
        from .observers.streaming_multi_strategy_tracer import StreamingMultiStrategyTracer
        logger.info(f"Using StreamingMultiStrategyTracer for {max_bars} bars")
    else:
        from .observers.multi_strategy_tracer import MultiStrategyTracer
    from .types import EventType
    
    # Get workspace path from trace settings
    trace_settings = context['config'].get('execution', {}).get('trace_settings', {})
    workspace_path = trace_settings.get('storage', {}).get('base_dir', './workspaces')
    
    # Get study configuration for organized workspace structure
    results_dir = context['config'].get('results_dir')
    wfv_window = context['config'].get('wfv_window')
    phase = context['config'].get('phase')
    
    # Create workspace directory based on study organization
    if results_dir and wfv_window and phase:
        # WFV execution: study_name/window_XX_phase/
        workspace_name = f"window_{wfv_window:02d}_{phase}"
        full_workspace_path = os.path.join(workspace_path, results_dir, workspace_name)
        logger.info(f"WFV workspace: {results_dir}/window_{wfv_window:02d}_{phase}")
    elif results_dir:
        # Study execution without WFV: study_name/run_unique_id/
        unique_run_id = str(uuid.uuid4())[:8]
        workspace_name = f"run_{unique_run_id}"
        full_workspace_path = os.path.join(workspace_path, results_dir, workspace_name)
        logger.info(f"Study workspace: {results_dir}/{workspace_name}")
    else:
        # Fallback to legacy naming for backwards compatibility
        config_name = context.get('metadata', {}).get('config_name', 'unknown_config')
        if not config_name or config_name == 'unknown_config':
            config_file = context.get('metadata', {}).get('config_file', '')
            if config_file:
                config_name = Path(config_file).stem
            else:
                config_name = 'signal_generation'
        
        unique_run_id = str(uuid.uuid4())[:8]
        workspace_name = f"{config_name}_{unique_run_id}"
        full_workspace_path = os.path.join(workspace_path, workspace_name)
        logger.info(f"Legacy workspace: {workspace_name}")
    
    # Get all strategy and classifier IDs from expanded configurations
    strategy_ids = []
    classifier_ids = []
    
    # Extract strategy IDs from expanded strategies
    for strategy in context['config'].get('strategies', []):
        strategy_name = strategy.get('name', '')
        if strategy_name:
            # Add symbol prefix if we have symbols
            symbols = context['config'].get('symbols', ['SPY'])
            for symbol in symbols:
                strategy_ids.append(f"{symbol}_{strategy_name}")
    
    # Extract classifier IDs from expanded classifiers
    for classifier in context['config'].get('classifiers', []):
        classifier_name = classifier.get('name', '')
        if classifier_name:
            # Add symbol prefix if we have symbols
            symbols = context['config'].get('symbols', ['SPY'])
            for symbol in symbols:
                classifier_ids.append(f"{symbol}_{classifier_name}")
    
    # Extract data source configuration for source metadata
    data_source_config = {
        'data_dir': context['config'].get('data_dir', './data'),
        'data_source': context['config'].get('data_source', 'csv'),
        'symbols': context['config'].get('symbols', []),
        'timeframes': context['config'].get('timeframes', ['1m'])
    }
    
    # Create the multi-strategy tracer
    if use_streaming:
        # Get write settings from config (default to no periodic writes)
        write_interval = trace_settings.get('write_interval', 0)
        write_on_changes = trace_settings.get('write_on_changes', 0)
        
        tracer = StreamingMultiStrategyTracer(
            workspace_path=full_workspace_path,
            workflow_id=config_name,
            managed_strategies=strategy_ids if strategy_ids else None,
            managed_classifiers=classifier_ids if classifier_ids else None,
            data_source_config=data_source_config,
            write_interval=write_interval,
            write_on_changes=write_on_changes
        )
    else:
        tracer = MultiStrategyTracer(
            workspace_path=full_workspace_path,
            workflow_id=config_name,
            managed_strategies=strategy_ids if strategy_ids else None,
            managed_classifiers=classifier_ids if classifier_ids else None,
            data_source_config=data_source_config
        )
    
    # Attach to root event bus - use the actual root container's bus if available
    root_container = topology.get('containers', {}).get('root')
    if root_container and hasattr(root_container, 'event_bus'):
        # Use the actual root container's event bus
        root_bus = root_container.event_bus
        root_bus.attach_observer(tracer)
    else:
        # Fallback to context bus
        root_bus = context.get('root_event_bus')
        if root_bus:
            root_bus.attach_observer(tracer)
        else:
            logger.warning("No event bus found for MultiStrategyTracer attachment")
            return
    
    # Store tracer in topology for finalization
    topology['multi_strategy_tracer'] = tracer
    logger.info(f"MultiStrategyTracer attached to event bus with workspace: {full_workspace_path}")


def finalize_multi_strategy_tracer(topology: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Finalize MultiStrategyTracer if present in topology."""
    if 'multi_strategy_tracer' not in topology:
        return None
        
    logger.info("Finalizing MultiStrategyTracer")
    try:
        tracer = topology['multi_strategy_tracer']
        tracer_results = tracer.finalize()
        logger.debug(f"MultiStrategyTracer finalized: {tracer_results.get('compression_ratio', 0):.1f}% compression")
        logger.info(f"Saved {len(tracer_results.get('components', {}))} component signal files (compression: {tracer_results.get('compression_ratio', 0):.1f}%)")
        return tracer_results
    except Exception as e:
        logger.error(f"Error finalizing MultiStrategyTracer: {e}")
        raise