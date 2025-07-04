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
    # Check if streaming tracer should be used
    # Use streaming tracer for:
    # 1. Large runs (>2000 bars)
    # 2. When explicitly requested
    # 3. When signal generation mode is active (to get metadata storage)
    # 4. When universal mode is active (to get portfolio/execution traces)
    max_bars = context['config'].get('max_bars', 0)
    mode = context.get('mode')
    is_signal_generation = mode == 'signal_generation'
    is_universal = mode == 'universal'
    use_streaming = max_bars > 2000 or context['config'].get('streaming_tracer', False) or is_signal_generation or is_universal
    
    if use_streaming:
        from .observers.global_streaming_tracer import GlobalStreamingTracer
        logger.info(f"Using GlobalStreamingTracer for {max_bars} bars (with global trace storage)")
    else:
        from .observers.multi_strategy_tracer import MultiStrategyTracer
    from .types import EventType
    
    # Get config file path to determine where to save results
    config_file = context.get('config', {}).get('metadata', {}).get('config_file', '')
    if not config_file:
        # Try context metadata for config_file
        config_file = context.get('metadata', {}).get('config_file', '')
    
    # If we have a config file, save results relative to it
    if config_file:
        config_path = Path(config_file)
        # Save results in a results/ subdirectory next to the config file
        workspace_base = config_path.parent / 'results'
    else:
        # Fallback to current directory if no config file specified
        workspace_base = Path('./results')
    
    # Create timestamp for this run
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get study configuration for organized workspace structure
    results_dir = context['config'].get('results_dir')
    wfv_window = context['config'].get('wfv_window')
    phase = context['config'].get('phase')
    
    # Create workspace directory based on configuration
    if results_dir and wfv_window and phase:
        # WFV execution: results/<results_dir>/window_XX_phase/
        run_dir = f"window_{wfv_window:02d}_{phase}"
        full_workspace_path = workspace_base / results_dir / run_dir
        logger.info(f"WFV workspace: {full_workspace_path}")
    elif results_dir:
        # Study execution without WFV: results/<results_dir>/<timestamp>/
        run_dir = timestamp
        full_workspace_path = workspace_base / results_dir / run_dir
        logger.info(f"Study workspace: {full_workspace_path}")
    else:
        # Standard execution: results/<timestamp>/
        run_dir = timestamp
        full_workspace_path = workspace_base / run_dir
        
        # Create results directory if it doesn't exist
        workspace_base.mkdir(parents=True, exist_ok=True)
        
        # Log the full workspace path for debugging
        logger.info(f"📁 Creating workspace at: {full_workspace_path}")
        
        # Create symlink to latest run in results directory
        latest_link = workspace_base / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(run_dir)
            logger.info(f"Workspace: {full_workspace_path} (latest -> {run_dir})")
        except Exception as e:
            # Symlinks might not work on all systems
            logger.debug(f"Could not create symlink: {e}")
            logger.info(f"Workspace: {full_workspace_path}")
    
    # For signal generation mode, we don't need to pre-specify strategy IDs
    # The tracer will capture all signals based on strategy hashes
    strategy_ids = None
    classifier_ids = None
    
    # Log that we're using hash-based strategy identification
    logger.info("Using hash-based strategy identification - will trace all signals")
    
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
        trace_settings = context['config'].get('execution', {}).get('trace_settings', {})
        write_interval = trace_settings.get('write_interval', 0)
        write_on_changes = trace_settings.get('write_on_changes', 0)
        
        # Include mode in full config for metadata
        full_config = context.get('config', {}).copy()
        if 'mode' in context:
            full_config['mode'] = context['mode']
            
        tracer = GlobalStreamingTracer(
            workspace_path=str(full_workspace_path),
            workflow_id=workspace_base.name,  # Use parent directory name as workflow ID
            managed_strategies=strategy_ids if strategy_ids else None,
            managed_classifiers=classifier_ids if classifier_ids else None,
            data_source_config=data_source_config,
            write_interval=write_interval,
            write_on_changes=write_on_changes,
            full_config=full_config
        )
    else:
        tracer = MultiStrategyTracer(
            workspace_path=str(full_workspace_path),
            workflow_id=workspace_base.parent.name,  # Use config directory name as workflow ID
            managed_strategies=strategy_ids if strategy_ids else None,
            managed_classifiers=classifier_ids if classifier_ids else None,
            data_source_config=data_source_config,
            full_config=context.get('config', {})
        )
    
    # Attach to root event bus - use the actual root container's bus if available
    root_container = topology.get('containers', {}).get('root')
    if root_container and hasattr(root_container, 'event_bus'):
        # Use the actual root container's event bus
        root_bus = root_container.event_bus
        root_bus.attach_observer(tracer)
        logger.info(f"Tracer attached as observer to root event bus")
        # Also explicitly subscribe to portfolio events to ensure we receive them
        from .types import EventType
        # ORDER events don't require a filter
        root_bus.subscribe(EventType.ORDER.value, tracer.on_event)
        # FILL events require a filter - use a permissive filter that accepts all events
        root_bus.subscribe(EventType.FILL.value, tracer.on_event, filter_func=lambda event: True)
        # POSITION events don't require filters
        root_bus.subscribe(EventType.POSITION_OPEN.value, tracer.on_event)
        root_bus.subscribe(EventType.POSITION_CLOSE.value, tracer.on_event)
        logger.info(f"Tracer explicitly subscribed to ORDER, FILL, and POSITION events")
    else:
        # Fallback to context bus
        root_bus = context.get('root_event_bus')
        if root_bus:
            root_bus.attach_observer(tracer)
            # Also subscribe to all portfolio events
            from .types import EventType
            root_bus.subscribe(EventType.ORDER.value, tracer.on_event)
            root_bus.subscribe(EventType.FILL.value, tracer.on_event, filter_func=lambda event: True)
            root_bus.subscribe(EventType.POSITION_OPEN.value, tracer.on_event)
            root_bus.subscribe(EventType.POSITION_CLOSE.value, tracer.on_event)
            logger.info(f"Tracer subscribed to all portfolio events on fallback bus")
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