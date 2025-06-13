"""
Sparse Portfolio Tracer

Only stores signal changes, not every redundant signal.
"""

from typing import Optional, Dict, Any, List, Set
import logging
from datetime import datetime

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.temporal_sparse_storage import TemporalSparseStorage

logger = logging.getLogger(__name__)


class SparsePortfolioTracer(EventObserverProtocol):
    """
    Portfolio tracer that only stores signal changes.
    
    Instead of storing 50 identical signals, stores only the transitions.
    """
    
    def __init__(self, 
                 container_id: str,
                 workflow_id: str,
                 managed_strategies: List[str],
                 managed_classifiers: Optional[List[str]] = None,
                 storage_config: Optional[Dict[str, Any]] = None,
                 portfolio_container: Optional[Any] = None):
        """Initialize sparse tracer."""
        self.container_id = container_id
        self.workflow_id = workflow_id
        self.managed_strategies = set(managed_strategies)
        self.managed_classifiers = set(managed_classifiers or [])
        self.portfolio_container = portfolio_container
        
        # Use temporal sparse storage with proper workspace organization
        config = storage_config or {}
        base_dir = config.get('base_dir', './workspaces')  # Use workspaces to match system
        
        # Use 'tmp' instead of 'unknown' for unspecified workflows
        if workflow_id == 'unknown' or not workflow_id:
            workflow_id = 'tmp'
        
        # Create run_id based on timestamp and workflow (not container)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Organize by workflow/run_id/container_id
        self.storage = TemporalSparseStorage(
            base_dir=f"{base_dir}/{workflow_id}/{run_id}",
            run_id=container_id
        )
        
        logger.info(f"Sparse storage path: {base_dir}/{workflow_id}/{run_id}")
        
        self._total_signals = 0
        self._stored_changes = 0
        self._current_bar_count = 0
        
        logger.info(f"SparsePortfolioTracer initialized for {container_id} "
                   f"managing strategies: {managed_strategies}, classifiers: {managed_classifiers or []}")
    
    def on_event(self, event: Event) -> None:
        """Process event - store signal and classification changes."""
        # Track bar count
        if event.event_type == EventType.BAR.value:
            self._current_bar_count += 1
        # Process both SIGNAL and CLASSIFICATION events
        elif event.event_type == EventType.SIGNAL.value:
            self._process_signal_event(event)
        elif event.event_type == EventType.CLASSIFICATION.value:
            self._process_classification_event(event)
    
    def _process_signal_event(self, event: Event) -> None:
        """Process SIGNAL events from strategies."""
        payload = event.payload
        strategy_id = payload.get('strategy_id', '')
        
        # Check if this signal is from a managed strategy
        is_managed = False
        for strategy_name in self.managed_strategies:
            if strategy_name in strategy_id:
                is_managed = True
                break
                
        if not is_managed:
            return
        
        # Process the signal
        self._total_signals += 1
        
        # Get trading direction
        direction = payload.get('direction', 'flat')
        
        was_change = self.storage.process_signal(
            symbol=payload.get('symbol', 'UNKNOWN'),
            direction=str(direction),
            strategy_id=strategy_id,
            timestamp=event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
            price=payload.get('price', 0.0),
            bar_index=self._current_bar_count
        )
        
        if was_change:
            self._stored_changes += 1
            
        # Log compression ratio periodically
        if self._total_signals % 10 == 0:
            ratio = (self._stored_changes / self._total_signals * 100) if self._total_signals > 0 else 0
            logger.info(f"Sparse storage: {self._stored_changes}/{self._total_signals} "
                       f"signals stored ({ratio:.1f}%)")
    
    def _process_classification_event(self, event: Event) -> None:
        """Process CLASSIFICATION events from classifiers."""
        payload = event.payload
        classifier_id = payload.get('classifier_id', '')
        
        # Check if this classification is from a managed classifier
        is_managed = False
        for classifier_name in self.managed_classifiers:
            if classifier_name in classifier_id:
                is_managed = True
                break
                
        if not is_managed:
            return
        
        # Process the classification
        self._total_signals += 1
        
        # Get regime classification
        regime = payload.get('regime', 'unknown')
        
        # Classifications are already sparse (only published on change)
        # So every event is a change that should be stored
        was_change = self.storage.process_signal(
            symbol=payload.get('symbol', 'UNKNOWN'),
            direction=str(regime),  # Store regime as direction
            strategy_id=classifier_id,
            timestamp=event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
            price=0.0,  # Classifications don't have prices
            bar_index=self._current_bar_count
        )
        
        if was_change:
            self._stored_changes += 1
            
        # Log compression ratio periodically
        if self._total_signals % 10 == 0:
            ratio = (self._stored_changes / self._total_signals * 100) if self._total_signals > 0 else 0
            logger.info(f"Sparse storage: {self._stored_changes}/{self._total_signals} "
                       f"events stored ({ratio:.1f}%)")
    
    def flush(self) -> None:
        """Save signal changes to disk with performance metrics and strategy parameters."""
        if self._stored_changes > 0:
            # Get performance metrics from portfolio container if available
            performance_metrics = None
            if self.portfolio_container and hasattr(self.portfolio_container, 'get_metrics'):
                performance_metrics = self.portfolio_container.get_metrics()
                logger.info(f"Including performance metrics in sparse storage: {performance_metrics}")
            
            # Get strategy and classifier parameters from container config
            strategy_params = {}
            classifier_params = {}
            if self.portfolio_container:
                config = self.portfolio_container.config.config
                logger.info(f"Portfolio config keys: {list(config.keys())}")
                
                # Look for strategies in different places
                strategies = config.get('strategies', [])
                if not strategies:
                    # Try managed_strategies
                    managed = config.get('managed_strategies', [])
                    logger.info(f"Managed strategies: {managed}")
                    # Try to get from strategy names
                    if 'strategy_names' in config:
                        logger.info(f"Strategy names: {config['strategy_names']}")
                
                # Look for strategy configs
                if strategies:
                    logger.info(f"Found {len(strategies)} strategies in config")
                    for strategy in strategies:
                        strategy_name = strategy.get('name', strategy.get('type', 'unknown'))
                        strategy_params[strategy_name] = {
                            'type': strategy.get('type'),
                            'params': strategy.get('params', {}),
                            'name': strategy_name
                        }
                        # Store in sparse storage for strategy metadata
                        self.storage.set_strategy_metadata(strategy_name, strategy_params[strategy_name])
                else:
                    logger.info("No strategies found in portfolio config")
                
                # Look for classifiers
                classifiers = config.get('classifiers', [])
                if classifiers:
                    logger.info(f"Found {len(classifiers)} classifiers in config")
                    for classifier in classifiers:
                        classifier_name = classifier.get('name', classifier.get('type', 'unknown'))
                        classifier_params[classifier_name] = {
                            'type': classifier.get('type'),
                            'params': classifier.get('params', {}),
                            'name': classifier_name
                        }
                        # Store in sparse storage for classifier metadata
                        self.storage.set_strategy_metadata(f"classifier_{classifier_name}", classifier_params[classifier_name])
                else:
                    logger.info("No classifiers found in portfolio config")
            
            # Combine strategy and classifier params for saving
            all_params = {**strategy_params}
            for name, params in classifier_params.items():
                all_params[f"classifier_{name}"] = params
            
            filepath = self.storage.save(
                tag=self.container_id, 
                performance_metrics=performance_metrics,
                strategy_params=all_params
            )
            
            compression_ratio = self._total_signals / self._stored_changes if self._stored_changes > 0 else 0
            logger.info(f"Flushed sparse signals for {self.container_id}: "
                       f"{self._stored_changes} changes from {self._total_signals} signals "
                       f"({compression_ratio:.1f}x compression)")
        else:
            logger.info(f"No signal changes to flush for {self.container_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        return {
            'container_id': self.container_id,
            'managed_strategies': list(self.managed_strategies),
            'managed_classifiers': list(self.managed_classifiers),
            'total_signals_seen': self._total_signals,
            'signal_changes_stored': self._stored_changes,
            'compression_ratio': self._total_signals / self._stored_changes if self._stored_changes > 0 else 0,
            'signal_ranges': self.storage.get_signal_ranges()
        }
    
    # Required EventObserverProtocol methods
    def on_publish(self, event: Event) -> None:
        """Called when event is published."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler_count: int) -> None:
        """Called after event delivery."""
        pass
    
    def on_error(self, event: Event, error: Exception, handler: Any = None) -> None:
        """Called on event error."""
        logger.error(f"Sparse tracer error for event {event.event_type}: {error}")