"""
Backtest container factories implementing BACKTEST.MD architecture.

This module provides three standardized patterns:
1. FULL BACKTEST: Market Data → Indicators → Classifiers → Strategies → Risk & Portfolio → Execution
2. SIGNAL REPLAY: Signal Logs → Ensemble Weights → Risk & Portfolio → Execution  
3. SIGNAL GENERATION: Market Data → Indicators → Classifiers → Strategies → Analysis
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
import logging

from ..protocols import Container, ContainerFactory
from ..universal import UniversalScopedContainer

logger = logging.getLogger(__name__)


class BacktestPattern(Enum):
    """Three standardized backtest patterns per BACKTEST.MD."""
    FULL = "full"
    SIGNAL_REPLAY = "signal_replay"
    SIGNAL_GENERATION = "signal_generation"


@dataclass
class BacktestConfig:
    """Configuration for backtest container creation."""
    container_id: str
    pattern: BacktestPattern
    data_config: Dict[str, Any]
    indicator_config: Dict[str, Any]
    classifiers: List[Dict[str, Any]]
    execution_config: Dict[str, Any]
    
    # Optional configs
    signal_log_path: Optional[str] = None  # For signal replay
    ensemble_weights: Optional[Dict[str, float]] = None  # For signal replay
    analysis_config: Optional[Dict[str, Any]] = None  # For signal generation


class BacktestContainerFactory:
    """
    Creates standardized backtest containers per BACKTEST.MD.
    
    Ensures every backtest follows identical creation pattern:
    1. Same component creation order
    2. Same event wiring
    3. Same resource limits
    4. Complete isolation
    """
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> Container:
        """
        Create backtest container based on pattern.
        
        Args:
            config: Backtest configuration
            
        Returns:
            Configured container ready for execution
        """
        if config.pattern == BacktestPattern.FULL:
            return FullBacktestContainerFactory.create_instance(config)
        elif config.pattern == BacktestPattern.SIGNAL_REPLAY:
            return SignalReplayContainerFactory.create_instance(config)
        elif config.pattern == BacktestPattern.SIGNAL_GENERATION:
            return SignalGenerationContainerFactory.create_instance(config)
        else:
            raise ValueError(f"Unknown backtest pattern: {config.pattern}")


class FullBacktestContainerFactory:
    """
    Creates full backtest containers with complete hierarchy:
    
    BacktestContainer
        ├── DataStreamer
        ├── IndicatorHub (Shared computation)
        ├── Classifier Containers
        │   └── Risk & Portfolio Containers
        │       └── Strategies
        └── BacktestEngine
    """
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> Container:
        """Create full backtest container with all components."""
        logger.info(f"Creating full backtest container: {config.container_id}")
        
        # Create top-level container
        container = UniversalScopedContainer(
            container_id=config.container_id,
            container_type="backtest_full"
        )
        
        # 1. Data layer - ALWAYS created first
        container.register_singleton(
            "data_streamer",
            lambda: _create_data_streamer(config.data_config)
        )
        
        # 2. Shared computation layer
        container.register_singleton(
            "indicator_hub", 
            lambda: _create_indicator_hub(config.indicator_config)
        )
        
        # 3. Classifier layer with nested containers
        for classifier_config in config.classifiers:
            classifier_id = f"{config.container_id}_{classifier_config['type']}"
            
            # Create classifier subcontainer
            classifier_container = container.create_subcontainer(
                container_id=classifier_id,
                container_type="classifier"
            )
            
            # Add classifier component
            classifier_container.register_singleton(
                "classifier",
                lambda cc=classifier_config: _create_classifier(cc)
            )
            
            # Create Risk & Portfolio subcontainers
            for risk_profile in classifier_config.get('risk_profiles', []):
                risk_id = f"{classifier_id}_{risk_profile['name']}"
                
                # Create risk & portfolio subcontainer
                risk_container = classifier_container.create_subcontainer(
                    container_id=risk_id,
                    container_type="risk_portfolio"
                )
                
                # Add risk manager
                risk_container.register_singleton(
                    "risk_manager",
                    lambda rp=risk_profile: _create_risk_manager(rp)
                )
                
                # Add portfolio
                risk_container.register_singleton(
                    "portfolio",
                    lambda rp=risk_profile: _create_portfolio(rp)
                )
                
                # Add strategies
                for strategy_config in risk_profile.get('strategies', []):
                    strategy_name = strategy_config['name']
                    risk_container.register_singleton(
                        f"strategy_{strategy_name}",
                        lambda sc=strategy_config: _create_strategy(sc)
                    )
        
        # 4. Execution layer - ALWAYS created last
        container.register_singleton(
            "backtest_engine",
            lambda: _create_backtest_engine(config.execution_config)
        )
        
        # 5. Wire event flows - ALWAYS in same order
        _wire_full_backtest_events(container)
        
        logger.info(f"Full backtest container created: {config.container_id}")
        return container


class SignalReplayContainerFactory:
    """
    Creates signal replay containers for ensemble optimization:
    
    SignalReplayContainer
        ├── SignalLogStreamer
        ├── EnsembleOptimizer
        ├── Risk & Portfolio Container
        └── BacktestEngine
    """
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> Container:
        """Create signal replay container."""
        logger.info(f"Creating signal replay container: {config.container_id}")
        
        container = UniversalScopedContainer(
            container_id=config.container_id,
            container_type="backtest_signal_replay"
        )
        
        # 1. Signal replay layer
        container.register_singleton(
            "signal_streamer",
            lambda: _create_signal_streamer(config.signal_log_path)
        )
        
        # 2. Ensemble optimizer
        container.register_singleton(
            "ensemble_optimizer",
            lambda: _create_ensemble_optimizer(config.ensemble_weights)
        )
        
        # 3. Risk & Portfolio container (simplified - one profile)
        risk_container = container.create_subcontainer(
            container_id=f"{config.container_id}_risk",
            container_type="risk_portfolio"
        )
        
        risk_container.register_singleton(
            "risk_manager",
            lambda: _create_risk_manager({'name': 'replay'})
        )
        
        risk_container.register_singleton(
            "portfolio",
            lambda: _create_portfolio({'name': 'replay'})
        )
        
        # 4. Execution layer
        container.register_singleton(
            "backtest_engine",
            lambda: _create_backtest_engine(config.execution_config)
        )
        
        # 5. Wire signal replay events
        _wire_signal_replay_events(container)
        
        logger.info(f"Signal replay container created: {config.container_id}")
        return container


class SignalGenerationContainerFactory:
    """
    Creates signal generation containers for analysis:
    
    SignalGenerationContainer
        ├── DataStreamer
        ├── IndicatorHub
        ├── Classifier Containers
        │   └── Strategies (no risk/portfolio)
        └── SignalAnalysisEngine
    """
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> Container:
        """Create signal generation container."""
        logger.info(f"Creating signal generation container: {config.container_id}")
        
        container = UniversalScopedContainer(
            container_id=config.container_id,
            container_type="backtest_signal_generation"
        )
        
        # 1. Data layer
        container.register_singleton(
            "data_streamer",
            lambda: _create_data_streamer(config.data_config)
        )
        
        # 2. Indicator layer
        container.register_singleton(
            "indicator_hub",
            lambda: _create_indicator_hub(config.indicator_config)
        )
        
        # 3. Classifier layer (simplified - no risk containers)
        for classifier_config in config.classifiers:
            classifier_id = f"{config.container_id}_{classifier_config['type']}"
            
            classifier_container = container.create_subcontainer(
                container_id=classifier_id,
                container_type="classifier"
            )
            
            # Add classifier
            classifier_container.register_singleton(
                "classifier",
                lambda cc=classifier_config: _create_classifier(cc)
            )
            
            # Add strategies directly (no risk/portfolio needed)
            for strategy_config in classifier_config.get('strategies', []):
                strategy_name = strategy_config['name']
                classifier_container.register_singleton(
                    f"strategy_{strategy_name}",
                    lambda sc=strategy_config: _create_strategy(sc)
                )
        
        # 4. Analysis engine instead of execution
        container.register_singleton(
            "signal_analysis_engine",
            lambda: _create_signal_analysis_engine(config.analysis_config)
        )
        
        # 5. Wire signal generation events
        _wire_signal_generation_events(container)
        
        logger.info(f"Signal generation container created: {config.container_id}")
        return container


# Component creation functions (to be implemented)
def _create_data_streamer(config: Dict[str, Any]):
    """Create data streamer component."""
    # Import here to avoid circular dependencies
    from ...data.streamers import HistoricalDataStreamer
    return HistoricalDataStreamer(config)


def _create_indicator_hub(config: Dict[str, Any]):
    """Create indicator hub for shared computation."""
    from ...strategy.indicators.indicator_hub import IndicatorHub
    return IndicatorHub(config)


def _create_classifier(config: Dict[str, Any]):
    """Create classifier based on type."""
    from ...strategy.classifiers import create_classifier
    return create_classifier(config['type'], config.get('parameters', {}))


def _create_risk_manager(config: Dict[str, Any]):
    """Create risk manager."""
    from ...risk.manager import RiskManager
    return RiskManager(config)


def _create_portfolio(config: Dict[str, Any]):
    """Create portfolio."""
    from ...risk.portfolio import Portfolio
    return Portfolio(config)


def _create_strategy(config: Dict[str, Any]):
    """Create strategy based on type."""
    from ...strategy import create_strategy
    return create_strategy(config['type'], config.get('parameters', {}))


def _create_backtest_engine(config: Dict[str, Any]):
    """Create backtest execution engine."""
    from ...execution.backtest_engine import BacktestEngine
    return BacktestEngine(config)


def _create_signal_streamer(signal_log_path: str):
    """Create signal log streamer."""
    from ...data.streamers import SignalLogStreamer
    return SignalLogStreamer(signal_log_path)


def _create_ensemble_optimizer(weights: Dict[str, float]):
    """Create ensemble optimizer."""
    from ...strategy.ensemble import EnsembleOptimizer
    return EnsembleOptimizer(weights)


def _create_signal_analysis_engine(config: Dict[str, Any]):
    """Create signal analysis engine."""
    from ...execution.analysis import SignalAnalysisEngine
    return SignalAnalysisEngine(config)


# Event wiring functions
def _wire_full_backtest_events(container: Container):
    """Wire events for full backtest pattern."""
    # This will be implemented when we have the event system ready
    pass


def _wire_signal_replay_events(container: Container):
    """Wire events for signal replay pattern."""
    pass


def _wire_signal_generation_events(container: Container):
    """Wire events for signal generation pattern."""
    pass