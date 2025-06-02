"""
File: src/risk/step2_container_factory.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#container-factory
Step: 2 - Add Risk Container
Dependencies: core.logging, risk components

Factory function for creating configured Step 2 risk containers.
Provides easy setup and integration with existing systems.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from ..core.logging.structured import ContainerLogger
from .models import RiskConfig
from .risk_container import RiskContainer


def create_risk_container(container_id: str, config: Dict[str, Any]) -> RiskContainer:
    """
    Factory function to create configured risk container for Step 2.
    
    This factory handles all the setup required to create a properly
    configured risk container with all components initialized.
    
    Args:
        container_id: Unique container identifier
        config: Risk configuration parameters
        
    Returns:
        Configured and ready-to-use RiskContainer
        
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Provides: Simple creation of risk containers
        - Enables: Easy integration with existing systems
        - Dependencies: All Step 2 risk components
    
    Example:
        config = {
            'initial_capital': 100000,
            'sizing_method': 'percent_risk',
            'max_position_size': 0.1,
            'percent_risk_per_trade': 0.01
        }
        container = create_risk_container("risk_001", config)
    """
    logger = ContainerLogger("RiskContainerFactory", container_id, "risk_factory")
    
    logger.info(
        "Creating risk container",
        container_id=container_id,
        config_keys=list(config.keys())
    )
    
    # Create risk configuration
    risk_config = _create_risk_config(config)
    
    # Validate configuration
    _validate_risk_config(risk_config, logger)
    
    # Create risk container
    container = RiskContainer(container_id, risk_config)
    
    logger.info(
        "Risk container created successfully",
        container_id=container_id,
        sizing_method=risk_config.sizing_method,
        initial_capital=risk_config.initial_capital
    )
    
    return container


def _create_risk_config(config: Dict[str, Any]) -> RiskConfig:
    """
    Create RiskConfig from dictionary with defaults.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RiskConfig object with validated parameters
    """
    # Default configuration
    default_config = {
        'initial_capital': 100000.0,
        'sizing_method': 'fixed',
        'max_position_size': 0.1,
        'max_portfolio_risk': 0.02,
        'max_correlation': 0.7,
        'max_drawdown': 0.2,
        'fixed_position_size': 1000.0,
        'percent_risk_per_trade': 0.01,
        'volatility_lookback': 20,
        'max_leverage': 1.0,
        'max_concentration': 0.2,
        'max_orders_per_minute': 10,
        'cooldown_period_seconds': 60,
        'default_stop_loss_pct': 0.05,
        'use_trailing_stops': False,
        'trailing_stop_pct': 0.03
    }
    
    # Update with provided config
    default_config.update(config)
    
    return RiskConfig(**default_config)


def _validate_risk_config(config: RiskConfig, logger: ContainerLogger) -> None:
    """
    Validate risk configuration parameters.
    
    Args:
        config: Risk configuration to validate
        logger: Logger for warnings
    """
    warnings = []
    
    # Validate sizing method
    valid_methods = ['fixed', 'percent_risk', 'volatility']
    if config.sizing_method not in valid_methods:
        warnings.append(f"Invalid sizing method: {config.sizing_method}")
    
    # Validate percentage values (should be between 0 and 1)
    percentage_fields = [
        ('max_position_size', config.max_position_size),
        ('max_portfolio_risk', config.max_portfolio_risk),
        ('max_correlation', config.max_correlation),
        ('max_drawdown', config.max_drawdown),
        ('max_concentration', config.max_concentration),
        ('percent_risk_per_trade', config.percent_risk_per_trade),
        ('default_stop_loss_pct', config.default_stop_loss_pct),
        ('trailing_stop_pct', config.trailing_stop_pct)
    ]
    
    for field_name, value in percentage_fields:
        if not (0 <= value <= 1):
            warnings.append(f"{field_name} should be between 0 and 1, got {value}")
    
    # Validate positive values
    positive_fields = [
        ('initial_capital', config.initial_capital),
        ('fixed_position_size', config.fixed_position_size),
        ('volatility_lookback', config.volatility_lookback),
        ('max_orders_per_minute', config.max_orders_per_minute),
        ('cooldown_period_seconds', config.cooldown_period_seconds)
    ]
    
    for field_name, value in positive_fields:
        if value <= 0:
            warnings.append(f"{field_name} should be positive, got {value}")
    
    # Validate leverage
    if config.max_leverage < 1.0:
        warnings.append(f"max_leverage should be >= 1.0, got {config.max_leverage}")
    
    # Log warnings
    if warnings:
        for warning in warnings:
            logger.warning(f"Risk config validation: {warning}")
    else:
        logger.debug("Risk configuration validation passed")


def create_test_risk_container(
    container_id: str, 
    initial_capital: float = 10000.0,
    sizing_method: str = "fixed"
) -> RiskContainer:
    """
    Create a risk container for testing with minimal configuration.
    
    Args:
        container_id: Container identifier
        initial_capital: Starting capital
        sizing_method: Position sizing method
        
    Returns:
        Risk container configured for testing
    """
    test_config = {
        'initial_capital': initial_capital,
        'sizing_method': sizing_method,
        'max_position_size': 0.2,  # 20% for testing
        'fixed_position_size': 500.0,  # $500 per trade
        'percent_risk_per_trade': 0.02,  # 2% risk for testing
        'max_drawdown': 0.5,  # 50% max drawdown for testing
        'max_concentration': 0.5  # 50% concentration for testing
    }
    
    return create_risk_container(container_id, test_config)


def create_conservative_risk_container(container_id: str, initial_capital: float = 100000.0) -> RiskContainer:
    """
    Create a conservative risk container for production use.
    
    Args:
        container_id: Container identifier
        initial_capital: Starting capital
        
    Returns:
        Conservatively configured risk container
    """
    conservative_config = {
        'initial_capital': initial_capital,
        'sizing_method': 'percent_risk',
        'max_position_size': 0.05,  # 5% max position
        'percent_risk_per_trade': 0.005,  # 0.5% risk per trade
        'max_portfolio_risk': 0.01,  # 1% max portfolio risk
        'max_drawdown': 0.1,  # 10% max drawdown
        'max_concentration': 0.1,  # 10% max concentration
        'max_leverage': 1.0,  # No leverage
        'default_stop_loss_pct': 0.03,  # 3% stop loss
        'use_trailing_stops': True,
        'trailing_stop_pct': 0.02  # 2% trailing stop
    }
    
    return create_risk_container(container_id, conservative_config)


def create_aggressive_risk_container(container_id: str, initial_capital: float = 100000.0) -> RiskContainer:
    """
    Create an aggressive risk container for higher risk tolerance.
    
    Args:
        container_id: Container identifier
        initial_capital: Starting capital
        
    Returns:
        Aggressively configured risk container
    """
    aggressive_config = {
        'initial_capital': initial_capital,
        'sizing_method': 'percent_risk',
        'max_position_size': 0.2,  # 20% max position
        'percent_risk_per_trade': 0.02,  # 2% risk per trade
        'max_portfolio_risk': 0.05,  # 5% max portfolio risk
        'max_drawdown': 0.3,  # 30% max drawdown
        'max_concentration': 0.3,  # 30% max concentration
        'max_leverage': 2.0,  # 2x leverage allowed
        'default_stop_loss_pct': 0.08,  # 8% stop loss
        'use_trailing_stops': True,
        'trailing_stop_pct': 0.05  # 5% trailing stop
    }
    
    return create_risk_container(container_id, aggressive_config)


# Preset configurations for common use cases
PRESET_CONFIGS = {
    'conservative': {
        'sizing_method': 'percent_risk',
        'max_position_size': 0.05,
        'percent_risk_per_trade': 0.005,
        'max_drawdown': 0.1,
        'max_concentration': 0.1
    },
    'moderate': {
        'sizing_method': 'percent_risk',
        'max_position_size': 0.1,
        'percent_risk_per_trade': 0.01,
        'max_drawdown': 0.2,
        'max_concentration': 0.2
    },
    'aggressive': {
        'sizing_method': 'percent_risk',
        'max_position_size': 0.2,
        'percent_risk_per_trade': 0.02,
        'max_drawdown': 0.3,
        'max_concentration': 0.3
    },
    'test': {
        'sizing_method': 'fixed',
        'max_position_size': 0.2,
        'fixed_position_size': 500.0,
        'max_drawdown': 0.5,
        'max_concentration': 0.5
    }
}


def create_preset_risk_container(
    container_id: str, 
    preset: str, 
    initial_capital: float = 100000.0,
    overrides: Optional[Dict[str, Any]] = None
) -> RiskContainer:
    """
    Create risk container using preset configuration.
    
    Args:
        container_id: Container identifier
        preset: Preset name ('conservative', 'moderate', 'aggressive', 'test')
        initial_capital: Starting capital
        overrides: Optional configuration overrides
        
    Returns:
        Risk container with preset configuration
        
    Raises:
        ValueError: If preset name is invalid
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Invalid preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    # Start with preset config
    config = PRESET_CONFIGS[preset].copy()
    config['initial_capital'] = initial_capital
    
    # Apply overrides if provided
    if overrides:
        config.update(overrides)
    
    return create_risk_container(container_id, config)