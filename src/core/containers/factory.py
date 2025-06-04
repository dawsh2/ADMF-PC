"""
Container Factory

This module implements the factory infrastructure for creating Container
instances from patterns. Used by the Coordinator for workflow orchestration.
"""

from typing import Dict, List, Any, Optional, Set, Type, Callable
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass, field

from .protocols import (
    ComposableContainer, ContainerComposition, 
    ContainerRole, ContainerMetadata, ContainerLimits,
    ContainerState
)


logger = logging.getLogger(__name__)


@dataclass
class ContainerPattern:
    """Definition of a container arrangement pattern."""
    name: str
    description: str
    structure: Dict[str, Any]
    required_capabilities: Set[str] = field(default_factory=set)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> 'ContainerPattern':
        """Create pattern from YAML configuration."""
        return cls(
            name=yaml_data['name'],
            description=yaml_data.get('description', ''),
            structure=yaml_data['structure'],
            required_capabilities=set(yaml_data.get('required_capabilities', [])),
            default_config=yaml_data.get('default_config', {})
        )


class ContainerRegistry:
    """Registry for container types and patterns."""
    
    def __init__(self):
        self._container_factories: Dict[ContainerRole, Callable] = {}
        self._container_capabilities: Dict[ContainerRole, Set[str]] = {}
        self._patterns: Dict[str, ContainerPattern] = {}
        self._load_default_patterns()
    
    def register_container_type(
        self,
        role: ContainerRole,
        factory_func: Callable,
        capabilities: Set[str] = None
    ) -> None:
        """Register a container factory for a role."""
        self._container_factories[role] = factory_func
        self._container_capabilities[role] = capabilities or set()
        logger.info(f"Registered container type: {role.value}")
    
    def get_container_factory(self, role: ContainerRole) -> Optional[Callable]:
        """Get factory function for container role."""
        return self._container_factories.get(role)
    
    def register_pattern(self, pattern: ContainerPattern) -> None:
        """Register a composition pattern."""
        self._patterns[pattern.name] = pattern
        logger.info(f"Registered container pattern: {pattern.name}")
    
    def get_pattern(self, pattern_name: str) -> Optional[ContainerPattern]:
        """Get pattern by name."""
        return self._patterns.get(pattern_name)
    
    def list_available_patterns(self) -> List[str]:
        """List all available pattern names."""
        return list(self._patterns.keys())
    
    def get_container_capabilities(self, role: ContainerRole) -> Set[str]:
        """Get capabilities for container role."""
        return self._container_capabilities.get(role, set())
    
    def _load_default_patterns(self) -> None:
        """Load default container patterns."""
        # Full Backtest Pattern
        full_backtest = ContainerPattern(
            name="full_backtest",
            description="Complete backtest with data → features → classifier → risk → portfolio → strategy → execution",
            structure={
                "root": {
                    "role": "data",
                    "children": {
                        "features": {
                            "role": "feature",
                            "children": {
                                "classifier": {
                                    "role": "classifier", 
                                    "children": {
                                        "risk": {
                                            "role": "risk",
                                            "children": {
                                                "portfolio": {
                                                    "role": "portfolio",
                                                    "children": {
                                                        "strategy": {"role": "strategy"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "execution": {"role": "execution"}
                    }
                }
            },
            required_capabilities={"data.historical", "execution.backtest"},
            default_config={
                "data": {"source": "historical"},
                "execution": {"mode": "backtest"}
            }
        )
        
        # Signal Generation Pattern
        signal_generation = ContainerPattern(
            name="signal_generation",
            description="Signal generation only - no execution",
            structure={
                "root": {
                    "role": "data",
                    "children": {
                        "features": {
                            "role": "feature",
                            "children": {
                                "classifier": {
                                    "role": "classifier",
                                    "children": {
                                        "strategy": {"role": "strategy"}
                                    }
                                }
                            }
                        },
                        "analysis": {"role": "analysis"}
                    }
                }
            },
            required_capabilities={"data.historical", "analysis.signals"},
            default_config={
                "data": {"source": "historical"},
                "analysis": {"mode": "signal_generation"}
            }
        )
        
        # Signal Replay Pattern
        signal_replay = ContainerPattern(
            name="signal_replay",
            description="Replay signals for ensemble optimization",
            structure={
                "root": {
                    "role": "signal_log",
                    "children": {
                        "ensemble": {
                            "role": "ensemble",
                            "children": {
                                "risk": {
                                    "role": "risk",
                                    "children": {
                                        "portfolio": {
                                            "role": "portfolio"
                                        }
                                    }
                                }
                            }
                        },
                        "execution": {"role": "execution"}
                    }
                }
            },
            required_capabilities={"signal_log.replay", "execution.backtest"},
            default_config={
                "signal_log": {"source": "phase1_output"},
                "execution": {"mode": "backtest"}
            }
        )
        
        # Simple Backtest Pattern  
        simple_backtest = ContainerPattern(
            name="simple_backtest",
            description="Simple backtest - backtest container with peer containers: data, features, classifier(risk→portfolio→strategy), execution",
            structure={
                "root": {
                    "role": "backtest",  # Use backtest container as root for peer containers
                    "children": {
                        "data": {
                            "role": "data"
                        },
                        "features": {
                            "role": "feature"
                        },
                        "classifier": {
                            "role": "classifier",
                            "children": {
                                "risk": {
                                    "role": "risk",
                                    "children": {
                                        "portfolio": {
                                            "role": "portfolio",
                                            "children": {
                                                "strategy": {"role": "strategy"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "execution": {"role": "execution"}
                    }
                }
            },
            required_capabilities={"data.historical", "feature.computation", "risk.management", "execution.backtest"},
            default_config={
                "data": {"source": "historical"},
                "risk": {"initial_capital": 10000},
                "execution": {"mode": "backtest"}
            }
        )
        
        # Register all patterns
        for pattern in [full_backtest, signal_generation, signal_replay, simple_backtest]:
            self.register_pattern(pattern)


class ContainerFactory:
    """Factory for creating containers according to patterns."""
    
    def __init__(self, registry: ContainerRegistry = None):
        self.registry = registry or ContainerRegistry()
    
    def create_container(
        self,
        role: ContainerRole,
        config: Dict[str, Any],
        container_id: Optional[str] = None
    ) -> ComposableContainer:
        """Create a container of specified role."""
        factory = self.registry.get_container_factory(role)
        if not factory:
            raise ValueError(f"No factory registered for container role: {role.value}")
        
        # Create container with config
        container = factory(
            config=config,
            container_id=container_id
        )
        
        logger.debug(f"Created container: {role.value} with ID: {container.metadata.container_id}")
        return container
    
    def _infer_required_features(self, config: Dict[str, Any]) -> Set[str]:
        """Infer required features from strategy configurations."""
        required_features = set()
        
        # Collect strategies from multiple possible locations
        all_strategies = []
        
        # Check top-level strategies
        strategies = config.get('strategies', [])
        if not isinstance(strategies, list):
            strategies = [strategies]
        all_strategies.extend(strategies)
        
        # Check backtest.strategies section
        backtest_config = config.get('backtest', {})
        backtest_strategies = backtest_config.get('strategies', [])
        if not isinstance(backtest_strategies, list):
            backtest_strategies = [backtest_strategies]
        all_strategies.extend(backtest_strategies)
        
        # Check optimization.strategies section (fallback)
        optimization_config = config.get('optimization', {})
        opt_strategies = optimization_config.get('strategies', [])
        if not isinstance(opt_strategies, list):
            opt_strategies = [opt_strategies]
        all_strategies.extend(opt_strategies)
        
        # Process all found strategies
        for strategy_config in all_strategies:
            strategy_type = strategy_config.get('type', '')
            parameters = strategy_config.get('parameters', {})
            
            # Infer features based on strategy type and parameters
            if strategy_type == 'momentum':
                # Momentum strategies typically need SMA and RSI
                lookback_period = parameters.get('lookback_period', 20)
                required_features.add(f'SMA_{lookback_period}')
                
                rsi_period = parameters.get('rsi_period', 14)
                required_features.add(f'RSI_{rsi_period}' if rsi_period != 14 else 'RSI')
                
            elif strategy_type == 'mean_reversion':
                # Mean reversion strategies typically need moving averages and Bollinger Bands
                required_features.update(['SMA_20', 'BB_20', 'RSI'])
                
            elif strategy_type == 'trend_following':
                # Trend following strategies typically need multiple timeframe MAs
                required_features.update(['SMA_10', 'SMA_20', 'SMA_50', 'MACD'])
                
            # Add explicit features if specified in config
            explicit_features = parameters.get('features', [])
            required_features.update(explicit_features)
        
        return required_features

    def compose_pattern(
        self,
        pattern_name: str,
        config_overrides: Dict[str, Any] = None
    ) -> ComposableContainer:
        """Compose containers according to named pattern."""
        pattern = self.registry.get_pattern(pattern_name)
        if not pattern:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        # Merge default config with overrides
        base_config = pattern.default_config.copy()
        if config_overrides:
            base_config.update(config_overrides)
        
        # Infer required features from strategy configurations
        required_features = self._infer_required_features(base_config)
        if required_features:
            logger.info(f"Inferred features from strategy config: {required_features}")
            # Inject features into feature container config (use 'feature' key to match ContainerRole.FEATURE.value)
            base_config['feature'] = {
                'required_features': list(required_features)
            }
        
        # Validate required capabilities
        if not self._validate_pattern_capabilities(pattern):
            raise ValueError(f"Pattern '{pattern_name}' requirements not met")
        
        # Build container hierarchy
        root_container = self._build_container_tree(
            pattern.structure["root"],
            base_config
        )
        
        logger.info(f"Composed pattern '{pattern_name}' with root container: {root_container.metadata.container_id}")
        return root_container
    
    def compose_custom_pattern(
        self,
        structure: Dict[str, Any],
        config: Dict[str, Any] = None
    ) -> ComposableContainer:
        """Compose containers according to custom structure."""
        if "root" not in structure:
            raise ValueError("Custom pattern must have 'root' element")
        
        config = config or {}
        
        # Build container hierarchy
        root_container = self._build_container_tree(
            structure["root"],
            config
        )
        
        logger.info(f"Composed custom pattern with root container: {root_container.metadata.container_id}")
        return root_container
    
    def validate_pattern(self, pattern: ContainerPattern) -> bool:
        """Validate that pattern is valid and composable."""
        try:
            # Check structure validity
            if "root" not in pattern.structure:
                return False
            
            # Check all required roles are registered
            required_roles = self._extract_roles_from_structure(pattern.structure["root"])
            for role in required_roles:
                if not self.registry.get_container_factory(role):
                    logger.warning(f"No factory for required role: {role.value}")
                    return False
            
            # Check capabilities
            if not self._validate_pattern_capabilities(pattern):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pattern validation error: {e}")
            return False
    
    def _build_container_tree(
        self,
        node: Dict[str, Any],
        base_config: Dict[str, Any],
        parent_container: ComposableContainer = None
    ) -> ComposableContainer:
        """Recursively build container tree from structure definition."""
        # Extract role and config for this container
        role = ContainerRole(node["role"])
        
        # Get container-specific config
        container_config = base_config.get(role.value, {}).copy()
        if "config" in node:
            container_config.update(node["config"])
        
        # Create container
        container = self.create_container(
            role=role,
            config=container_config
        )
        
        # Add to parent if provided
        if parent_container:
            parent_container.add_child_container(container)
        
        # Recursively create children
        if "children" in node:
            for child_name, child_def in node["children"].items():
                child_container = self._build_container_tree(
                    child_def,
                    base_config,
                    container
                )
        
        return container
    
    def _extract_roles_from_structure(self, node: Dict[str, Any]) -> Set[ContainerRole]:
        """Extract all roles used in structure definition."""
        roles = {ContainerRole(node["role"])}
        
        if "children" in node:
            for child_def in node["children"].values():
                roles.update(self._extract_roles_from_structure(child_def))
        
        return roles
    
    def _validate_pattern_capabilities(self, pattern: ContainerPattern) -> bool:
        """Validate pattern capability requirements."""
        if not pattern.required_capabilities:
            return True
        
        # Extract roles and check if they support required capabilities
        roles = self._extract_roles_from_structure(pattern.structure["root"])
        
        available_capabilities = set()
        for role in roles:
            available_capabilities.update(self.registry.get_container_capabilities(role))
        
        missing_capabilities = pattern.required_capabilities - available_capabilities
        if missing_capabilities:
            logger.warning(f"Missing capabilities for pattern '{pattern.name}': {missing_capabilities}")
            return False
        
        return True


class PatternManager:
    """Manager for loading and saving container patterns."""
    
    def __init__(self, patterns_dir: Path = None):
        self.patterns_dir = patterns_dir or Path("config/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pattern_from_file(self, pattern_file: Path) -> ContainerPattern:
        """Load pattern from YAML file."""
        with open(pattern_file, 'r') as f:
            pattern_data = yaml.safe_load(f)
        
        return ContainerPattern.from_yaml(pattern_data)
    
    def save_pattern_to_file(self, pattern: ContainerPattern, filename: str = None) -> Path:
        """Save pattern to YAML file."""
        filename = filename or f"{pattern.name}.yaml"
        pattern_file = self.patterns_dir / filename
        
        pattern_data = {
            'name': pattern.name,
            'description': pattern.description,
            'structure': pattern.structure,
            'required_capabilities': list(pattern.required_capabilities),
            'default_config': pattern.default_config
        }
        
        with open(pattern_file, 'w') as f:
            yaml.dump(pattern_data, f, indent=2, default_flow_style=False)
        
        logger.info(f"Saved pattern '{pattern.name}' to {pattern_file}")
        return pattern_file
    
    def load_all_patterns(self) -> List[ContainerPattern]:
        """Load all patterns from patterns directory."""
        patterns = []
        
        for pattern_file in self.patterns_dir.glob("*.yaml"):
            try:
                pattern = self.load_pattern_from_file(pattern_file)
                patterns.append(pattern)
                logger.info(f"Loaded pattern: {pattern.name}")
            except Exception as e:
                logger.error(f"Error loading pattern from {pattern_file}: {e}")
        
        return patterns
    
    def create_pattern_from_config(
        self,
        name: str,
        description: str,
        structure: Dict[str, Any],
        required_capabilities: Set[str] = None,
        default_config: Dict[str, Any] = None
    ) -> ContainerPattern:
        """Create pattern from configuration data."""
        return ContainerPattern(
            name=name,
            description=description,
            structure=structure,
            required_capabilities=required_capabilities or set(),
            default_config=default_config or {}
        )


# Global registry instance
_global_registry = ContainerRegistry()
_global_factory = ContainerFactory(_global_registry)


def get_global_registry() -> ContainerRegistry:
    """Get the global container registry."""
    return _global_registry


def get_global_factory() -> ContainerFactory:
    """Get the global container factory."""
    return _global_factory


def register_container_type(
    role: ContainerRole,
    factory_func: Callable,
    capabilities: Set[str] = None
) -> None:
    """Register a container type with the global registry."""
    _global_registry.register_container_type(role, factory_func, capabilities)


def compose_pattern(
    pattern_name: str,
    config_overrides: Dict[str, Any] = None
) -> ComposableContainer:
    """Compose containers using global factory."""
    return _global_factory.compose_pattern(pattern_name, config_overrides)