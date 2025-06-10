"""
Pattern Loading Utilities

Consolidates YAML and Python pattern loading logic used by 
Coordinator, Sequencer, and TopologyBuilder.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class PatternLoader:
    """
    Unified pattern loader for workflow, sequence, and topology patterns.
    
    Handles:
    - YAML pattern loading from config/patterns/{type}/*.yaml
    - Python pattern loading for backward compatibility
    - Built-in pattern definitions
    - Directory auto-creation and error handling
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize pattern loader.
        
        Args:
            project_root: Project root directory. If None, auto-detected.
        """
        if project_root is None:
            # Navigate to project root from this file's location
            project_root = Path(__file__).parent.parent.parent.parent.parent
        
        self.project_root = project_root
        self.patterns_dir = project_root / 'config' / 'patterns'
        
    def load_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """
        Load all patterns of a specific type.
        
        Args:
            pattern_type: Pattern type ('workflows', 'sequences', 'topologies')
            
        Returns:
            Dictionary of pattern_name -> pattern_definition
        """
        patterns = {}
        
        # Load YAML patterns
        yaml_patterns = self._load_yaml_patterns(pattern_type)
        patterns.update(yaml_patterns)
        
        # Load Python patterns for backward compatibility
        python_patterns = self._load_python_patterns(pattern_type)
        patterns.update(python_patterns)
        
        # Load built-in patterns
        builtin_patterns = self._get_builtin_patterns(pattern_type)
        patterns.update(builtin_patterns)
        
        logger.info(f"Loaded {len(patterns)} {pattern_type} patterns")
        return patterns
    
    def _load_yaml_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Load patterns from YAML files."""
        patterns = {}
        pattern_dir = self.patterns_dir / pattern_type
        
        # Create directory if it doesn't exist
        if not pattern_dir.exists():
            pattern_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {pattern_type} patterns directory: {pattern_dir}")
            return patterns
        
        # Load YAML files
        for pattern_file in pattern_dir.glob('*.yaml'):
            try:
                with open(pattern_file) as f:
                    pattern = yaml.safe_load(f)
                    patterns[pattern_file.stem] = pattern
                    logger.info(f"Loaded {pattern_type} pattern: {pattern_file.stem}")
            except Exception as e:
                logger.error(f"Failed to load {pattern_type} pattern {pattern_file}: {e}")
        
        return patterns
    
    def _load_python_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Load Python patterns for backward compatibility."""
        patterns = {}
        
        # Try to import pattern modules
        module_names = {
            'workflows': 'workflow_patterns',
            'sequences': 'sequence_patterns', 
            'topologies': 'patterns'
        }
        
        module_name = module_names.get(pattern_type)
        if not module_name:
            return patterns
        
        try:
            # Import from coordinator module
            full_module_name = f'..{module_name}'
            module = importlib.import_module(full_module_name, __name__)
            
            # Look for _PATTERN suffixed attributes
            for name in dir(module):
                if name.endswith('_PATTERN'):
                    pattern_name = name[:-8].lower()  # Remove _PATTERN suffix
                    patterns[pattern_name] = getattr(module, name)
                    logger.info(f"Loaded Python {pattern_type} pattern: {pattern_name}")
                    
        except ImportError:
            # Module doesn't exist - that's ok
            pass
        except Exception as e:
            logger.warning(f"Error loading Python {pattern_type} patterns: {e}")
        
        return patterns
    
    def _get_builtin_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Get built-in patterns for each type."""
        if pattern_type == 'sequences':
            return self._get_builtin_sequence_patterns()
        elif pattern_type == 'workflows':
            return self._get_builtin_workflow_patterns()
        elif pattern_type == 'topologies':
            return self._get_builtin_topology_patterns()
        else:
            return {}
    
    def _get_builtin_sequence_patterns(self) -> Dict[str, Any]:
        """Built-in sequence patterns (moved from Sequencer)."""
        return {
            'single_pass': {
                'name': 'single_pass',
                'description': 'Execute phase once',
                'iterations': {
                    'type': 'single',
                    'count': 1
                },
                'aggregation': {
                    'type': 'none'  # No aggregation needed
                }
            },
            
            'walk_forward': {
                'name': 'walk_forward',
                'description': 'Rolling window analysis',
                'iterations': {
                    'type': 'windowed',
                    'window_generator': {
                        'type': 'rolling',
                        'train_periods': {'from_config': 'walk_forward.train_periods', 'default': 252},
                        'test_periods': {'from_config': 'walk_forward.test_periods', 'default': 63},
                        'step_size': {'from_config': 'walk_forward.step_size', 'default': 21}
                    }
                },
                'config_modifiers': [
                    {
                        'type': 'set_dates',
                        'train_start': '{window.train_start}',
                        'train_end': '{window.train_end}',
                        'test_start': '{window.test_start}',
                        'test_end': '{window.test_end}'
                    }
                ],
                'aggregation': {
                    'type': 'time_series',
                    'primary_metric': 'sharpe_ratio',
                    'include_equity_curve': True
                }
            },
            
            'parameter_sweep': {
                'name': 'parameter_sweep',
                'description': 'Test multiple parameter combinations',
                'iterations': {
                    'type': 'grid',
                    'parameters': {'from_config': 'parameter_sweep.parameters'}
                },
                'config_modifiers': [
                    {
                        'type': 'set_parameters',
                        'parameters': '{iteration.parameters}'
                    }
                ],
                'aggregation': {
                    'type': 'optimization',
                    'primary_metric': 'sharpe_ratio',
                    'sort_descending': True
                }
            }
        }
    
    def _get_builtin_workflow_patterns(self) -> Dict[str, Any]:
        """Built-in workflow patterns."""
        return {
            # Basic patterns can be added here if needed
        }
    
    def _get_builtin_topology_patterns(self) -> Dict[str, Any]:
        """Built-in topology patterns."""
        return {
            # Basic patterns can be added here if needed
        }