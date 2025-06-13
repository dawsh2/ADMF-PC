#!/usr/bin/env python3
"""
Generate Strategy and Classifier Configurations from Grid

Takes a grid search configuration and expands it into individual
strategy and classifier configurations for the signal generation topology.
"""

import yaml
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class GridConfigGenerator:
    """Generate expanded configurations from parameter grids."""
    
    def __init__(self, grid_config_path: str):
        self.grid_config_path = Path(grid_config_path)
        with open(self.grid_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.strategy_configs = []
        self.classifier_configs = []
        
    def generate_configurations(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate all strategy and classifier configurations from grids."""
        
        # Generate strategies
        if 'strategy_grids' in self.config:
            self._expand_strategy_grids()
            
        # Generate classifiers  
        if 'classifier_grids' in self.config:
            self._expand_classifier_grids()
            
        print(f"Generated {len(self.strategy_configs)} strategy configurations")
        print(f"Generated {len(self.classifier_configs)} classifier configurations")
        
        return self.strategy_configs, self.classifier_configs
        
    def _expand_strategy_grids(self) -> None:
        """Expand strategy parameter grids into individual configurations."""
        
        for category, strategies in self.config['strategy_grids'].items():
            for strategy_spec in strategies:
                strategy_type = strategy_spec['type']
                base_name = strategy_spec['base_name']
                param_grid = strategy_spec['param_grid']
                
                # Generate all parameter combinations
                param_names = list(param_grid.keys())
                param_values = [param_grid[name] if isinstance(param_grid[name], list) 
                               else [param_grid[name]] for name in param_names]
                
                for i, param_combo in enumerate(itertools.product(*param_values)):
                    # Create parameter dict
                    params = dict(zip(param_names, param_combo))
                    
                    # Generate unique name
                    name = f"{base_name}_{i:03d}"
                    
                    # Add some metadata to the name for readability
                    if 'fast_period' in params and 'slow_period' in params:
                        name = f"{base_name}_f{params['fast_period']}_s{params['slow_period']}"
                    elif 'sma_period' in params and 'rsi_threshold_long' in params:
                        name = f"{base_name}_sma{params['sma_period']}_rsi{params['rsi_threshold_long']}"
                    elif 'lookback_period' in params and 'entry_threshold' in params:
                        name = f"{base_name}_lb{params['lookback_period']}_et{params['entry_threshold']}"
                        
                    config = {
                        'type': strategy_type,
                        'name': name,
                        'params': params
                    }
                    
                    self.strategy_configs.append(config)
                    
    def _expand_classifier_grids(self) -> None:
        """Expand classifier parameter grids into individual configurations."""
        
        for category, classifiers in self.config['classifier_grids'].items():
            for classifier_spec in classifiers:
                classifier_type = classifier_spec['type']
                base_name = classifier_spec['base_name']
                param_grid = classifier_spec['param_grid']
                
                # Generate all parameter combinations
                param_names = list(param_grid.keys())
                param_values = [param_grid[name] if isinstance(param_grid[name], list) 
                               else [param_grid[name]] for name in param_names]
                
                for i, param_combo in enumerate(itertools.product(*param_values)):
                    # Create parameter dict
                    params = dict(zip(param_names, param_combo))
                    
                    # Generate unique name with key params
                    name = f"{base_name}_{i:03d}"
                    
                    # Add readable param info to name
                    if 'momentum_threshold' in params:
                        name = f"{base_name}_mt{int(params['momentum_threshold']*100):02d}"
                    elif 'trend_threshold' in params:
                        name = f"{base_name}_tt{int(params['trend_threshold']*1000):02d}"
                    elif 'high_vol_threshold' in params:
                        name = f"{base_name}_hv{int(params['high_vol_threshold']*10):02d}"
                        
                    config = {
                        'type': classifier_type,
                        'name': name,
                        'params': params
                    }
                    
                    self.classifier_configs.append(config)
                    
    def save_expanded_config(self, output_path: str) -> None:
        """Save the expanded configuration for signal generation."""
        
        # Create full config with expanded strategies and classifiers
        expanded_config = self.config.copy()
        
        # Replace grids with expanded configurations
        expanded_config['strategies'] = self.strategy_configs
        expanded_config['classifiers'] = self.classifier_configs
        
        # Remove grid specifications
        expanded_config.pop('strategy_grids', None)
        expanded_config.pop('classifier_grids', None)
        
        # Add expansion metadata
        expanded_config['metadata']['expansion_info'] = {
            'expanded_from': str(self.grid_config_path),
            'expansion_date': datetime.now().isoformat(),
            'total_strategies': len(self.strategy_configs),
            'total_classifiers': len(self.classifier_configs),
            'total_combinations': len(self.strategy_configs) * len(self.classifier_configs)
        }
        
        # Save to file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            yaml.dump(expanded_config, f, default_flow_style=False)
            
        print(f"\nExpanded configuration saved to: {output_path}")
        
    def save_parameter_manifest(self, output_dir: str) -> None:
        """Save a manifest of all parameter combinations for reference."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save strategy manifest
        strategy_manifest = {
            'total_configurations': len(self.strategy_configs),
            'configurations': self.strategy_configs
        }
        
        with open(output_dir / 'strategy_manifest.json', 'w') as f:
            json.dump(strategy_manifest, f, indent=2)
            
        # Save classifier manifest
        classifier_manifest = {
            'total_configurations': len(self.classifier_configs),
            'configurations': self.classifier_configs
        }
        
        with open(output_dir / 'classifier_manifest.json', 'w') as f:
            json.dump(classifier_manifest, f, indent=2)
            
        # Save summary
        summary = {
            'grid_config': str(self.grid_config_path),
            'expansion_date': datetime.now().isoformat(),
            'strategy_types': list(set(s['type'] for s in self.strategy_configs)),
            'classifier_types': list(set(c['type'] for c in self.classifier_configs)),
            'total_strategies': len(self.strategy_configs),
            'total_classifiers': len(self.classifier_configs),
            'parameter_ranges': self._extract_parameter_ranges()
        }
        
        with open(output_dir / 'expansion_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Parameter manifests saved to: {output_dir}")
        
    def _extract_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter ranges for each strategy/classifier type."""
        ranges = {}
        
        # Strategy ranges
        for config in self.strategy_configs:
            strat_type = config['type']
            if strat_type not in ranges:
                ranges[strat_type] = {}
                
            for param, value in config['params'].items():
                if param not in ranges[strat_type]:
                    ranges[strat_type][param] = {'min': value, 'max': value}
                else:
                    ranges[strat_type][param]['min'] = min(ranges[strat_type][param]['min'], value)
                    ranges[strat_type][param]['max'] = max(ranges[strat_type][param]['max'], value)
                    
        # Classifier ranges
        for config in self.classifier_configs:
            class_type = f"classifier_{config['type']}"
            if class_type not in ranges:
                ranges[class_type] = {}
                
            for param, value in config['params'].items():
                if param not in ranges[class_type]:
                    ranges[class_type][param] = {'min': value, 'max': value}
                else:
                    ranges[class_type][param]['min'] = min(ranges[class_type][param]['min'], value)
                    ranges[class_type][param]['max'] = max(ranges[class_type][param]['max'], value)
                    
        return ranges
        
    def estimate_runtime(self, bars_per_config: int = 10000, 
                        seconds_per_bar: float = 0.001) -> None:
        """Estimate total runtime for the grid search."""
        
        total_configs = len(self.strategy_configs) * len(self.classifier_configs)
        total_bars = total_configs * bars_per_config
        estimated_seconds = total_bars * seconds_per_bar
        
        print(f"\nRuntime Estimate:")
        print(f"  Total configurations: {total_configs:,}")
        print(f"  Bars per config: {bars_per_config:,}")
        print(f"  Total bars to process: {total_bars:,}")
        print(f"  Estimated runtime: {estimated_seconds/3600:.1f} hours")
        
        if total_configs > 1000:
            print("\n⚠️  WARNING: Large parameter space!")
            print("  Consider using sampling or reducing parameter ranges")


def main():
    """Main function to generate configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate configurations from parameter grid')
    parser.add_argument('--grid-config', type=str, required=True,
                       help='Path to grid search configuration file')
    parser.add_argument('--output-config', type=str, required=True,
                       help='Path for expanded configuration output')
    parser.add_argument('--manifest-dir', type=str, default='./grid_manifests',
                       help='Directory for parameter manifests')
    parser.add_argument('--estimate-runtime', action='store_true',
                       help='Estimate runtime for full grid search')
    
    args = parser.parse_args()
    
    # Generate configurations
    generator = GridConfigGenerator(args.grid_config)
    generator.generate_configurations()
    
    # Save outputs
    generator.save_expanded_config(args.output_config)
    generator.save_parameter_manifest(args.manifest_dir)
    
    # Runtime estimate
    if args.estimate_runtime:
        generator.estimate_runtime()
        
    print("\nConfiguration generation complete!")


if __name__ == "__main__":
    main()