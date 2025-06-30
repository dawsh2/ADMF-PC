#!/usr/bin/env python3
"""Validate strategy definitions and feature discovery."""

import sys
sys.path.append('.')

from src.core.components.discovery import discover_components
from src.strategy.types import FeatureSpec
import traceback

def validate_strategies():
    """Validate all discovered strategies."""
    print("Validating all strategies...")
    print("=" * 80)
    
    strategies = discover_components('strategy')
    
    if not strategies:
        print("❌ No strategies found!")
        return False
    
    print(f"Found {len(strategies)} strategies\n")
    
    issues = []
    validated = []
    
    for name, info in strategies.items():
        print(f"Checking {name}...")
        
        # Check for required attributes
        if 'function' not in info:
            issues.append(f"❌ {name}: Missing function")
            continue
            
        # Check for feature_discovery
        if 'feature_discovery' not in info:
            # Check if it's using legacy feature_config
            if 'feature_config' in info:
                issues.append(f"⚠️  {name}: Using legacy feature_config - needs migration to feature_discovery")
            else:
                issues.append(f"❌ {name}: Missing feature_discovery")
            continue
        
        # Test feature discovery with default params
        try:
            params = {}
            for param, config in info.get('parameter_space', {}).items():
                params[param] = config.get('default', 0)
            
            features = info['feature_discovery'](params)
            
            # Validate each feature spec
            if not isinstance(features, list):
                issues.append(f"❌ {name}: feature_discovery must return a list")
                continue
                
            for i, spec in enumerate(features):
                if not isinstance(spec, FeatureSpec):
                    issues.append(f"❌ {name}: feature_discovery[{i}] must be a FeatureSpec object, got {type(spec)}")
                elif not hasattr(spec, 'canonical_name'):
                    issues.append(f"❌ {name}: FeatureSpec[{i}] missing canonical_name")
                else:
                    # Validate the spec has required fields
                    if not spec.canonical_name:
                        issues.append(f"❌ {name}: FeatureSpec[{i}] has empty canonical_name")
                    if not hasattr(spec, 'params') or spec.params is None:
                        issues.append(f"❌ {name}: FeatureSpec[{i}] missing params")
            
            validated.append(f"✅ {name}: {len(features)} features configured")
            
        except Exception as e:
            issues.append(f"❌ {name}: feature_discovery failed with error: {str(e)}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    print("\nValidation Results:")
    print("-" * 80)
    
    # Print validated strategies
    for msg in validated:
        print(msg)
    
    # Print issues if any
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        
        # Count legacy strategies
        legacy_count = sum(1 for issue in issues if "legacy feature_config" in issue)
        if legacy_count > 0:
            print(f"\n⚠️  {legacy_count} strategies need migration from feature_config to feature_discovery")
            print("   Run: python migrate_strategies.py")
        
        return False
    else:
        print(f"\n✅ All {len(strategies)} strategies validated successfully!")
        return True

def check_feature_availability():
    """Check which features are available in FeatureHub."""
    print("\nAvailable Features in FeatureHub:")
    print("-" * 80)
    
    try:
        from src.strategy.components.features.hub import FeatureHub
        features = FeatureHub.available_features()
        
        categories = {}
        for feature in sorted(features):
            # Group by category (before first underscore or whole name)
            parts = feature.split('_')
            category = parts[0] if len(parts) > 1 else 'other'
            if category not in categories:
                categories[category] = []
            categories[category].append(feature)
        
        for category, feature_list in sorted(categories.items()):
            print(f"\n{category.upper()}:")
            for feature in feature_list:
                print(f"  - {feature}")
                
    except Exception as e:
        print(f"❌ Could not load FeatureHub: {e}")

if __name__ == "__main__":
    success = validate_strategies()
    
    if "--features" in sys.argv:
        check_feature_availability()
    
    sys.exit(0 if success else 1)