#!/usr/bin/env python3
"""
Diagnostic script to analyze why strategies aren't generating signals.
Tests each missing strategy individually to identify the root cause.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.handlers import SimpleHistoricalDataHandler
from src.data.loaders import SimpleCSVLoader
from src.strategy.components.features.hub import FeatureHub
from src.strategy.state import StrategyState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyDiagnostics:
    """Diagnose why strategies aren't generating signals."""
    
    def __init__(self):
        self.missing_strategies = [
            "accumulation_distribution_grid",
            "adx_trend_strength_grid", 
            "aroon_crossover_grid",
            "bollinger_breakout_grid",
            "donchian_breakout_grid",
            "fibonacci_retracement_grid",
            "ichimoku_grid",
            "keltner_breakout_grid",
            "linear_regression_slope_grid",
            "macd_crossover_grid",
            "obv_trend_grid",
            "parabolic_sar_grid",
            "pivot_points_grid",
            "price_action_swing_grid",
            "roc_threshold_grid",
            "stochastic_crossover_grid",
            "stochastic_rsi_grid",
            "supertrend_grid",
            "support_resistance_breakout_grid",
            "ultimate_oscillator_grid",
            "vortex_crossover_grid",
            "vwap_deviation_grid"
        ]
        
        # Load config to get strategy parameters
        with open('config/expansive_grid_search.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create strategy lookup
        self.strategy_config = {}
        for strategy in self.config['strategies']:
            self.strategy_config[strategy['name']] = strategy
    
    def diagnose_strategy(self, strategy_name: str, num_bars: int = 2000) -> Dict[str, Any]:
        """Diagnose a single strategy."""
        result = {
            'strategy': strategy_name,
            'status': 'unknown',
            'signals_generated': 0,
            'first_signal_bar': None,
            'errors': [],
            'warnings': [],
            'feature_issues': [],
            'bar_requirements': None
        }
        
        # Get strategy config
        if strategy_name not in self.strategy_config:
            result['status'] = 'config_missing'
            result['errors'].append(f"Strategy {strategy_name} not found in config")
            return result
        
        strategy_cfg = self.strategy_config[strategy_name]
        strategy_type = strategy_cfg['type']
        
        # Get first parameter set for testing
        params = {}
        for key, values in strategy_cfg['params'].items():
            params[key] = values[0] if isinstance(values, list) else values
        
        logger.info(f"Testing {strategy_name} with params: {params}")
        
        try:
            # Load data
            data_handler = CsvDataHandler(
                file_path='./data/SPY_1m.csv',
                symbol='SPY',
                timeframe='1m'
            )
            
            # Initialize feature hub
            feature_hub = FeatureHub()
            
            # Discover strategies
            discovery = StrategyDiscovery()
            available_strategies = discovery.discover_strategies()
            
            if strategy_type not in available_strategies:
                result['status'] = 'strategy_not_found'
                result['errors'].append(f"Strategy type '{strategy_type}' not found in registry")
                return result
            
            # Get strategy function
            strategy_func = available_strategies[strategy_type]
            
            # Create strategy state
            state = StrategyState(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                config={'params': params}
            )
            
            # Process bars
            signals_by_bar = []
            feature_readiness = {}
            feature_values_log = []
            
            for i, bar in enumerate(data_handler.stream_bars(max_bars=num_bars)):
                # Update features
                features = feature_hub.update(bar)
                
                # Track feature readiness
                if i % 100 == 0:  # Sample every 100 bars
                    ready_features = [k for k, v in features.items() if v is not None]
                    feature_readiness[i] = len(ready_features)
                
                # Try to generate signal
                try:
                    signal = strategy_func(features, bar, params)
                    
                    if signal is not None:
                        signals_by_bar.append((i, signal))
                        if result['first_signal_bar'] is None:
                            result['first_signal_bar'] = i
                        
                        # Log first few signals
                        if len(signals_by_bar) <= 3:
                            logger.info(f"Signal at bar {i}: {signal}")
                    
                    # Log feature values for first potential signal
                    if i >= 50 and len(feature_values_log) < 5:
                        relevant_features = self._get_relevant_features(strategy_type, features)
                        feature_values_log.append({
                            'bar': i,
                            'features': relevant_features
                        })
                        
                except Exception as e:
                    if f"Error at bar {i}" not in [err[:15] for err in result['errors']]:
                        result['errors'].append(f"Error at bar {i}: {str(e)}")
                    if len(result['errors']) > 5:
                        result['errors'] = result['errors'][:5] + [f"... and {len(result['errors']) - 5} more errors"]
            
            # Analyze results
            result['signals_generated'] = len(signals_by_bar)
            result['feature_readiness'] = feature_readiness
            
            if len(signals_by_bar) > 0:
                result['status'] = 'working'
                result['signal_bars'] = [bar for bar, _ in signals_by_bar[:10]]  # First 10
            else:
                result['status'] = 'no_signals'
                
                # Check feature availability
                self._check_feature_availability(
                    strategy_type, feature_values_log, result
                )
                
                # Estimate bar requirements
                if feature_readiness:
                    max_features = max(feature_readiness.values())
                    bars_to_90pct = None
                    for bar, count in sorted(feature_readiness.items()):
                        if count >= 0.9 * max_features:
                            bars_to_90pct = bar
                            break
                    result['bar_requirements'] = bars_to_90pct
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(f"Fatal error: {str(e)}")
            logger.error(f"Error testing {strategy_name}: {e}", exc_info=True)
        
        return result
    
    def _get_relevant_features(self, strategy_type: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get features relevant to a strategy type."""
        # Map strategy types to their expected features
        feature_patterns = {
            'aroon_crossover': ['aroon_'],
            'supertrend': ['supertrend_'],
            'adx_trend_strength': ['adx_', 'plus_di_', 'minus_di_'],
            'macd_crossover': ['macd_', 'macd_signal_', 'macd_histogram_'],
            'ichimoku': ['ichimoku_'],
            'parabolic_sar': ['sar_'],
            'bollinger_breakout': ['bbands_'],
            'keltner_breakout': ['keltner_'],
            'donchian_breakout': ['donchian_'],
            'ultimate_oscillator': ['ultosc_'],
            'stochastic_crossover': ['stoch_'],
            'stochastic_rsi': ['stochrsi_'],
            'vortex_crossover': ['vortex_'],
            'obv_trend': ['obv'],
            'accumulation_distribution': ['ad_'],
            'vwap_deviation': ['vwap'],
            'linear_regression_slope': ['linearreg_'],
            'roc_threshold': ['roc_'],
            'pivot_points': ['pivot_'],
            'fibonacci_retracement': ['fib_'],
            'support_resistance_breakout': ['support_', 'resistance_'],
            'price_action_swing': ['swing_']
        }
        
        relevant = {}
        patterns = feature_patterns.get(strategy_type, [])
        
        for key, value in features.items():
            for pattern in patterns:
                if pattern in key:
                    relevant[key] = value
                    break
        
        return relevant
    
    def _check_feature_availability(self, strategy_type: str, 
                                   feature_logs: List[Dict], 
                                   result: Dict[str, Any]):
        """Check if required features are available."""
        if not feature_logs:
            result['feature_issues'].append("No feature data collected")
            return
        
        # Analyze feature availability
        all_none = True
        partial_none = False
        
        for log in feature_logs:
            features = log['features']
            if not features:
                result['feature_issues'].append(f"No relevant features found at bar {log['bar']}")
                continue
                
            none_count = sum(1 for v in features.values() if v is None)
            if none_count == 0:
                all_none = False
            elif none_count < len(features):
                all_none = False
                partial_none = True
                result['feature_issues'].append(
                    f"Bar {log['bar']}: {none_count}/{len(features)} features are None"
                )
        
        if all_none:
            result['feature_issues'].append("All required features are None throughout test")
            result['warnings'].append("Strategy may need more warmup bars")
        elif partial_none:
            result['warnings'].append("Some features intermittently None")
    
    def run_diagnostics(self):
        """Run diagnostics on all missing strategies."""
        results = []
        
        print("\n=== Strategy Signal Generation Diagnostics ===\n")
        
        for i, strategy in enumerate(self.missing_strategies):
            print(f"[{i+1}/{len(self.missing_strategies)}] Testing {strategy}...")
            result = self.diagnose_strategy(strategy)
            results.append(result)
            
            # Print summary
            status_icon = {
                'working': '✅',
                'no_signals': '❌', 
                'error': '🔥',
                'strategy_not_found': '❓',
                'config_missing': '📄'
            }.get(result['status'], '❔')
            
            print(f"  {status_icon} Status: {result['status']}")
            print(f"     Signals: {result['signals_generated']}")
            
            if result['first_signal_bar']:
                print(f"     First signal at bar: {result['first_signal_bar']}")
            
            if result['bar_requirements']:
                print(f"     Estimated bars needed: {result['bar_requirements']}")
                
            if result['errors']:
                print(f"     Errors: {result['errors'][0]}")
                
            if result['warnings']:
                print(f"     Warnings: {', '.join(result['warnings'])}")
            
            print()
        
        # Summary analysis
        self._print_summary(results)
        
        # Save detailed results
        self._save_results(results)
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary analysis."""
        print("\n=== SUMMARY ===\n")
        
        # Group by status
        by_status = defaultdict(list)
        for r in results:
            by_status[r['status']].append(r['strategy'])
        
        print("Status breakdown:")
        for status, strategies in by_status.items():
            print(f"  {status}: {len(strategies)} strategies")
            if len(strategies) <= 5:
                for s in strategies:
                    print(f"    - {s}")
        
        # Common issues
        print("\nCommon issues:")
        feature_issues = sum(1 for r in results if r['feature_issues'])
        high_bar_req = sum(1 for r in results if r.get('bar_requirements', 0) > 1000)
        
        print(f"  - Feature availability issues: {feature_issues} strategies")
        print(f"  - Need >1000 bars for warmup: {high_bar_req} strategies")
        
        # Recommendations
        print("\nRecommendations:")
        if high_bar_req > 0:
            print("  1. Increase --bars to 3000+ for complex indicators")
        if feature_issues > 0:
            print("  2. Check feature naming consistency")
        print("  3. Add debug logging to strategy functions")
        print("  4. Consider reducing warmup requirements for 1-min data")
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save detailed results to file."""
        import json
        
        output_file = 'strategy_diagnostics_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    diagnostics = StrategyDiagnostics()
    diagnostics.run_diagnostics()