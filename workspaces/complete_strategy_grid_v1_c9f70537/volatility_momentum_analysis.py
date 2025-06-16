#!/usr/bin/env python3
"""
Comprehensive Analysis of Volatility Momentum Strategies with Correct Trade Boundary Logic

This script implements the correct trade counting logic:
- 0 = always trade closure only
- +1 to -1 (or -1 to +1) = close current + open new = 2 trade boundaries  
- Any signal to 0 = 1 trade boundary (closure)
- 0 to any signal = 1 trade boundary (opening)

Returns are calculated using actual SPY prices between trade boundaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VolatilityMomentumAnalyzer:
    """Analyzer for volatility momentum strategies with correct trade boundary logic."""
    
    def __init__(self, source_data_path: str, signals_dir: str, classifier_path: str):
        """Initialize analyzer with data paths."""
        self.source_data_path = source_data_path
        self.signals_dir = Path(signals_dir)
        self.classifier_path = classifier_path
        
        # Load source data and classifier
        self.source_data = self._load_source_data()
        self.classifier_data = self._load_classifier_data()
        
        # Strategy configurations
        self.strategies = {
            'MACD Strategies': [
                'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet',
                'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet'
            ],
            'Williams %R': [
                'traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet',
                'traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_14_-85_-15.parquet'
            ],
            'RSI Threshold': [
                'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_50.parquet',
                'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_7_45.parquet'
            ],
            'CCI Bands': [
                'traces/SPY_1m/signals/cci_bands_grid/SPY_cci_bands_grid_11_-100_100.parquet',
                'traces/SPY_1m/signals/cci_bands_grid/SPY_cci_bands_grid_19_-80_80.parquet'
            ]
        }
        
    def _load_source_data(self) -> pd.DataFrame:
        """Load and prepare source SPY data."""
        data = pd.read_parquet(self.source_data_path)
        
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp').sort_index()
        
        print(f"Loaded source data: {data.shape[0]} rows from {data.index.min()} to {data.index.max()}")
        return data
    
    def _load_classifier_data(self) -> pd.DataFrame:
        """Load and prepare volatility momentum classifier data."""
        data = pd.read_parquet(self.classifier_path)
        
        # Convert timestamp to datetime
        data['ts'] = pd.to_datetime(data['ts'])
        data = data.set_index('ts').sort_index()
        
        print(f"Loaded classifier data: {data.shape[0]} rows")
        print(f"Regime distribution: {data['val'].value_counts().to_dict()}")
        return data
    
    def _calculate_trade_boundaries(self, signals: pd.Series) -> List[Dict]:
        """
        Calculate trade boundaries with correct logic:
        - 0 = always trade closure only
        - +1 to -1 (or -1 to +1) = close current + open new = 2 trade boundaries  
        - Any signal to 0 = 1 trade boundary (closure)
        - 0 to any signal = 1 trade boundary (opening)
        """
        boundaries = []
        prev_signal = 0  # Start with no position
        
        for timestamp, signal in signals.items():
            if signal != prev_signal:
                
                # Case 1: Any signal to 0 (closure)
                if signal == 0 and prev_signal != 0:
                    boundaries.append({
                        'timestamp': timestamp,
                        'action': 'close',
                        'position': prev_signal,
                        'new_position': 0
                    })
                
                # Case 2: 0 to any signal (opening)
                elif prev_signal == 0 and signal != 0:
                    boundaries.append({
                        'timestamp': timestamp,
                        'action': 'open',
                        'position': 0,
                        'new_position': signal
                    })
                
                # Case 3: +1 to -1 or -1 to +1 (close current + open new = 2 boundaries)
                elif (prev_signal == 1 and signal == -1) or (prev_signal == -1 and signal == 1):
                    # Close current position
                    boundaries.append({
                        'timestamp': timestamp,
                        'action': 'close',
                        'position': prev_signal,
                        'new_position': 0
                    })
                    # Open new position
                    boundaries.append({
                        'timestamp': timestamp,
                        'action': 'open',
                        'position': 0,
                        'new_position': signal
                    })
                
                prev_signal = signal
        
        return boundaries
    
    def _calculate_returns(self, boundaries: List[Dict]) -> List[Dict]:
        """Calculate actual returns between trade boundaries using SPY prices."""
        returns_data = []
        
        # Group boundaries into trade pairs (open -> close)
        open_positions = {}
        
        for boundary in boundaries:
            timestamp = boundary['timestamp']
            
            if boundary['action'] == 'open':
                # Store open position
                position = boundary['new_position']
                price = self._get_price_at_timestamp(timestamp)
                open_positions[position] = {
                    'entry_timestamp': timestamp,
                    'entry_price': price,
                    'position': position
                }
            
            elif boundary['action'] == 'close':
                # Calculate return for closed position
                position = boundary['position']
                exit_price = self._get_price_at_timestamp(timestamp)
                
                if position in open_positions:
                    entry_data = open_positions[position]
                    entry_price = entry_data['entry_price']
                    
                    # Calculate return based on position direction
                    if position == 1:  # Long position
                        trade_return = (exit_price - entry_price) / entry_price
                    else:  # Short position (position == -1)
                        trade_return = (entry_price - exit_price) / entry_price
                    
                    returns_data.append({
                        'entry_timestamp': entry_data['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'return': trade_return
                    })
                    
                    # Remove from open positions
                    del open_positions[position]
        
        return returns_data
    
    def _get_price_at_timestamp(self, timestamp: pd.Timestamp) -> float:
        """Get SPY close price at specific timestamp."""
        try:
            # Find the closest timestamp in source data
            idx = self.source_data.index.get_indexer([timestamp], method='nearest')[0]
            return self.source_data.iloc[idx]['close']
        except:
            # Fallback to the last available price
            return self.source_data['close'].iloc[-1]
    
    def _assign_regimes_to_returns(self, returns_data: List[Dict]) -> List[Dict]:
        """Assign volatility momentum regimes to return periods."""
        enhanced_returns = []
        
        for trade in returns_data:
            entry_time = trade['entry_timestamp']
            exit_time = trade['exit_timestamp']
            
            # Find regime during the trade period
            trade_period_regimes = self.classifier_data[
                (self.classifier_data.index >= entry_time) & 
                (self.classifier_data.index <= exit_time)
            ]['val']
            
            if len(trade_period_regimes) > 0:
                # Use most common regime during trade period
                regime = trade_period_regimes.mode().iloc[0] if len(trade_period_regimes.mode()) > 0 else 'neutral'
            else:
                # Find nearest regime
                nearest_idx = self.classifier_data.index.get_indexer([entry_time], method='nearest')[0]
                regime = self.classifier_data.iloc[nearest_idx]['val']
            
            enhanced_returns.append({
                **trade,
                'regime': regime
            })
        
        return enhanced_returns
    
    def analyze_strategy(self, strategy_path: str) -> Dict:
        """Analyze a single strategy file."""
        try:
            # Load strategy signals
            signals_data = pd.read_parquet(strategy_path)
            signals_data['ts'] = pd.to_datetime(signals_data['ts'])
            signals_data = signals_data.set_index('ts').sort_index()
            
            # Extract strategy name
            strategy_name = Path(strategy_path).stem
            
            # Create signals series
            signals = signals_data['val']
            
            # Calculate trade boundaries
            boundaries = self._calculate_trade_boundaries(signals)
            
            # Calculate returns
            returns_data = self._calculate_returns(boundaries)
            
            # Assign regimes
            enhanced_returns = self._assign_regimes_to_returns(returns_data)
            
            # Calculate regime-specific statistics
            results = self._calculate_regime_statistics(enhanced_returns, strategy_name)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing {strategy_path}: {e}")
            return None
        
    def _calculate_regime_statistics(self, returns_data: List[Dict], strategy_name: str) -> Dict:
        """Calculate statistics per regime."""
        if not returns_data:
            return {'strategy_name': strategy_name, 'regimes': {}}
        
        df = pd.DataFrame(returns_data)
        
        # Total time period (approximately 0.8 years based on requirements)
        total_days = (self.source_data.index.max() - self.source_data.index.min()).total_seconds() / (24 * 3600)
        regime_days = total_days / 3  # Each regime ~0.27 years
        
        regime_stats = {}
        
        for regime in ['low_vol_bearish', 'low_vol_bullish', 'neutral']:
            regime_returns = df[df['regime'] == regime]['return']
            
            if len(regime_returns) > 0:
                mean_return = regime_returns.mean()
                std_return = regime_returns.std() if len(regime_returns) > 1 else 0
                
                # Calculate trades per day
                trades_per_day = len(regime_returns) / regime_days
                
                # Calculate annualized Sharpe ratio
                if std_return > 0:
                    # Annualized Sharpe = (mean_return / std_return) × sqrt(trades_per_year)
                    trades_per_year = trades_per_day * 365.25
                    annualized_sharpe = (mean_return / std_return) * np.sqrt(trades_per_year)
                else:
                    annualized_sharpe = 0
                
                regime_stats[regime] = {
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'num_trades': len(regime_returns),
                    'trades_per_day': trades_per_day,
                    'annualized_sharpe': annualized_sharpe
                }
            else:
                regime_stats[regime] = {
                    'mean_return': 0,
                    'std_return': 0,
                    'num_trades': 0,
                    'trades_per_day': 0,
                    'annualized_sharpe': 0
                }
        
        return {
            'strategy_name': strategy_name,
            'regimes': regime_stats,
            'total_trades': len(returns_data)
        }
    
    def analyze_all_strategies(self) -> Dict:
        """Analyze all strategies and return results."""
        all_results = {}
        
        # Flatten strategy paths
        all_strategy_paths = []
        for category, paths in self.strategies.items():
            all_strategy_paths.extend(paths)
        
        for strategy_path in all_strategy_paths:
            print(f"Analyzing: {strategy_path}")
            result = self.analyze_strategy(strategy_path)
            if result:
                all_results[result['strategy_name']] = result
        
        return all_results
    
    def get_top_strategies_per_regime(self, results: Dict, top_n: int = 10) -> Dict:
        """Get top N strategies per regime based on annualized Sharpe ratio."""
        regime_rankings = {
            'low_vol_bearish': [],
            'low_vol_bullish': [],
            'neutral': []
        }
        
        # Collect all strategies for each regime
        for strategy_name, result in results.items():
            for regime, stats in result['regimes'].items():
                if stats['num_trades'] > 0:  # Only include strategies with trades
                    regime_rankings[regime].append({
                        'strategy': strategy_name,
                        'annualized_sharpe': stats['annualized_sharpe'],
                        'trades_per_day': stats['trades_per_day'],
                        'num_trades': stats['num_trades'],
                        'mean_return': stats['mean_return']
                    })
        
        # Sort and get top N for each regime
        top_strategies = {}
        for regime in regime_rankings:
            sorted_strategies = sorted(
                regime_rankings[regime], 
                key=lambda x: x['annualized_sharpe'], 
                reverse=True
            )
            top_strategies[regime] = sorted_strategies[:top_n]
        
        return top_strategies
    
    def print_results(self, top_strategies: Dict):
        """Print formatted results."""
        print("\n" + "="*80)
        print("VOLATILITY MOMENTUM STRATEGY ANALYSIS RESULTS")
        print("="*80)
        
        for regime, strategies in top_strategies.items():
            print(f"\n{regime.upper().replace('_', ' ')} - TOP 10 STRATEGIES:")
            print("-" * 60)
            print(f"{'Rank':<4} {'Strategy':<40} {'Ann. Sharpe':<12} {'Trades/Day':<12}")
            print("-" * 60)
            
            for i, strategy in enumerate(strategies, 1):
                print(f"{i:<4} {strategy['strategy'][:38]:<40} "
                      f"{strategy['annualized_sharpe']:<12.4f} {strategy['trades_per_day']:<12.2f}")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = VolatilityMomentumAnalyzer(
        source_data_path='/Users/daws/ADMF-PC/data/SPY_1m.parquet',
        signals_dir='traces/SPY_1m/signals/',
        classifier_path='traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet'
    )
    
    print("Starting comprehensive volatility momentum analysis...")
    print(f"Using classifier: {analyzer.classifier_path}")
    
    # Run analysis
    results = analyzer.analyze_all_strategies()
    
    # Get top strategies per regime
    top_strategies = analyzer.get_top_strategies_per_regime(results, top_n=10)
    
    # Print results
    analyzer.print_results(top_strategies)
    
    # Save detailed results
    import json
    with open('volatility_momentum_detailed_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(results), f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: volatility_momentum_detailed_results.json")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main()