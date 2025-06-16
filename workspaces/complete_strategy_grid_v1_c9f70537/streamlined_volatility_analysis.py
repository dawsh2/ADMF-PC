#!/usr/bin/env python3
"""
Streamlined Volatility Momentum Analysis with Correct Trade Boundary Logic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all required data efficiently."""
    print("Loading data...")
    
    # Load source SPY data
    spy_data = pd.read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'])
    spy_data = spy_data.set_index('timestamp').sort_index()
    
    # Load classifier
    classifier_path = 'traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet'
    classifier_data = pd.read_parquet(classifier_path)
    classifier_data['ts'] = pd.to_datetime(classifier_data['ts'])
    classifier_data = classifier_data.set_index('ts').sort_index()
    
    print(f"Loaded SPY data: {spy_data.shape[0]} rows")
    print(f"Loaded classifier: {classifier_data.shape[0]} rows")
    print(f"Regime distribution: {classifier_data['val'].value_counts().to_dict()}")
    
    return spy_data, classifier_data

def calculate_trade_boundaries_and_returns(signals, spy_data, strategy_name):
    """
    Calculate trade boundaries and returns with correct logic:
    - 0 = always trade closure only
    - +1 to -1 (or -1 to +1) = close current + open new = 2 trade boundaries  
    - Any signal to 0 = 1 trade boundary (closure)
    - 0 to any signal = 1 trade boundary (opening)
    """
    returns_data = []
    prev_signal = 0
    open_position = None
    
    for timestamp, signal in signals.items():
        if signal != prev_signal:
            
            # Get price at this timestamp
            try:
                price_idx = spy_data.index.get_indexer([timestamp], method='nearest')[0]
                price = spy_data.iloc[price_idx]['close']
            except:
                price = spy_data['close'].iloc[-1]
            
            # Case 1: Any signal to 0 (closure)
            if signal == 0 and prev_signal != 0:
                if open_position:
                    # Calculate return
                    if prev_signal == 1:  # Long position
                        trade_return = (price - open_position['entry_price']) / open_position['entry_price']
                    else:  # Short position
                        trade_return = (open_position['entry_price'] - price) / open_position['entry_price']
                    
                    returns_data.append({
                        'entry_timestamp': open_position['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': open_position['entry_price'],
                        'exit_price': price,
                        'position': prev_signal,
                        'return': trade_return
                    })
                    open_position = None
            
            # Case 2: 0 to any signal (opening)
            elif prev_signal == 0 and signal != 0:
                open_position = {
                    'entry_timestamp': timestamp,
                    'entry_price': price,
                    'position': signal
                }
            
            # Case 3: +1 to -1 or -1 to +1 (close current + open new)
            elif (prev_signal == 1 and signal == -1) or (prev_signal == -1 and signal == 1):
                # Close current position
                if open_position:
                    if prev_signal == 1:  # Long position
                        trade_return = (price - open_position['entry_price']) / open_position['entry_price']
                    else:  # Short position
                        trade_return = (open_position['entry_price'] - price) / open_position['entry_price']
                    
                    returns_data.append({
                        'entry_timestamp': open_position['entry_timestamp'],
                        'exit_timestamp': timestamp,
                        'entry_price': open_position['entry_price'],
                        'exit_price': price,
                        'position': prev_signal,
                        'return': trade_return
                    })
                
                # Open new position
                open_position = {
                    'entry_timestamp': timestamp,
                    'entry_price': price,
                    'position': signal
                }
            
            prev_signal = signal
    
    return returns_data

def assign_regimes_to_trades(trades, classifier_data):
    """Assign regimes to trades based on trade timing."""
    enhanced_trades = []
    
    for trade in trades:
        entry_time = trade['entry_timestamp']
        
        # Find nearest regime classification
        try:
            nearest_idx = classifier_data.index.get_indexer([entry_time], method='nearest')[0]
            regime = classifier_data.iloc[nearest_idx]['val']
        except:
            regime = 'neutral'
        
        enhanced_trades.append({
            **trade,
            'regime': regime
        })
    
    return enhanced_trades

def calculate_regime_statistics(trades, strategy_name):
    """Calculate statistics per regime."""
    if not trades:
        return {'strategy_name': strategy_name, 'regimes': {}}
    
    df = pd.DataFrame(trades)
    
    # Assume 0.8 years total, ~0.27 years per regime
    regime_days = 0.27 * 365.25
    
    regime_stats = {}
    
    for regime in ['low_vol_bearish', 'low_vol_bullish', 'neutral']:
        regime_trades = df[df['regime'] == regime]['return']
        
        if len(regime_trades) > 0:
            mean_return = regime_trades.mean()
            std_return = regime_trades.std() if len(regime_trades) > 1 else 0
            
            trades_per_day = len(regime_trades) / regime_days
            
            # Annualized Sharpe = (mean_return / std_return) × sqrt(trades_per_year)
            if std_return > 0:
                trades_per_year = trades_per_day * 365.25
                annualized_sharpe = (mean_return / std_return) * np.sqrt(trades_per_year)
            else:
                annualized_sharpe = 0
            
            regime_stats[regime] = {
                'annualized_sharpe': annualized_sharpe,
                'trades_per_day': trades_per_day,
                'num_trades': len(regime_trades),
                'mean_return': mean_return
            }
        else:
            regime_stats[regime] = {
                'annualized_sharpe': 0,
                'trades_per_day': 0,
                'num_trades': 0,
                'mean_return': 0
            }
    
    return {
        'strategy_name': strategy_name,
        'regimes': regime_stats
    }

def analyze_strategy(strategy_path, spy_data, classifier_data):
    """Analyze a single strategy."""
    try:
        print(f"Analyzing: {Path(strategy_path).stem}")
        
        # Load signals
        signals_data = pd.read_parquet(strategy_path)
        signals_data['ts'] = pd.to_datetime(signals_data['ts'])
        signals_data = signals_data.set_index('ts').sort_index()
        signals = signals_data['val']
        
        strategy_name = Path(strategy_path).stem
        
        # Calculate trades and returns
        trades = calculate_trade_boundaries_and_returns(signals, spy_data, strategy_name)
        
        # Assign regimes
        enhanced_trades = assign_regimes_to_trades(trades, classifier_data)
        
        # Calculate statistics
        results = calculate_regime_statistics(enhanced_trades, strategy_name)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {strategy_path}: {e}")
        return None

def main():
    """Main execution function."""
    # Load data
    spy_data, classifier_data = load_data()
    
    # Strategy files
    strategies = [
        'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet',
        'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet',
        'traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet',
        'traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_14_-85_-15.parquet',
        'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_50.parquet',
        'traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_7_45.parquet',
        'traces/SPY_1m/signals/cci_bands_grid/SPY_cci_bands_grid_11_-100_100.parquet',
        'traces/SPY_1m/signals/cci_bands_grid/SPY_cci_bands_grid_19_-80_80.parquet'
    ]
    
    # Analyze all strategies
    all_results = {}
    for strategy_path in strategies:
        result = analyze_strategy(strategy_path, spy_data, classifier_data)
        if result:
            all_results[result['strategy_name']] = result
    
    # Get top strategies per regime
    regime_rankings = {
        'low_vol_bearish': [],
        'low_vol_bullish': [],
        'neutral': []
    }
    
    for strategy_name, result in all_results.items():
        for regime, stats in result['regimes'].items():
            if stats['num_trades'] > 0:
                regime_rankings[regime].append({
                    'strategy': strategy_name,
                    'annualized_sharpe': stats['annualized_sharpe'],
                    'trades_per_day': stats['trades_per_day'],
                    'num_trades': stats['num_trades']
                })
    
    # Sort and display results
    print("\n" + "="*80)
    print("VOLATILITY MOMENTUM STRATEGY ANALYSIS RESULTS")
    print("="*80)
    
    for regime, strategies in regime_rankings.items():
        sorted_strategies = sorted(strategies, key=lambda x: x['annualized_sharpe'], reverse=True)
        
        print(f"\n{regime.upper().replace('_', ' ')} - TOP 10 STRATEGIES:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Strategy':<45} {'Ann. Sharpe':<12} {'Trades/Day':<12}")
        print("-" * 70)
        
        for i, strategy in enumerate(sorted_strategies[:10], 1):
            print(f"{i:<4} {strategy['strategy'][:43]:<45} "
                  f"{strategy['annualized_sharpe']:<12.4f} {strategy['trades_per_day']:<12.2f}")
    
    print("\n" + "="*80)
    print("Analysis complete!")

if __name__ == "__main__":
    main()