#!/usr/bin/env python3
"""
Analyze Keltner strategy performance across different market regimes:
- Volatility levels
- Volume conditions
- Trend strength
- VWAP positioning
- Time of day
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/daws/ADMF-PC')

class RegimeAnalyzer:
    def __init__(self, workspace_path: str, ohlc_path: str = None):
        self.workspace_path = Path(workspace_path)
        self.signals_path = self.workspace_path / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"
        
        # Load OHLC data
        if ohlc_path:
            self.ohlc_df = pd.read_csv(ohlc_path)
        else:
            # Try to find SPY data
            possible_paths = [
                "/Users/daws/ADMF-PC/data/SPY_5m.csv",
                "/Users/daws/ADMF-PC/data/SPY_5m.parquet"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    if path.endswith('.parquet'):
                        self.ohlc_df = pd.read_parquet(path)
                    else:
                        self.ohlc_df = pd.read_csv(path)
                    break
        
        # Ensure we have the required columns
        self._prepare_ohlc_data()
        
    def _prepare_ohlc_data(self):
        """Prepare OHLC data with technical indicators."""
        df = self.ohlc_df
        
        # Ensure we have an index column
        if 'idx' not in df.columns:
            df['idx'] = range(len(df))
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(78) * 100  # Annualized
        df['volatility_5'] = df['returns'].rolling(5).std() * np.sqrt(78) * 100
        df['atr'] = self._calculate_atr(df, 14)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['dollar_volume'] = df['close'] * df['volume']
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['trend_direction'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # VWAP
        df['vwap'] = self._calculate_vwap(df)
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['time_of_day'] = df['hour'] + df['minute'] / 60
        else:
            # Assume 5-minute bars from 9:30 to 4:00
            df['bar_of_day'] = df['idx'] % 78
            df['time_of_day'] = 9.5 + (df['bar_of_day'] * 5 / 60)
        
        # Market regimes
        df['volatility_regime'] = pd.qcut(df['volatility_20'].dropna(), q=3, labels=['Low', 'Medium', 'High'])
        df['volume_regime'] = pd.qcut(df['volume_ratio'].dropna(), q=3, labels=['Low', 'Medium', 'High'])
        df['trend_regime'] = pd.cut(df['trend_strength'].dropna(), bins=[-np.inf, -1, 1, np.inf], labels=['Down', 'Neutral', 'Up'])
        
        self.ohlc_df = df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily VWAP."""
        # Reset VWAP each day (every 78 bars for 5-min data)
        df['day'] = df['idx'] // 78
        
        vwap_values = []
        for day, group in df.groupby('day'):
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            cumulative_tpv = (typical_price * group['volume']).cumsum()
            cumulative_volume = group['volume'].cumsum()
            vwap = cumulative_tpv / cumulative_volume
            vwap_values.extend(vwap.values)
        
        return pd.Series(vwap_values, index=df.index)
    
    def analyze_strategy_by_regime(self, signals_df: pd.DataFrame) -> Dict:
        """Analyze strategy performance across different regimes."""
        # Merge signals with OHLC data
        signals_df = signals_df.sort_values('idx').reset_index(drop=True)
        
        # Create trades from signals
        trades = []
        current_position = None
        
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            signal = row['val']
            price = row['px']
            idx = row['idx']
            
            if signal != 0:
                if current_position is not None:
                    # Close existing position
                    exit_idx = idx
                    
                    # Get regime information at entry
                    entry_regimes = self._get_regimes_at_index(current_position['entry_idx'])
                    
                    trade = {
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': exit_idx,
                        'entry_price': current_position['entry_price'],
                        'exit_price': price,
                        'direction': current_position['direction'],
                        **entry_regimes
                    }
                    
                    # Calculate return
                    if trade['direction'] == 'long':
                        trade['return_bps'] = np.log(price / current_position['entry_price']) * 10000
                    else:
                        trade['return_bps'] = -np.log(price / current_position['entry_price']) * 10000
                    
                    trades.append(trade)
                
                # Open new position
                current_position = {
                    'entry_idx': idx,
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                # Exit signal
                exit_idx = idx
                entry_regimes = self._get_regimes_at_index(current_position['entry_idx'])
                
                trade = {
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': exit_idx,
                    'entry_price': current_position['entry_price'],
                    'exit_price': price,
                    'direction': current_position['direction'],
                    **entry_regimes
                }
                
                if trade['direction'] == 'long':
                    trade['return_bps'] = np.log(price / current_position['entry_price']) * 10000
                else:
                    trade['return_bps'] = -np.log(price / current_position['entry_price']) * 10000
                
                trades.append(trade)
                current_position = None
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            return {}
        
        # Apply execution costs
        exec_mult = 1 - (0.5 / 10000)
        trades_df['return_bps'] = trades_df['return_bps'] * exec_mult
        
        return trades_df
    
    def _get_regimes_at_index(self, idx: int) -> Dict:
        """Get regime information at a specific index."""
        if idx >= len(self.ohlc_df):
            idx = len(self.ohlc_df) - 1
        
        row = self.ohlc_df.iloc[idx]
        
        return {
            'volatility': row.get('volatility_20', np.nan),
            'volatility_regime': row.get('volatility_regime', 'Unknown'),
            'volume_ratio': row.get('volume_ratio', np.nan),
            'volume_regime': row.get('volume_regime', 'Unknown'),
            'trend_strength': row.get('trend_strength', np.nan),
            'trend_regime': row.get('trend_regime', 'Unknown'),
            'trend_direction': row.get('trend_direction', 0),
            'vwap_distance': row.get('vwap_distance', np.nan),
            'time_of_day': row.get('time_of_day', np.nan),
            'atr': row.get('atr', np.nan)
        }
    
    def generate_regime_report(self, trades_df: pd.DataFrame) -> None:
        """Generate comprehensive regime analysis report."""
        print("\n" + "="*80)
        print("REGIME-BASED PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Overall performance
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']
        
        print(f"\nOverall Performance:")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Average return: {trades_df['return_bps'].mean():.2f} bps")
        print(f"  Long trades: {len(long_trades)} ({trades_df['return_bps'][trades_df['direction'] == 'long'].mean():.2f} bps)")
        print(f"  Short trades: {len(short_trades)} ({trades_df['return_bps'][trades_df['direction'] == 'short'].mean():.2f} bps)")
        
        # 1. Volatility Analysis
        print(f"\n{'='*60}")
        print("1. VOLATILITY REGIME ANALYSIS")
        print(f"{'='*60}")
        
        for regime in ['Low', 'Medium', 'High']:
            regime_trades = trades_df[trades_df['volatility_regime'] == regime]
            if len(regime_trades) > 0:
                long_regime = regime_trades[regime_trades['direction'] == 'long']
                short_regime = regime_trades[regime_trades['direction'] == 'short']
                
                print(f"\n{regime} Volatility:")
                print(f"  Trades: {len(regime_trades)} ({len(regime_trades)/len(trades_df)*100:.1f}%)")
                print(f"  Avg return: {regime_trades['return_bps'].mean():.2f} bps")
                print(f"  Long: {len(long_regime)} trades, {long_regime['return_bps'].mean():.2f} bps" if len(long_regime) > 0 else "  Long: No trades")
                print(f"  Short: {len(short_regime)} trades, {short_regime['return_bps'].mean():.2f} bps" if len(short_regime) > 0 else "  Short: No trades")
        
        # 2. Volume Analysis
        print(f"\n{'='*60}")
        print("2. VOLUME REGIME ANALYSIS")
        print(f"{'='*60}")
        
        for regime in ['Low', 'Medium', 'High']:
            regime_trades = trades_df[trades_df['volume_regime'] == regime]
            if len(regime_trades) > 0:
                long_regime = regime_trades[regime_trades['direction'] == 'long']
                short_regime = regime_trades[regime_trades['direction'] == 'short']
                
                print(f"\n{regime} Volume:")
                print(f"  Trades: {len(regime_trades)} ({len(regime_trades)/len(trades_df)*100:.1f}%)")
                print(f"  Avg return: {regime_trades['return_bps'].mean():.2f} bps")
                print(f"  Long: {len(long_regime)} trades, {long_regime['return_bps'].mean():.2f} bps" if len(long_regime) > 0 else "  Long: No trades")
                print(f"  Short: {len(short_regime)} trades, {short_regime['return_bps'].mean():.2f} bps" if len(short_regime) > 0 else "  Short: No trades")
        
        # 3. Trend Analysis
        print(f"\n{'='*60}")
        print("3. TREND REGIME ANALYSIS")
        print(f"{'='*60}")
        
        for regime in ['Down', 'Neutral', 'Up']:
            regime_trades = trades_df[trades_df['trend_regime'] == regime]
            if len(regime_trades) > 0:
                long_regime = regime_trades[regime_trades['direction'] == 'long']
                short_regime = regime_trades[regime_trades['direction'] == 'short']
                
                print(f"\n{regime} Trend:")
                print(f"  Trades: {len(regime_trades)} ({len(regime_trades)/len(trades_df)*100:.1f}%)")
                print(f"  Avg return: {regime_trades['return_bps'].mean():.2f} bps")
                print(f"  Long: {len(long_regime)} trades, {long_regime['return_bps'].mean():.2f} bps" if len(long_regime) > 0 else "  Long: No trades")
                print(f"  Short: {len(short_regime)} trades, {short_regime['return_bps'].mean():.2f} bps" if len(short_regime) > 0 else "  Short: No trades")
        
        # 4. VWAP Analysis
        print(f"\n{'='*60}")
        print("4. VWAP POSITIONING ANALYSIS")
        print(f"{'='*60}")
        
        trades_df['vwap_position'] = pd.cut(trades_df['vwap_distance'], 
                                           bins=[-np.inf, -0.1, 0.1, np.inf],
                                           labels=['Below', 'Near', 'Above'])
        
        for position in ['Below', 'Near', 'Above']:
            position_trades = trades_df[trades_df['vwap_position'] == position]
            if len(position_trades) > 0:
                long_position = position_trades[position_trades['direction'] == 'long']
                short_position = position_trades[position_trades['direction'] == 'short']
                
                print(f"\n{position} VWAP:")
                print(f"  Trades: {len(position_trades)} ({len(position_trades)/len(trades_df)*100:.1f}%)")
                print(f"  Avg return: {position_trades['return_bps'].mean():.2f} bps")
                print(f"  Long: {len(long_position)} trades, {long_position['return_bps'].mean():.2f} bps" if len(long_position) > 0 else "  Long: No trades")
                print(f"  Short: {len(short_position)} trades, {short_position['return_bps'].mean():.2f} bps" if len(short_position) > 0 else "  Short: No trades")
        
        # 5. Time of Day Analysis
        print(f"\n{'='*60}")
        print("5. TIME OF DAY ANALYSIS")
        print(f"{'='*60}")
        
        trades_df['time_period'] = pd.cut(trades_df['time_of_day'], 
                                         bins=[9.5, 10.5, 12, 14.5, 16],
                                         labels=['Open', 'Morning', 'Midday', 'Close'])
        
        for period in ['Open', 'Morning', 'Midday', 'Close']:
            period_trades = trades_df[trades_df['time_period'] == period]
            if len(period_trades) > 0:
                long_period = period_trades[period_trades['direction'] == 'long']
                short_period = period_trades[period_trades['direction'] == 'short']
                
                print(f"\n{period}:")
                print(f"  Trades: {len(period_trades)} ({len(period_trades)/len(trades_df)*100:.1f}%)")
                print(f"  Avg return: {period_trades['return_bps'].mean():.2f} bps")
                print(f"  Long: {len(long_period)} trades, {long_period['return_bps'].mean():.2f} bps" if len(long_period) > 0 else "  Long: No trades")
                print(f"  Short: {len(short_period)} trades, {short_period['return_bps'].mean():.2f} bps" if len(short_period) > 0 else "  Short: No trades")
        
        # 6. Key Insights
        print(f"\n{'='*60}")
        print("KEY INSIGHTS")
        print(f"{'='*60}")
        
        # Find best regimes
        best_volatility = trades_df.groupby('volatility_regime')['return_bps'].mean().idxmax()
        best_volume = trades_df.groupby('volume_regime')['return_bps'].mean().idxmax()
        best_trend = trades_df.groupby('trend_regime')['return_bps'].mean().idxmax()
        
        print(f"\nBest Performing Regimes:")
        print(f"  Volatility: {best_volatility} ({trades_df[trades_df['volatility_regime'] == best_volatility]['return_bps'].mean():.2f} bps)")
        print(f"  Volume: {best_volume} ({trades_df[trades_df['volume_regime'] == best_volume]['return_bps'].mean():.2f} bps)")
        print(f"  Trend: {best_trend} ({trades_df[trades_df['trend_regime'] == best_trend]['return_bps'].mean():.2f} bps)")
        
        # Long vs Short insights
        print(f"\nDirectional Insights:")
        long_up_trend = trades_df[(trades_df['direction'] == 'long') & (trades_df['trend_regime'] == 'Up')]
        short_down_trend = trades_df[(trades_df['direction'] == 'short') & (trades_df['trend_regime'] == 'Down')]
        
        if len(long_up_trend) > 0:
            print(f"  Long in Uptrend: {long_up_trend['return_bps'].mean():.2f} bps ({len(long_up_trend)} trades)")
        if len(short_down_trend) > 0:
            print(f"  Short in Downtrend: {short_down_trend['return_bps'].mean():.2f} bps ({len(short_down_trend)} trades)")
        
        # Save detailed results
        trades_df.to_csv("keltner_regime_analysis.csv", index=False)
        print(f"\nDetailed results saved to keltner_regime_analysis.csv")


def main():
    # Analyze the best strategies from both workspaces
    workspaces = [
        ("/Users/daws/ADMF-PC/configs/optimize_keltner_with_filters/20250622_112210", 0),  # Strategy 0
        ("/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448", 4)  # Strategy 4
    ]
    
    for workspace_path, strategy_num in workspaces:
        print(f"\n{'='*80}")
        print(f"ANALYZING WORKSPACE: {workspace_path.split('/')[-1]}")
        print(f"STRATEGY: {strategy_num}")
        print(f"{'='*80}")
        
        analyzer = RegimeAnalyzer(workspace_path)
        
        # Load strategy signals
        signals_file = Path(workspace_path) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands" / f"SPY_5m_compiled_strategy_{strategy_num}.parquet"
        
        if signals_file.exists():
            signals_df = pd.read_parquet(signals_file)
            trades_df = analyzer.analyze_strategy_by_regime(signals_df)
            
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                analyzer.generate_regime_report(trades_df)
            else:
                print("No trades found for analysis")
        else:
            print(f"Strategy file not found: {signals_file}")


if __name__ == "__main__":
    main()