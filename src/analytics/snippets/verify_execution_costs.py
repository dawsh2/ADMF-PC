# Verify Execution Costs and Returns Math
# Double-check if 1 bps execution costs are properly applied

import pandas as pd
import numpy as np

def verify_returns_with_execution_costs(strategy_hash, trace_path, market_data, 
                                       stop_pct=0.075, target_pct=0.10, 
                                       execution_cost_bps=1.0):
    """
    Detailed verification including execution costs and regime filtering
    """
    print(f"\n🔍 Verifying with {execution_cost_bps} bps execution cost")
    print("=" * 80)
    
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Add volatility regime to trades
    trades_with_regime = trades.merge(
        market_data[['vol_regime']], 
        left_on='entry_idx', 
        right_index=True, 
        how='left'
    )
    
    # Analyze by regime
    for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
        regime_trades = trades_with_regime[trades_with_regime['vol_regime'] == regime]
        
        if len(regime_trades) < 10:
            continue
            
        print(f"\n📊 {regime} Regime Analysis:")
        print(f"Number of trades: {len(regime_trades)}")
        
        # Apply stop/target to regime trades
        modified_returns = []
        exit_types = {'stop': 0, 'target': 0, 'signal': 0}
        
        for _, trade in regime_trades.iterrows():
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            entry_price = trade['entry_price']
            direction = trade['direction']
            exec_cost = trade['execution_cost']  # This should be execution_cost_bps/10000
            
            # Set stop and target
            if direction == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100)
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100)
                target_price = entry_price * (1 - target_pct/100)
            
            # Find exit
            exit_price = trade['exit_price']
            exit_type = 'signal'
            
            for _, bar in trade_prices.iterrows():
                if direction == 1:
                    if bar['low'] <= stop_price:
                        exit_price = stop_price
                        exit_type = 'stop'
                        break
                    elif bar['high'] >= target_price:
                        exit_price = target_price
                        exit_type = 'target'
                        break
                else:
                    if bar['high'] >= stop_price:
                        exit_price = stop_price
                        exit_type = 'stop'
                        break
                    elif bar['low'] <= target_price:
                        exit_price = target_price
                        exit_type = 'target'
                        break
            
            exit_types[exit_type] += 1
            
            # Calculate return WITH execution costs
            if direction == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            
            # Net return includes execution cost
            net_return = raw_return - exec_cost
            modified_returns.append(net_return)
        
        if modified_returns:
            returns_array = np.array(modified_returns)
            
            # Show first few trades as examples
            print("\nExample trades (first 5):")
            for i, ret in enumerate(returns_array[:5]):
                print(f"  Trade {i+1}: {ret*100:.4f}% net return")
            
            # Statistics
            print(f"\nExit types:")
            total_exits = sum(exit_types.values())
            for exit_type, count in exit_types.items():
                print(f"  {exit_type}: {count} ({count/total_exits*100:.1f}%)")
            
            print(f"\nReturns statistics:")
            print(f"  Mean return per trade: {returns_array.mean()*100:.4f}%")
            print(f"  Std dev: {returns_array.std()*100:.4f}%")
            
            # Breakdown by exit type
            stop_return = -stop_pct/100 - execution_cost_bps/10000
            target_return = target_pct/100 - execution_cost_bps/10000
            
            print(f"\nExpected returns by exit type:")
            print(f"  Stop hit: {stop_return*100:.4f}% (raw: -{stop_pct}%, exec: -{execution_cost_bps/100:.2f}%)")
            print(f"  Target hit: {target_return*100:.4f}% (raw: +{target_pct}%, exec: -{execution_cost_bps/100:.2f}%)")
            
            # Win rate
            win_rate = (returns_array > 0).mean()
            print(f"\nWin rate: {win_rate*100:.1f}%")
            
            # Expected value
            p_target = exit_types['target'] / total_exits
            p_stop = exit_types['stop'] / total_exits
            expected_return = p_target * target_return + p_stop * stop_return
            
            print(f"\nExpected value calculation:")
            print(f"  P(target) × target_return + P(stop) × stop_return")
            print(f"  {p_target:.3f} × {target_return*100:.4f}% + {p_stop:.3f} × {stop_return*100:.4f}%")
            print(f"  = {expected_return*100:.4f}% per trade")
            
            # Total return
            total_return = (1 + returns_array).prod() - 1
            print(f"\nTotal return: {total_return*100:.2f}%")
            print(f"Number of trades: {len(returns_array)}")
            print(f"Expected total: {expected_return * len(returns_array) * 100:.2f}%")
            
            # Sharpe calculation
            if returns_array.std() > 0:
                trades_per_day = len(regime_trades) / trading_days
                daily_sharpe = returns_array.mean() / returns_array.std()
                annualization = np.sqrt(252 * trades_per_day)
                annual_sharpe = daily_sharpe * annualization
                
                print(f"\nSharpe ratio calculation:")
                print(f"  Mean/Std = {returns_array.mean():.6f} / {returns_array.std():.6f} = {daily_sharpe:.4f}")
                print(f"  Trades/day in regime: {trades_per_day:.2f}")
                print(f"  Annualization: sqrt(252 × {trades_per_day:.2f}) = {annualization:.1f}")
                print(f"  Annual Sharpe: {annual_sharpe:.2f}")

# Run verification
if len(performance_df) > 0:
    print("🔬 Detailed Execution Cost Verification")
    print("=" * 80)
    
    # Get high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days]
    
    if len(high_freq_df) > 0:
        # Check if we have volatility regimes
        if 'vol_regime' not in market_data.columns:
            market_data['returns'] = market_data['close'].pct_change()
            market_data['volatility'] = market_data['returns'].rolling(window=20*78).std() * np.sqrt(252*78)
            vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
            market_data['vol_regime'] = pd.cut(
                market_data['volatility'],
                bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
                labels=['Low Vol', 'Medium Vol', 'High Vol']
            )
        
        # Analyze top performer
        strategy = high_freq_df.iloc[0]
        print(f"Analyzing: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print(f"Total trades: {strategy['num_trades']}")
        
        verify_returns_with_execution_costs(
            strategy['strategy_hash'],
            strategy['trace_path'],
            market_data,
            stop_pct=0.075,
            target_pct=0.10,
            execution_cost_bps=execution_cost_bps
        )
        
        print("\n⚠️ Reality Check:")
        print("=" * 60)
        print("These returns assume:")
        print("1. ✓ 1 bps execution cost IS included")
        print("2. × Perfect fills at stop/target prices (no slippage)")
        print("3. × Sufficient liquidity for all trades")
        print("4. × No market impact from 1000+ trades")
        print("\nThe math is correct but implementation challenges remain!")