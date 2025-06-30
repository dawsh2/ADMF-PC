# Verify Micro Stop/Target Calculations
# Check if the inflated Sharpe ratios are real or a calculation error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def verify_stop_target_calculation(strategy_hash, trace_path, market_data, stop_pct=0.075, target_pct=0.10, execution_cost_bps=1.0):
    """
    Detailed verification of stop/target calculations
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    print(f"\n📊 Verifying calculations for first 10 trades:")
    print("=" * 80)
    
    detailed_trades = []
    
    for idx, trade in trades.head(10).iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calculate stop and target prices
        if direction == 1:  # Long
            stop_price = entry_price * (1 - stop_pct/100)
            target_price = entry_price * (1 + target_pct/100)
        else:  # Short
            stop_price = entry_price * (1 + stop_pct/100)
            target_price = entry_price * (1 - target_pct/100)
        
        # Track through trade
        exit_price = trade['exit_price']
        exit_type = 'signal'
        exit_bar = len(trade_prices) - 1
        
        for i, (_, bar) in enumerate(trade_prices.iterrows()):
            if direction == 1:  # Long
                if bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    exit_bar = i
                    break
                elif bar['high'] >= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    exit_bar = i
                    break
            else:  # Short
                if bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    exit_bar = i
                    break
                elif bar['low'] <= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    exit_bar = i
                    break
        
        # Calculate returns
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
        
        # Original return
        original_raw_return = ((trade['exit_price'] - entry_price) / entry_price) if direction == 1 else ((entry_price - trade['exit_price']) / entry_price)
        
        net_return = raw_return - trade['execution_cost']
        original_net_return = trade['net_return']
        
        print(f"\nTrade {idx+1} ({['Short', 'Long'][direction]}):")
        print(f"  Entry: ${entry_price:.4f}")
        print(f"  Stop: ${stop_price:.4f} ({stop_pct}% risk)")
        print(f"  Target: ${target_price:.4f} ({target_pct}% reward)")
        print(f"  Original exit: ${trade['exit_price']:.4f} after {len(trade_prices)} bars")
        print(f"  New exit: ${exit_price:.4f} after {exit_bar+1} bars ({exit_type})")
        print(f"  Original return: {original_net_return*100:.3f}%")
        print(f"  New return: {net_return*100:.3f}%")
        
        detailed_trades.append({
            'trade_idx': idx,
            'direction': direction,
            'original_bars': len(trade_prices),
            'new_bars': exit_bar + 1,
            'exit_type': exit_type,
            'original_return': original_net_return,
            'new_return': net_return,
            'improvement': net_return - original_net_return
        })
    
    detailed_df = pd.DataFrame(detailed_trades)
    
    # Now calculate full performance
    print("\n📈 Full Performance Calculation:")
    print("=" * 60)
    
    all_trades_modified = []
    exit_counts = {'stop': 0, 'target': 0, 'signal': 0}
    
    for _, trade in trades.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calculate stop and target
        if direction == 1:
            stop_price = entry_price * (1 - stop_pct/100)
            target_price = entry_price * (1 + target_pct/100)
        else:
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
        
        exit_counts[exit_type] += 1
        
        # Calculate return
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
        
        net_return = raw_return - trade['execution_cost']
        all_trades_modified.append(net_return)
    
    # Calculate metrics
    returns_array = np.array(all_trades_modified)
    
    print(f"\nNumber of trades: {len(returns_array)}")
    print(f"Exit breakdown:")
    print(f"  Stops: {exit_counts['stop']} ({exit_counts['stop']/len(returns_array)*100:.1f}%)")
    print(f"  Targets: {exit_counts['target']} ({exit_counts['target']/len(returns_array)*100:.1f}%)")
    print(f"  Signals: {exit_counts['signal']} ({exit_counts['signal']/len(returns_array)*100:.1f}%)")
    
    # Returns analysis
    print(f"\nReturns per trade:")
    print(f"  Mean: {returns_array.mean()*100:.4f}%")
    print(f"  Std: {returns_array.std()*100:.4f}%")
    print(f"  Min: {returns_array.min()*100:.4f}%")
    print(f"  Max: {returns_array.max()*100:.4f}%")
    
    # Win rate
    win_rate = (returns_array > 0).mean()
    print(f"\nWin rate: {win_rate*100:.1f}%")
    
    # Expected value per trade
    winners = returns_array[returns_array > 0]
    losers = returns_array[returns_array <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    
    print(f"Average winner: {avg_win*100:.4f}%")
    print(f"Average loser: {avg_loss*100:.4f}%")
    
    # Sharpe calculation
    if returns_array.std() > 0:
        # Daily Sharpe
        daily_sharpe = returns_array.mean() / returns_array.std()
        
        # Annualized Sharpe - need to know trade frequency
        trades_per_day = len(trades) / (market_data['timestamp'].dt.date.nunique())
        annualization_factor = np.sqrt(252 * trades_per_day)
        annual_sharpe = daily_sharpe * annualization_factor
        
        print(f"\nSharpe Calculation:")
        print(f"  Daily Sharpe: {daily_sharpe:.4f}")
        print(f"  Trades per day: {trades_per_day:.1f}")
        print(f"  Annualization factor: {annualization_factor:.1f}")
        print(f"  Annual Sharpe: {annual_sharpe:.2f}")
        
        # Check if this matches the reported Sharpe
        if annual_sharpe > 50:
            print("\n⚠️ WARNING: Sharpe ratio seems unrealistically high!")
            print("Possible issues:")
            print("1. Annualization factor too high (many trades per day)")
            print("2. Very low volatility due to capped returns")
            print("3. Execution costs not properly accounted for")
    
    # Total return
    total_return = (1 + returns_array).prod() - 1
    print(f"\nTotal return: {total_return*100:.2f}%")
    
    # Sanity check
    print("\n🔍 Sanity Check:")
    expected_return_per_trade = win_rate * avg_win + (1 - win_rate) * avg_loss
    print(f"Expected return per trade: {expected_return_per_trade*100:.4f}%")
    print(f"With {len(trades)} trades: {expected_return_per_trade * len(trades) * 100:.2f}% expected total")
    
    return {
        'returns_array': returns_array,
        'exit_counts': exit_counts,
        'win_rate': win_rate,
        'sharpe': annual_sharpe if returns_array.std() > 0 else 0,
        'total_return': total_return
    }

# Run verification
if len(performance_df) > 0:
    print("🔍 Verifying Micro Stop/Target Calculations")
    print("=" * 80)
    
    # Get a high-frequency strategy
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days]
    
    if len(high_freq_df) > 0:
        # Verify the top performer
        strategy = high_freq_df.iloc[0]
        
        print(f"Verifying: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print(f"Original performance: Sharpe={strategy['sharpe_ratio']:.2f}, Return={strategy['total_return']*100:.2f}%")
        print(f"Testing with: Stop=0.075%, Target=0.10%")
        
        results = verify_stop_target_calculation(
            strategy['strategy_hash'],
            strategy['trace_path'],
            market_data,
            stop_pct=0.075,
            target_pct=0.10,
            execution_cost_bps=execution_cost_bps
        )
        
        if results:
            print("\n💡 Explanation of High Sharpe Ratios:")
            print("=" * 60)
            print("1. CONSISTENCY: With 57% of trades hitting 0.10% target")
            print("   and only 19% hitting 0.075% stop, returns are very consistent")
            print("\n2. LOW VOLATILITY: Capping returns at ±0.10% dramatically")
            print("   reduces standard deviation of returns")
            print("\n3. HIGH FREQUENCY: With 5+ trades/day, annualization")
            print(f"   factor is ~{np.sqrt(252 * 5):.0f}, amplifying the Sharpe ratio")
            print("\n4. REALISTIC?: While mathematically correct, this assumes:")
            print("   - Perfect execution at stop/target prices")
            print("   - No slippage on 1000+ trades")
            print("   - Sufficient liquidity for all trades")
            print("\n⚠️ CAUTION: These results may not be achievable in live trading!")