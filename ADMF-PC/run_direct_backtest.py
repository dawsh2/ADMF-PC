"""Run backtest directly on synthetic data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.execution.simple_backtest_engine import run_simple_backtest


def main():
    """Run direct backtest test."""
    print("=" * 70)
    print("DIRECT BACKTEST ON SYNTHETIC DATA")
    print("=" * 70)
    
    # Test with increasing number of bars
    for num_bars in [100, 500, 1000]:
        print(f"\n\nTest with {num_bars} bars:")
        print("-" * 40)
        
        try:
            result = run_simple_backtest(
                config_path="configs/simple_synthetic_backtest.yaml",
                max_bars=num_bars
            )
            
            # Additional analysis
            if result.num_trades > 0:
                avg_trade_return = result.total_return / result.num_trades
                print(f"\nAdditional Metrics:")
                print(f"Avg Return per Trade: {avg_trade_return:.2%}")
                print(f"Profit Factor: {abs(result.avg_win/result.avg_loss) if result.avg_loss != 0 else 'N/A'}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()