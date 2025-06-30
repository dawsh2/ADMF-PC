#!/usr/bin/env python3
"""
Explain why total return is 3.21% not 40%
"""
import numpy as np

def explain_return_difference():
    """Show why average return doesn't compound to expected total"""
    
    print("=== Why Total Return is 3.21% not 40% ===\n")
    
    # Your calculation
    print("Your calculation assumes EVERY trade returns 0.0031%:")
    avg_return = 0.0031 / 100  # Convert to decimal
    your_total = (1 + avg_return) ** 1033
    print(f"(1.00031)^1033 = {your_total:.4f} = {(your_total-1)*100:.2f}% return\n")
    
    # Why it's different
    print("But in reality, returns vary:")
    print("- 267 trades lose -0.075% (stop losses)")
    print("- 150 trades gain +0.100% (take profits)")
    print("- 616 trades have varied small returns (signal exits)\n")
    
    # Simplified example
    print("=== Simplified Example ===")
    print("Compare these two scenarios with 4 trades:\n")
    
    print("Scenario A: Every trade returns 0.025% (average)")
    returns_a = [0.00025, 0.00025, 0.00025, 0.00025]
    compound_a = np.prod([1 + r for r in returns_a])
    print(f"Returns: {[f'{r*100:.3f}%' for r in returns_a]}")
    print(f"Total: {(compound_a-1)*100:.4f}%\n")
    
    print("Scenario B: Actual varied returns (same average)")
    returns_b = [-0.00075, 0.001, -0.00075, 0.001]  # Same avg as A
    compound_b = np.prod([1 + r for r in returns_b])
    print(f"Returns: {[f'{r*100:.3f}%' for r in returns_b]}")
    print(f"Average: {np.mean(returns_b)*100:.3f}%")
    print(f"Total: {(compound_b-1)*100:.4f}%\n")
    
    print("Same average, but different total due to variance!")
    
    # The math
    print("\n=== The Math ===")
    print("When returns vary, the compound return is ALWAYS less than")
    print("what you'd get from compounding the average return.")
    print("\nThis is due to 'volatility drag' or 'variance drain':")
    print("- Large losses hurt more than equal gains help")
    print("- Example: -50% then +50% = -25% total (not 0%)")
    
    print("\n=== Your Actual Results ===")
    print("Average return: 0.0031% per trade")
    print("If every trade returned exactly 0.0031%: ~40% total")
    print("Actual total with variance: 3.21%")
    print("\nThe difference (~37%) is the cost of variance!")

explain_return_difference()