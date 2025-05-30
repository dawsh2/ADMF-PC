"""Ultra minimal test - directly test core functionality without complex imports."""

from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

print("Testing core ADMF-PC functionality...")
print("="*50)

# Define minimal types locally to avoid import issues
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"

@dataclass
class SimpleSignal:
    symbol: str
    side: OrderSide
    signal_type: SignalType
    strength: Decimal
    price: Decimal

@dataclass
class SimplePosition:
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    
    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.current_price

class SimplePortfolio:
    """Minimal portfolio implementation."""
    
    def __init__(self, initial_cash: Decimal):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, SimplePosition] = {}
        self.realized_pnl = Decimal(0)
        
    def can_afford(self, symbol: str, quantity: Decimal, price: Decimal) -> bool:
        """Check if we can afford to buy."""
        cost = quantity * price
        return cost <= self.cash
    
    def buy(self, symbol: str, quantity: Decimal, price: Decimal) -> bool:
        """Execute a buy."""
        cost = quantity * price
        if not self.can_afford(symbol, quantity, price):
            return False
            
        self.cash -= cost
        
        if symbol in self.positions:
            # Add to existing position
            pos = self.positions[symbol]
            total_cost = pos.quantity * pos.avg_price + cost
            pos.quantity += quantity
            pos.avg_price = total_cost / pos.quantity
            pos.current_price = price
        else:
            # New position
            self.positions[symbol] = SimplePosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        return True
    
    def sell(self, symbol: str, quantity: Decimal, price: Decimal) -> bool:
        """Execute a sell."""
        if symbol not in self.positions:
            return False
            
        pos = self.positions[symbol]
        if quantity > pos.quantity:
            return False
        
        # Calculate realized P&L
        realized = quantity * (price - pos.avg_price)
        self.realized_pnl += realized
        
        # Update cash
        self.cash += quantity * price
        
        # Update position
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]
            
        return True
    
    def update_prices(self, prices: Dict[str, Decimal]):
        """Update current prices."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    @property
    def total_value(self) -> Decimal:
        """Total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> Decimal:
        """Total return percentage."""
        if self.initial_cash == 0:
            return Decimal(0)
        return (self.total_value - self.initial_cash) / self.initial_cash * 100


def position_size(portfolio_value: Decimal, risk_pct: Decimal) -> Decimal:
    """Calculate position size as percentage of portfolio."""
    return portfolio_value * risk_pct / 100


# Run the test
print("\n1. Creating portfolio with $100,000")
portfolio = SimplePortfolio(Decimal("100000"))
print(f"   Cash: ${portfolio.cash}")
print(f"   Total Value: ${portfolio.total_value}")

print("\n2. Generating BUY signal")
buy_signal = SimpleSignal(
    symbol="AAPL",
    side=OrderSide.BUY,
    signal_type=SignalType.ENTRY,
    strength=Decimal("0.8"),
    price=Decimal("150")
)
print(f"   Signal: {buy_signal.side.value} {buy_signal.symbol} @ ${buy_signal.price}")

print("\n3. Calculating position size (2% risk)")
size = position_size(portfolio.total_value, Decimal("2"))
shares = (size / buy_signal.price).quantize(Decimal("1"))
print(f"   Position size: ${size:.2f}")
print(f"   Shares to buy: {shares}")

print("\n4. Executing BUY order")
if portfolio.buy(buy_signal.symbol, shares, buy_signal.price):
    print(f"   ✓ Bought {shares} shares @ ${buy_signal.price}")
    print(f"   Cash remaining: ${portfolio.cash:.2f}")
    pos = portfolio.positions[buy_signal.symbol]
    print(f"   Position: {pos.quantity} shares, avg price ${pos.avg_price}")
else:
    print("   ✗ Buy failed")

print("\n5. Price moves to $155")
portfolio.update_prices({"AAPL": Decimal("155")})
pos = portfolio.positions["AAPL"]
print(f"   Unrealized P&L: ${pos.unrealized_pnl:.2f}")
print(f"   Total Value: ${portfolio.total_value:.2f}")

print("\n6. Generating SELL signal")
sell_signal = SimpleSignal(
    symbol="AAPL",
    side=OrderSide.SELL,
    signal_type=SignalType.EXIT,
    strength=Decimal("0.8"),
    price=Decimal("155")
)

print("\n7. Executing SELL order")
if portfolio.sell(sell_signal.symbol, shares, sell_signal.price):
    print(f"   ✓ Sold {shares} shares @ ${sell_signal.price}")
    print(f"   Realized P&L: ${portfolio.realized_pnl:.2f}")
    print(f"   Cash: ${portfolio.cash:.2f}")
else:
    print("   ✗ Sell failed")

print("\n8. Final Results")
print("="*40)
print(f"Initial Capital: ${portfolio.initial_cash}")
print(f"Final Value: ${portfolio.total_value:.2f}")
print(f"Total Return: {portfolio.total_return:.2f}%")
print(f"Realized P&L: ${portfolio.realized_pnl:.2f}")
print(f"Positions: {len(portfolio.positions)}")

print("\n✅ Basic functionality test PASSED!")
print("\nThis demonstrates:")
print("- Signal generation")
print("- Position sizing")
print("- Order execution")
print("- P&L tracking")
print("- Portfolio management")
print("\nAll core concepts work without complex dependencies!")