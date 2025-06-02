"""
File: src/risk/step2_position_sizer.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#position-sizing
Step: 2 - Add Risk Container
Dependencies: core.logging, risk.models

Position sizing implementation for Step 2.
Calculates appropriate position sizes based on risk parameters.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from decimal import Decimal
import math

from ..core.logging.structured import ContainerLogger
from .models import RiskConfig, TradingSignal


class PositionSizer:
    """
    Calculates position sizes based on risk parameters.
    
    Implements multiple sizing methods:
    - Fixed dollar amount
    - Percentage risk per trade
    - Volatility-based sizing
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Protocol-based position sizing without inheritance
        - Provides: Risk-adjusted trade sizing for signals
        - Dependencies: Structured logging for audit trail
    
    Example:
        config = RiskConfig(sizing_method='percent_risk', percent_risk_per_trade=0.01)
        sizer = PositionSizer('percent_risk', config)
        size = sizer.calculate_size(signal, portfolio_state)
    """
    
    def __init__(self, sizing_method: str, config: RiskConfig):
        """
        Initialize position sizer.
        
        Args:
            sizing_method: Method to use for sizing ('fixed', 'percent_risk', 'volatility')
            config: Risk configuration parameters
        """
        self.sizing_method = sizing_method
        self.config = config
        
        # Sizing parameters
        self.fixed_position_size = config.fixed_position_size
        self.percent_risk_per_trade = config.percent_risk_per_trade
        self.volatility_lookback = config.volatility_lookback
        
        # Logging
        self.logger = ContainerLogger("PositionSizer", "position_sizer", "position_sizer")
        
        # Validate sizing method
        valid_methods = ['fixed', 'percent_risk', 'volatility']
        if sizing_method not in valid_methods:
            raise ValueError(f"Invalid sizing method: {sizing_method}. Must be one of {valid_methods}")
        
        self.logger.info(
            "PositionSizer initialized",
            sizing_method=sizing_method,
            fixed_size=self.fixed_position_size,
            percent_risk=self.percent_risk_per_trade
        )
    
    def calculate_size(self, signal: TradingSignal, portfolio_state) -> Decimal:
        """
        Calculate position size for a trading signal.
        
        Args:
            signal: Trading signal to size
            portfolio_state: Current portfolio state
            
        Returns:
            Position size in shares/units
        """
        self.logger.trace(
            "Calculating position size",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            method=self.sizing_method,
            strength=float(signal.strength)
        )
        
        # Get current prices
        current_prices = portfolio_state.get_current_prices()
        current_price = current_prices.get(signal.symbol)
        
        if not current_price:
            self.logger.warning(
                "No current price available for sizing",
                symbol=signal.symbol
            )
            return Decimal('0')
        
        current_price = Decimal(str(current_price))
        
        # Calculate base size using selected method
        if self.sizing_method == 'fixed':
            size = self._calculate_fixed_size(signal, current_price, portfolio_state)
        elif self.sizing_method == 'percent_risk':
            size = self._calculate_percent_risk_size(signal, current_price, portfolio_state)
        elif self.sizing_method == 'volatility':
            size = self._calculate_volatility_size(signal, current_price, portfolio_state)
        else:
            self.logger.error(f"Unknown sizing method: {self.sizing_method}")
            return Decimal('0')
        
        # Apply signal strength scaling
        size = self._apply_signal_strength(size, signal.strength)
        
        # Apply minimum/maximum size constraints
        size = self._apply_size_constraints(size, current_price, portfolio_state)
        
        self.logger.debug(
            "Position size calculated",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            method=self.sizing_method,
            raw_size=float(size),
            current_price=float(current_price)
        )
        
        return size
    
    def _calculate_fixed_size(
        self, 
        signal: TradingSignal, 
        current_price: Decimal, 
        portfolio_state
    ) -> Decimal:
        """
        Calculate fixed dollar amount position size.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_state: Portfolio state
            
        Returns:
            Position size in shares
        """
        fixed_amount = Decimal(str(self.fixed_position_size))
        size = fixed_amount / current_price
        
        self.logger.trace(
            "Fixed size calculation",
            fixed_amount=float(fixed_amount),
            current_price=float(current_price),
            calculated_size=float(size)
        )
        
        return size
    
    def _calculate_percent_risk_size(
        self, 
        signal: TradingSignal, 
        current_price: Decimal, 
        portfolio_state
    ) -> Decimal:
        """
        Calculate position size based on percentage risk per trade.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_state: Portfolio state
            
        Returns:
            Position size in shares
        """
        portfolio_value = portfolio_state.total_value
        if portfolio_value <= 0:
            self.logger.warning("Cannot calculate percent risk with zero portfolio value")
            return Decimal('0')
        
        # Calculate risk amount
        risk_amount = portfolio_value * Decimal(str(self.percent_risk_per_trade))
        
        # Estimate stop distance (simplified - assume 5% stop loss)
        stop_distance_pct = Decimal(str(self.config.default_stop_loss_pct))
        stop_distance = current_price * stop_distance_pct
        
        if stop_distance <= 0:
            self.logger.warning("Invalid stop distance for percent risk sizing")
            return Decimal('0')
        
        # Calculate size: risk_amount / stop_distance
        size = risk_amount / stop_distance
        
        self.logger.trace(
            "Percent risk size calculation",
            portfolio_value=float(portfolio_value),
            risk_amount=float(risk_amount),
            stop_distance=float(stop_distance),
            calculated_size=float(size)
        )
        
        return size
    
    def _calculate_volatility_size(
        self, 
        signal: TradingSignal, 
        current_price: Decimal, 
        portfolio_state
    ) -> Decimal:
        """
        Calculate position size based on volatility.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_state: Portfolio state
            
        Returns:
            Position size in shares
        """
        # For Step 2, implement a simplified volatility calculation
        # More sophisticated volatility models can be added in later steps
        
        portfolio_value = portfolio_state.total_value
        if portfolio_value <= 0:
            return Decimal('0')
        
        # Assume average volatility for now (would need price history for real calculation)
        estimated_volatility = Decimal('0.02')  # 2% daily volatility
        
        # Calculate volatility-adjusted risk
        base_risk = portfolio_value * Decimal(str(self.percent_risk_per_trade))
        volatility_adjustment = Decimal('1') / (estimated_volatility * Decimal('10'))  # Scale factor
        
        # Calculate position value
        position_value = base_risk * volatility_adjustment
        size = position_value / current_price
        
        self.logger.trace(
            "Volatility size calculation",
            portfolio_value=float(portfolio_value),
            estimated_volatility=float(estimated_volatility),
            position_value=float(position_value),
            calculated_size=float(size)
        )
        
        return size
    
    def _apply_signal_strength(self, base_size: Decimal, strength: Decimal) -> Decimal:
        """
        Apply signal strength scaling to position size.
        
        Args:
            base_size: Base position size
            strength: Signal strength (0-1)
            
        Returns:
            Strength-adjusted position size
        """
        # Scale position size by signal strength
        adjusted_size = base_size * strength
        
        self.logger.trace(
            "Signal strength adjustment",
            base_size=float(base_size),
            strength=float(strength),
            adjusted_size=float(adjusted_size)
        )
        
        return adjusted_size
    
    def _apply_size_constraints(
        self, 
        size: Decimal, 
        current_price: Decimal, 
        portfolio_state
    ) -> Decimal:
        """
        Apply minimum and maximum size constraints.
        
        Args:
            size: Calculated position size
            current_price: Current market price
            portfolio_state: Portfolio state
            
        Returns:
            Constrained position size
        """
        original_size = size
        
        # Minimum size constraint (avoid tiny positions)
        min_position_value = Decimal('100')  # $100 minimum
        min_size = min_position_value / current_price
        
        if size < min_size and size > 0:
            self.logger.trace(
                "Applying minimum size constraint",
                original_size=float(size),
                min_size=float(min_size)
            )
            size = min_size
        
        # Maximum size constraint (based on portfolio value)
        portfolio_value = portfolio_state.total_value
        if portfolio_value > 0:
            max_position_value = portfolio_value * Decimal(str(self.config.max_position_size))
            max_size = max_position_value / current_price
            
            if size > max_size:
                self.logger.trace(
                    "Applying maximum size constraint",
                    original_size=float(size),
                    max_size=float(max_size)
                )
                size = max_size
        
        # Cash constraint (can't buy more than available cash for long positions)
        cash_available = portfolio_state.cash
        max_affordable_size = cash_available / current_price
        
        if size > max_affordable_size:
            self.logger.trace(
                "Applying cash constraint",
                original_size=float(size),
                max_affordable=float(max_affordable_size),
                cash_available=float(cash_available)
            )
            size = max_affordable_size
        
        # Ensure size is positive and reasonable
        if size < 0:
            size = Decimal('0')
        
        # Round to reasonable precision (allow fractional shares for better execution)
        size = size.quantize(Decimal('0.01'))  # Allow 2 decimal places for fractional shares
        
        if size != original_size:
            self.logger.debug(
                "Size constraints applied",
                original_size=float(original_size),
                final_size=float(size)
            )
        
        return size
    
    def update_config(self, new_config: RiskConfig) -> None:
        """
        Update position sizing configuration.
        
        Args:
            new_config: New risk configuration
        """
        old_config = {
            'fixed_position_size': self.fixed_position_size,
            'percent_risk_per_trade': self.percent_risk_per_trade,
            'volatility_lookback': self.volatility_lookback
        }
        
        self.config = new_config
        self.fixed_position_size = new_config.fixed_position_size
        self.percent_risk_per_trade = new_config.percent_risk_per_trade
        self.volatility_lookback = new_config.volatility_lookback
        
        self.logger.info(
            "Position sizer configuration updated",
            old_config=old_config,
            new_config={
                'fixed_position_size': self.fixed_position_size,
                'percent_risk_per_trade': self.percent_risk_per_trade,
                'volatility_lookback': self.volatility_lookback
            }
        )
    
    def get_sizing_info(self, signal: TradingSignal, portfolio_state) -> Dict[str, Any]:
        """
        Get detailed sizing information for analysis.
        
        Args:
            signal: Trading signal
            portfolio_state: Portfolio state
            
        Returns:
            Dictionary containing sizing analysis
        """
        current_prices = portfolio_state.get_current_prices()
        current_price = current_prices.get(signal.symbol, 0)
        
        info = {
            'sizing_method': self.sizing_method,
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'signal_strength': float(signal.strength),
            'current_price': current_price,
            'portfolio_value': float(portfolio_state.total_value),
            'cash_available': float(portfolio_state.cash)
        }
        
        if current_price > 0:
            calculated_size = self.calculate_size(signal, portfolio_state)
            position_value = float(calculated_size) * current_price
            
            info.update({
                'calculated_size': float(calculated_size),
                'position_value': position_value,
                'position_pct_of_portfolio': position_value / float(portfolio_state.total_value) if portfolio_state.total_value > 0 else 0
            })
        
        return info
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current position sizer state.
        
        Returns:
            Dictionary containing position sizer state
        """
        return {
            'sizing_method': self.sizing_method,
            'config': {
                'fixed_position_size': self.fixed_position_size,
                'percent_risk_per_trade': self.percent_risk_per_trade,
                'volatility_lookback': self.volatility_lookback
            }
        }