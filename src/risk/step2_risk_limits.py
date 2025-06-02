"""
File: src/risk/step2_risk_limits.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#risk-limits
Step: 2 - Add Risk Container
Dependencies: core.logging, risk.models

Risk limits implementation for Step 2.
Simplified risk enforcement following complexity guide specifications.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal

from ..core.logging.structured import ContainerLogger
from .models import RiskConfig, TradingSignal, Order


class RiskLimits:
    """
    Enforces various risk constraints for Step 2.
    
    Implements the core risk checks required by the complexity guide:
    - Position size limits
    - Portfolio risk limits
    - Drawdown limits
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Protocol-based risk checking without inheritance
        - Provides: Simple, fast risk validation for trading signals
        - Dependencies: Structured logging for audit trail
    
    Example:
        config = RiskConfig(max_position_size=0.1, max_drawdown=0.2)
        limits = RiskLimits(config)
        can_trade = limits.can_trade(portfolio_state, signal)
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk limits with configuration.
        
        Args:
            config: Risk configuration parameters
        """
        self.config = config
        
        # Risk parameters
        self.max_position_size = config.max_position_size
        self.max_portfolio_risk = config.max_portfolio_risk
        self.max_correlation = config.max_correlation
        self.max_drawdown = config.max_drawdown
        self.max_leverage = config.max_leverage
        self.max_concentration = config.max_concentration
        
        # Violation tracking
        self.violations_history: List[Dict[str, Any]] = []
        self.total_violations = 0
        
        # Logging
        self.logger = ContainerLogger("RiskLimits", "risk_limits", "risk_limits")
        
        self.logger.info(
            "RiskLimits initialized",
            max_position_size=self.max_position_size,
            max_portfolio_risk=self.max_portfolio_risk,
            max_drawdown=self.max_drawdown
        )
    
    def can_trade(self, portfolio_state, signal: TradingSignal) -> bool:
        """
        Check if trade is allowed under current risk limits.
        
        This method performs all risk checks required by Step 2:
        1. Position size limit
        2. Portfolio risk limit
        3. Drawdown limit
        4. Concentration limit
        
        Args:
            portfolio_state: Current portfolio state
            signal: Trading signal to validate
            
        Returns:
            True if trade is allowed, False if any limit is violated
        """
        self.logger.trace(
            "Checking risk limits",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side.value,
            strength=float(signal.strength)
        )
        
        # Check position limit
        if not self._check_position_limit(portfolio_state, signal):
            self._record_violation("position_limit", signal, "Position size limit exceeded")
            return False
        
        # Check portfolio risk
        if not self._check_portfolio_risk(portfolio_state, signal):
            self._record_violation("portfolio_risk", signal, "Portfolio risk limit exceeded")
            return False
        
        # Check drawdown limit
        if not self._check_drawdown_limit(portfolio_state):
            self._record_violation("drawdown_limit", signal, "Drawdown limit exceeded")
            return False
        
        # Check concentration limit
        if not self._check_concentration_limit(portfolio_state, signal):
            self._record_violation("concentration_limit", signal, "Concentration limit exceeded")
            return False
        
        self.logger.debug(
            "All risk checks passed",
            signal_id=signal.signal_id,
            symbol=signal.symbol
        )
        
        return True
    
    def _check_position_limit(self, portfolio_state, signal: TradingSignal) -> bool:
        """
        Ensure position size is within limits.
        
        Args:
            portfolio_state: Current portfolio state
            signal: Trading signal to check
            
        Returns:
            True if position limit check passes
        """
        current_position = portfolio_state.get_position(signal.symbol)
        current_prices = portfolio_state.get_current_prices()
        
        if not current_prices.get(signal.symbol):
            self.logger.warning(
                "No current price for position limit check",
                symbol=signal.symbol
            )
            return False
        
        current_price = Decimal(str(current_prices[signal.symbol]))
        
        # Calculate current position value
        if current_position and current_position.quantity != 0:
            position_value = abs(current_position.quantity * current_price)
            position_pct = float(position_value / portfolio_state.total_value) if portfolio_state.total_value > 0 else 0.0
            
            if position_pct > self.max_position_size:
                self.logger.warning(
                    "Position size limit exceeded",
                    symbol=signal.symbol,
                    current_position_pct=position_pct,
                    limit=self.max_position_size
                )
                return False
        
        return True
    
    def _check_portfolio_risk(self, portfolio_state, signal: TradingSignal) -> bool:
        """
        Check portfolio risk limit.
        
        Args:
            portfolio_state: Current portfolio state
            signal: Trading signal to check
            
        Returns:
            True if portfolio risk check passes
        """
        # For Step 2, implement a simple check based on signal strength
        # More sophisticated risk models can be added in later steps
        
        if signal.strength > Decimal(str(self.max_portfolio_risk * 10)):  # Scale up for comparison
            self.logger.warning(
                "Portfolio risk limit exceeded",
                signal_strength=float(signal.strength),
                risk_limit=self.max_portfolio_risk
            )
            return False
        
        return True
    
    def _check_drawdown_limit(self, portfolio_state) -> bool:
        """
        Check if current drawdown exceeds limit.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            True if drawdown check passes
        """
        if hasattr(portfolio_state, 'current_drawdown'):
            current_drawdown = portfolio_state.current_drawdown
            
            if current_drawdown > Decimal(str(self.max_drawdown)):
                self.logger.warning(
                    "Drawdown limit exceeded",
                    current_drawdown=float(current_drawdown),
                    limit=self.max_drawdown
                )
                return False
        
        return True
    
    def _check_concentration_limit(self, portfolio_state, signal: TradingSignal) -> bool:
        """
        Check concentration limits.
        
        Args:
            portfolio_state: Current portfolio state
            signal: Trading signal to check
            
        Returns:
            True if concentration check passes
        """
        # Get portfolio exposure
        exposure = portfolio_state.get_exposure()
        total_value = portfolio_state.total_value
        
        if total_value == 0:
            return True  # No concentration risk with no portfolio value
        
        # Check if gross exposure would exceed concentration limit
        gross_exposure_pct = exposure['gross_exposure'] / float(total_value)
        
        if gross_exposure_pct > self.max_concentration:
            self.logger.warning(
                "Concentration limit exceeded",
                gross_exposure_pct=gross_exposure_pct,
                limit=self.max_concentration
            )
            return False
        
        return True
    
    def _record_violation(self, violation_type: str, signal: TradingSignal, reason: str) -> None:
        """
        Record a risk limit violation.
        
        Args:
            violation_type: Type of violation
            signal: Signal that caused violation
            reason: Reason for violation
        """
        violation = {
            'timestamp': datetime.now(),
            'violation_type': violation_type,
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'side': signal.side.value,
            'strength': float(signal.strength),
            'reason': reason
        }
        
        self.violations_history.append(violation)
        self.total_violations += 1
        
        # Keep only recent violations
        if len(self.violations_history) > 100:
            self.violations_history = self.violations_history[-100:]
        
        self.logger.warning(
            "Risk limit violation recorded",
            violation_type=violation_type,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            reason=reason
        )
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """
        Get violation statistics.
        
        Returns:
            Dictionary containing violation statistics
        """
        # Count violations by type
        violation_counts = {}
        for violation in self.violations_history:
            v_type = violation['violation_type']
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        return {
            'total_violations': self.total_violations,
            'recent_violations': len(self.violations_history),
            'violations_by_type': violation_counts,
            'last_violation': self.violations_history[-1] if self.violations_history else None
        }
    
    def update_config(self, new_config: RiskConfig) -> None:
        """
        Update risk configuration.
        
        Args:
            new_config: New risk configuration
        """
        old_config = {
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_drawdown': self.max_drawdown
        }
        
        self.config = new_config
        self.max_position_size = new_config.max_position_size
        self.max_portfolio_risk = new_config.max_portfolio_risk
        self.max_correlation = new_config.max_correlation
        self.max_drawdown = new_config.max_drawdown
        self.max_leverage = new_config.max_leverage
        self.max_concentration = new_config.max_concentration
        
        self.logger.info(
            "Risk limits configuration updated",
            old_config=old_config,
            new_config={
                'max_position_size': self.max_position_size,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_drawdown': self.max_drawdown
            }
        )
    
    def reset_violations(self) -> None:
        """Reset violation history."""
        violations_cleared = len(self.violations_history)
        self.violations_history.clear()
        
        self.logger.info(
            "Risk violations reset",
            violations_cleared=violations_cleared,
            total_violations_retained=self.total_violations
        )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current risk limits state.
        
        Returns:
            Dictionary containing risk limits state
        """
        return {
            'config': self.config.to_dict(),
            'total_violations': self.total_violations,
            'recent_violations_count': len(self.violations_history),
            'violation_stats': self.get_violation_stats()
        }