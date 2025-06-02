"""
File: src/core/types.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#shared-types
Dependencies: enum, decimal, datetime

Shared types used across multiple modules.
Breaks circular dependencies by providing common type definitions.
"""

from enum import Enum
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = 1
    SELL = -1


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class SignalType(Enum):
    """Signal type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    RISK_EXIT = "risk_exit"


class FillType(Enum):
    """Fill type enumeration."""
    PARTIAL = "partial"
    FULL = "full"


class FillStatus(Enum):
    """Fill status enumeration."""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"