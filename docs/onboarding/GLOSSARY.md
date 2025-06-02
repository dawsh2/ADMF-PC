# ADMF-PC Glossary

Key terms and concepts used throughout ADMF-PC documentation.

## A

**ADMF-PC**
: Adaptive Decision Making Framework - Protocol Composition. The complete system name.

**Allocation**
: The percentage of capital assigned to a strategy or position.

**ATR (Average True Range)**
: A volatility indicator used for position sizing and stop losses.

## B

**Backtest**
: Historical simulation of a trading strategy using past market data.

**BAR Event**
: Event containing price data (Open, High, Low, Close, Volume) for a specific time period.

**Batch Processing**
: Processing multiple operations together for efficiency, especially in optimization.

## C

**Capability**
: Cross-cutting functionality (logging, monitoring) that can be added to any component.

**Classifier**
: Component that identifies market conditions or regimes without generating trading signals.

**Composition**
: Building complex behavior by combining simple components, rather than inheritance.

**Configuration-Driven**
: System behavior defined entirely through YAML configuration files, not code.

**Container**
: Isolated execution environment for components, ensuring no state leakage.

**Coordinator**
: Central orchestrator that manages workflow execution and phase transitions.

## D

**Data Handler**
: Component responsible for loading and streaming market data.

**Dependency Injection**
: Automatic wiring of component dependencies based on protocols.

**Drawdown**
: Peak-to-trough decline in portfolio value, key risk metric.

## E

**Ensemble**
: Strategy that combines multiple sub-strategies with weighted voting.

**Event**
: Message passed between components (BAR, INDICATOR, SIGNAL, ORDER, FILL).

**Event Bus**
: Message routing system that connects components through events.

**Event-Driven**
: Architecture where components react to events rather than calling each other.

**Execution Engine**
: Component that simulates or executes trades based on orders.

## F

**Factory**
: Component that creates and configures other components.

**FILL Event**
: Confirmation that an order has been executed at a specific price.

**Filter**
: Condition that must be met before a signal is generated.

## G

**Grid Search**
: Optimization method that tests all parameter combinations.

## H

**Heat**
: Total portfolio risk exposure at any moment.

**HMM (Hidden Markov Model)**
: Statistical model used for regime detection.

## I

**Indicator**
: Technical calculation on price data (SMA, RSI, MACD, etc.).

**INDICATOR Event**
: Event containing calculated indicator values.

**Isolation**
: Property where containers cannot access each other's state.

## K

**Kelly Criterion**
: Mathematical formula for optimal position sizing.

## L

**Liquidity**
: Ease of entering/exiting positions without significant price impact.

**Live Trading**
: Real-time trading with actual money (vs backtesting).

## M

**MAE (Maximum Adverse Excursion)**
: Largest loss experienced during a trade before exit.

**Market Impact**
: Price movement caused by executing large orders.

**Market Regime**
: Current market state (trending, ranging, volatile, etc.).

**MFE (Maximum Favorable Excursion)**
: Largest profit experienced during a trade before exit.

**Monte Carlo**
: Statistical simulation using random sampling for validation.

**Multi-Phase Workflow**
: Complex operation broken into sequential phases.

## O

**Optimization**
: Process of finding best parameters for a strategy.

**ORDER Event**
: Request to execute a trade at specific terms.

**Order Router**
: Component that directs orders to appropriate execution venues.

## P

**Paper Trading**
: Simulated trading with fake money for testing.

**Parameter Space**
: Range of values to test during optimization.

**PnL (Profit and Loss)**
: Financial performance of positions or strategies.

**Portfolio**
: Collection of positions across multiple instruments.

**Position**
: Current holding in a specific instrument.

**Position Sizing**
: Determining how much capital to allocate per trade.

**Protocol**
: Interface defining how components interact through events.

**Protocol + Composition**
: Core philosophy of building through protocols, not inheritance.

## R

**Regime**
: Market state or condition affecting strategy behavior.

**Risk Container**
: Component enforcing risk limits and position sizing.

**Risk Parity**
: Allocation method equalizing risk contribution across strategies.

## S

**Sharpe Ratio**
: Risk-adjusted return metric (return per unit of volatility).

**SIGNAL Event**
: Trading recommendation from a strategy (BUY/SELL with strength).

**Signal Replay**
: Optimization technique reusing previously generated signals.

**Slippage**
: Difference between expected and actual execution price.

**Strategy**
: Component that generates trading signals from market data.

**Synthetic Data**
: Artificially generated data for testing edge cases.

## T

**Three-Tier Testing**
: Testing approach with unit, integration, and system tests.

**Tick**
: Single price update from the market.

**Transaction Cost**
: Total cost of executing trades (commissions, spread, impact).

## V

**Validation**
: Testing strategy on unseen data to prevent overfitting.

**VaR (Value at Risk)**
: Statistical risk measure of potential losses.

**Vectorized**
: Operations performed on entire arrays simultaneously for speed.

**Volatility**
: Measure of price variation over time.

## W

**Walk-Forward**
: Validation method using rolling train/test windows.

**Warmup Period**
: Historical data required before strategy can trade.

**Workflow**
: Complete end-to-end process (backtest, optimization, etc.).

## Y

**YAML**
: Human-readable data format used for all ADMF-PC configuration.

## Z

**Zero-Code**
: Philosophy where users configure behavior without writing code.

**Z-Score**
: Statistical measure of deviation from mean, used in pairs trading.

---

## Common Abbreviations

- **API**: Application Programming Interface
- **AUM**: Assets Under Management
- **CSV**: Comma-Separated Values
- **DMA**: Direct Market Access
- **EMA**: Exponential Moving Average
- **FIFO**: First In, First Out
- **HFT**: High-Frequency Trading
- **JSON**: JavaScript Object Notation
- **MACD**: Moving Average Convergence Divergence
- **PM**: Portfolio Manager
- **RSI**: Relative Strength Index
- **SMA**: Simple Moving Average
- **SOR**: Smart Order Router
- **TWAP**: Time-Weighted Average Price
- **VWAP**: Volume-Weighted Average Price

---

## Architecture Terms

**Container Hierarchy**
: Nested structure of containers, each managing specific responsibilities.

**Duck Typing**
: If it walks like a duck and quacks like a duck, it's a duck (protocol compatibility).

**Event Sourcing**
: Storing all events for replay and audit capabilities.

**Idempotent**
: Operation that produces same result regardless of how many times it's executed.

**Immutable**
: Data that cannot be changed after creation, ensuring consistency.

**Lazy Evaluation**
: Computing values only when needed, not in advance.

**Stateless**
: Components that don't maintain internal state between calls.

---

*Can't find a term? Check the [FAQ](FAQ.md) or [Architecture Documentation](../SYSTEM_ARCHITECTURE_V5.MD).*

[← Back to Hub](README.md)