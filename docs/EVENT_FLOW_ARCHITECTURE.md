# ADMF-PC Event Flow Architecture

## Overview

This document visualizes the complete event flow in ADMF-PC's stateless service architecture, showing:
- **[Containerized Components]** - Stateful components that maintain state
- **--EVENTS--** - Events passed between components
- **(Service Pools)** - Stateless service pools that process data without maintaining state

## Complete Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          ADMF-PC Event Flow Architecture                            │
│                                                                                     │
│  [Data Container]                                                                   │
│        │                                                                            │
│        │ --BAR--                                                                    │
│        ▼                                                                            │
│  [FeatureHub Container] ◄─── ONE-TO-MANY BROADCAST ───►                           │
│        │                                                                            │
│        │ --FEATURES, BAR-- ┌──────────────┬──────────────┬──────────────┐          │
│        └──────────────────► │              │              │              │          │
│                             ▼              ▼              ▼              ▼          │
│  ┌─────────────────────────────────────┬───────────────────────────────────────┐   │
│  │   Strategy Service Pool             │   Classifier Service Pool              │   │
│  │  (strategy_1) (strategy_2) ...      │  (classifier_1) (classifier_2) ...    │   │
│  │   (strategy_n)                      │   (classifier_m)                      │   │
│  │                                     │                                        │   │
│  │  Pure Functions - No State          │  Pure Functions - No State            │   │
│  └─────────────────────────────────────┴───────────────────────────────────────┘   │
│               │        │        │                    │           │                  │
│               ▼        ▼        ▼                    ▼           ▼                  │
│        ┌─────────────────────────────────────┐    ┌─────────────────────────────────┐   │
│        │ --SIGNALS-- (broadcast to       │    │ --CLASSIFICATIONS-- (broadcast  │   │
│        │  relevant portfolios)           │    │  to relevant portfolios)        │   │
│        └─────────────────────────────────────┘    └─────────────────────────────────┘   │
│                             │                                    │                  │
│                             └────────────┬───────────────────────┘                  │
│                                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Portfolio Containers                                   │  │
│  │    [Portfolio_1]    [Portfolio_2]    ...    [Portfolio_n]                   │  │
│  │         ▲                ▲                        ▲                         │  │
│  │         │                │                        │                         │  │
│  │    Each portfolio receives specific signals/classifications                  │  │
│  │    based on its strategy/classifier combination                              │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                              │
│                                      │ --no-event-- (internal processing)          │
│                                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                          Risk Service Pool                                    │  │
│  │    (risk_1, risk_2, ..., risk_k)                                            │  │
│  │                                                                              │  │
│  │    Pure Functions - Validate orders based on portfolio state                │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                              │
│                                      │ --ORDERS-- (approved orders only)           │
│                                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                      [Execution Container]                                    │  │
│  │                                                                              │  │
│  │    Shared stateful container managing order lifecycle                        │  │
│  │    Uses stateless execution simulators for fill calculation                  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                              │
│                                      │ --FILLS-- ◄─── ONE-TO-MANY BROADCAST ───► │
│                                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        Portfolio Containers                                   │  │
│  │    [Portfolio_1]    [Portfolio_2]    ...    [Portfolio_n]                   │  │
│  │         ▲                ▲                        ▲                         │  │
│  │         │                │                        │                         │  │
│  │    Each portfolio updates its state based on fills                          │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow Breakdown

### 1. Market Data Flow
```
[Data Container] --BAR--> [FeatureHub Container]
```
- **[Data Container]**: Stateful container that maintains streaming position and data cache
- **--BAR--**: Market bar events (OHLCV data)
- **[FeatureHub Container]**: Stateful container that calculates and caches indicators

### 2. Feature Broadcasting
```
[FeatureHub] --FEATURES, BAR--> (strategy_1, ..., strategy_n) | (classifier_1, ..., classifier_m)
```
- **[FeatureHub]**: Broadcasts calculated features and latest bar
- **--FEATURES, BAR--**: Feature vectors and market data
- **(Strategy Pool)**: Stateless services that generate trading signals
- **(Classifier Pool)**: Stateless services that classify market regimes

### 3. Signal and Classification Routing
```
(Strategies) | (Classifiers) --SIGNALS, CLASSIFICATIONS--> [Portfolio_1, ..., Portfolio_n]
```
- **(Strategies)**: Generate signals based on features
- **(Classifiers)**: Generate regime classifications
- **--SIGNALS, CLASSIFICATIONS--**: Trading signals and market regime states
- **[Portfolios]**: Stateful containers maintaining positions and P&L

### 4. Risk Validation (Internal)
```
[Portfolios] --no-event--> (risk_1, ..., risk_k)
```
- **[Portfolios]**: Internally call risk services (not event-driven)
- **--no-event--**: Direct function calls, not event broadcasts
- **(Risk Pool)**: Stateless services that validate orders

### 5. Order Execution
```
(Risk Services) --ORDERS--> [Execution Container]
```
- **(Risk Services)**: Approve or reject orders
- **--ORDERS--**: Only approved orders are sent
- **[Execution Container]**: Stateful container managing order lifecycle

### 6. Fill Distribution
```
[Execution] --FILLS--> [Portfolio_1, ..., Portfolio_n]
```
- **[Execution]**: Processes orders and generates fills
- **--FILLS--**: Execution confirmations with price and quantity
- **[Portfolios]**: Update positions and cash based on fills

## Key Architecture Benefits

### Stateful Containers (Maintain State)
- **[Data]**: Streaming position, timeline coordination
- **[FeatureHub]**: Indicator cache for performance
- **[Portfolio_1...n]**: Position tracking, cash, P&L
- **[Execution]**: Order lifecycle management

### Stateless Services (Pure Functions)
- **(strategy_1...n)**: Signal generation logic
- **(classifier_1...m)**: Regime detection logic
- **(risk_1...k)**: Order validation logic

### Event-Driven Benefits
1. **Clear Boundaries**: Each component has well-defined inputs/outputs
2. **Parallel Processing**: Stateless services process in parallel
3. **Fault Isolation**: Failures in one service don't affect others
4. **Perfect Tracing**: Every event can be tracked with correlation IDs
5. **Resource Efficiency**: 60% fewer containers needed

## Example Configuration

For a configuration with:
- 6 strategy parameter combinations
- 4 classifier parameter combinations
- 24 total portfolio combinations (6 × 4)

The system creates:
- **4 Stateful Containers**: [Data], [FeatureHub], [Execution], + shared infrastructure
- **24 Portfolio Containers**: One per parameter combination
- **10 Stateless Services**: 6 strategies + 4 classifiers
- **Total**: 28 containers + 10 lightweight services (vs 75+ containers in pure container architecture)

## Implementation Notes

### Event Routing
- Strategy services route signals to portfolios based on combination ID
- Classifier services broadcast classifications to relevant portfolios
- Execution container routes fills back to originating portfolio

### Parallelization
- All stateless services execute in parallel
- Portfolio containers process independently
- Risk validation happens in parallel across portfolios

### Analytics Integration
- Every event includes correlation ID
- Service execution times are tracked
- Parameter combinations are linked to performance metrics