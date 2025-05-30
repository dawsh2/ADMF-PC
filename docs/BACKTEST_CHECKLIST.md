# BACKTEST_CHECKLIST.md - Implementation Handover

## Overview
This checklist documents what needs to be completed to finish the backtest implementation. The architectural foundation is complete - we need to bridge existing components with the new composable container system.

## Current Status ✅

### **COMPLETED - Architecture & Coordinator**
- ✅ **Canonical Coordinator** (`src/core/coordinator/coordinator.py`) with clean imports
- ✅ **Composable Container System** (`src/core/containers/composable.py`) - protocol-based, zero inheritance
- ✅ **Container Composition Engine** (`src/core/containers/composition_engine.py`) - pattern-based container creation
- ✅ **Composable Workflow Manager** (`src/core/coordinator/composable_workflow_manager.py`) - bridges coordinator with containers
- ✅ **Three Container Patterns**: `full_backtest`, `signal_generation`, `signal_replay`
- ✅ **Multi-Asset & Multi-Timeframe Support** - verified and preserved
- ✅ **Coordinator directory cleanup** - removed examples, tests, duplicates

### **COMPLETED - Documentation**
- ✅ **BACKTEST_README.md** - Complete architecture with Classifier/Risk/Portfolio/Strategy hierarchy
- ✅ **COORDINATOR_MIGRATION.md** - Migration guide for users
- ✅ **Container patterns documented** in composition engine

## REMAINING IMPLEMENTATION TASKS 🚧

### **HIGH PRIORITY**

#### **1. Fix Broken Imports in Container Implementations** 
**File**: `src/execution/containers.py`
**Issue**: References non-existent modules
```python
# BROKEN IMPORTS (lines 45-49):
from ..data.loaders import HistoricalDataLoader  # ❌ Doesn't exist
from ..data.loaders import LiveDataLoader        # ❌ Doesn't exist

# BROKEN IMPORTS (lines ~162):
from ..strategy.components.indicator_hub import IndicatorHub  # ❌ Doesn't exist
```

**TASK**: 
- [ ] Replace with actual data loader imports from existing modules
- [ ] Replace with actual indicator hub imports from existing modules  
- [ ] Check `src/data/` directory for correct loader classes
- [ ] Check `src/strategy/` directory for correct indicator classes

#### **2. Implement Missing Container Types**
**Files**: Need to create in `src/execution/containers.py`

**MISSING CONTAINERS**:
- [ ] **RiskContainer** - Risk management container
- [ ] **PortfolioContainer** - Portfolio allocation container  
- [ ] **ClassifierContainer** - Regime detection container
- [ ] **AnalysisContainer** - Signal analysis container
- [ ] **SignalLogContainer** - Signal replay source container
- [ ] **EnsembleContainer** - Signal ensemble optimization container

**TEMPLATE TO FOLLOW**:
```python
class RiskContainer(BaseComposableContainer):
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.RISK,
            name="RiskContainer", 
            config=config,
            container_id=container_id
        )
        # Risk-specific initialization
        
    def get_required_indicators(self) -> Set[str]:
        # Return required indicators
        
    async def process_event(self, event: Event) -> Optional[Event]:
        # Handle SIGNAL events, produce ORDER events
```

**REFERENCE**: Use existing containers in same file as template

#### **3. Complete Factory Registrations**
**File**: `src/execution/containers.py` (bottom of file)
**Current**: Only 4 container types registered
**MISSING**: 6+ container types need registration

**TASK**:
- [ ] Add factory functions for all missing containers
- [ ] Register with `register_container_type()` 
- [ ] Add appropriate capabilities for each container type

#### **4. Bridge Existing Engines with Container System**
**Files**: Multiple backtest engines exist but aren't connected

**EXISTING ENGINES**:
- `src/execution/backtest_engine.py` - UnifiedBacktestEngine
- `src/execution/simple_backtest_engine.py` - SimpleBacktestEngine  
- `src/execution/signal_generation_engine.py` - Exists but not integrated
- `src/execution/signal_replay_engine.py` - Exists but not integrated

**TASK**:
- [ ] Update `ExecutionContainer` to use existing backtest engines
- [ ] Connect signal generation/replay engines to analysis containers
- [ ] Ensure event flow between containers and engines works

### **MEDIUM PRIORITY**

#### **5. Complete Event Flow Implementation**
**Issue**: Event bus wiring between containers is incomplete

**MISSING EVENT TYPES** (check `src/core/events/types.py`):
- [ ] `EventType.INDICATORS` - For indicator distribution  
- [ ] `EventType.REGIME` - For classifier state changes
- [ ] `EventType.RISK_UPDATE` - For risk limit changes

**TASK**:
- [ ] Add missing event types
- [ ] Complete event handling in each container
- [ ] Test event flow: Data → Indicator → Classifier → Risk → Portfolio → Strategy → Execution

#### **6. Connect to Existing Risk/Strategy Modules**
**EXISTING MODULES**:
- `src/risk/` - Has risk management protocols and implementations
- `src/strategy/` - Has strategy protocols and implementations

**TASK**:
- [ ] Import and use existing risk managers in RiskContainer
- [ ] Import and use existing strategies in StrategyContainer
- [ ] Ensure protocol compatibility

#### **7. Data Loader Integration**
**EXISTING**: `src/data/` has data loading capabilities
**ISSUE**: DataContainer references non-existent loaders

**TASK**:
- [ ] Check what data loaders actually exist in `src/data/`
- [ ] Update DataContainer to use existing loaders
- [ ] Ensure multi-asset, multi-timeframe support works

### **LOW PRIORITY**

#### **8. Advanced Container Features**
- [ ] Resource limits enforcement in containers
- [ ] Container metrics and monitoring
- [ ] Container persistence and recovery
- [ ] Advanced error handling and rollback

#### **9. Performance Optimization**
- [ ] Indicator computation caching
- [ ] Event batching for high-frequency data
- [ ] Memory-efficient result streaming
- [ ] Parallel container execution

#### **10. Testing Integration**
- [ ] Container unit tests
- [ ] Integration tests with coordinator
- [ ] End-to-end backtest tests
- [ ] Performance benchmarking

## IMPLEMENTATION GUIDE

### **Directory Structure**
```
src/
├── core/
│   ├── coordinator/
│   │   ├── coordinator.py              ✅ DONE
│   │   ├── composable_workflow_manager.py  ✅ DONE
│   │   └── ...
│   └── containers/
│       ├── composable.py               ✅ DONE
│       ├── composition_engine.py       ✅ DONE
│       └── ...
├── execution/
│   ├── containers.py                   🚧 NEEDS COMPLETION
│   ├── backtest_engine.py             ✅ EXISTS - NEEDS INTEGRATION
│   ├── simple_backtest_engine.py      ✅ EXISTS - NEEDS INTEGRATION
│   └── ...
├── data/                               ✅ EXISTS - NEEDS CONNECTION
├── strategy/                           ✅ EXISTS - NEEDS CONNECTION  
└── risk/                               ✅ EXISTS - NEEDS CONNECTION
```

### **Key Integration Points**

#### **1. Container to Engine Connection**
```python
# In ExecutionContainer
async def _initialize_self(self) -> None:
    execution_mode = self._metadata.config.get('mode', 'backtest')
    
    if execution_mode == 'backtest':
        from .backtest_engine import UnifiedBacktestEngine  # Use existing
        self.execution_engine = UnifiedBacktestEngine(config)
```

#### **2. Container to Risk/Strategy Connection** 
```python
# In RiskContainer
async def _initialize_self(self) -> None:
    from ..risk.risk_portfolio import RiskPortfolioContainer  # Use existing
    self.risk_manager = RiskPortfolioContainer(config)

# In StrategyContainer  
async def _initialize_self(self) -> None:
    from ..strategy.strategies.momentum import MomentumStrategy  # Use existing
    self.strategy = MomentumStrategy(config)
```

#### **3. Event Flow Pattern**
```python
# Market Data → Indicators → Signals → Orders → Fills
async def process_event(self, event: Event) -> Optional[Event]:
    if event.event_type == EventType.BAR:
        # Process market data
        result = self.process_market_data(event.payload)
        
        # Create output event
        output_event = Event(
            event_type=EventType.INDICATORS,  # or SIGNAL, ORDER, etc.
            payload=result,
            timestamp=event.timestamp
        )
        
        return output_event
```

### **Testing Strategy**

#### **1. Unit Test Each Container**
```python
# Test individual container functionality
async def test_data_container():
    config = {'source': 'historical', 'symbols': ['SPY']}
    container = DataContainer(config)
    await container.initialize()
    # Test streaming
```

#### **2. Integration Test Container Patterns**
```python
# Test full container patterns  
async def test_simple_backtest_pattern():
    coordinator = Coordinator()
    config = WorkflowConfig(...)
    result = await coordinator.execute_workflow(config)
    assert result.success
```

#### **3. End-to-End Test**
```python
# Test complete workflow
async def test_full_backtest_with_multi_assets():
    config = WorkflowConfig(
        data_config={'symbols': ['AAPL', 'GOOGL', 'MSFT']},
        parameters={'container_pattern': 'full_backtest'}
    )
    # Verify multi-asset results
```

## COMPLETION CRITERIA ✅

### **Minimum Viable Implementation**
- [ ] All container imports work (no ImportError)
- [ ] Basic container pattern executes successfully
- [ ] Simple backtest with single asset completes
- [ ] Multi-asset backtest completes
- [ ] Signal generation pattern works
- [ ] Signal replay pattern works

### **Full Implementation**
- [ ] All container types implemented
- [ ] All existing engines integrated
- [ ] Complete event flow working
- [ ] Performance meets requirements
- [ ] All tests passing

## GETTING STARTED

### **Recommended Order**
1. **Fix imports** - Start with `src/execution/containers.py` imports
2. **Implement RiskContainer** - Most critical missing piece
3. **Implement PortfolioContainer** - Second most critical
4. **Test simple pattern** - Get basic backtest working
5. **Add remaining containers** - Complete the system
6. **Integration testing** - Verify everything works together

### **Quick Verification**
```python
# Test if system works
from src.core.coordinator.coordinator import Coordinator

coordinator = Coordinator()
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    data_config={'symbols': ['SPY']},
    parameters={'container_pattern': 'simple_backtest'}
)

result = await coordinator.execute_workflow(config)
print(f"Success: {result.success}")
```

## NOTES

- **Architecture is solid** - focus on implementation, not design
- **Protocol-based** - no inheritance, clean interfaces
- **Multi-asset/timeframe support preserved** - don't break this
- **Backward compatibility maintained** - traditional mode still works
- **Clean imports** - avoid deep dependency chains
- **Event-driven** - containers communicate via events only

The foundation is complete. The remaining work is connecting existing components and implementing the missing container types following the established patterns.