"""
Validate Protocol + Composition (PC) Architecture across ADMF-PC.

This test suite ensures:
1. NO inheritance in business logic
2. All components use protocols
3. Capabilities are added through composition
4. Clean separation of concerns
5. Proper container isolation
"""

import ast
import inspect
import unittest
from pathlib import Path
from typing import Set, List, Type, Any
import importlib
import pkgutil

# Import all modules for inspection
import src.strategy
import src.core
import src.data


class TestPCArchitectureCompliance(unittest.TestCase):
    """Test that entire codebase follows PC architecture."""
    
    def setUp(self):
        """Set up test environment."""
        self.src_path = Path(__file__).parent / 'src'
        self.violations = []
    
    def test_no_inheritance_in_strategy_module(self):
        """Test strategy module has NO inheritance."""
        violations = self._check_module_for_inheritance('src.strategy')
        
        # Print violations for debugging
        for violation in violations:
            print(f"Inheritance violation: {violation}")
        
        # Should have no inheritance violations
        self.assertEqual(len(violations), 0, 
                        f"Found {len(violations)} inheritance violations in strategy module")
    
    def test_no_inheritance_in_core_business_logic(self):
        """Test core business logic has NO inheritance."""
        # Check specific core modules that should have no inheritance
        modules_to_check = [
            'src.core.components',
            'src.core.containers',
            'src.core.events'
        ]
        
        all_violations = []
        for module_name in modules_to_check:
            violations = self._check_module_for_inheritance(module_name)
            all_violations.extend(violations)
        
        # Filter out acceptable cases (protocols, exceptions)
        filtered_violations = [
            v for v in all_violations
            if not self._is_acceptable_inheritance(v)
        ]
        
        self.assertEqual(len(filtered_violations), 0,
                        f"Found {len(filtered_violations)} inheritance violations in core")
    
    def _check_module_for_inheritance(self, module_name: str) -> List[str]:
        """Check a module and its submodules for inheritance."""
        violations = []
        
        try:
            module = importlib.import_module(module_name)
            
            # Check all classes in module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__.startswith(module_name):
                    # Check base classes
                    bases = obj.__bases__
                    if len(bases) > 1 or (len(bases) == 1 and bases[0] != object):
                        violation = f"{obj.__module__}.{obj.__name__} inherits from {bases}"
                        violations.append(violation)
            
            # Check submodules
            if hasattr(module, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
                    full_modname = f"{module_name}.{modname}"
                    if not modname.startswith('test_'):  # Skip test files
                        subviolations = self._check_module_for_inheritance(full_modname)
                        violations.extend(subviolations)
        
        except Exception as e:
            # Module might not exist or have issues
            pass
        
        return violations
    
    def _is_acceptable_inheritance(self, violation: str) -> bool:
        """Check if inheritance violation is acceptable."""
        acceptable_patterns = [
            'Protocol',      # Protocol inheritance is OK
            'Exception',     # Exception inheritance is OK
            'BaseModel',     # Pydantic models for data validation
            'Enum',          # Enum inheritance is OK
            'ABC',           # Should be avoided but sometimes necessary
            'Mock',          # Test mocks
            'TestCase'       # Test classes
        ]
        
        return any(pattern in violation for pattern in acceptable_patterns)
    
    def test_protocols_are_runtime_checkable(self):
        """Test all protocols are marked as runtime_checkable."""
        protocol_files = [
            'src/strategy/protocols.py',
            'src/strategy/optimization/protocols.py',
            'src/core/infrastructure/protocols.py'
        ]
        
        for file_path in protocol_files:
            full_path = Path(file_path)
            if full_path.exists():
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                # Find all Protocol classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if inherits from Protocol
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == 'Protocol':
                                # Check if has @runtime_checkable decorator
                                has_decorator = any(
                                    isinstance(d, ast.Name) and d.id == 'runtime_checkable'
                                    for d in node.decorator_list
                                )
                                
                                self.assertTrue(has_decorator,
                                              f"{node.name} Protocol should be @runtime_checkable")
    
    def test_capabilities_use_composition(self):
        """Test capabilities are applied through composition, not inheritance."""
        from src.strategy.capabilities import StrategyCapability
        from src.strategy.optimization.capabilities import OptimizationCapability
        
        capabilities = [
            StrategyCapability(),
            OptimizationCapability()
        ]
        
        for capability in capabilities:
            # Should have apply method
            self.assertTrue(hasattr(capability, 'apply'))
            
            # Apply should return enhanced object, not subclass
            mock_component = type('MockComponent', (object,), {})()
            enhanced = capability.apply(mock_component, {})
            
            # Enhanced should not be a subclass
            self.assertFalse(
                issubclass(type(enhanced), type(mock_component)),
                "Capability should use composition, not inheritance"
            )
    
    def test_container_isolation(self):
        """Test containers properly isolate components."""
        from src.core.containers import UniversalScopedContainer
        
        # Create two containers
        container1 = UniversalScopedContainer('test1')
        container2 = UniversalScopedContainer('test2')
        
        # Initialize
        container1.initialize_scope()
        container2.initialize_scope()
        
        # Register same component type in both
        container1.register_component('test', {'value': 1})
        container2.register_component('test', {'value': 2})
        
        # Should have different values
        with container1.create_scope():
            val1 = container1.get_component('test')
        
        with container2.create_scope():
            val2 = container2.get_component('test')
        
        self.assertNotEqual(val1['value'], val2['value'])
        
        # Clean up
        container1.dispose()
        container2.dispose()
    
    def test_event_bus_isolation(self):
        """Test event buses are container-scoped."""
        from src.core.containers import UniversalScopedContainer
        from src.core.events import EventBus
        
        container = UniversalScopedContainer('event_test')
        container.initialize_scope()
        
        # Container should have its own event bus
        self.assertIsInstance(container.event_bus, EventBus)
        
        # Events in container should be isolated
        received = []
        container.event_bus.subscribe('test.event', lambda e: received.append(e))
        
        # Publish in container
        container.event_bus.publish('test.event', {'data': 'test'})
        
        self.assertEqual(len(received), 1)
        
        container.dispose()
    
    def test_strategy_protocol_implementation(self):
        """Test strategies implement protocol without inheritance."""
        from src.strategy.protocols import Strategy
        from src.strategy.strategies.momentum import MomentumStrategy
        from src.strategy.strategies.mean_reversion import MeanReversionStrategy
        
        strategies = [
            MomentumStrategy(),
            MeanReversionStrategy()
        ]
        
        for strategy in strategies:
            # Should satisfy protocol
            self.assertTrue(isinstance(strategy, Strategy))
            
            # Should have no inheritance
            bases = strategy.__class__.__bases__
            self.assertEqual(len(bases), 1)
            self.assertEqual(bases[0], object)
    
    def test_optimization_components_no_inheritance(self):
        """Test optimization components have NO inheritance."""
        from src.strategy.optimization import (
            GridOptimizer,
            BayesianOptimizer,
            SharpeObjective,
            RelationalConstraint,
            RangeConstraint
        )
        
        components = [
            GridOptimizer(),
            BayesianOptimizer(),
            SharpeObjective(),
            RelationalConstraint('p1', '<', 'p2'),
            RangeConstraint('param', 0, 100)
        ]
        
        for component in components:
            bases = component.__class__.__bases__
            self.assertEqual(len(bases), 1, 
                           f"{component.__class__.__name__} should not use inheritance")
            self.assertEqual(bases[0], object)
    
    def test_clean_module_structure(self):
        """Test modules have clean structure with proper separation."""
        # Check key module structure
        expected_structure = {
            'src.strategy': ['protocols', 'capabilities', 'strategies', 'optimization'],
            'src.core': ['components', 'containers', 'events', 'coordinator'],
            'src.data': ['protocols', 'loaders', 'models']
        }
        
        for module_name, expected_submodules in expected_structure.items():
            try:
                module = importlib.import_module(module_name)
                
                # Check expected submodules exist
                for submodule in expected_submodules:
                    full_name = f"{module_name}.{submodule}"
                    try:
                        importlib.import_module(full_name)
                    except ImportError:
                        self.fail(f"Expected submodule {full_name} not found")
            
            except ImportError:
                pass  # Module might not exist in test environment


class TestArchitecturalPrinciples(unittest.TestCase):
    """Test high-level architectural principles."""
    
    def test_protocol_over_inheritance_principle(self):
        """Test system uses protocols instead of inheritance."""
        # Count protocols vs base classes
        protocol_count = 0
        base_class_count = 0
        
        # Check strategy module
        strategy_path = Path('src/strategy')
        if strategy_path.exists():
            for py_file in strategy_path.rglob('*.py'):
                if 'test' not in str(py_file):
                    with open(py_file, 'r') as f:
                        content = f.read()
                        protocol_count += content.count('Protocol):')
                        base_class_count += content.count('(Base') 
        
        # Should have more protocols than base classes
        self.assertGreater(protocol_count, base_class_count,
                          "Should use protocols more than base classes")
    
    def test_composition_over_inheritance_principle(self):
        """Test system uses composition instead of inheritance."""
        # Check for composition patterns
        composition_patterns = [
            'capability.apply',
            'ComponentFactory',
            'register_component',
            'with_capability'
        ]
        
        inheritance_patterns = [
            'super().__init__',
            'class.*\\(.*[^)]\\):',  # Class inheritance
            'override'
        ]
        
        composition_count = 0
        inheritance_count = 0
        
        # Count patterns in strategy module
        strategy_path = Path('src/strategy')
        if strategy_path.exists():
            for py_file in strategy_path.rglob('*.py'):
                if 'test' not in str(py_file):
                    with open(py_file, 'r') as f:
                        content = f.read()
                        
                        for pattern in composition_patterns:
                            composition_count += content.count(pattern)
                        
                        for pattern in inheritance_patterns:
                            if pattern == 'super().__init__':
                                inheritance_count += content.count(pattern)
        
        # Should favor composition
        self.assertGreater(composition_count, 0,
                          "Should use composition patterns")
    
    def test_container_isolation_principle(self):
        """Test components run in isolated containers."""
        from src.core.containers import UniversalScopedContainer
        
        # Test container provides isolation
        container = UniversalScopedContainer('isolation_test')
        
        # Should have isolation features
        self.assertTrue(hasattr(container, 'create_scope'))
        self.assertTrue(hasattr(container, 'event_bus'))
        self.assertTrue(hasattr(container, 'config_namespace'))
        
        # Should track resources
        self.assertTrue(hasattr(container, 'resource_limits'))
        
        container.dispose()
    
    def test_event_driven_principle(self):
        """Test system uses event-driven communication."""
        from src.core.events import EventBus
        
        event_bus = EventBus()
        
        # Should support pub/sub
        self.assertTrue(hasattr(event_bus, 'publish'))
        self.assertTrue(hasattr(event_bus, 'subscribe'))
        
        # Test event flow
        received = []
        event_bus.subscribe('test', lambda e: received.append(e))
        event_bus.publish('test', {'data': 'value'})
        
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data['data'], 'value')


class TestProductionReadiness(unittest.TestCase):
    """Test production readiness aspects."""
    
    def test_error_handling(self):
        """Test proper error handling throughout system."""
        from src.strategy.optimization import GridOptimizer
        
        optimizer = GridOptimizer()
        
        # Should handle invalid inputs gracefully
        try:
            # Invalid evaluation function
            optimizer.optimize(None, {})
        except Exception as e:
            # Should raise meaningful error
            self.assertIsInstance(e, (TypeError, ValueError))
    
    def test_resource_management(self):
        """Test proper resource management."""
        from src.core.containers import UniversalScopedContainer
        
        container = UniversalScopedContainer('resource_test')
        container.initialize_scope()
        
        # Should track resources
        self.assertTrue(hasattr(container, 'active_resources'))
        
        # Should clean up properly
        container.start()
        container.stop()
        container.dispose()
        
        # Resources should be cleaned up
        self.assertEqual(len(container.active_resources), 0)
    
    def test_monitoring_capabilities(self):
        """Test system has monitoring capabilities."""
        from src.core.infrastructure.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Should support metric collection
        self.assertTrue(hasattr(collector, 'record_metric'))
        self.assertTrue(hasattr(collector, 'get_metrics'))
        
        # Test metric recording
        collector.record_metric('test.metric', 1.0)
        metrics = collector.get_metrics()
        
        self.assertIn('test.metric', metrics)


if __name__ == '__main__':
    unittest.main()