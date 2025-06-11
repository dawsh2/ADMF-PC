# Coordinator Refactor Example

## How the Coordinator would change to support topology-first architecture:

```python
class Coordinator:
    def __init__(self, ...):
        # Base components
        self.topology_runner = TopologyRunner()
        self.sequencer = Sequencer()
        self.workflow_manager = WorkflowManager()
        
    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart entry point that routes to appropriate execution method.
        """
        # Direct topology execution
        if 'topology' in config and 'workflow' not in config:
            return self.run_topology(config['topology'], config)
            
        # Sequence execution (topology + iteration)
        elif 'sequence' in config and 'topology' in config:
            return self.run_sequence(
                sequence=config['sequence'],
                topology=config['topology'], 
                config=config
            )
            
        # Full workflow execution
        elif 'workflow' in config:
            return self.run_workflow(config['workflow'], config)
            
        else:
            # Default to simple backtest workflow for compatibility
            return self.run_workflow('simple_backtest', config)
    
    def run_topology(self, topology_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a topology directly - the simplest execution mode.
        """
        return self.topology_runner.run_topology(topology_name, config)
    
    def run_sequence(self, sequence: str, topology: str, 
                    config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a topology with sequence iteration (walk-forward, param sweep, etc).
        """
        sequence_config = {
            'sequence': sequence,
            'topology': topology,
            'config': config
        }
        
        # Sequencer handles iteration and calls topology_runner for each
        return self.sequencer.run_sequence(sequence_config, self.topology_runner)
    
    def run_workflow(self, workflow_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a full workflow - composition of multiple topology+sequence phases.
        """
        # Get workflow pattern
        workflow = self.workflow_patterns.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        # Execute phases
        results = {}
        for phase in workflow['phases']:
            phase_result = self.run_sequence(
                sequence=phase.get('sequence', 'single_pass'),
                topology=phase['topology'],
                config=self._merge_configs(config, phase.get('config', {}))
            )
            results[phase['name']] = phase_result
        
        return {
            'workflow': workflow_name,
            'phases': results,
            'success': all(r.get('success', True) for r in results.values())
        }
```

## Clean CLI Usage

```python
# main.py would support:

def main():
    args = parse_arguments()
    coordinator = Coordinator()
    
    # Build config based on CLI args
    config = load_config(args.config)
    
    # Direct topology execution
    if args.topology:
        config['topology'] = args.topology
        result = coordinator.run_topology(args.topology, config)
        
    # Sequence execution
    elif args.sequence and args.topology:
        result = coordinator.run_sequence(args.sequence, args.topology, config)
        
    # Workflow execution
    elif args.workflow:
        result = coordinator.run_workflow(args.workflow, config)
        
    # Config-driven (auto-detect mode)
    else:
        result = coordinator.run(config)
```

## Example Usage Progression

### 1. Start Simple - Just Run a Topology
```bash
# Direct topology test
python main.py --topology signal_generation --config data_config.yaml
```

### 2. Add Iteration - Use a Sequence
```bash
# Walk-forward analysis
python main.py --topology backtest --sequence walk_forward --config strategy_config.yaml
```

### 3. Compose Complex Pipelines - Use Workflows
```bash
# Full research pipeline
python main.py --workflow optimization_pipeline --config research_config.yaml
```

## Benefits of This Approach

1. **Natural Learning Curve**: Users start with topologies and gradually add complexity
2. **Clean Separation**: Each layer has a clear responsibility
3. **Better Testing**: Can test topologies in isolation
4. **No Magic**: Everything is explicit - no auto-wrapping
5. **Composable**: Each layer builds on the previous one naturally