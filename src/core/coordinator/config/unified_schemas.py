"""
Simplified configuration schemas for unified architecture.

These schemas replace the complex pattern-based schemas with simple,
mode-based configurations that work with the universal topology.
"""

from .schema_validator import ConfigSchema


# Unified backtest schema - works for all three modes
UNIFIED_WORKFLOW_SCHEMA = ConfigSchema(
    name="unified_workflow",
    version="2.0.0",
    description="Simplified schema for unified architecture workflows",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            # Basic workflow info
            "workflow_type": {
                "type": "string",
                "enum": ["backtest", "optimization", "walk_forward", "ensemble_creation"],
                "default": "backtest"
            },
            
            # Core parameters (minimal!)
            "parameters": {
                "type": "object",
                "properties": {
                    # Execution mode determines the pipeline behavior
                    "mode": {
                        "type": "string",
                        "enum": ["backtest", "signal_generation", "signal_replay"],
                        "description": "Execution mode for the workflow"
                    },
                    
                    # Data parameters (for backtest/signal_generation)
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    
                    # Signal I/O (mode-specific)
                    "signal_output_dir": {
                        "type": "string",
                        "description": "Directory to save signals (signal_generation mode)"
                    },
                    "signal_input_dir": {
                        "type": "string",
                        "description": "Directory to load signals from (signal_replay mode)"
                    },
                    "market_data_file": {
                        "type": "string",
                        "description": "Optional market data for signal replay pricing"
                    },
                    
                    # Feature configuration
                    "features": {
                        "type": "object",
                        "properties": {
                            "indicators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "period": {"type": "integer", "minimum": 1}
                                    },
                                    "required": ["name", "type"]
                                }
                            }
                        }
                    },
                    
                    # Strategy parameters (creates parameter grid)
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["momentum", "mean_reversion", "trend_following"]
                                },
                                # All other properties are parameter arrays for grid expansion
                                "additionalProperties": {
                                    "oneOf": [
                                        {"type": "number"},
                                        {"type": "array", "items": {"type": "number"}}
                                    ]
                                }
                            },
                            "required": ["type"]
                        }
                    },
                    
                    # Risk profiles (creates parameter grid)
                    "risk_profiles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["conservative", "moderate", "aggressive"]
                                },
                                "max_position_size": {"type": "number"},
                                "max_portfolio_risk": {"type": "number"},
                                "max_drawdown": {"type": "number"},
                                "stop_loss_pct": {"type": "number"}
                            },
                            "required": ["type"]
                        }
                    },
                    
                    # Classifiers (optional)
                    "classifiers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["trend", "volatility", "composite"]
                                },
                                "additionalProperties": true
                            },
                            "required": ["type"]
                        }
                    },
                    
                    # Execution configuration
                    "execution": {
                        "type": "object",
                        "properties": {
                            "slippage_bps": {
                                "oneOf": [
                                    {"type": "number"},
                                    {"type": "array", "items": {"type": "number"}}
                                ]
                            },
                            "commission_per_share": {"type": "number"}
                        }
                    }
                },
                "required": ["mode"]
            },
            
            # Data configuration (simplified)
            "data_config": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["csv", "database", "api"],
                        "default": "csv"
                    },
                    "file_path": {"type": "string"},
                    "api_config": {"type": "object"}
                }
            },
            
            # Infrastructure (optional)
            "infrastructure_config": {
                "type": "object",
                "properties": {
                    "parallel_execution": {"type": "boolean", "default": true},
                    "max_workers": {"type": "integer", "minimum": 1, "default": 4},
                    "memory_limit_mb": {"type": "integer", "minimum": 512}
                },
                "additionalProperties": false
            }
        },
        "required": ["workflow_type", "parameters"],
        
        # Mode-specific requirements
        "allOf": [
            {
                "if": {
                    "properties": {
                        "parameters": {
                            "properties": {
                                "mode": {"const": "signal_generation"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "parameters": {
                            "required": ["symbols", "start_date", "end_date", "strategies"]
                        }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "parameters": {
                            "properties": {
                                "mode": {"const": "signal_replay"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "parameters": {
                            "required": ["signal_input_dir", "risk_profiles"]
                        }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "parameters": {
                            "properties": {
                                "mode": {"const": "backtest"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "parameters": {
                            "required": ["symbols", "start_date", "end_date", "strategies", "risk_profiles"]
                        }
                    }
                }
            }
        ]
    },
    examples=[
        {
            "workflow_type": "backtest",
            "parameters": {
                "mode": "backtest",
                "symbols": ["SPY"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "features": {
                    "indicators": [
                        {"name": "sma_fast", "type": "sma", "period": 10},
                        {"name": "sma_slow", "type": "sma", "period": 20},
                        {"name": "rsi", "type": "rsi", "period": 14}
                    ]
                },
                "strategies": [
                    {
                        "type": "momentum",
                        "momentum_threshold": [0.01, 0.02, 0.03],
                        "rsi_oversold": 30,
                        "rsi_overbought": 70
                    }
                ],
                "risk_profiles": [
                    {
                        "type": "conservative",
                        "max_position_size": 0.1,
                        "max_portfolio_risk": 0.02
                    }
                ]
            },
            "data_config": {
                "source": "csv",
                "file_path": "./data/SPY.csv"
            }
        },
        {
            "workflow_type": "backtest",
            "parameters": {
                "mode": "signal_generation",
                "symbols": ["SPY", "QQQ"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "signal_output_dir": "./signals/exploration",
                "strategies": [
                    {
                        "type": "momentum",
                        "momentum_threshold": [0.005, 0.01, 0.015, 0.02],
                        "rsi_oversold": [25, 30, 35],
                        "rsi_overbought": [65, 70, 75]
                    }
                ]
            }
        },
        {
            "workflow_type": "backtest",
            "parameters": {
                "mode": "signal_replay",
                "signal_input_dir": "./signals/exploration",
                "risk_profiles": [
                    {"type": "conservative", "max_position_size": 0.05},
                    {"type": "moderate", "max_position_size": 0.1},
                    {"type": "aggressive", "max_position_size": 0.2}
                ],
                "execution": {
                    "slippage_bps": [5, 10, 15]
                }
            }
        }
    ]
)


# Multi-phase workflow schema (for complex workflows)
MULTI_PHASE_SCHEMA = ConfigSchema(
    name="multi_phase_workflow",
    version="2.0.0", 
    description="Schema for multi-phase workflows using unified architecture",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "workflow_type": {
                "type": "string",
                "enum": ["optimization", "walk_forward", "ensemble_creation"]
            },
            
            "parameters": {
                "type": "object",
                "properties": {
                    # Phase definitions
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["optimization", "backtest", "analysis"]
                                },
                                "execution_mode": {
                                    "type": "string",
                                    "enum": ["signal_generation", "backtest", "signal_replay"]
                                },
                                "parameters": {"type": "object"},
                                "inherit_best_from": {"type": "string"},
                                "signal_source": {"type": "string"}
                            },
                            "required": ["name", "type"]
                        },
                        "minItems": 1
                    }
                }
            }
        },
        "required": ["workflow_type", "parameters"]
    }
)


# Export all schemas
__all__ = [
    'UNIFIED_WORKFLOW_SCHEMA',
    'MULTI_PHASE_SCHEMA'
]