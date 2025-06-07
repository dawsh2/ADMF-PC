"""Configuration schemas for ADMF-PC.

This module defines JSON schemas for all configuration types used in the system.
These schemas ensure configurations are valid before execution.
"""

from .schema_validator import ConfigSchema


# Base component schemas (reusable)
STRATEGY_COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "enabled": {"type": "boolean", "default": True},
        "allocation": {"type": "number", "minimum": 0, "maximum": 1},
        "parameters": {"type": "object"},
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "config": {"type": "object"}
                },
                "required": ["name"]
            }
        }
    },
    "required": ["name", "type"]
}

RISK_LIMIT_SCHEMA = {
    "type": "object", 
    "properties": {
        "type": {"type": "string", "enum": [
            "position", "exposure", "drawdown", "var", 
            "concentration", "leverage", "daily_loss", "symbol_restriction"
        ]},
        "enabled": {"type": "boolean", "default": True},
        # Type-specific parameters
        "max_position": {"type": "number"},
        "max_exposure_pct": {"type": "number"},
        "max_drawdown_pct": {"type": "number"},
        "reduce_at_pct": {"type": "number"},
        "confidence_level": {"type": "number"},
        "max_var_pct": {"type": "number"},
        "max_position_pct": {"type": "number"},
        "max_sector_pct": {"type": "number"},
        "max_leverage": {"type": "number"},
        "max_daily_loss": {"type": "number"},
        "max_daily_loss_pct": {"type": "number"},
        "allowed_symbols": {"type": "array", "items": {"type": "string"}},
        "blocked_symbols": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["type"]
}

POSITION_SIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string", "enum": ["fixed", "percentage", "volatility", "kelly", "atr"]},
        # Type-specific parameters
        "size": {"type": "number"},
        "percentage": {"type": "number"},
        "risk_per_trade": {"type": "number"},
        "lookback_period": {"type": "integer"},
        "kelly_fraction": {"type": "number"},
        "max_leverage": {"type": "number"},
        "risk_amount": {"type": "number"},
        "atr_multiplier": {"type": "number"}
    },
    "required": ["name", "type"]
}

# Backtest configuration schema
BACKTEST_SCHEMA = ConfigSchema(
    name="backtest",
    version="1.0.0",
    description="Schema for backtest workflow configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Backtest name"},
            "type": {"type": "string", "enum": ["backtest"], "default": "backtest"},
            "description": {"type": "string"},
            "execution_mode": {
                "type": "string",
                "enum": ["vectorized", "event_driven"],
                "default": "vectorized"
            },
            "data": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    "frequency": {"type": "string", "enum": ["1min", "5min", "15min", "30min", "1h", "1d"]},
                    "format": {"type": "string", "enum": ["csv", "parquet", "hdf5"]},
                    "timezone": {"type": "string", "default": "UTC"}
                },
                "required": ["symbols", "start_date", "end_date", "frequency"]
            },
            "portfolio": {
                "type": "object",
                "properties": {
                    "initial_capital": {"type": "number", "minimum": 0},
                    "currency": {"type": "string", "default": "USD"},
                    "commission": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["fixed", "percentage"]},
                            "value": {"type": "number", "minimum": 0}
                        }
                    },
                    "slippage": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["fixed", "percentage"]},
                            "value": {"type": "number", "minimum": 0}
                        }
                    }
                },
                "required": ["initial_capital"]
            },
            "strategies": {
                "type": "array",
                "items": STRATEGY_COMPONENT_SCHEMA,
                "minItems": 1
            },
            "risk": {
                "type": "object",
                "properties": {
                    "position_sizers": {
                        "type": "array",
                        "items": POSITION_SIZER_SCHEMA
                    },
                    "limits": {
                        "type": "array", 
                        "items": RISK_LIMIT_SCHEMA
                    }
                }
            },
            "analysis": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "plots": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "export": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string", "enum": ["csv", "excel", "html"]},
                            "path": {"type": "string"}
                        }
                    }
                }
            }
        },
        "required": ["name", "data", "portfolio", "strategies"]
    },
    examples=[{
        "name": "MA Crossover Backtest",
        "type": "backtest",
        "data": {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01", 
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "ma_crossover",
            "type": "moving_average_crossover",
            "parameters": {
                "fast_period": 10,
                "slow_period": 30
            }
        }]
    }]
)

# Optimization configuration schema
OPTIMIZATION_SCHEMA = ConfigSchema(
    name="optimization",
    version="1.0.0", 
    description="Schema for optimization workflow configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["optimization"], "default": "optimization"},
            "description": {"type": "string"},
            "base_config": {
                "type": "object",
                "description": "Base backtest configuration to optimize",
                "$ref": "#/definitions/backtest_config"
            },
            "optimization": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["grid_search", "random_search", "bayesian", "genetic", "differential_evolution"]
                    },
                    "parameter_space": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["int", "float", "choice"]},
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                                "step": {"type": "number"},
                                "choices": {"type": "array"}
                            }
                        }
                    },
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "expression": {"type": "string"}
                            }
                        }
                    },
                    "objectives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric": {"type": "string"},
                                "direction": {"type": "string", "enum": ["minimize", "maximize"]},
                                "weight": {"type": "number", "default": 1.0}
                            },
                            "required": ["metric", "direction"]
                        },
                        "minItems": 1
                    },
                    "n_trials": {"type": "integer", "minimum": 1},
                    "n_jobs": {"type": "integer", "minimum": 1},
                    "timeout": {"type": "integer", "minimum": 0},
                    "early_stopping": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "patience": {"type": "integer"},
                            "min_delta": {"type": "number"}
                        }
                    }
                },
                "required": ["method", "parameter_space", "objectives"]
            },
            "parallel": {"type": "boolean", "default": True},
            "max_workers": {"type": "integer", "minimum": 1, "default": 4},
            "output": {
                "type": "object",
                "properties": {
                    "save_all_results": {"type": "boolean", "default": False},
                    "save_top_n": {"type": "integer", "minimum": 1},
                    "export_path": {"type": "string"}
                }
            }
        },
        "required": ["name", "base_config", "optimization"],
        "definitions": {
            "backtest_config": BACKTEST_SCHEMA.schema
        }
    },
    examples=[{
        "name": "MA Strategy Optimization",
        "type": "optimization",
        "base_config": {
            "data": {
                "symbols": ["SPY"],
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
                "frequency": "1h"
            },
            "portfolio": {
                "initial_capital": 100000
            },
            "strategies": [{
                "name": "ma_strategy",
                "type": "moving_average_crossover"
            }]
        },
        "optimization": {
            "method": "bayesian",
            "parameter_space": {
                "fast_period": {
                    "type": "int",
                    "min": 5,
                    "max": 50
                },
                "slow_period": {
                    "type": "int", 
                    "min": 20,
                    "max": 200
                }
            },
            "constraints": [{
                "type": "inequality",
                "expression": "slow_period > fast_period + 10"
            }],
            "objectives": [{
                "metric": "sharpe_ratio",
                "direction": "maximize"
            }],
            "n_trials": 100
        }
    }]
)

# Live trading configuration schema
LIVE_TRADING_SCHEMA = ConfigSchema(
    name="live_trading",
    version="1.0.0",
    description="Schema for live trading workflow configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["live_trading", "live"], "default": "live_trading"},
            "description": {"type": "string"},
            "paper_trading": {"type": "boolean", "default": True},
            "broker": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["alpaca", "interactive_brokers", "paper"]},
                    "api_key": {"type": "string"},
                    "api_secret": {"type": "string"},
                    "base_url": {"type": "string"},
                    "account_id": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1},
                    "retry_count": {"type": "integer", "minimum": 0}
                },
                "required": ["name"]
            },
            "data": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["broker", "yahoo", "polygon", "alpaca"]},
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "frequency": {"type": "string"},
                    "lookback_days": {"type": "integer", "minimum": 1},
                    "warmup_period": {"type": "integer", "minimum": 0}
                },
                "required": ["provider", "symbols", "frequency"]
            },
            "portfolio": {
                "type": "object",
                "properties": {
                    "initial_capital": {"type": "number", "minimum": 0},
                    "currency": {"type": "string", "default": "USD"},
                    "max_positions": {"type": "integer", "minimum": 1},
                    "position_sizing": {"type": "string", "enum": ["equal", "risk_parity", "kelly"]}
                },
                "required": ["initial_capital"]
            },
            "strategies": {
                "type": "array",
                "items": STRATEGY_COMPONENT_SCHEMA,
                "minItems": 1
            },
            "risk": {
                "type": "object",
                "properties": {
                    "check_frequency": {"type": "string", "enum": ["pre_trade", "post_trade", "continuous"]},
                    "position_sizers": {
                        "type": "array",
                        "items": POSITION_SIZER_SCHEMA
                    },
                    "limits": {
                        "type": "array",
                        "items": RISK_LIMIT_SCHEMA
                    },
                    "stop_loss": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "type": {"type": "string", "enum": ["fixed", "trailing", "atr"]},
                            "value": {"type": "number"}
                        }
                    },
                    "emergency_shutdown": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "max_daily_loss_pct": {"type": "number"},
                            "max_drawdown_pct": {"type": "number"}
                        }
                    }
                }
            },
            "execution": {
                "type": "object", 
                "properties": {
                    "order_type": {"type": "string", "enum": ["market", "limit", "adaptive"]},
                    "time_in_force": {"type": "string", "enum": ["DAY", "GTC", "IOC", "FOK"]},
                    "retry_failed_orders": {"type": "boolean"},
                    "max_retry": {"type": "integer"},
                    "split_orders": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "max_order_size": {"type": "number"}
                        }
                    }
                }
            },
            "monitoring": {
                "type": "object",
                "properties": {
                    "heartbeat_interval": {"type": "integer", "minimum": 1},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "alerts": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "format": "email"},
                            "webhook": {"type": "string", "format": "uri"}
                        }
                    },
                    "metrics_export": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "interval": {"type": "integer"},
                            "path": {"type": "string"}
                        }
                    }
                }
            }
        },
        "required": ["name", "broker", "data", "portfolio", "strategies"]
    },
    examples=[{
        "name": "Momentum Live Trading",
        "type": "live_trading",
        "paper_trading": True,
        "broker": {
            "name": "paper"
        },
        "data": {
            "provider": "yahoo",
            "symbols": ["SPY", "QQQ"],
            "frequency": "5min",
            "lookback_days": 30
        },
        "portfolio": {
            "initial_capital": 100000
        },
        "strategies": [{
            "name": "momentum",
            "type": "momentum_strategy",
            "parameters": {
                "lookback": 20,
                "threshold": 0.02
            }
        }],
        "risk": {
            "limits": [{
                "type": "position",
                "max_position": 10000
            }, {
                "type": "daily_loss",
                "max_daily_loss_pct": 2.0
            }]
        }
    }]
)

# Additional component schemas
STRATEGY_SCHEMA = ConfigSchema(
    name="strategy",
    version="1.0.0",
    description="Schema for strategy component configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {
                "type": "object",
                "additionalProperties": True
            },
            "indicators": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "parameters": {"type": "object"}
                    },
                    "required": ["name", "type"]
                }
            },
            "rules": {
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "exit": {
                        "type": "array", 
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "required": ["name", "type"]
    }
)

RISK_SCHEMA = ConfigSchema(
    name="risk",
    version="1.0.0",
    description="Schema for risk management configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "initial_capital": {"type": "number", "minimum": 0},
            "position_sizers": {
                "type": "array",
                "items": POSITION_SIZER_SCHEMA
            },
            "limits": {
                "type": "array",
                "items": RISK_LIMIT_SCHEMA
            },
            "risk_per_trade": {"type": "number", "minimum": 0, "maximum": 100},
            "max_correlation": {"type": "number", "minimum": 0, "maximum": 1},
            "rebalance_frequency": {"type": "string", "enum": ["daily", "weekly", "monthly", "none"]}
        }
    }
)

DATA_SCHEMA = ConfigSchema(
    name="data",
    version="1.0.0",
    description="Schema for data source configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "type": {"type": "string", "enum": ["historical", "realtime", "stream"]},
            "symbols": {
                "type": "array",
                "items": {"type": "string"}
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["open", "high", "low", "close", "volume"]
            },
            "frequency": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            "timezone": {"type": "string"},
            "adjustments": {
                "type": "object",
                "properties": {
                    "splits": {"type": "boolean", "default": True},
                    "dividends": {"type": "boolean", "default": True}
                }
            }
        },
        "required": ["source", "type", "symbols"]
    }
)

EXECUTION_SCHEMA = ConfigSchema(
    name="execution",
    version="1.0.0",
    description="Schema for execution configurations",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["simulation", "paper", "live"]},
            "broker": {"type": "string"},
            "order_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"]}
            },
            "slippage": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ["fixed", "linear", "square_root"]},
                    "value": {"type": "number", "minimum": 0}
                }
            },
            "commission": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ["fixed", "percentage", "tiered"]},
                    "value": {"type": "number", "minimum": 0},
                    "minimum": {"type": "number", "minimum": 0}
                }
            },
            "latency": {
                "type": "object",
                "properties": {
                    "mean_ms": {"type": "number", "minimum": 0},
                    "std_ms": {"type": "number", "minimum": 0}
                }
            }
        },
        "required": ["mode"]
    }
)


# Results collection and storage schema
RESULTS_SCHEMA = ConfigSchema(
    name="results",
    version="1.0.0",
    description="Schema for results collection and storage configuration",
    schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "collection": {
                "type": "object",
                "description": "What data to collect during execution",
                "properties": {
                    "streaming_metrics": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Use memory-efficient streaming metrics calculation"
                    },
                    "store_trades": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Store individual trade records"
                    },
                    "store_equity_curve": {
                        "type": "boolean", 
                        "default": False,
                        "description": "Store full equity curve (memory intensive)"
                    },
                    "snapshot_interval": {
                        "type": "integer", 
                        "default": 100,
                        "minimum": 1,
                        "description": "Save snapshots every N bars"
                    },
                    "store_order_book": {
                        "type": "boolean",
                        "default": False,
                        "description": "Store order book snapshots (very memory intensive)"
                    },
                    "trade_summary_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Store only trade entry/exit, not intermediate updates"
                    },
                    "max_equity_points": {
                        "type": "integer",
                        "default": 10000,
                        "minimum": 100,
                        "description": "Maximum equity curve points to store before downsampling"
                    }
                },
                "additionalProperties": False
            },
            "storage": {
                "type": "object",
                "description": "Where and how to save results",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["parquet", "json", "csv", "hdf5"],
                        "default": "json",
                        "description": "Output format for results"
                    },
                    "location": {
                        "type": "string",
                        "default": "./results/{workflow_id}/{phase_name}/",
                        "description": "Template for result storage location"
                    },
                    "compress": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to compress output files"
                    },
                    "partition_by": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["date", "symbol", "strategy", "phase"]},
                        "description": "How to partition output files"
                    }
                },
                "additionalProperties": False
            },
            "memory": {
                "type": "object",
                "description": "Memory management configuration",
                "properties": {
                    "global_limit_mb": {
                        "type": "integer",
                        "default": 1000,
                        "minimum": 100,
                        "description": "Global memory limit for results across all containers"
                    },
                    "storage_mode": {
                        "type": "string",
                        "enum": ["memory", "disk", "auto"],
                        "default": "auto",
                        "description": "Force memory or disk storage, or auto-decide based on size"
                    },
                    "memory_threshold_mb": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "description": "Size threshold for automatic disk storage"
                    },
                    "cache_results": {
                        "type": "boolean",
                        "default": True,
                        "description": "Cache frequently accessed results in memory"
                    }
                },
                "additionalProperties": False
            },
            "retention_policy": {
                "type": "string",
                "enum": ["trade_complete", "sliding_window", "minimal"],
                "default": "trade_complete",
                "description": "Event retention policy for metrics calculation"
            },
            "max_events": {
                "type": "integer",
                "default": 1000,
                "minimum": 100,
                "description": "Maximum events to retain in memory (for sliding_window policy)"
            },
            "testing_mode": {
                "type": "object",
                "description": "Special settings for testing/debugging",
                "properties": {
                    "store_all_events": {
                        "type": "boolean",
                        "default": False,
                        "description": "Store all events (different from tracing)"
                    },
                    "detailed_timing": {
                        "type": "boolean",
                        "default": False,
                        "description": "Track detailed execution timing"
                    },
                    "memory_profiling": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable memory profiling"
                    }
                },
                "additionalProperties": False
            }
        },
        "additionalProperties": False
    }
)