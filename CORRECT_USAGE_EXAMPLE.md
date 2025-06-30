# Correct Usage Example

## How the New Structure Works

When you run a config file like `config/mean_reversion_research/config.yaml`:

```bash
python main.py --signal-generation --config config/mean_reversion_research/config.yaml
```

The system will:

1. Extract the parent directory name (`mean_reversion_research`) from the config path
2. Create results under: `configs/mean_reversion_research/results/<timestamp>/`
3. NOT create a directory named "config"

## Example Directory Structure

```
config/                              # Your config files directory
├── mean_reversion_research/
│   ├── config.yaml                  # The config file you run
│   └── README.md                    # Optional documentation
│
├── keltner_optimization/
│   └── config.yaml
│
└── test_new_workspace_structure.yaml  # Standalone config

configs/                             # Results storage directory (separate from config/)
├── mean_reversion_research/         # Matches the config's parent directory
│   └── results/
│       ├── 20241220_143022/
│       │   ├── metadata.json
│       │   └── traces/
│       │       ├── bollinger_bands/
│       │       │   ├── bb_baseline_params.parquet
│       │       │   └── bb_tight_params.parquet
│       │       └── rsi_threshold/
│       │           └── rsi_standard_params.parquet
│       └── latest -> 20241220_143022/
│
├── keltner_optimization/            # Results for keltner config
│   └── results/
│       └── ...
│
└── test_new_workspace_structure/    # For standalone configs
    └── results/
        └── ...
```

## Key Points

1. **Config files stay in `config/` directory** - your source configs remain organized there
2. **Results go to `configs/` directory** - a separate directory for all execution results
3. **Directory naming uses parent folder** - `mean_reversion_research/config.yaml` creates results in `configs/mean_reversion_research/`
4. **No "config" directories created** - the system extracts the meaningful parent directory name

## Special Cases

- If config is directly in `config/` (like `config/test.yaml`), it uses the filename: `configs/test/results/`
- If config path doesn't have a meaningful parent, it falls back to the filename without extension