"""
Papermill-based notebook runner for ADMF-PC analysis.

This module uses papermill to parameterize and execute Jupyter notebooks,
replacing the complex notebook generation with template-based analysis.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Check if papermill is available
try:
    import papermill as pm
    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False
    logger.warning("Papermill not available. Install with: pip install papermill")


class PapermillNotebookRunner:
    """Run analysis notebooks using papermill for parameterization and execution"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        
        if not PAPERMILL_AVAILABLE:
            raise ImportError("Papermill is required. Install with: pip install papermill")
            
        if not self.template_dir.exists():
            self.template_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created template directory: {self.template_dir}")
    
    def run_analysis(self, 
                     run_dir: Path, 
                     config: Dict[str, Any],
                     execute: bool = True,
                     launch: bool = False,
                     generate_html: bool = False,
                     template_override: Optional[str] = None) -> Optional[Path]:
        """
        Run analysis notebook with papermill.
        
        Args:
            run_dir: Path to results directory
            config: Configuration dictionary from the run
            execute: Whether to execute the notebook (vs just parameterize)
            launch: Whether to launch Jupyter after generation
            generate_html: Whether to generate HTML report
            template_override: Override template selection (e.g., 'signal_analysis', 'replay_analysis')
            
        Returns:
            Path to the generated notebook (or None if failed)
        """
        # Select template based on override or auto-detection
        if template_override:
            template_name = template_override
            if not template_name.endswith('.ipynb'):
                template_name += '.ipynb'
            template = self.template_dir / template_name
            logger.info(f"Using template override: {template_name}")
        else:
            # Auto-detect based on traces
            is_universal = False
            is_signal_only = False
            
            # Check for traces in run directory (for universal/backtest runs)
            if run_dir.exists():
                traces_dir = run_dir / "traces"
                if traces_dir.exists():
                    # Check what traces exist
                    has_signals = (traces_dir / "signals").exists()
                    has_portfolio = (traces_dir / "portfolio").exists()
                    has_execution = (traces_dir / "execution").exists()
                    
                    is_universal = has_signals and has_portfolio and has_execution
                    is_signal_only = has_signals and not has_portfolio and not has_execution
            
            # For signal generation mode, check global traces directory
            if not is_universal and not is_signal_only:
                # Check if this was a signal generation run by looking for metadata
                metadata_json = run_dir / "metadata.json"
                if metadata_json.exists():
                    try:
                        import json
                        with open(metadata_json) as f:
                            metadata = json.load(f)
                        # Check if this was signal generation mode
                        if metadata.get('mode') == 'signal_generation':
                            is_signal_only = True
                            logger.info("Detected signal generation run from metadata (mode field)")
                        # Also check for signal-only pattern: has signals but no orders/fills/positions
                        elif (metadata.get('total_signals', 0) > 0 and 
                              metadata.get('total_orders', 0) == 0 and
                              metadata.get('total_fills', 0) == 0 and
                              metadata.get('total_positions', 0) == 0):
                            is_signal_only = True
                            logger.info("Detected signal generation run from metadata (signal-only pattern)")
                    except Exception as e:
                        logger.debug(f"Could not read metadata: {e}")
            
            # Select appropriate template
            if is_universal:
                template = self.template_dir / "analysis.ipynb"  # Comprehensive analysis
                logger.info("Detected full system run - using comprehensive analysis template")
            elif is_signal_only:
                template = self.template_dir / "signal_analysis.ipynb"
                logger.info("Detected signal-only run - using signal_analysis template")
            else:
                # Default to comprehensive analysis for universal topology (new default)
                template = self.template_dir / "analysis.ipynb"
                logger.info("Using comprehensive analysis template (default)")
        
        if not template.exists():
            logger.error(f"Template not found: {template}")
            logger.info("Creating default template...")
            self._create_default_template()
            
        # Extract parameters from config
        # First check if data field exists and parse it
        if 'data' in config:
            data_str = config['data']
            if isinstance(data_str, str):
                # Parse symbol from data string like "SPY_5m"
                for tf in ['1m', '5m', '15m', '30m', '1h', '1d']:
                    if data_str.endswith(f'_{tf}'):
                        symbols = [data_str[:-len(f'_{tf}')]]
                        timeframe = tf
                        break
                else:
                    # No timeframe found, use whole string
                    symbols = [data_str]
                    timeframe = config.get('timeframe', '5m')
            else:
                # Handle list or other data formats
                symbols = config.get('symbols', ['SPY'])
                timeframe = config.get('timeframe', '5m')
        else:
            # Fallback to symbols field
            symbols = config.get('symbols', ['SPY'])
            timeframe = config.get('timeframe', '5m')
            
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Parameters for the notebook
        params = {
            'run_dir': str(run_dir.resolve()),  # Use absolute path to avoid duplication
            'config_name': config.get('name', 'unnamed'),
            'symbols': symbols,
            'timeframe': timeframe,
        }
        
        # Add analysis-specific parameters only for signal_analysis template
        if template_override == 'signal_analysis':
            params.update({
                # Global traces directory for signal analysis
                'global_traces_dir': str(Path.cwd() / 'traces'),
                # Analysis parameters (can be overridden by config)
                'min_strategies_to_analyze': config.get('analysis', {}).get('min_strategies', 20),
                'sharpe_threshold': config.get('analysis', {}).get('sharpe_threshold', 1.0),
                'correlation_threshold': config.get('analysis', {}).get('correlation_threshold', 0.7),
                'top_n_strategies': config.get('analysis', {}).get('top_n', 10),
                'ensemble_size': config.get('analysis', {}).get('ensemble_size', 5),
                'calculate_all_performance': config.get('analysis', {}).get('calculate_all', True),
                'performance_limit': config.get('analysis', {}).get('performance_limit', 100)
            })
        
        # Output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_notebook = run_dir / f"analysis_{timestamp}.ipynb"
        
        # Create data symlink in run directory for easier access
        try:
            data_symlink = run_dir / 'data'
            if not data_symlink.exists():
                # Try to find project root with data directory
                for parent in run_dir.parents:
                    if (parent / 'data').exists():
                        data_symlink.symlink_to(parent / 'data')
                        logger.info(f"Created symlink to data directory: {parent / 'data'}")
                        break
        except Exception as e:
            logger.debug(f"Could not create data symlink: {e}")
        
        try:
            if execute:
                logger.info(f"📓 Executing analysis notebook with papermill...")
                logger.info(f"   Template: {template}")
                logger.info(f"   Output: {output_notebook}")
                
                # Execute notebook
                pm.execute_notebook(
                    str(template),
                    str(output_notebook),
                    parameters=params,
                    kernel_name='admfpc-venv',  # Use the venv kernel
                    progress_bar=True
                )
                
                logger.info(f"✅ Notebook execution complete: {output_notebook}")
                
                # Generate HTML report if requested
                if generate_html:
                    html_path = self._generate_html_report(output_notebook)
                    if html_path:
                        logger.info(f"📄 HTML report generated: {html_path}")
                        
            else:
                logger.info(f"📓 Creating parameterized notebook...")
                
                # Read the template notebook
                import nbformat
                nb = nbformat.read(str(template), as_version=4)
                
                # Find and replace the parameters cell
                param_cell_idx = None
                for i, cell in enumerate(nb.cells):
                    if cell.cell_type == 'code' and ('parameters' in cell.source.lower() or 
                                                      'tags' in cell.metadata and 'parameters' in cell.metadata.get('tags', [])):
                        param_cell_idx = i
                        break
                
                # Create new parameters cell
                param_cell = nbformat.v4.new_code_cell(
                    source=f"# Parameters (auto-generated)\n" + 
                           "\n".join([f"{k} = {repr(v)}" for k, v in params.items()]),
                    metadata={"tags": ["parameters"]}
                )
                
                if param_cell_idx is not None:
                    # Replace existing parameters cell
                    nb.cells[param_cell_idx] = param_cell
                else:
                    # Insert after first cell if no parameters cell found
                    nb.cells.insert(1, param_cell)
                
                # Write the parameterized notebook
                nbformat.write(nb, str(output_notebook))
                
                logger.info(f"✅ Notebook parameterized: {output_notebook}")
            
            # Launch Jupyter if requested
            if launch:
                self._launch_notebook(output_notebook)
                
            return output_notebook
            
        except Exception as e:
            logger.error(f"Failed to run notebook: {e}", exc_info=True)
            return None
    
    def run_supplementary_analysis(self,
                                 run_dir: Path,
                                 template_name: str,
                                 params: Optional[Dict] = None,
                                 execute: bool = True) -> Optional[Path]:
        """
        Run supplementary analysis notebooks (correlation, regime, etc).
        
        Args:
            run_dir: Results directory
            template_name: Name of template (without .ipynb)
            params: Additional parameters
            execute: Whether to execute
            
        Returns:
            Path to generated notebook
        """
        template = self.template_dir / f"{template_name}.ipynb"
        
        if not template.exists():
            logger.error(f"Supplementary template not found: {template}")
            return None
            
        # Base parameters
        notebook_params = {
            'run_dir': str(run_dir),
            **(params or {})
        }
        
        # Output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_notebook = run_dir / f"{template_name}_{timestamp}.ipynb"
        
        try:
            if execute:
                pm.execute_notebook(
                    str(template),
                    str(output_notebook),
                    parameters=notebook_params,
                    kernel_name='admfpc-venv'  # Use the venv kernel
                )
            else:
                pm.parameterize_notebook(
                    str(template),
                    str(output_notebook),
                    parameters=notebook_params
                )
                
            logger.info(f"✅ Generated {template_name}: {output_notebook}")
            return output_notebook
            
        except Exception as e:
            logger.error(f"Failed to run {template_name}: {e}")
            return None
    
    def _generate_html_report(self, notebook_path: Path) -> Optional[Path]:
        """Convert executed notebook to HTML report"""
        try:
            html_path = notebook_path.with_suffix('.html')
            
            # Use nbconvert to generate HTML
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'html',
                '--no-input',  # Hide code cells
                '--output', str(html_path),
                str(notebook_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML: {e}")
            return None
    
    def _launch_notebook(self, notebook_path: Path):
        """Launch Jupyter with the notebook"""
        try:
            # Try jupyter lab first (non-blocking with Popen)
            subprocess.Popen(['jupyter', 'lab', str(notebook_path)])
            logger.info("🚀 Jupyter Lab launched in background")
        except FileNotFoundError:
            try:
                # Fall back to classic notebook (non-blocking with Popen)
                subprocess.Popen(['jupyter', 'notebook', str(notebook_path)])
                logger.info("🚀 Jupyter Notebook launched in background")
            except Exception as e:
                logger.error(f"Could not launch Jupyter: {e}")
                logger.info(f"You can manually open: jupyter lab {notebook_path}")
    
    def run_global_analysis(self,
                           strategy_type: Optional[str] = None,
                           symbol: Optional[str] = None,
                           timeframe: Optional[str] = None,
                           execute: bool = True,
                           launch: bool = False,
                           generate_html: bool = False,
                           output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Run global analysis directly from traces store without run directory.
        
        Args:
            strategy_type: Filter by strategy type (e.g., 'bollinger_bands')
            symbol: Filter by symbol (e.g., 'SPY')
            timeframe: Filter by timeframe (e.g., '5m')
            execute: Whether to execute the notebook
            launch: Whether to launch Jupyter after generation
            generate_html: Whether to generate HTML report
            output_dir: Where to save the analysis (defaults to current directory)
            
        Returns:
            Path to the generated notebook
        """
        # Use canonical signal_analysis template
        template = self.template_dir / 'signal_analysis.ipynb'
        
        if not template.exists():
            logger.error(f"Global analysis template not found: {template}")
            return None
        
        # Parameters for the notebook
        params = {
            'strategy_type': strategy_type,
            'symbol': symbol or 'SPY',
            'timeframe': timeframe or '5m',
            'traces_dir': str(Path.cwd() / 'traces'),
            'execution_cost_bps': 1.0  # Default execution cost
        }
        
        # Output path
        if output_dir is None:
            output_dir = Path.cwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build filename
        filename_parts = ['global_analysis']
        if strategy_type:
            filename_parts.append(strategy_type)
        if symbol:
            filename_parts.append(symbol)
        if timeframe:
            filename_parts.append(timeframe)
        filename_parts.append(timestamp)
        
        output_notebook = output_dir / f"{'_'.join(filename_parts)}.ipynb"
        
        try:
            if execute:
                logger.info(f"📓 Executing global analysis notebook...")
                logger.info(f"   Strategy type: {strategy_type or 'ALL'}")
                logger.info(f"   Symbol: {symbol or 'ALL'}")
                logger.info(f"   Timeframe: {timeframe or 'ALL'}")
                logger.info(f"   Output: {output_notebook}")
                
                # Execute notebook
                pm.execute_notebook(
                    str(template),
                    str(output_notebook),
                    parameters=params,
                    kernel_name='admfpc-venv',  # Use the venv kernel
                    progress_bar=True
                )
                
                logger.info(f"✅ Analysis complete: {output_notebook}")
                
                # Generate HTML if requested
                if generate_html:
                    html_path = self._generate_html_report(output_notebook)
                    if html_path:
                        logger.info(f"📄 HTML report generated: {html_path}")
                        
            else:
                logger.info(f"📓 Creating parameterized notebook...")
                
                # Just parameterize without executing
                pm.parameterize_notebook(
                    str(template),
                    str(output_notebook),
                    parameters=params
                )
                
                logger.info(f"✅ Notebook created: {output_notebook}")
            
            # Launch if requested
            if launch:
                self._launch_notebook(output_notebook)
                
            return output_notebook
            
        except Exception as e:
            logger.error(f"Failed to run global analysis: {e}", exc_info=True)
            return None
    
    def _create_default_template(self):
        """Create a minimal default template if none exists"""
        # This is a fallback - the real template should be in templates/
        minimal_template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Analysis Template\n\nThis is a minimal template. Please install the full template."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"tags": ["parameters"]},
                    "outputs": [],
                    "source": ["# parameters\nrun_dir = '.'\nconfig_name = 'test'"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        template_path = self.template_dir / "signal_analysis.ipynb"
        with open(template_path, 'w') as f:
            json.dump(minimal_template, f, indent=2)