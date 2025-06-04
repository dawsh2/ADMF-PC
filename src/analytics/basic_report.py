"""
Basic HTML Report Generator for ADMF-PC Backtest Results

This module creates static HTML reports from backtest results stored in the 
workspace management system.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime
import logging


class BacktestReportGenerator:
    """Generate HTML reports from backtest results"""
    
    def __init__(self, workspace_path: Path):
        self.workspace = Path(workspace_path)
        self.logger = logging.getLogger(__name__)
        
        # Validate workspace structure
        if not self.workspace.exists():
            raise ValueError(f"Workspace does not exist: {workspace_path}")
            
    def generate_report(self) -> Path:
        """Generate comprehensive backtest report"""
        self.logger.info(f"Generating report for workspace: {self.workspace}")
        
        try:
            # Load data
            metrics = self._load_performance_metrics()
            signals = self._load_signals()
            metadata = self._load_metadata()
            
            # Generate charts
            charts = self._generate_charts(metrics, signals)
            
            # Create HTML report
            html_content = self._create_html_report(metrics, charts, metadata)
            
            # Save report
            report_path = self.workspace / 'visualizations' / 'report.html'
            report_path.parent.mkdir(exist_ok=True)
            report_path.write_text(html_content)
            
            self.logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
    
    def _load_performance_metrics(self) -> Dict[str, Any]:
        """Load performance metrics from workspace"""
        # First try to load from actual backtest data
        backtest_file = self.workspace / 'backtest_data.json'
        if backtest_file.exists():
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)
                return self._calculate_metrics_from_backtest_data(backtest_data)
        
        # Fallback to performance directory
        performance_dir = self.workspace / 'performance'
        if performance_dir.exists():
            # Look for performance files
            performance_files = list(performance_dir.glob('*.json'))
            if performance_files:
                # Load the first performance file
                with open(performance_files[0]) as f:
                    return json.load(f)
        
        # Return actual empty state instead of fake data
        return self._create_empty_metrics()
    
    def _load_signals(self) -> List[Dict[str, Any]]:
        """Load signals from workspace"""
        signals_dir = self.workspace / 'signals'
        
        if not signals_dir.exists():
            return []
            
        signals = []
        for signal_file in signals_dir.glob('*.jsonl'):
            with open(signal_file) as f:
                for line in f:
                    signals.append(json.loads(line.strip()))
        
        return signals
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load workflow metadata"""
        metadata_dir = self.workspace / 'metadata'
        
        if metadata_dir.exists():
            config_file = metadata_dir / 'config.yaml'
            if config_file.exists():
                # Would normally parse YAML, but for now return basic info
                return {
                    'strategy_name': self.workspace.name,
                    'generated_at': datetime.now().isoformat(),
                    'workspace_path': str(self.workspace)
                }
        
        return {
            'strategy_name': self.workspace.name,
            'generated_at': datetime.now().isoformat(),
            'workspace_path': str(self.workspace)
        }
    
    def _generate_charts(self, metrics: Dict[str, Any], signals: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate Plotly charts as JSON"""
        charts = {}
        
        # Equity curve chart
        charts['equity_curve'] = self._create_equity_curve_chart(metrics)
        
        # Drawdown chart
        charts['drawdown'] = self._create_drawdown_chart(metrics)
        
        # Returns distribution
        charts['returns_dist'] = self._create_returns_distribution(metrics)
        
        # Signal timeline (if signals available)
        if signals:
            charts['signal_timeline'] = self._create_signal_timeline(signals)
        
        return charts
    
    def _create_equity_curve_chart(self, metrics: Dict[str, Any]) -> str:
        """Create equity curve chart"""
        # Use actual backtest data
        trades = metrics.get('trades', [])
        bars_processed = metrics.get('bars_processed', 1)
        
        # Create realistic timeline based on actual bars processed
        dates = pd.date_range('2024-01-01', periods=max(1, bars_processed), freq='1min')
        
        if not trades:
            # No trades - flat line at initial capital
            equity_curve = pd.Series([100000] * len(dates), index=dates)
        else:
            # Create equity curve showing the trade impact
            initial_capital = 100000
            equity_curve = pd.Series([initial_capital] * len(dates), index=dates)
            
            # Apply trade impact at the end (when trade occurred)
            for trade in trades:
                trade_cost = trade.get('price', 0) * trade.get('quantity', 0) + trade.get('commission', 0)
                equity_curve.iloc[-1] = initial_capital - trade_cost  # Cash after buying
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=400
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_drawdown_chart(self, metrics: Dict[str, Any]) -> str:
        """Create drawdown chart"""
        # Use actual bars processed for realistic timeline
        bars_processed = metrics.get('bars_processed', 1)
        dates = pd.date_range('2024-01-01', periods=max(1, bars_processed), freq='1min')
        
        # Minimal drawdown for our simple backtest (would calculate from equity curve in reality)
        drawdown = pd.Series([0.0] * len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=300
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_returns_distribution(self, metrics: Dict[str, Any]) -> str:
        """Create returns distribution histogram"""
        # Use actual trade data for returns if available
        trades = metrics.get('trades', [])
        
        if trades:
            # Calculate returns from actual trades (simplified)
            returns = []
            for trade in trades:
                trade_value = trade.get('price', 0) * trade.get('quantity', 0)
                trade_return = (trade_value / 100000) * 100  # As percentage of initial capital
                returns.append(trade_return)
            returns = pd.Series(returns)
        else:
            # No trades - show single zero return
            returns = pd.Series([0.0])
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=20,
            name='Daily Returns',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_signal_timeline(self, signals: List[Dict[str, Any]]) -> str:
        """Create signal timeline chart"""
        if not signals:
            return json.dumps({}, cls=PlotlyJSONEncoder)
            
        # Convert signals to DataFrame
        df = pd.DataFrame(signals)
        
        if 'timestamp' not in df.columns:
            return json.dumps({}, cls=PlotlyJSONEncoder)
        
        # Count signals by date
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        signal_counts = df.groupby(df['timestamp'].dt.date).size()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            name='Signal Count',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Trading Signals Timeline',
            xaxis_title='Date',
            yaxis_title='Number of Signals',
            template='plotly_white',
            height=300
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _calculate_metrics_from_backtest_data(self, backtest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from actual backtest data"""
        # Extract backtest data from the nested structure
        actual_backtest = backtest_data.get('backtest_data', {})
        
        trades = actual_backtest.get('trades', [])
        portfolio_value = actual_backtest.get('portfolio_value', 0)
        cash = actual_backtest.get('cash', 0)
        positions = actual_backtest.get('positions', {})
        bars_processed = actual_backtest.get('bars_processed', 0)
        
        # Calculate basic metrics
        total_trades = len(trades)
        
        # Calculate returns from trades
        total_pnl = 0
        winning_trades = 0
        
        for trade in trades:
            # Calculate cash impact of trade (negative for buys, positive for sells)
            trade_cost = trade.get('price', 0) * trade.get('quantity', 0) + trade.get('commission', 0)
            
            if trade.get('side') == 'OrderSide.BUY':
                cash_impact = -trade_cost  # Buying reduces cash
            else:  # SELL
                cash_impact = trade_cost  # Selling increases cash
            
            total_pnl += cash_impact
            
            # For now, don't count unrealized gains since we don't have position values
            # In a real system, winning_trades would be calculated from realized P&L
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade_return = (total_pnl / total_trades) if total_trades > 0 else 0
        
        # Portfolio metrics
        total_portfolio_value = portfolio_value + cash
        position_value = actual_backtest.get('position_value', 0)
        
        # Use portfolio value for return calculation if available, otherwise use cash impact
        initial_capital = 100000
        if total_portfolio_value > 0 or position_value > 0:
            # Portfolio return: (current_value - initial_capital) / initial_capital
            current_value = total_portfolio_value + position_value
            total_return_pct = ((current_value - initial_capital) / initial_capital) * 100
            total_pnl = current_value - initial_capital
        else:
            # Fallback to cash impact calculation
            total_return_pct = (total_pnl / initial_capital) * 100 if total_pnl != 0 else 0
        
        return {
            'total_return': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'portfolio_value': round(total_portfolio_value, 2),
            'cash': round(cash, 2),
            'sharpe_ratio': 0.0,  # Would need time series data to calculate
            'max_drawdown': 0.0,  # Would need equity curve to calculate
            'win_rate': round(win_rate, 1),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'avg_trade_return': round(avg_trade_return, 2),
            'bars_processed': bars_processed,
            'positions': positions,
            'trades': trades[:10],  # Include first 10 trades for display
            'generated_at': datetime.now().isoformat(),
        }
    
    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics when no data is available"""
        return {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'portfolio_value': 0.0,
            'cash': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_return': 0.0,
            'bars_processed': 0,
            'positions': {},
            'trades': [],
            'generated_at': datetime.now().isoformat(),
            'message': 'No backtest data available'
        }
    
    def _create_html_report(self, metrics: Dict[str, Any], charts: Dict[str, str], metadata: Dict[str, Any]) -> str:
        """Create complete HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 0.9em;
            font-weight: 500;
            margin-bottom: 8px;
            opacity: 0.9;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            margin: 0;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
            margin-top: 40px;
        }}
        .positive {{ color: #00c851; }}
        .negative {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Backtest Report</h1>
            <p>{strategy_name} | Generated on {generated_at}</p>
        </div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_class}">{total_return}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe_ratio}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{max_drawdown}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{profit_factor}</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-container">
            <div id="equity-curve-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="drawdown-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="returns-dist-chart"></div>
        </div>
        
        {signal_chart_section}
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by ADMF-PC Reporting System | Workspace: {workspace_path}</p>
        </div>
    </div>
    
    <script>
        // Render charts
        Plotly.newPlot('equity-curve-chart', {equity_curve_data});
        Plotly.newPlot('drawdown-chart', {drawdown_data});
        Plotly.newPlot('returns-dist-chart', {returns_dist_data});
        {signal_chart_script}
    </script>
</body>
</html>
        """
        
        # Prepare template variables
        return_class = 'positive' if metrics.get('total_return_pct', 0) > 0 else 'negative'
        
        signal_chart_section = ""
        signal_chart_script = ""
        
        if 'signal_timeline' in charts:
            signal_chart_section = """
        <div class="chart-container">
            <div id="signal-timeline-chart"></div>
        </div>
            """
            signal_chart_script = f"Plotly.newPlot('signal-timeline-chart', {charts['signal_timeline']});"
        
        return html_template.format(
            strategy_name=metadata.get('strategy_name', 'Unknown Strategy'),
            generated_at=datetime.fromisoformat(metadata.get('generated_at')).strftime('%B %d, %Y at %I:%M %p'),
            workspace_path=metadata.get('workspace_path', 'Unknown'),
            total_return=metrics.get('total_return_pct', 0),
            return_class=return_class,
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            win_rate=metrics.get('win_rate', 0),
            total_trades=metrics.get('total_trades', 0),
            profit_factor=metrics.get('profit_factor', 0),
            equity_curve_data=charts.get('equity_curve', '{}'),
            drawdown_data=charts.get('drawdown', '{}'),
            returns_dist_data=charts.get('returns_dist', '{}'),
            signal_chart_section=signal_chart_section,
            signal_chart_script=signal_chart_script
        )


def generate_report_from_workspace(workspace_path: str) -> Path:
    """Convenience function to generate report from workspace path"""
    generator = BacktestReportGenerator(workspace_path)
    return generator.generate_report()