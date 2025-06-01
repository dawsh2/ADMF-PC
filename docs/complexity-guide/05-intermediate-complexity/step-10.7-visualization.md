# Step 10.7: Advanced Visualization

**Status**: Intermediate Complexity Step  
**Complexity**: High  
**Prerequisites**: [Step 10.6: Custom Indicators](step-10.6-custom-indicators.md) completed  
**Architecture Ref**: [Visualization Architecture](../architecture/visualization-architecture.md)

## 🎯 Objective

Implement comprehensive visualization and dashboard system:
- Real-time market data visualization
- Interactive portfolio performance dashboards
- Risk monitoring and alert systems
- Multi-timeframe chart analysis
- Strategy performance attribution
- Custom visualization components

## 📋 Required Reading

Before starting:
1. [Financial Data Visualization Best Practices](../references/finviz-best-practices.md)
2. [Real-time Dashboard Design](../references/realtime-dashboard-design.md)
3. [Interactive Chart Libraries](../references/chart-libraries.md)

## 🏗️ Implementation Tasks

### 1. Core Visualization Framework

```python
# src/visualization/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

class ChartType(Enum):
    """Types of financial charts"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    VOLUME = "volume"
    INDICATOR = "indicator"
    PORTFOLIO = "portfolio"
    RISK = "risk"

class TimeFrame(Enum):
    """Chart timeframes"""
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"

@dataclass
class ChartConfig:
    """Configuration for chart appearance"""
    title: str = ""
    width: int = 800
    height: int = 400
    theme: str = "plotly_dark"
    
    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    background_color: str = "#2e2e2e"
    grid_color: str = "#444444"
    
    # Layout
    show_legend: bool = True
    show_grid: bool = True
    show_rangeslider: bool = False
    
    # Interactivity
    zoom_enabled: bool = True
    pan_enabled: bool = True
    crossfilter_enabled: bool = True

@dataclass
class ChartData:
    """Data container for charts"""
    data: pd.DataFrame
    x_column: str
    y_columns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional OHLCV columns
    open_column: Optional[str] = None
    high_column: Optional[str] = None
    low_column: Optional[str] = None
    close_column: Optional[str] = None
    volume_column: Optional[str] = None

class BaseChart(ABC):
    """Base class for all chart types"""
    
    def __init__(self, chart_type: ChartType, config: ChartConfig):
        self.chart_type = chart_type
        self.config = config
        self.figure = None
        self.logger = ComponentLogger(f"Chart_{chart_type.value}", "visualization")
    
    @abstractmethod
    def create_chart(self, data: ChartData) -> go.Figure:
        """Create the chart figure"""
        pass
    
    def update_chart(self, data: ChartData) -> go.Figure:
        """Update existing chart with new data"""
        # Default implementation recreates chart
        return self.create_chart(data)
    
    def add_annotation(self, text: str, x: Any, y: Any, 
                      arrow: bool = True) -> None:
        """Add annotation to chart"""
        if self.figure is None:
            return
        
        self.figure.add_annotation(
            text=text,
            x=x,
            y=y,
            showarrow=arrow,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.config.primary_color,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=self.config.primary_color,
            borderwidth=1
        )
    
    def add_shape(self, shape_type: str, x0: Any, y0: Any, 
                  x1: Any, y1: Any, **kwargs) -> None:
        """Add shape to chart"""
        if self.figure is None:
            return
        
        self.figure.add_shape(
            type=shape_type,
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=self.config.secondary_color, width=2),
            **kwargs
        )
    
    def apply_theme(self) -> None:
        """Apply theme to chart"""
        if self.figure is None:
            return
        
        self.figure.update_layout(
            template=self.config.theme,
            paper_bgcolor=self.config.background_color,
            plot_bgcolor=self.config.background_color,
            font=dict(color="white" if "dark" in self.config.theme else "black"),
            title=dict(text=self.config.title, x=0.5),
            showlegend=self.config.show_legend,
            width=self.config.width,
            height=self.config.height
        )
        
        # Grid configuration
        self.figure.update_xaxes(
            showgrid=self.config.show_grid,
            gridcolor=self.config.grid_color,
            zeroline=False
        )
        self.figure.update_yaxes(
            showgrid=self.config.show_grid,
            gridcolor=self.config.grid_color,
            zeroline=False
        )

class FinancialChart(BaseChart):
    """Financial candlestick/OHLC chart"""
    
    def __init__(self, config: ChartConfig):
        super().__init__(ChartType.CANDLESTICK, config)
    
    def create_chart(self, data: ChartData) -> go.Figure:
        """Create candlestick chart"""
        df = data.data
        
        if not all(col in df.columns for col in [
            data.open_column, data.high_column, 
            data.low_column, data.close_column
        ]):
            raise ValueError("Missing OHLC columns for candlestick chart")
        
        # Create candlestick
        candlestick = go.Candlestick(
            x=df[data.x_column],
            open=df[data.open_column],
            high=df[data.high_column],
            low=df[data.low_column],
            close=df[data.close_column],
            name="Price",
            increasing_line_color=self.config.primary_color,
            decreasing_line_color=self.config.secondary_color
        )
        
        # Create subplot with volume if available
        if data.volume_column and data.volume_column in df.columns:
            self.figure = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price', 'Volume'),
                row_width=[0.2, 0.7]
            )
            
            # Add candlestick
            self.figure.add_trace(candlestick, row=1, col=1)
            
            # Add volume
            colors = ['red' if df[data.close_column].iloc[i] < df[data.open_column].iloc[i] 
                     else 'green' for i in range(len(df))]
            
            self.figure.add_trace(
                go.Bar(
                    x=df[data.x_column],
                    y=df[data.volume_column],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        else:
            self.figure = go.Figure(data=[candlestick])
        
        # Apply theme and configuration
        self.apply_theme()
        
        # Add range slider if configured
        if self.config.show_rangeslider:
            self.figure.update_layout(xaxis_rangeslider_visible=True)
        
        return self.figure

class LineChart(BaseChart):
    """Multi-line chart for time series"""
    
    def __init__(self, config: ChartConfig):
        super().__init__(ChartType.LINE, config)
    
    def create_chart(self, data: ChartData) -> go.Figure:
        """Create line chart"""
        df = data.data
        
        self.figure = go.Figure()
        
        # Add lines for each y column
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(data.y_columns):
            if col not in df.columns:
                continue
            
            self.figure.add_trace(
                go.Scatter(
                    x=df[data.x_column],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{col}: %{{y:.4f}}<br>Time: %{{x}}<extra></extra>'
                )
            )
        
        self.apply_theme()
        
        return self.figure

class HeatmapChart(BaseChart):
    """Correlation/covariance heatmap"""
    
    def __init__(self, config: ChartConfig):
        super().__init__(ChartType.HEATMAP, config)
    
    def create_chart(self, data: ChartData) -> go.Figure:
        """Create heatmap chart"""
        df = data.data
        
        # Assume data is correlation matrix
        self.figure = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(df.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        self.apply_theme()
        
        return self.figure

class ChartFactory:
    """Factory for creating charts"""
    
    @staticmethod
    def create_chart(chart_type: ChartType, config: ChartConfig) -> BaseChart:
        """Create chart of specified type"""
        
        if chart_type == ChartType.CANDLESTICK:
            return FinancialChart(config)
        elif chart_type == ChartType.LINE:
            return LineChart(config)
        elif chart_type == ChartType.HEATMAP:
            return HeatmapChart(config)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
```

### 2. Real-time Dashboard Components

```python
# src/visualization/dashboard.py
class RealTimeDashboard:
    """Real-time trading dashboard"""
    
    def __init__(self, title: str = "Trading Dashboard"):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.title = title
        self.components = {}
        self.data_sources = {}
        self.update_intervals = {}
        
        # Layout components
        self.header = None
        self.sidebar = None
        self.main_content = None
        
        self.logger = ComponentLogger("RealTimeDashboard", "visualization")
        
        # Setup layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        
        # Header
        self.header = dbc.NavbarSimple(
            brand=self.title,
            brand_href="#",
            color="dark",
            dark=True,
            fluid=True,
            children=[
                dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                dbc.NavItem(dbc.NavLink("Analytics", href="#")),
                dbc.NavItem(dbc.NavLink("Risk", href="#")),
            ]
        )
        
        # Sidebar
        self.sidebar = dbc.Card([
            dbc.CardHeader("Controls"),
            dbc.CardBody([
                dbc.Label("Asset Selection"),
                dcc.Dropdown(
                    id="asset-dropdown",
                    options=[
                        {"label": "SPY", "value": "SPY"},
                        {"label": "QQQ", "value": "QQQ"},
                        {"label": "IWM", "value": "IWM"},
                    ],
                    value="SPY",
                    multi=True
                ),
                html.Br(),
                
                dbc.Label("Timeframe"),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[
                        {"label": "1 Minute", "value": "1min"},
                        {"label": "5 Minutes", "value": "5min"},
                        {"label": "1 Hour", "value": "1h"},
                        {"label": "1 Day", "value": "1d"},
                    ],
                    value="5min"
                ),
                html.Br(),
                
                dbc.Label("Indicators"),
                dcc.Checklist(
                    id="indicator-checklist",
                    options=[
                        {"label": "Moving Average", "value": "ma"},
                        {"label": "RSI", "value": "rsi"},
                        {"label": "Bollinger Bands", "value": "bb"},
                        {"label": "Volume", "value": "volume"},
                    ],
                    value=["ma", "volume"]
                ),
                html.Br(),
                
                dbc.Button(
                    "Start Live Feed",
                    id="live-feed-btn",
                    color="success",
                    className="me-1"
                ),
                html.Span(id="status-indicator", className="ms-2")
            ])
        ], style={"height": "100vh"})
        
        # Main content area
        self.main_content = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="price-chart", style={"height": "500px"})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id="indicator-chart", style={"height": "250px"}),
                    dcc.Graph(id="volume-chart", style={"height": "250px"})
                ], width=4)
            ]),
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="portfolio-chart", style={"height": "300px"})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="risk-chart", style={"height": "300px"})
                ], width=6)
            ]),
            html.Br(),
            
            # Performance metrics table
            dbc.Row([
                dbc.Col([
                    html.H4("Performance Metrics"),
                    html.Div(id="metrics-table")
                ], width=12)
            ])
        ], fluid=True)
        
        # Complete layout
        self.app.layout = dbc.Container([
            self.header,
            html.Br(),
            dbc.Row([
                dbc.Col(self.sidebar, width=3),
                dbc.Col(self.main_content, width=9)
            ])
        ], fluid=True)
        
        # Auto-refresh intervals
        self.app.layout.children.append(
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        )
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('indicator-chart', 'figure'),
             Output('volume-chart', 'figure')],
            [Input('asset-dropdown', 'value'),
             Input('timeframe-dropdown', 'value'),
             Input('indicator-checklist', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_charts(selected_assets, timeframe, indicators, n):
            return self._update_price_charts(selected_assets, timeframe, indicators)
        
        @self.app.callback(
            [Output('portfolio-chart', 'figure'),
             Output('risk-chart', 'figure'),
             Output('metrics-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_charts(n):
            return self._update_portfolio_charts()
        
        @self.app.callback(
            Output('status-indicator', 'children'),
            [Input('live-feed-btn', 'n_clicks')]
        )
        def toggle_live_feed(n_clicks):
            if n_clicks:
                return dbc.Badge("LIVE", color="success", pill=True)
            return dbc.Badge("PAUSED", color="secondary", pill=True)
    
    def _update_price_charts(self, assets, timeframe, indicators):
        """Update price and indicator charts"""
        
        # Generate sample data (replace with real data source)
        if isinstance(assets, str):
            assets = [assets]
        
        price_fig = go.Figure()
        indicator_fig = go.Figure()
        volume_fig = go.Figure()
        
        for asset in assets:
            # Sample price data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
                'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
                'low': prices * (1 + np.random.uniform(-0.01, 0, len(dates))),
                'close': prices,
                'volume': np.random.uniform(1e6, 1e7, len(dates))
            })
            
            # Price chart (candlestick)
            price_fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=asset
            ))
            
            # Add indicators
            if 'ma' in indicators:
                ma_20 = df['close'].rolling(20).mean()
                price_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=ma_20,
                    mode='lines',
                    name=f'{asset} MA(20)',
                    line=dict(width=2)
                ))
            
            if 'bb' in indicators:
                ma_20 = df['close'].rolling(20).mean()
                std_20 = df['close'].rolling(20).std()
                upper_bb = ma_20 + 2 * std_20
                lower_bb = ma_20 - 2 * std_20
                
                price_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=upper_bb,
                    mode='lines',
                    name=f'{asset} BB Upper',
                    line=dict(dash='dash', width=1)
                ))
                price_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=lower_bb,
                    mode='lines',
                    name=f'{asset} BB Lower',
                    line=dict(dash='dash', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.1)'
                ))
            
            # RSI indicator
            if 'rsi' in indicators:
                # Simple RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                indicator_fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=rsi,
                    mode='lines',
                    name=f'{asset} RSI',
                    line=dict(width=2)
                ))
            
            # Volume chart
            if 'volume' in indicators:
                volume_fig.add_trace(go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name=f'{asset} Volume',
                    opacity=0.7
                ))
        
        # Update layouts
        price_fig.update_layout(
            title="Price Chart",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            showlegend=True
        )
        
        indicator_fig.update_layout(
            title="RSI",
            xaxis_title="Time",
            yaxis_title="RSI",
            template="plotly_dark",
            yaxis=dict(range=[0, 100])
        )
        
        # Add RSI levels
        indicator_fig.add_hline(y=70, line_dash="dash", line_color="red")
        indicator_fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        volume_fig.update_layout(
            title="Volume",
            xaxis_title="Time",
            yaxis_title="Volume",
            template="plotly_dark"
        )
        
        return price_fig, indicator_fig, volume_fig
    
    def _update_portfolio_charts(self):
        """Update portfolio and risk charts"""
        
        # Sample portfolio data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        portfolio_value = 100000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.001))
        
        # Portfolio performance chart
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_value,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=3)
        ))
        
        # Add benchmark
        benchmark_value = 100000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.0008))
        portfolio_fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_value,
            mode='lines',
            name='Benchmark',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        portfolio_fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_dark"
        )
        
        # Risk metrics chart (drawdown)
        cummax = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - cummax) / cummax * 100
        
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        risk_fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark"
        )
        
        # Performance metrics table
        metrics = {
            'Total Return': f"{(portfolio_value[-1] / portfolio_value[0] - 1) * 100:.2f}%",
            'Sharpe Ratio': f"{np.random.uniform(0.5, 2.0):.2f}",
            'Max Drawdown': f"{drawdown.min():.2f}%",
            'Volatility': f"{np.std(np.diff(portfolio_value) / portfolio_value[:-1]) * np.sqrt(252) * 100:.2f}%",
            'Win Rate': f"{np.random.uniform(45, 65):.1f}%"
        }
        
        metrics_table = dbc.Table.from_dataframe(
            pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']),
            striped=True,
            bordered=True,
            hover=True,
            dark=True
        )
        
        return portfolio_fig, risk_fig, metrics_table
    
    def run(self, debug=True, host='0.0.0.0', port=8050):
        """Run the dashboard"""
        self.logger.info(f"Starting dashboard on http://{host}:{port}")
        self.app.run_server(debug=debug, host=host, port=port)
```

### 3. Advanced Chart Components

```python
# src/visualization/advanced_charts.py
class StrategyPerformanceChart:
    """Specialized chart for strategy performance analysis"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
        self.logger = ComponentLogger("StrategyPerformanceChart", "visualization")
    
    def create_performance_attribution_chart(self, 
                                           performance_data: pd.DataFrame,
                                           attribution_data: pd.DataFrame) -> go.Figure:
        """Create performance attribution chart"""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Cumulative Returns', 'Attribution', 'Drawdown'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=performance_data.index,
                y=performance_data['cumulative_return'],
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        if 'benchmark_return' in performance_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data['benchmark_return'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Attribution breakdown
        attribution_sources = attribution_data.columns
        colors = px.colors.qualitative.Set1
        
        for i, source in enumerate(attribution_sources):
            fig.add_trace(
                go.Bar(
                    x=attribution_data.index,
                    y=attribution_data[source],
                    name=source,
                    marker_color=colors[i % len(colors)]
                ),
                row=2, col=1
            )
        
        # Drawdown
        if 'drawdown' in performance_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Strategy Performance Attribution",
            template=self.config.theme,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_risk_decomposition_chart(self, risk_data: pd.DataFrame) -> go.Figure:
        """Create risk decomposition chart"""
        
        # Risk factors
        factors = risk_data.columns
        values = risk_data.iloc[-1]  # Latest values
        
        # Create sunburst chart for risk decomposition
        fig = go.Figure(go.Sunburst(
            labels=factors,
            values=values,
            parents=[""] * len(factors),
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Risk Decomposition",
            template=self.config.theme,
            height=500
        )
        
        return fig

class MultiAssetCorrelationChart:
    """Chart for multi-asset correlation analysis"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def create_correlation_matrix(self, correlation_data: pd.DataFrame) -> go.Figure:
        """Create interactive correlation matrix"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            template=self.config.theme,
            height=600,
            width=600
        )
        
        return fig
    
    def create_rolling_correlation_chart(self, 
                                       correlation_time_series: pd.DataFrame,
                                       asset_pair: Tuple[str, str]) -> go.Figure:
        """Create rolling correlation chart for asset pair"""
        
        asset1, asset2 = asset_pair
        column_name = f"{asset1}_{asset2}"
        
        if column_name not in correlation_time_series.columns:
            # Try reverse order
            column_name = f"{asset2}_{asset1}"
        
        if column_name not in correlation_time_series.columns:
            raise ValueError(f"Correlation data not found for {asset_pair}")
        
        fig = go.Figure()
        
        # Rolling correlation
        fig.add_trace(go.Scatter(
            x=correlation_time_series.index,
            y=correlation_time_series[column_name],
            mode='lines',
            name=f'{asset1}-{asset2} Correlation',
            line=dict(width=2)
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", 
                     annotation_text="High Correlation")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="orange",
                     annotation_text="High Negative Correlation")
        
        fig.update_layout(
            title=f"Rolling Correlation: {asset1} vs {asset2}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            template=self.config.theme,
            yaxis=dict(range=[-1, 1])
        )
        
        return fig

class RiskMonitoringChart:
    """Real-time risk monitoring visualizations"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def create_var_chart(self, var_data: pd.DataFrame) -> go.Figure:
        """Create Value at Risk monitoring chart"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Portfolio P&L', 'VaR Limits'),
            vertical_spacing=0.1
        )
        
        # Portfolio P&L
        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data['portfolio_pnl'],
                mode='lines',
                name='Portfolio P&L',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # VaR limits
        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data['var_95'],
                mode='lines',
                name='95% VaR',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=var_data.index,
                y=var_data['var_99'],
                mode='lines',
                name='99% VaR',
                line=dict(color='darkred', width=2, dash='dot')
            ),
            row=2, col=1
        )
        
        # Highlight breaches
        breaches = var_data[var_data['portfolio_pnl'] < var_data['var_95']]
        if not breaches.empty:
            fig.add_trace(
                go.Scatter(
                    x=breaches.index,
                    y=breaches['portfolio_pnl'],
                    mode='markers',
                    name='VaR Breaches',
                    marker=dict(color='red', size=10, symbol='x')
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title="Value at Risk Monitoring",
            template=self.config.theme,
            height=600
        )
        
        return fig
    
    def create_risk_dashboard(self, risk_metrics: Dict[str, float]) -> go.Figure:
        """Create risk dashboard with gauges"""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Portfolio VaR', 'Sharpe Ratio', 
                          'Max Drawdown', 'Beta')
        )
        
        # VaR Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get('var_95', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "95% VaR (%)"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 5
                    }
                }
            ),
            row=1, col=1
        )
        
        # Sharpe Ratio Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get('sharpe_ratio', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [-2, 0], 'color': "red"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 3], 'color': "green"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Max Drawdown Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(risk_metrics.get('max_drawdown', 0)),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Max Drawdown (%)"},
                gauge={
                    'axis': {'range': [0, 30]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 5], 'color': "green"},
                        {'range': [5, 15], 'color': "yellow"},
                        {'range': [15, 30], 'color': "red"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Beta Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_metrics.get('beta', 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Beta"},
                gauge={
                    'axis': {'range': [0, 2]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightblue"},
                        {'range': [0.5, 1.5], 'color': "blue"},
                        {'range': [1.5, 2], 'color': "darkblue"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Risk Dashboard",
            template=self.config.theme,
            height=600
        )
        
        return fig
```

### 4. Custom Visualization Components

```python
# src/visualization/custom_components.py
class TradingSignalChart:
    """Specialized chart for trading signals visualization"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def create_signal_chart(self, price_data: pd.DataFrame,
                           signals: pd.DataFrame) -> go.Figure:
        """Create chart with trading signals overlaid"""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Price & Signals', 'Signal Strength', 'P&L'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = signals[signals['direction'] == 'BUY']
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green'
                    )
                ),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = signals[signals['direction'] == 'SELL']
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red'
                    )
                ),
                row=1, col=1
            )
        
        # Signal strength
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['strength'],
                mode='lines',
                name='Signal Strength',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # P&L
        if 'pnl' in signals.columns:
            cumulative_pnl = signals['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Trading Signals Analysis",
            template=self.config.theme,
            height=800,
            showlegend=True
        )
        
        return fig

class PortfolioAllocationChart:
    """Portfolio allocation and rebalancing visualization"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def create_allocation_pie_chart(self, allocations: Dict[str, float]) -> go.Figure:
        """Create portfolio allocation pie chart"""
        
        assets = list(allocations.keys())
        weights = list(allocations.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=weights,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            template=self.config.theme,
            annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_allocation_timeline(self, allocation_history: pd.DataFrame) -> go.Figure:
        """Create stacked area chart of allocation over time"""
        
        fig = go.Figure()
        
        assets = allocation_history.columns
        colors = px.colors.qualitative.Set1
        
        # Create stacked area chart
        for i, asset in enumerate(assets):
            fig.add_trace(go.Scatter(
                x=allocation_history.index,
                y=allocation_history[asset],
                mode='lines',
                stackgroup='one',
                name=asset,
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title="Portfolio Allocation Over Time",
            xaxis_title="Date",
            yaxis_title="Weight",
            template=self.config.theme,
            yaxis=dict(range=[0, 1])
        )
        
        return fig

class VisualizationManager:
    """Manages all visualization components"""
    
    def __init__(self):
        self.charts = {}
        self.dashboards = {}
        self.config = ChartConfig()
        self.logger = ComponentLogger("VisualizationManager", "visualization")
    
    def create_dashboard(self, name: str, **kwargs) -> RealTimeDashboard:
        """Create and register a dashboard"""
        dashboard = RealTimeDashboard(title=name, **kwargs)
        self.dashboards[name] = dashboard
        return dashboard
    
    def create_chart(self, chart_type: ChartType, name: str, 
                    config: Optional[ChartConfig] = None) -> BaseChart:
        """Create and register a chart"""
        if config is None:
            config = self.config
        
        chart = ChartFactory.create_chart(chart_type, config)
        self.charts[name] = chart
        return chart
    
    def export_chart(self, chart_name: str, filepath: str, 
                    format: str = 'png', **kwargs) -> None:
        """Export chart to file"""
        if chart_name not in self.charts:
            raise ValueError(f"Chart '{chart_name}' not found")
        
        chart = self.charts[chart_name]
        if chart.figure is None:
            raise ValueError(f"Chart '{chart_name}' has no figure to export")
        
        if format.lower() == 'html':
            chart.figure.write_html(filepath, **kwargs)
        elif format.lower() in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            chart.figure.write_image(filepath, format=format, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Chart '{chart_name}' exported to {filepath}")
    
    def get_dashboard_url(self, dashboard_name: str) -> str:
        """Get URL for dashboard"""
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard '{dashboard_name}' not found")
        
        return f"http://localhost:8050"  # Default Dash port
```

### 5. Testing Framework

```python
# tests/unit/test_step10_7_visualization.py
class TestVisualizationFramework:
    """Test visualization components"""
    
    def test_chart_creation(self):
        """Test basic chart creation"""
        config = ChartConfig(title="Test Chart", theme="plotly_white")
        chart = LineChart(config)
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'price': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        chart_data = ChartData(
            data=data,
            x_column='date',
            y_columns=['price']
        )
        
        figure = chart.create_chart(chart_data)
        
        assert figure is not None
        assert len(figure.data) == 1
        assert figure.data[0].name == 'price'
    
    def test_candlestick_chart(self):
        """Test candlestick chart creation"""
        config = ChartConfig()
        chart = FinancialChart(config)
        
        # Create OHLCV data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.randn(50) * 0.01)
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 50)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 50)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, 50)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 50)
        })
        
        chart_data = ChartData(
            data=data,
            x_column='date',
            y_columns=[],
            open_column='open',
            high_column='high',
            low_column='low',
            close_column='close',
            volume_column='volume'
        )
        
        figure = chart.create_chart(chart_data)
        
        assert figure is not None
        # Should have candlestick and volume
        assert len(figure.data) >= 1

class TestDashboardComponents:
    """Test dashboard functionality"""
    
    def test_dashboard_initialization(self):
        """Test dashboard setup"""
        dashboard = RealTimeDashboard("Test Dashboard")
        
        assert dashboard.app is not None
        assert dashboard.title == "Test Dashboard"
        assert dashboard.header is not None
        assert dashboard.sidebar is not None
    
    def test_chart_factory(self):
        """Test chart factory"""
        config = ChartConfig()
        
        # Test different chart types
        line_chart = ChartFactory.create_chart(ChartType.LINE, config)
        assert isinstance(line_chart, LineChart)
        
        candlestick_chart = ChartFactory.create_chart(ChartType.CANDLESTICK, config)
        assert isinstance(candlestick_chart, FinancialChart)
        
        heatmap_chart = ChartFactory.create_chart(ChartType.HEATMAP, config)
        assert isinstance(heatmap_chart, HeatmapChart)

class TestAdvancedCharts:
    """Test advanced chart components"""
    
    def test_performance_attribution_chart(self):
        """Test performance attribution visualization"""
        config = ChartConfig()
        chart = StrategyPerformanceChart(config)
        
        # Create test performance data
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        performance_data = pd.DataFrame({
            'cumulative_return': np.cumprod(1 + np.random.normal(0.0005, 0.01, 252)) - 1,
            'benchmark_return': np.cumprod(1 + np.random.normal(0.0003, 0.008, 252)) - 1,
            'drawdown': np.random.uniform(-0.1, 0, 252)
        }, index=dates)
        
        attribution_data = pd.DataFrame({
            'alpha': np.random.normal(0.0002, 0.001, 252),
            'beta': np.random.normal(0.0003, 0.002, 252),
            'residual': np.random.normal(0, 0.001, 252)
        }, index=dates)
        
        figure = chart.create_performance_attribution_chart(
            performance_data, attribution_data
        )
        
        assert figure is not None
        assert len(figure.data) >= 3  # Multiple traces
    
    def test_correlation_matrix(self):
        """Test correlation matrix visualization"""
        config = ChartConfig()
        chart = MultiAssetCorrelationChart(config)
        
        # Create correlation matrix
        assets = ['SPY', 'TLT', 'GLD', 'VNQ']
        correlation_matrix = pd.DataFrame(
            np.random.uniform(0.1, 0.9, (4, 4)),
            index=assets,
            columns=assets
        )
        np.fill_diagonal(correlation_matrix.values, 1.0)
        
        figure = chart.create_correlation_matrix(correlation_matrix)
        
        assert figure is not None
        assert figure.data[0].type == 'heatmap'
```

## ✅ Validation Checklist

### Core Framework
- [ ] Chart factory working for all types
- [ ] Configuration system functional
- [ ] Theme application working
- [ ] Export functionality operational

### Dashboard Components
- [ ] Real-time updates working
- [ ] Interactive controls functional
- [ ] Multiple chart layouts supported
- [ ] Responsive design implemented

### Advanced Charts
- [ ] Performance attribution charts
- [ ] Risk monitoring visualizations
- [ ] Correlation analysis charts
- [ ] Signal visualization working

### Custom Components
- [ ] Trading signal overlays
- [ ] Portfolio allocation charts
- [ ] Custom indicators displayed
- [ ] Interactive features working

### Integration
- [ ] Dashboard serves successfully
- [ ] Charts update in real-time
- [ ] Export functions work
- [ ] Performance acceptable

## 📊 Performance Benchmarks

### Chart Rendering
- Simple charts: < 100ms
- Complex charts: < 500ms
- Dashboard load: < 2 seconds
- Real-time updates: < 50ms

### Memory Usage
- Dashboard: < 200MB
- Chart cache: < 100MB
- Update frequency: 5 seconds max
- Browser compatibility: Modern browsers

### User Experience
- Chart interaction: < 100ms response
- Zoom/pan: Smooth 60fps
- Data updates: Seamless
- Mobile responsive: Working

## 🐛 Common Issues

1. **Performance Degradation**
   - Limit data points displayed
   - Use data sampling for large datasets
   - Implement efficient update mechanisms
   - Cache expensive calculations

2. **Browser Compatibility**
   - Test across major browsers
   - Use fallbacks for older browsers
   - Optimize JavaScript bundle size
   - Handle memory limitations

3. **Real-time Updates**
   - Implement efficient WebSocket connections
   - Use appropriate update frequencies
   - Handle connection failures gracefully
   - Manage data buffer sizes

## 🎯 Success Criteria

Step 10.7 is complete when:
1. ✅ All chart types implemented and working
2. ✅ Real-time dashboard functional
3. ✅ Advanced visualizations operational
4. ✅ Performance benchmarks met
5. ✅ Export functionality working

## 🚀 Next Steps

Once all validations pass, proceed to:
**Phase 6: Going Beyond (Steps 11-18)**
- [Step 11: Alternative Data Integration](../06-going-beyond/step-11-alternative-data.md)

## 📚 Additional Resources

- [Financial Data Visualization Guide](../references/financial-dataviz.md)
- [Real-time Dashboard Architecture](../references/realtime-architecture.md)
- [Interactive Chart Best Practices](../references/interactive-charts.md)
- [Performance Optimization for Dashboards](../references/dashboard-performance.md)