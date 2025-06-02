#!/usr/bin/env python3
"""
Examples of heatmap visualizations available in ADMF-PC

This demonstrates the various heatmap types that can be generated:
1. Parameter optimization heatmaps
2. Correlation matrices  
3. Risk factor heatmaps
4. Performance attribution heatmaps
5. Regime transition matrices
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


def create_parameter_optimization_heatmap():
    """Create heatmap showing performance across parameter combinations"""
    
    # Sample parameter grid results
    lookback_periods = [10, 15, 20, 25, 30]
    threshold_values = [0.01, 0.015, 0.02, 0.025, 0.03]
    
    # Generate sample performance matrix (Sharpe ratios)
    np.random.seed(42)
    performance_matrix = np.random.uniform(0.5, 2.5, (len(threshold_values), len(lookback_periods)))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=performance_matrix,
        x=lookback_periods,
        y=threshold_values,
        colorscale='Viridis',
        text=np.round(performance_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(title="Sharpe Ratio")
    ))
    
    fig.update_layout(
        title="Parameter Optimization Results - Sharpe Ratio Heatmap",
        xaxis_title="Lookback Period (days)",
        yaxis_title="Signal Threshold",
        font=dict(size=12),
        height=500
    )
    
    return fig


def create_correlation_matrix_heatmap():
    """Create correlation matrix heatmap for multiple assets"""
    
    # Sample correlation data
    assets = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM']
    
    # Generate realistic correlation matrix
    np.random.seed(42)
    base_corr = np.random.uniform(0.3, 0.8, (len(assets), len(assets)))
    correlation_matrix = (base_corr + base_corr.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_matrix, 1.0)  # Perfect self-correlation
    
    corr_df = pd.DataFrame(correlation_matrix, index=assets, columns=assets)
    
    # Create heatmap with custom colorscale
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=assets,
        y=assets,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        xaxis_title="Assets",
        yaxis_title="Assets",
        height=600,
        width=600
    )
    
    return fig


def create_risk_factor_heatmap():
    """Create risk factor exposure heatmap"""
    
    # Sample data: strategies vs risk factors
    strategies = ['Momentum', 'Mean Reversion', 'Breakout', 'Pairs Trading', 'Market Making']
    risk_factors = ['Market Beta', 'Size', 'Value', 'Momentum', 'Volatility', 'Quality']
    
    # Generate sample risk exposures
    np.random.seed(42)
    exposures = np.random.uniform(-0.5, 0.5, (len(strategies), len(risk_factors)))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=exposures,
        x=risk_factors,
        y=strategies,
        colorscale='RdBu',
        zmid=0,
        text=np.round(exposures, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Factor Exposure")
    ))
    
    fig.update_layout(
        title="Strategy Risk Factor Exposures",
        xaxis_title="Risk Factors",
        yaxis_title="Strategies",
        height=400,
        width=700
    )
    
    return fig


def create_regime_transition_heatmap():
    """Create market regime transition matrix"""
    
    regimes = ['Bull Market', 'Bear Market', 'High Volatility', 'Low Volatility', 'Trending', 'Ranging']
    
    # Sample transition probabilities (rows = from, columns = to)
    np.random.seed(42)
    transitions = np.random.dirichlet([1] * len(regimes), len(regimes))
    
    fig = go.Figure(data=go.Heatmap(
        z=transitions,
        x=regimes,
        y=regimes,
        colorscale='Blues',
        text=np.round(transitions, 3),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="Transition Probability")
    ))
    
    fig.update_layout(
        title="Market Regime Transition Matrix",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=500,
        width=700
    )
    
    return fig


def create_performance_attribution_heatmap():
    """Create performance attribution heatmap by time period and strategy"""
    
    # Sample data: monthly returns by strategy
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    strategies = ['Alpha Strategy', 'Beta Strategy', 'Gamma Strategy', 'Delta Strategy']
    
    # Generate sample monthly returns (%)
    np.random.seed(42)
    returns = np.random.normal(0.5, 2.0, (len(strategies), len(months)))
    
    fig = go.Figure(data=go.Heatmap(
        z=returns,
        x=months,
        y=strategies,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(returns, 2),
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Monthly Return (%)")
    ))
    
    fig.update_layout(
        title="Monthly Performance Attribution by Strategy",
        xaxis_title="Month",
        yaxis_title="Strategy",
        height=400,
        width=800
    )
    
    return fig


def create_multi_timeframe_volatility_heatmap():
    """Create volatility heatmap across assets and timeframes"""
    
    assets = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    timeframes = ['1D', '1W', '1M', '3M', '6M', '1Y']
    
    # Generate sample volatility data (annualized %)
    np.random.seed(42)
    volatilities = np.random.uniform(10, 40, (len(assets), len(timeframes)))
    
    fig = go.Figure(data=go.Heatmap(
        z=volatilities,
        x=timeframes,
        y=assets,
        colorscale='Reds',
        text=np.round(volatilities, 1),
        texttemplate="%{text}%",
        textfont={"size": 11},
        colorbar=dict(title="Volatility (%)")
    ))
    
    fig.update_layout(
        title="Asset Volatility Across Timeframes",
        xaxis_title="Timeframe",
        yaxis_title="Asset",
        height=400,
        width=600
    )
    
    return fig


def create_3d_parameter_surface():
    """Create 3D surface plot for parameter optimization"""
    
    # Parameter ranges
    x = np.linspace(5, 30, 20)  # Lookback period
    y = np.linspace(0.005, 0.05, 20)  # Threshold
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Generate sample performance surface (Sharpe ratio)
    Z = 2 * np.exp(-(X-15)**2/100 - (Y-0.02)**2/0.001) + np.random.normal(0, 0.1, X.shape)
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Sharpe Ratio")
    )])
    
    fig.update_layout(
        title="3D Parameter Optimization Surface",
        scene=dict(
            xaxis_title="Lookback Period",
            yaxis_title="Signal Threshold", 
            zaxis_title="Sharpe Ratio"
        ),
        height=600
    )
    
    return fig


def generate_all_heatmap_examples():
    """Generate all heatmap examples and save as HTML"""
    
    examples = [
        ("parameter_optimization", create_parameter_optimization_heatmap()),
        ("correlation_matrix", create_correlation_matrix_heatmap()),
        ("risk_factors", create_risk_factor_heatmap()),
        ("regime_transitions", create_regime_transition_heatmap()),
        ("performance_attribution", create_performance_attribution_heatmap()),
        ("volatility_timeframes", create_multi_timeframe_volatility_heatmap()),
        ("3d_parameter_surface", create_3d_parameter_surface())
    ]
    
    # Create combined dashboard
    from plotly.subplots import make_subplots
    
    print("ADMF-PC Heatmap Examples")
    print("=" * 40)
    
    for name, fig in examples:
        filename = f"heatmap_{name}.html"
        fig.write_html(filename)
        print(f"✓ {name.replace('_', ' ').title()}: {filename}")
    
    print(f"\n🔥 Generated {len(examples)} different heatmap types!")
    print("\nHeatmap types available in ADMF-PC:")
    print("• Parameter optimization landscapes")
    print("• Asset correlation matrices")  
    print("• Risk factor exposures")
    print("• Market regime transitions")
    print("• Performance attribution")
    print("• Multi-timeframe analysis")
    print("• 3D parameter surfaces")
    
    return examples


if __name__ == "__main__":
    examples = generate_all_heatmap_examples()
    
    print("\n💡 These heatmaps integrate seamlessly with:")
    print("   • Existing workspace management")
    print("   • Signal analysis engine") 
    print("   • Multi-phase optimization workflows")
    print("   • Interactive dashboards")
    
    print("\n🚀 Ready to implement in ADMF-PC reporting system!")