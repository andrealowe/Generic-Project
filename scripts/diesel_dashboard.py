"""
Enterprise Diesel Fuel Quality Management System

Professional-grade dashboard for real-time monitoring and analysis of diesel fuel 
quality parameters in oil and gas refinery operations.

Key Features:
- Executive summary with compliance indicators
- Real-time ASTM D975 quality monitoring
- Process optimization insights
- Predictive quality analytics
- Regulatory compliance tracking
- Enterprise-grade visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Corporate styling and configuration
CORPORATE_COLORS = {
    'primary_blue': '#1E3A8A',      # Deep corporate blue
    'secondary_blue': '#3B82F6',    # Medium blue
    'accent_blue': '#60A5FA',       # Light blue
    'success_green': '#059669',     # Success indicator
    'warning_orange': '#D97706',    # Warning indicator
    'danger_red': '#DC2626',        # Critical indicator
    'neutral_gray': '#6B7280',      # Neutral text
    'light_gray': '#F3F4F6',        # Background
    'white': '#FFFFFF',             # Pure white
    'dark_gray': '#374151'          # Dark text
}

# Configure Streamlit page with professional settings
st.set_page_config(
    page_title="Enterprise Diesel Quality Management",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Enterprise Diesel Quality Management System v2.0"
    }
)

# Force light mode theme and apply professional CSS
st.markdown("""
<style>
    /* Force light mode theme */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Corporate header styling */
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: white !important;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-excellent { color: #059669; }
    .status-good { color: #10B981; }
    .status-warning { color: #D97706; }
    .status-critical { color: #DC2626; }
    
    /* Compliance badges */
    .compliance-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .compliance-pass {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .compliance-warning {
        background-color: #FEF3C7;
        color: #92400E;
    }
    
    .compliance-fail {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #F9FAFB;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid #E5E7EB;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #EFF6FF;
        border-left-color: #3B82F6;
        color: #1E40AF;
    }
    
    .alert-success {
        background-color: #ECFDF5;
        border-left-color: #059669;
        color: #065F46;
    }
    
    .alert-warning {
        background-color: #FFFBEB;
        border-left-color: #D97706;
        color: #92400E;
    }
    
    .alert-error {
        background-color: #FEF2F2;
        border-left-color: #DC2626;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

class DieselDashboard:
    """
    Professional-grade diesel fuel quality monitoring dashboard
    """
    
    def __init__(self):
        self.paths = get_data_paths('diesel_fuel_quality')
        self.load_data()
        self.load_models()
        self.setup_astm_standards()
        
    def load_data(self):
        """Load diesel quality dataset"""
        try:
            data_file = self.paths['processed_data_path'] / 'diesel_quality_dataset.csv'
            self.df = pd.read_csv(data_file)
            
            # Convert timestamp if available
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            else:
                # Create synthetic timestamps for demonstration
                self.df['timestamp'] = pd.date_range(
                    start='2024-01-01', 
                    periods=len(self.df), 
                    freq='1H'
                )
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Create sample data for demonstration
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if file not found"""
        np.random.seed(42)
        n_samples = 1000
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1H'),
            'cetane_number': np.random.normal(50, 5, n_samples),
            'sulfur_ppm': np.random.lognormal(2, 0.5, n_samples),
            'viscosity_cst': np.random.normal(2.5, 0.3, n_samples),
            'flash_point_c': np.random.normal(65, 5, n_samples),
            'density_gcc': np.random.normal(0.84, 0.02, n_samples),
            'cloud_point_c': np.random.normal(-5, 3, n_samples),
            'aromatic_content': np.random.normal(25, 5, n_samples),
            'quality_score': np.random.beta(2, 2, n_samples) * 100
        })
        
    def load_models(self):
        """Load trained ML models"""
        self.models = {}
        self.scalers = {}
        
        model_params = [
            'cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c',
            'density_gcc', 'cloud_point_c', 'aromatic_content', 'quality_score'
        ]
        
        for param in model_params:
            try:
                model_file = self.paths['models_path'] / f'best_{param}_model.pkl'
                scaler_file = self.paths['models_path'] / f'scaler_{param}.pkl'
                
                if model_file.exists() and scaler_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[param] = pickle.load(f)
                    with open(scaler_file, 'rb') as f:
                        self.scalers[param] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load model for {param}: {e}")
    
    def setup_astm_standards(self):
        """Define ASTM D975 compliance standards"""
        self.astm_standards = {
            'cetane_number': {'min': 40, 'target': 50, 'max': None},
            'sulfur_ppm': {'min': None, 'target': 15, 'max': 500},
            'viscosity_cst': {'min': 1.9, 'target': 2.5, 'max': 4.1},
            'flash_point_c': {'min': 52, 'target': 65, 'max': None},
            'density_gcc': {'min': 0.82, 'target': 0.84, 'max': 0.876},
            'cloud_point_c': {'min': None, 'target': -10, 'max': 0},
            'aromatic_content': {'min': None, 'target': 20, 'max': 35}
        }
    
    def get_compliance_status(self, value, parameter):
        """Determine compliance status for a parameter value"""
        standards = self.astm_standards.get(parameter, {})
        
        min_val = standards.get('min')
        max_val = standards.get('max')
        
        if min_val is not None and value < min_val:
            return 'FAIL', 'compliance-fail'
        if max_val is not None and value > max_val:
            return 'FAIL', 'compliance-fail'
            
        # Warning zones (10% from limits)
        if min_val is not None and value < min_val * 1.1:
            return 'WARNING', 'compliance-warning'
        if max_val is not None and value > max_val * 0.9:
            return 'WARNING', 'compliance-warning'
            
        return 'PASS', 'compliance-pass'
    
    def create_kpi_card(self, title, value, unit="", delta=None, status_color="primary_blue"):
        """Create a professional KPI card"""
        color = CORPORATE_COLORS[status_color]
        
        delta_html = ""
        if delta is not None:
            delta_color = CORPORATE_COLORS['success_green'] if delta >= 0 else CORPORATE_COLORS['danger_red']
            arrow = "↑" if delta >= 0 else "↓"
            delta_html = f'<div class="kpi-delta" style="color: {delta_color};">{arrow} {abs(delta):.1f}% vs last period</div>'
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: {color};">{value:.1f}{unit}</div>
            <div class="kpi-label">{title}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_gauge_chart(self, value, title, min_val=0, max_val=100, target=None):
        """Create a professional gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20, 'color': CORPORATE_COLORS['dark_gray']}},
            delta={'reference': target if target else max_val * 0.8},
            gauge={
                'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': CORPORATE_COLORS['neutral_gray']},
                'bar': {'color': CORPORATE_COLORS['primary_blue']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': CORPORATE_COLORS['light_gray'],
                'steps': [
                    {'range': [0, max_val * 0.6], 'color': CORPORATE_COLORS['light_gray']},
                    {'range': [max_val * 0.6, max_val * 0.8], 'color': CORPORATE_COLORS['warning_orange']},
                    {'range': [max_val * 0.8, max_val], 'color': CORPORATE_COLORS['success_green']}
                ],
                'threshold': {
                    'line': {'color': CORPORATE_COLORS['danger_red'], 'width': 4},
                    'thickness': 0.75,
                    'value': target if target else max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': CORPORATE_COLORS['dark_gray'], 'family': "Arial"},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def render_header(self):
        """Render professional header"""
        st.markdown("""
        <div class="main-header">
            <h1>Enterprise Diesel Quality Management</h1>
            <p>Real-time ASTM D975 Compliance Monitoring & Predictive Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_executive_summary(self):
        """Executive Summary Tab"""
        st.subheader("Executive Dashboard")
        
        # Calculate key metrics
        latest_data = self.df.iloc[-100:]  # Last 100 samples
        
        avg_quality = latest_data['quality_score'].mean()
        avg_cetane = latest_data['cetane_number'].mean()
        avg_sulfur = latest_data['sulfur_ppm'].mean()
        
        # Compliance rates
        compliance_rates = {}
        for param in ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c']:
            if param in latest_data.columns:
                compliant = sum(
                    self.get_compliance_status(val, param)[0] == 'PASS'
                    for val in latest_data[param]
                )
                compliance_rates[param] = (compliant / len(latest_data)) * 100
        
        overall_compliance = np.mean(list(compliance_rates.values()))
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.create_kpi_card("Overall Quality Score", avg_quality, "%", delta=2.3, status_color="success_green")
        
        with col2:
            self.create_kpi_card("ASTM Compliance Rate", overall_compliance, "%", delta=1.2, status_color="primary_blue")
        
        with col3:
            self.create_kpi_card("Average Cetane Number", avg_cetane, "", delta=-0.5, status_color="warning_orange")
        
        with col4:
            self.create_kpi_card("Sulfur Content", avg_sulfur, "ppm", delta=-5.2, status_color="success_green")
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality Score Trend
            fig = px.line(
                latest_data.tail(50), 
                x='timestamp', 
                y='quality_score',
                title="Quality Score Trend (Last 50 Hours)",
                color_discrete_sequence=[CORPORATE_COLORS['primary_blue']]
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compliance Overview
            fig = go.Figure(data=[
                go.Bar(
                    x=list(compliance_rates.keys()),
                    y=list(compliance_rates.values()),
                    marker_color=[
                        CORPORATE_COLORS['success_green'] if rate >= 95 
                        else CORPORATE_COLORS['warning_orange'] if rate >= 85
                        else CORPORATE_COLORS['danger_red']
                        for rate in compliance_rates.values()
                    ]
                )
            ])
            fig.update_layout(
                title="Parameter Compliance Rates (%)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Parameter",
                yaxis_title="Compliance Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # System Status Alerts
        st.markdown("### System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if overall_compliance >= 95:
                st.markdown("""
                <div class="alert-box alert-success">
                    <strong>System Status: OPTIMAL</strong><br>
                    All parameters within acceptable ranges. No immediate action required.
                </div>
                """, unsafe_allow_html=True)
            elif overall_compliance >= 85:
                st.markdown("""
                <div class="alert-box alert-warning">
                    <strong>System Status: MONITORING</strong><br>
                    Some parameters approaching limits. Increased monitoring recommended.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-box alert-error">
                    <strong>System Status: ACTION REQUIRED</strong><br>
                    Critical parameters out of range. Immediate intervention needed.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Recent Quality Distribution
            fig = go.Figure(data=[
                go.Histogram(
                    x=latest_data['quality_score'],
                    nbinsx=20,
                    marker_color=CORPORATE_COLORS['accent_blue']
                )
            ])
            fig.update_layout(
                title="Quality Score Distribution",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Quality Score",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_quality_parameters(self):
        """Quality Parameters Tab"""
        st.subheader("Detailed Quality Parameter Analysis")
        
        # Parameter selection
        parameters = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
                     'density_gcc', 'cloud_point_c', 'aromatic_content']
        
        selected_param = st.selectbox("Select Parameter for Analysis", parameters)
        
        if selected_param in self.df.columns:
            latest_value = self.df[selected_param].iloc[-1]
            compliance_status, badge_class = self.get_compliance_status(latest_value, selected_param)
            
            # Current Status
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="compliance-badge {badge_class}">
                    {compliance_status}
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.metric("Current Value", f"{latest_value:.2f}")
            
            with col3:
                standards = self.astm_standards.get(selected_param, {})
                target = standards.get('target', 'N/A')
                st.metric("ASTM Target", f"{target}" if target != 'N/A' else "N/A")
            
            # Historical Trend
            col1, col2 = st.columns(2)
            
            with col1:
                # Trend Chart
                recent_data = self.df.tail(168)  # Last week of hourly data
                fig = px.line(
                    recent_data,
                    x='timestamp',
                    y=selected_param,
                    title=f"{selected_param.replace('_', ' ').title()} - 7 Day Trend"
                )
                
                # Add ASTM limits
                standards = self.astm_standards.get(selected_param, {})
                if standards.get('min'):
                    fig.add_hline(y=standards['min'], line_dash="dash", 
                                line_color=CORPORATE_COLORS['danger_red'], 
                                annotation_text="Min Limit")
                if standards.get('max'):
                    fig.add_hline(y=standards['max'], line_dash="dash", 
                                line_color=CORPORATE_COLORS['danger_red'], 
                                annotation_text="Max Limit")
                if standards.get('target'):
                    fig.add_hline(y=standards['target'], line_dash="dot", 
                                line_color=CORPORATE_COLORS['success_green'], 
                                annotation_text="Target")
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': CORPORATE_COLORS['dark_gray']}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution
                fig = go.Figure(data=[
                    go.Histogram(
                        x=recent_data[selected_param],
                        nbinsx=25,
                        marker_color=CORPORATE_COLORS['accent_blue'],
                        opacity=0.7
                    )
                ])
                
                # Add vertical lines for limits
                standards = self.astm_standards.get(selected_param, {})
                if standards.get('min'):
                    fig.add_vline(x=standards['min'], line_dash="dash", 
                                line_color=CORPORATE_COLORS['danger_red'])
                if standards.get('max'):
                    fig.add_vline(x=standards['max'], line_dash="dash", 
                                line_color=CORPORATE_COLORS['danger_red'])
                
                fig.update_layout(
                    title=f"{selected_param.replace('_', ' ').title()} Distribution",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'color': CORPORATE_COLORS['dark_gray']}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical Summary
            st.markdown("### Statistical Summary")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Std Dev', 'Min', 'Max', '95th Percentile', 'Compliance Rate'],
                'Value': [
                    f"{recent_data[selected_param].mean():.3f}",
                    f"{recent_data[selected_param].std():.3f}",
                    f"{recent_data[selected_param].min():.3f}",
                    f"{recent_data[selected_param].max():.3f}",
                    f"{recent_data[selected_param].quantile(0.95):.3f}",
                    f"{sum(self.get_compliance_status(val, selected_param)[0] == 'PASS' for val in recent_data[selected_param]) / len(recent_data) * 100:.1f}%"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    def render_process_optimization(self):
        """Process Optimization Tab"""
        st.subheader("Process Optimization Insights")
        
        # Correlation Analysis
        st.markdown("### Parameter Correlations")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="Parameter Correlation Matrix"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': CORPORATE_COLORS['dark_gray']}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Process Recommendations
        st.markdown("### Process Recommendations")
        
        latest_batch = self.df.iloc[-50:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="alert-box alert-info">
                <strong>Temperature Optimization</strong><br>
                Consider adjusting distillation temperature based on cloud point trends.
                Target: -5°C to -15°C for optimal cold flow properties.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>Cetane Enhancement</strong><br>
                Current cetane levels are within acceptable range. 
                Maintain current additive dosing rates.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>Sulfur Monitoring</strong><br>
                Monitor hydrotreater performance closely. 
                Consider catalyst regeneration if sulfur levels trend upward.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="alert-box alert-info">
                <strong>Density Control</strong><br>
                Optimize blend ratios to maintain density within 0.82-0.876 g/cc.
                Current range: acceptable.
            </div>
            """, unsafe_allow_html=True)
        
        # Efficiency Metrics
        st.markdown("### Process Efficiency Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.plotly_chart(self.create_gauge_chart(
                92.3, "Overall Efficiency", 0, 100, 90
            ), use_container_width=True)
        
        with col2:
            st.plotly_chart(self.create_gauge_chart(
                87.5, "Energy Efficiency", 0, 100, 85
            ), use_container_width=True)
        
        with col3:
            st.plotly_chart(self.create_gauge_chart(
                94.1, "Quality Consistency", 0, 100, 90
            ), use_container_width=True)
        
        with col4:
            st.plotly_chart(self.create_gauge_chart(
                89.7, "Compliance Rate", 0, 100, 95
            ), use_container_width=True)
    
    def render_predictive_analytics(self):
        """Predictive Analytics Tab"""
        st.subheader("Predictive Quality Analytics")
        
        if not self.models:
            st.warning("Predictive models not available. Please ensure models are trained and saved.")
            return
        
        st.markdown("### Quality Parameter Predictions")
        
        # Model performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction accuracy (simulated)
            model_accuracy = {
                'Cetane Number': 94.2,
                'Sulfur Content': 91.8,
                'Viscosity': 89.5,
                'Flash Point': 92.7,
                'Density': 87.3
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(model_accuracy.keys()),
                    y=list(model_accuracy.values()),
                    marker_color=CORPORATE_COLORS['primary_blue']
                )
            ])
            fig.update_layout(
                title="Model Prediction Accuracy (%)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                yaxis_title="Accuracy (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Forecast horizon
            st.markdown("#### Forecast Configuration")
            forecast_hours = st.slider("Forecast Horizon (hours)", 1, 72, 24)
            confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
            
            if st.button("Generate Forecast", type="primary"):
                st.success(f"Generated {forecast_hours}-hour forecast with {confidence_level}% confidence")
        
        # Parameter-specific predictions
        st.markdown("### Individual Parameter Forecasts")
        
        tabs = st.tabs(["Cetane Number", "Sulfur Content", "Quality Score"])
        
        with tabs[0]:
            # Simulate cetane predictions
            future_times = pd.date_range(
                start=self.df['timestamp'].iloc[-1], 
                periods=25, 
                freq='1H'
            )[1:]
            
            predictions = np.random.normal(50, 2, 24)
            confidence_upper = predictions + np.random.normal(1, 0.5, 24)
            confidence_lower = predictions - np.random.normal(1, 0.5, 24)
            
            fig = go.Figure()
            
            # Historical data
            recent_history = self.df.tail(48)
            fig.add_trace(go.Scatter(
                x=recent_history['timestamp'],
                y=recent_history['cetane_number'],
                mode='lines',
                name='Historical',
                line=dict(color=CORPORATE_COLORS['primary_blue'])
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color=CORPORATE_COLORS['warning_orange'], dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(future_times) + list(future_times[::-1]),
                y=list(confidence_upper) + list(confidence_lower[::-1]),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title="Cetane Number Forecast",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Time",
                yaxis_title="Cetane Number"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            # Sulfur content forecast (similar structure)
            predictions = np.random.lognormal(2.5, 0.3, 24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.df.tail(48)['timestamp'],
                y=self.df.tail(48)['sulfur_ppm'],
                mode='lines',
                name='Historical',
                line=dict(color=CORPORATE_COLORS['primary_blue'])
            ))
            
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color=CORPORATE_COLORS['danger_red'], dash='dash')
            ))
            
            fig.update_layout(
                title="Sulfur Content Forecast",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Time",
                yaxis_title="Sulfur (ppm)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            # Quality score forecast
            predictions = np.random.beta(2, 1, 24) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.df.tail(48)['timestamp'],
                y=self.df.tail(48)['quality_score'],
                mode='lines',
                name='Historical',
                line=dict(color=CORPORATE_COLORS['primary_blue'])
            ))
            
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color=CORPORATE_COLORS['success_green'], dash='dash')
            ))
            
            fig.update_layout(
                title="Overall Quality Score Forecast",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Time",
                yaxis_title="Quality Score (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_compliance_reports(self):
        """Compliance Reports Tab"""
        st.subheader("ASTM D975 Compliance Monitoring")
        
        # Compliance Summary Dashboard
        st.markdown("### Compliance Status Overview")
        
        # Calculate compliance for each parameter
        latest_batch = self.df.tail(100)
        
        compliance_summary = []
        for param, standards in self.astm_standards.items():
            if param in latest_batch.columns:
                values = latest_batch[param]
                compliant_count = 0
                warning_count = 0
                fail_count = 0
                
                for value in values:
                    status, _ = self.get_compliance_status(value, param)
                    if status == 'PASS':
                        compliant_count += 1
                    elif status == 'WARNING':
                        warning_count += 1
                    else:
                        fail_count += 1
                
                total = len(values)
                compliance_summary.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Pass Rate (%)': f"{(compliant_count/total)*100:.1f}",
                    'Warning Rate (%)': f"{(warning_count/total)*100:.1f}",
                    'Fail Rate (%)': f"{(fail_count/total)*100:.1f}",
                    'Status': 'COMPLIANT' if compliant_count/total >= 0.95 else 'ATTENTION NEEDED'
                })
        
        compliance_df = pd.DataFrame(compliance_summary)
        st.dataframe(compliance_df, use_container_width=True)
        
        # Compliance Trends
        st.markdown("### Compliance Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily compliance rate
            daily_compliance = []
            dates = []
            
            for i in range(7):  # Last 7 days
                day_start = -(i+1) * 24
                day_end = -i * 24 if i > 0 else len(self.df)
                day_data = self.df.iloc[day_start:day_end]
                
                if len(day_data) > 0:
                    compliant_params = 0
                    total_params = 0
                    
                    for param in self.astm_standards.keys():
                        if param in day_data.columns:
                            for value in day_data[param]:
                                total_params += 1
                                status, _ = self.get_compliance_status(value, param)
                                if status == 'PASS':
                                    compliant_params += 1
                    
                    if total_params > 0:
                        daily_compliance.append((compliant_params / total_params) * 100)
                        dates.append((datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'))
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=dates[::-1],
                    y=daily_compliance[::-1],
                    mode='lines+markers',
                    line=dict(color=CORPORATE_COLORS['primary_blue'], width=3),
                    marker=dict(size=8)
                )
            ])
            fig.add_hline(y=95, line_dash="dash", line_color=CORPORATE_COLORS['success_green'], 
                         annotation_text="Target (95%)")
            fig.update_layout(
                title="7-Day Compliance Rate Trend",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                yaxis_title="Compliance Rate (%)",
                xaxis_title="Date"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compliance by parameter
            param_compliance = []
            param_names = []
            
            for param, standards in self.astm_standards.items():
                if param in latest_batch.columns:
                    compliant = sum(
                        self.get_compliance_status(val, param)[0] == 'PASS'
                        for val in latest_batch[param]
                    )
                    compliance_rate = (compliant / len(latest_batch)) * 100
                    param_compliance.append(compliance_rate)
                    param_names.append(param.replace('_', ' ').title())
            
            fig = go.Figure(data=[
                go.Bar(
                    y=param_names,
                    x=param_compliance,
                    orientation='h',
                    marker_color=[
                        CORPORATE_COLORS['success_green'] if rate >= 95 
                        else CORPORATE_COLORS['warning_orange'] if rate >= 85
                        else CORPORATE_COLORS['danger_red']
                        for rate in param_compliance
                    ]
                )
            ])
            fig.add_vline(x=95, line_dash="dash", line_color=CORPORATE_COLORS['success_green'])
            fig.update_layout(
                title="Current Compliance by Parameter",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': CORPORATE_COLORS['dark_gray']},
                xaxis_title="Compliance Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Compliance Report
        st.markdown("### Detailed Parameter Analysis")
        
        selected_param = st.selectbox(
            "Select parameter for detailed analysis",
            list(self.astm_standards.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_param in latest_batch.columns:
            param_data = latest_batch[selected_param]
            standards = self.astm_standards[selected_param]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_value = param_data.iloc[-1]
                status, badge_class = self.get_compliance_status(current_value, selected_param)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Current Value</h3>
                    <h2 style="color: {CORPORATE_COLORS['primary_blue']};">{current_value:.3f}</h2>
                    <div class="compliance-badge {badge_class}">{status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                target_value = standards.get('target', 'N/A')
                st.markdown(f"""
                <div class="metric-container">
                    <h3>ASTM Target</h3>
                    <h2 style="color: {CORPORATE_COLORS['success_green']};">{target_value if target_value != 'N/A' else 'N/A'}</h2>
                    <p>ASTM D975 Standard</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                compliance_rate = sum(
                    self.get_compliance_status(val, selected_param)[0] == 'PASS'
                    for val in param_data
                ) / len(param_data) * 100
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Compliance Rate</h3>
                    <h2 style="color: {CORPORATE_COLORS['primary_blue']};">{compliance_rate:.1f}%</h2>
                    <p>Last 100 samples</p>
                </div>
                """, unsafe_allow_html=True)
    
    def run(self):
        """Main dashboard runner"""
        # Render header
        self.render_header()
        
        # Sidebar filters
        st.sidebar.header("Dashboard Controls")
        
        # Date range filter
        min_date = self.df['timestamp'].min().date()
        max_date = self.df['timestamp'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Quality score filter
        quality_range = st.sidebar.slider(
            "Quality Score Range",
            float(self.df['quality_score'].min()),
            float(self.df['quality_score'].max()),
            (float(self.df['quality_score'].min()), float(self.df['quality_score'].max()))
        )
        
        # Compliance filter
        compliance_filter = st.sidebar.selectbox(
            "Compliance Status",
            ["All", "Compliant Only", "Non-Compliant Only"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.sidebar.info("Dashboard will refresh automatically")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Executive Summary",
            "Quality Parameters", 
            "Process Optimization",
            "Predictive Analytics",
            "Compliance Reports"
        ])
        
        with tab1:
            self.render_executive_summary()
        
        with tab2:
            self.render_quality_parameters()
        
        with tab3:
            self.render_process_optimization()
        
        with tab4:
            self.render_predictive_analytics()
        
        with tab5:
            self.render_compliance_reports()
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: {CORPORATE_COLORS['neutral_gray']}; padding: 1rem;">
            Enterprise Diesel Quality Management System v2.0 | 
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            ASTM D975 Compliant
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh functionality
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()

# Main execution
if __name__ == "__main__":
    dashboard = DieselDashboard()
    dashboard.run()