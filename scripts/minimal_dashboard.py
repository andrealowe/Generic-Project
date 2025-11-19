#!/usr/bin/env python3

"""
Minimal Diesel Fuel Quality Management Dashboard

A functional dashboard without complex styling to ensure reliability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

# Configure Streamlit page
st.set_page_config(
    page_title="Diesel Quality Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear cache to ensure fresh data loading
st.cache_data.clear()

class MinimalDieselDashboard:
    """Minimal but functional dashboard for diesel fuel quality management."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.paths = get_data_paths()
        self.target_columns = [
            'cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
            'density_gcc', 'cloud_point_c', 'aromatic_content', 'quality_score'
        ]
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # ASTM D975 compliance standards for diesel fuel
        self.astm_standards = {
            'cetane_number': {'min': 40, 'target': 45, 'excellent': 50},
            'sulfur_ppm': {'max': 15, 'target': 10, 'excellent': 5},
            'viscosity_cst': {'min': 1.9, 'max': 4.1, 'target': 3.0},
            'flash_point_c': {'min': 52, 'target': 60, 'excellent': 70},
            'density_gcc': {'min': 0.82, 'max': 0.845, 'target': 0.835},
            'cloud_point_c': {'max': -12, 'target': -15, 'excellent': -20},
            'aromatic_content': {'max': 35, 'target': 25, 'excellent': 15},
            'quality_score': {'min': 70, 'target': 85, 'excellent': 95}
        }
        
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load trained models and dataset."""
        try:
            # Load dataset
            dataset_path = self.paths['processed_data_path'] / 'diesel_quality_dataset.csv'
            self.df = pd.read_csv(dataset_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Load feature columns
            feature_path = self.paths['models_path'] / 'feature_columns.json'
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    self.feature_columns = json.load(f)
            
            # Load models and scalers
            models_dir = self.paths['models_path']
            for target in self.target_columns:
                model_path = models_dir / f"best_{target}_model.pkl"
                scaler_path = models_dir / f"scaler_{target}.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[target] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[target] = pickle.load(f)
            
            # Load performance metrics
            perf_path = models_dir / 'model_performance.json'
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    self.performance_data = json.load(f)
            else:
                self.performance_data = {}
            
        except Exception as e:
            st.error(f"Error loading models and data: {e}")
            self.df = pd.DataFrame()
            self.performance_data = {}
    
    def get_compliance_status(self, parameter, value):
        """Determine ASTM D975 compliance status for a parameter."""
        if parameter not in self.astm_standards:
            return 'unknown', '#6B7280'
        
        standards = self.astm_standards[parameter]
        
        # Handle different parameter types
        if parameter in ['sulfur_ppm', 'cloud_point_c', 'aromatic_content']:
            # Lower is better
            if 'excellent' in standards and value <= standards['excellent']:
                return 'excellent', '#059669'
            elif value <= standards.get('target', standards.get('max', float('inf'))):
                return 'good', '#059669'
            elif value <= standards.get('max', float('inf')):
                return 'warning', '#D97706'
            else:
                return 'critical', '#DC2626'
        else:
            # Higher is better or range-based
            if 'min' in standards and 'max' in standards:
                # Range-based (e.g., viscosity)
                if standards['min'] <= value <= standards['max']:
                    return 'good', '#059669'
                else:
                    return 'critical', '#DC2626'
            else:
                # Higher is better
                if 'excellent' in standards and value >= standards['excellent']:
                    return 'excellent', '#059669'
                elif value >= standards.get('target', standards.get('min', 0)):
                    return 'good', '#059669'
                elif value >= standards.get('min', 0) * 0.9:
                    return 'warning', '#D97706'
                else:
                    return 'critical', '#DC2626'
    
    def render_header(self):
        """Render professional header."""
        st.markdown("# 🛢️ Diesel Fuel Quality Management System")
        st.markdown("**Real-time monitoring and compliance tracking for refinery operations**")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("Dashboard Controls")
        
        if self.df.empty:
            st.sidebar.warning("No data available")
            return {}
        
        # Time range filter
        if 'timestamp' in self.df.columns:
            min_date = self.df['timestamp'].min()
            max_date = self.df['timestamp'].max()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        else:
            date_range = None
        
        # Compliance filter
        compliance_filter = st.sidebar.selectbox(
            "ASTM Compliance Filter",
            ["All", "Non-Compliant", "Excellent"],
            index=0
        )
        
        # Quality score filter
        quality_range = st.sidebar.slider(
            "Quality Score Range",
            min_value=float(self.df['quality_score'].min()) if 'quality_score' in self.df.columns else 0.0,
            max_value=float(self.df['quality_score'].max()) if 'quality_score' in self.df.columns else 100.0,
            value=(70.0, 100.0),
            step=1.0
        )
        
        return {
            'date_range': date_range,
            'compliance_filter': compliance_filter,
            'quality_range': quality_range
        }
    
    def filter_data(self, filters):
        """Filter data based on sidebar controls."""
        if self.df.empty:
            return self.df
        
        df_filtered = self.df.copy()
        
        # Apply date filter
        if filters.get('date_range') and 'timestamp' in df_filtered.columns:
            if len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                df_filtered = df_filtered[
                    (df_filtered['timestamp'].dt.date >= start_date) &
                    (df_filtered['timestamp'].dt.date <= end_date)
                ]
        
        # Apply quality score filter
        if 'quality_score' in df_filtered.columns and filters.get('quality_range'):
            min_qual, max_qual = filters['quality_range']
            df_filtered = df_filtered[
                (df_filtered['quality_score'] >= min_qual) &
                (df_filtered['quality_score'] <= max_qual)
            ]
        
        return df_filtered
    
    def render_summary_kpis(self, df_filtered):
        """Render summary KPIs."""
        if df_filtered.empty:
            st.warning("No data to display with current filters")
            return
        
        st.subheader("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_batches = len(df_filtered)
            st.metric("Total Batches", total_batches)
        
        with col2:
            if 'quality_score' in df_filtered.columns:
                avg_quality = df_filtered['quality_score'].mean()
                st.metric("Average Quality Score", f"{avg_quality:.1f}")
            else:
                st.metric("Average Quality Score", "N/A")
        
        with col3:
            if 'quality_pass' in df_filtered.columns:
                pass_rate = (df_filtered['quality_pass'].sum() / len(df_filtered)) * 100
                st.metric("ASTM Pass Rate", f"{pass_rate:.1f}%")
            else:
                st.metric("ASTM Pass Rate", "N/A")
        
        with col4:
            if 'is_anomaly' in df_filtered.columns:
                anomaly_rate = (df_filtered['is_anomaly'].sum() / len(df_filtered)) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            else:
                st.metric("Anomaly Rate", "N/A")
    
    def render_quality_charts(self, df_filtered):
        """Render quality parameter charts."""
        if df_filtered.empty:
            return
        
        st.subheader("🎯 Quality Parameters")
        
        # Select parameters to display
        available_params = [col for col in self.target_columns if col in df_filtered.columns]
        
        if not available_params:
            st.warning("No quality parameters available in the data")
            return
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Trends", "📊 Current Status"])
        
        with tab1:
            # Time series charts
            if 'timestamp' in df_filtered.columns:
                for param in available_params[:4]:  # Limit to first 4 for performance
                    fig = px.line(
                        df_filtered.tail(1000),  # Last 1000 points for performance
                        x='timestamp',
                        y=param,
                        title=f"{param.replace('_', ' ').title()} Over Time"
                    )
                    
                    # Add ASTM limits if available
                    if param in self.astm_standards:
                        standards = self.astm_standards[param]
                        if 'max' in standards:
                            fig.add_hline(y=standards['max'], line_dash="dash", line_color="red",
                                        annotation_text=f"ASTM Max: {standards['max']}")
                        if 'min' in standards:
                            fig.add_hline(y=standards['min'], line_dash="dash", line_color="red",
                                        annotation_text=f"ASTM Min: {standards['min']}")
                        if 'target' in standards:
                            fig.add_hline(y=standards['target'], line_dash="dot", line_color="green",
                                        annotation_text=f"Target: {standards['target']}")
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Timestamp data not available for trend analysis")
        
        with tab2:
            # Current status gauge charts
            if len(df_filtered) > 0:
                latest_data = df_filtered.iloc[-1]
                
                cols = st.columns(2)
                for i, param in enumerate(available_params[:4]):
                    with cols[i % 2]:
                        value = latest_data[param]
                        status, color = self.get_compliance_status(param, value)
                        
                        # Simple metric display
                        st.metric(
                            param.replace('_', ' ').title(),
                            f"{value:.2f}",
                            delta=f"Status: {status.title()}"
                        )
    
    def run(self):
        """Run the minimal dashboard."""
        # Render header
        self.render_header()
        
        if self.df.empty:
            st.error("⚠️ **System Notice:** Historical dataset is not available. Please ensure data pipeline is operational and models have been trained before accessing the dashboard.")
            
            # Show system status when no data
            st.markdown("### System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Data Pipeline:** Disconnected")
            with col2:
                st.warning("**AI Models:** Not Loaded")
            with col3:
                st.error("**Real-time Feed:** Unavailable")
            
            return
        
        # Render sidebar controls
        filters = self.render_sidebar()
        
        # Filter data based on controls
        df_filtered = self.filter_data(filters)
        
        # Render main content
        self.render_summary_kpis(df_filtered)
        
        st.markdown("---")
        
        self.render_quality_charts(df_filtered)
        
        # Show data table
        if st.checkbox("Show Raw Data"):
            st.subheader("📋 Raw Data")
            st.dataframe(df_filtered.head(1000))  # Limit for performance
        
        # Footer
        st.markdown("---")
        st.markdown("**Diesel Fuel Quality Management System** | Enterprise Dashboard | Real-time Monitoring")

def main():
    """Main function to run the dashboard."""
    dashboard = MinimalDieselDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()