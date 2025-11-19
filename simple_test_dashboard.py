#!/usr/bin/env python3

"""Simple test dashboard to verify data loading works."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path  
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

st.set_page_config(page_title="Diesel Quality Test Dashboard", layout="wide")

# Clear cache
st.cache_data.clear()

# Test data loading
@st.cache_data
def load_test_data():
    try:
        paths = get_data_paths()
        dataset_path = paths['processed_data_path'] / 'diesel_quality_dataset.csv'
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            return df, None
        else:
            return None, f"Dataset not found at {dataset_path}"
    except Exception as e:
        return None, f"Error loading data: {e}"

def main():
    st.title("🔬 Diesel Quality Test Dashboard")
    st.write("Testing data loading functionality...")
    
    # Load data
    df, error = load_test_data()
    
    if error:
        st.error(f"❌ Data loading failed: {error}")
        return
    
    if df is None or df.empty:
        st.error("❌ No data loaded")
        return
        
    # Show success
    st.success("✅ Data loaded successfully!")
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Data Type", "Diesel Quality")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Show column info
    st.subheader("Column Information")
    st.write("Available columns:")
    st.write(list(df.columns))
    
    # Show basic stats
    if 'quality_score' in df.columns:
        st.subheader("Quality Score Statistics")
        st.write(df['quality_score'].describe())

if __name__ == "__main__":
    main()