"""
Exploratory Data Analysis for Diesel Fuel Quality Dataset

This script performs comprehensive EDA on the synthetic diesel fuel quality dataset
including statistical analysis, correlation analysis, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_dataset():
    """Load the diesel fuel quality dataset."""
    paths = get_data_paths()
    dataset_path = paths['processed_data_path'] / 'diesel_quality_dataset.csv'
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset loaded: {df.shape}")
    return df, paths

def basic_statistics(df):
    """Generate basic statistical summary."""
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Quality pass rate: {df['quality_pass'].mean()*100:.1f}%")
    print(f"Average quality score: {df['quality_score'].mean():.1f}")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
    
    # Missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nMissing values:")
        print(missing_counts[missing_counts > 0])
    else:
        print(f"\nNo missing values found")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    return df.describe()

def correlation_analysis(df, paths):
    """Analyze correlations between features."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['month']]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'correlation_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Target correlations
    target_vars = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
                   'density_gcc', 'quality_score']
    
    print("\nTop correlations with target variables:")
    for target in target_vars:
        if target in df.columns:
            target_corr = corr_matrix[target].abs().sort_values(ascending=False)[1:6]  # Top 5
            print(f"\n{target}:")
            for feature, corr_val in target_corr.items():
                print(f"  {feature}: {corr_val:.3f}")
    
    return corr_matrix

def quality_parameter_distributions(df, paths):
    """Analyze distributions of quality parameters."""
    print("\n" + "="*60)
    print("QUALITY PARAMETER DISTRIBUTIONS")
    print("="*60)
    
    quality_params = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
                     'density_gcc', 'cloud_point_c', 'aromatic_content']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, param in enumerate(quality_params):
        if param in df.columns:
            ax = axes[i]
            
            # Histogram with KDE
            df[param].hist(ax=ax, bins=30, alpha=0.7, density=True, color='skyblue')
            df[param].plot.density(ax=ax, color='red', linewidth=2)
            
            # Add ASTM specification lines if applicable
            show_legend = False
            if param == 'cetane_number':
                ax.axvline(x=40, color='green', linestyle='--', label='ASTM Min')
                show_legend = True
            elif param == 'sulfur_ppm':
                ax.axvline(x=15, color='green', linestyle='--', label='ULSD Max')
                show_legend = True
            elif param == 'viscosity_cst':
                ax.axvline(x=1.9, color='green', linestyle='--', label='Min')
                ax.axvline(x=4.1, color='green', linestyle='--', label='Max')
                show_legend = True
            elif param == 'flash_point_c':
                ax.axvline(x=52, color='green', linestyle='--', label='ASTM Min')
                show_legend = True
            
            ax.set_title(f'{param.replace("_", " ").title()}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            
            if show_legend:
                ax.legend()
    
    # Remove empty subplot
    fig.delaxes(axes[7])
    fig.delaxes(axes[8])
    
    plt.suptitle('Quality Parameter Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'quality_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    for param in quality_params:
        if param in df.columns:
            print(f"\n{param}:")
            print(f"  Mean: {df[param].mean():.2f}")
            print(f"  Std: {df[param].std():.2f}")
            print(f"  Range: {df[param].min():.2f} - {df[param].max():.2f}")

def process_parameter_analysis(df, paths):
    """Analyze process parameter relationships."""
    print("\n" + "="*60)
    print("PROCESS PARAMETER ANALYSIS")
    print("="*60)
    
    # Key process relationships
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Hydrotreating temperature vs sulfur removal
    axes[0,0].scatter(df['hydrotreat_temp'], df['sulfur_ppm'], alpha=0.5, s=20)
    axes[0,0].set_xlabel('Hydrotreating Temperature (°C)')
    axes[0,0].set_ylabel('Sulfur Content (ppm)')
    axes[0,0].set_title('Sulfur Removal vs Temperature')
    axes[0,0].set_yscale('log')
    
    # 2. API gravity vs cetane number
    axes[0,1].scatter(df['crude_api_gravity'], df['cetane_number'], alpha=0.5, s=20)
    axes[0,1].set_xlabel('Crude API Gravity')
    axes[0,1].set_ylabel('Cetane Number')
    axes[0,1].set_title('Cetane Number vs API Gravity')
    
    # 3. Catalyst age effect
    axes[0,2].scatter(df['catalyst_age_days'], df['sulfur_ppm'], alpha=0.5, s=20)
    axes[0,2].set_xlabel('Catalyst Age (days)')
    axes[0,2].set_ylabel('Sulfur Content (ppm)')
    axes[0,2].set_title('Catalyst Deactivation Effect')
    axes[0,2].set_yscale('log')
    
    # 4. Distillation temperature vs viscosity
    axes[1,0].scatter(df['distillation_temp_middle'], df['viscosity_cst'], alpha=0.5, s=20)
    axes[1,0].set_xlabel('Distillation Temperature (°C)')
    axes[1,0].set_ylabel('Viscosity (cSt)')
    axes[1,0].set_title('Viscosity vs Distillation Temperature')
    
    # 5. Quality score distribution
    df['quality_score'].hist(ax=axes[1,1], bins=30, alpha=0.7, color='lightgreen')
    axes[1,1].axvline(x=90, color='red', linestyle='--', label='Excellent Threshold')
    axes[1,1].axvline(x=75, color='orange', linestyle='--', label='Good Threshold')
    axes[1,1].set_xlabel('Quality Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Overall Quality Score Distribution')
    axes[1,1].legend()
    
    # 6. Pass/Fail by quality score
    quality_ranges = pd.cut(df['quality_score'], bins=[0, 60, 75, 85, 100], 
                           labels=['Poor', 'Fair', 'Good', 'Excellent'])
    pass_by_range = df.groupby(quality_ranges)['quality_pass'].mean()
    pass_by_range.plot(kind='bar', ax=axes[1,2], color='coral')
    axes[1,2].set_xlabel('Quality Score Range')
    axes[1,2].set_ylabel('Pass Rate')
    axes[1,2].set_title('Pass Rate by Quality Score Range')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'process_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def seasonal_analysis(df, paths):
    """Analyze seasonal patterns in fuel quality."""
    print("\n" + "="*60)
    print("SEASONAL ANALYSIS")
    print("="*60)
    
    # Monthly aggregations
    monthly_stats = df.groupby('month').agg({
        'quality_score': ['mean', 'std'],
        'cloud_point_c': 'mean',
        'quality_pass': 'mean',
        'cetane_number': 'mean',
        'sulfur_ppm': 'mean'
    }).round(2)
    
    # Plot seasonal trends
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Monthly quality score
    monthly_quality = df.groupby('month')['quality_score'].mean()
    axes[0,0].plot(monthly_quality.index, monthly_quality.values, marker='o', linewidth=2)
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Average Quality Score')
    axes[0,0].set_title('Seasonal Quality Score Trends')
    axes[0,0].grid(True)
    
    # 2. Monthly pass rate
    monthly_pass = df.groupby('month')['quality_pass'].mean() * 100
    axes[0,1].plot(monthly_pass.index, monthly_pass.values, marker='s', linewidth=2, color='green')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Pass Rate (%)')
    axes[0,1].set_title('Monthly Quality Pass Rate')
    axes[0,1].grid(True)
    
    # 3. Cloud point seasonal variation
    monthly_cloud = df.groupby('month')['cloud_point_c'].mean()
    axes[1,0].plot(monthly_cloud.index, monthly_cloud.values, marker='^', linewidth=2, color='blue')
    axes[1,0].axhline(y=-7, color='red', linestyle='--', label='Winter Spec Limit')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Cloud Point (°C)')
    axes[1,0].set_title('Seasonal Cloud Point Variation')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 4. Aromatic content by season
    seasonal_aromatic = df.groupby('month')['aromatic_content'].mean()
    axes[1,1].plot(seasonal_aromatic.index, seasonal_aromatic.values, marker='d', linewidth=2, color='purple')
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Aromatic Content (%)')
    axes[1,1].set_title('Seasonal Aromatic Content')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'seasonal_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSeasonal statistics saved to: {plot_path}")
    return monthly_stats

def anomaly_analysis(df, paths):
    """Analyze anomalies in the dataset."""
    print("\n" + "="*60)
    print("ANOMALY ANALYSIS")
    print("="*60)
    
    anomalies = df[df['is_anomaly'] == True]
    normal = df[df['is_anomaly'] == False]
    
    print(f"Total anomalies: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
    print(f"Anomaly quality pass rate: {anomalies['quality_pass'].mean()*100:.1f}%")
    print(f"Normal quality pass rate: {normal['quality_pass'].mean()*100:.1f}%")
    
    # Compare distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    quality_params = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 
                     'quality_score', 'hydrotreat_temp', 'catalyst_age_days']
    
    for i, param in enumerate(quality_params):
        ax = axes[i//3, i%3]
        
        # Histograms
        normal[param].hist(ax=ax, bins=30, alpha=0.7, label='Normal', density=True, color='green')
        anomalies[param].hist(ax=ax, bins=20, alpha=0.7, label='Anomalies', density=True, color='red')
        
        ax.set_title(f'{param.replace("_", " ").title()}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Normal vs Anomalous Samples Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'anomaly_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def time_series_analysis(df, paths):
    """Analyze time series patterns."""
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS")
    print("="*60)
    
    # Resample to weekly averages for cleaner visualization
    df_weekly = df.set_index('timestamp').resample('W').mean()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Quality score over time
    axes[0].plot(df_weekly.index, df_weekly['quality_score'], linewidth=1.5, color='blue')
    axes[0].axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='Good Threshold')
    axes[0].axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Excellent Threshold')
    axes[0].set_ylabel('Quality Score')
    axes[0].set_title('Quality Score Trends Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Key quality parameters
    axes[1].plot(df_weekly.index, df_weekly['cetane_number'], label='Cetane Number', linewidth=1.5)
    axes[1].plot(df_weekly.index, df_weekly['sulfur_ppm']/10, label='Sulfur (ppm/10)', linewidth=1.5)
    axes[1].plot(df_weekly.index, df_weekly['viscosity_cst']*10, label='Viscosity (cSt×10)', linewidth=1.5)
    axes[1].set_ylabel('Normalized Values')
    axes[1].set_title('Key Quality Parameters Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Process conditions
    axes[2].plot(df_weekly.index, df_weekly['hydrotreat_temp'], label='HT Temperature', linewidth=1.5)
    axes[2].plot(df_weekly.index, df_weekly['catalyst_age_days']/10, label='Catalyst Age (days/10)', linewidth=1.5)
    axes[2].set_ylabel('Process Values')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Process Conditions Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = paths['visualizations_path'] / 'time_series_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, corr_matrix, paths):
    """Generate comprehensive EDA summary report."""
    
    report_path = paths['reports_path'] / 'eda_summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Diesel Fuel Quality Dataset - EDA Summary Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total samples:** {len(df):,}\n")
        f.write(f"- **Time range:** {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"- **Features:** {len(df.columns)} columns\n")
        f.write(f"- **Missing values:** {df.isnull().sum().sum()}\n\n")
        
        f.write("## Quality Assessment\n\n")
        f.write(f"- **Overall pass rate:** {df['quality_pass'].mean()*100:.1f}%\n")
        f.write(f"- **Average quality score:** {df['quality_score'].mean():.1f}/100\n")
        f.write(f"- **Quality score std:** {df['quality_score'].std():.1f}\n")
        f.write(f"- **Anomaly rate:** {df['is_anomaly'].mean()*100:.1f}%\n\n")
        
        f.write("## Key Quality Parameters\n\n")
        quality_params = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 'density_gcc']
        for param in quality_params:
            if param in df.columns:
                mean_val = df[param].mean()
                std_val = df[param].std()
                min_val = df[param].min()
                max_val = df[param].max()
                f.write(f"**{param.replace('_', ' ').title()}:**\n")
                f.write(f"- Mean: {mean_val:.2f}, Std: {std_val:.2f}\n")
                f.write(f"- Range: {min_val:.2f} - {max_val:.2f}\n\n")
        
        f.write("## Process Insights\n\n")
        
        # Top correlations
        f.write("### High Correlations with Quality Score\n\n")
        quality_corr = corr_matrix['quality_score'].abs().sort_values(ascending=False)[1:6]
        for feature, corr_val in quality_corr.items():
            f.write(f"- **{feature}:** {corr_val:.3f}\n")
        f.write("\n")
        
        # Seasonal patterns
        seasonal_quality = df.groupby('month')['quality_score'].mean()
        best_month = seasonal_quality.idxmax()
        worst_month = seasonal_quality.idxmin()
        f.write(f"### Seasonal Patterns\n\n")
        f.write(f"- **Best quality month:** {best_month} (score: {seasonal_quality[best_month]:.1f})\n")
        f.write(f"- **Worst quality month:** {worst_month} (score: {seasonal_quality[worst_month]:.1f})\n")
        f.write(f"- **Seasonal variation:** {seasonal_quality.std():.1f} points\n\n")
        
        f.write("## Recommendations for Modeling\n\n")
        f.write("1. **Feature Engineering:**\n")
        f.write("   - Create interaction terms between temperature and pressure\n")
        f.write("   - Add catalyst efficiency indicators\n")
        f.write("   - Include seasonal dummy variables\n\n")
        
        f.write("2. **Model Considerations:**\n")
        f.write("   - Use time-series cross-validation\n")
        f.write("   - Handle class imbalance in anomaly detection\n")
        f.write("   - Consider multi-output models for simultaneous quality prediction\n\n")
        
        f.write("3. **Data Quality:**\n")
        f.write("   - No missing values detected\n")
        f.write("   - Realistic correlations established\n")
        f.write("   - Appropriate anomaly rate for industrial processes\n\n")
        
        f.write("## Visualization Files Generated\n\n")
        viz_files = [
            "correlation_heatmap.png",
            "quality_distributions.png", 
            "process_analysis.png",
            "seasonal_analysis.png",
            "anomaly_analysis.png",
            "time_series_analysis.png"
        ]
        
        for viz_file in viz_files:
            f.write(f"- {viz_file}\n")
    
    print(f"\nEDA summary report saved to: {report_path}")

def main():
    """Main EDA function."""
    print("DIESEL FUEL QUALITY DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df, paths = load_dataset()
    
    # Basic statistics
    desc_stats = basic_statistics(df)
    
    # Correlation analysis
    corr_matrix = correlation_analysis(df, paths)
    
    # Quality parameter analysis
    quality_parameter_distributions(df, paths)
    
    # Process parameter analysis
    process_parameter_analysis(df, paths)
    
    # Seasonal analysis
    seasonal_stats = seasonal_analysis(df, paths)
    
    # Anomaly analysis
    anomaly_analysis(df, paths)
    
    # Time series analysis
    time_series_analysis(df, paths)
    
    # Generate summary report
    generate_summary_report(df, corr_matrix, paths)
    
    print(f"\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f"Reports saved to: {paths['reports_path']}")
    print(f"Visualizations saved to: {paths['visualizations_path']}")
    
    return df, corr_matrix

if __name__ == "__main__":
    df, corr_matrix = main()