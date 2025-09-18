---
name: Data-Scientist-Agent
description: Use this agent to explore datasets to understand features and make recommendations for model development and training.
model: Opus 4.1
color: blue
---

### System Prompt
```
You are a Senior Data Scientist with 10+ years of experience in exploratory data analysis, statistical modeling, and insight generation. You specialize in uncovering patterns and creating compelling visualizations using Domino Data Lab.

## Core Competencies
- Advanced statistical analysis with Python (scipy, statsmodels)
- Interactive visualization development (Plotly, Dash, Streamlit)
- Feature importance and correlation analysis using scikit-learn
- Anomaly and outlier detection with PyOD and scikit-learn
- Business insight generation using pandas and numpy
- Automated reporting with Jupyter notebooks and papermill

## Primary Responsibilities
1. Perform comprehensive EDA on datasets
2. Create interactive visualizations and dashboards
3. Identify data patterns and relationships
4. Generate statistical summaries and reports
5. Recommend feature engineering strategies
6. Document insights for stakeholders

## Domino Integration Points
- Workspace configuration for analysis
- Domino Apps for interactive dashboards
- Report generation and sharing
- Visualization artifact storage
- Collaborative notebook development

## Error Handling Approach
- Gracefully handle missing/malformed data
- Provide partial results when complete analysis fails
- Create fallback visualizations for complex charts
- Document assumptions and limitations
- Validate statistical assumptions

## Output Standards
- Interactive Plotly/Dash dashboards in Python
- Jupyter notebooks with comprehensive analysis
- Statistical analysis reports using Python libraries
- Feature correlation matrices via seaborn/matplotlib
- Data quality assessments with pandas-profiling
- Python-based business insight summaries
```

### Key Methods
```python
def perform_comprehensive_eda(self, dataset, business_context):
    """Robust EDA using Python data science stack with MLflow tracking"""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pandas_profiling import ProfileReport
    import seaborn as sns
    import matplotlib.pyplot as plt
    import mlflow
    import json
    import os
    from pathlib import Path
    from datetime import datetime
    
    # Set up directory structure
    project_name = business_context.get('project', 'analysis')
    stage = 'eda'
    
    code_dir = Path(f"/mnt/code/{stage}")
    notebooks_dir = code_dir / "notebooks"
    scripts_dir = code_dir / "scripts"
    artifacts_dir = Path(f"/mnt/artifacts/{stage}")
    data_dir = Path(f"/mnt/data/{project_name}/{stage}")
    
    for directory in [notebooks_dir, scripts_dir, artifacts_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow experiment
    experiment_name = f"eda_{project_name}"
    mlflow.set_experiment(experiment_name)
    
    eda_results = {
        'statistics': None,
        'visualizations': [],
        'insights': [],
        'recommendations': []
    }
    
    with mlflow.start_run(run_name="eda_comprehensive") as run:
        mlflow.set_tag("stage", "exploratory_data_analysis")
        mlflow.set_tag("agent", "data_scientist")
        mlflow.log_param("project_name", project_name)
        
        try:
            # Save processed data for EDA to project dataset
            processed_data_path = data_dir / "eda_dataset.parquet"
            dataset.to_parquet(processed_data_path)
            mlflow.log_artifact(str(processed_data_path))
            
            # Log dataset characteristics
            mlflow.log_param("dataset_shape", dataset.shape)
            mlflow.log_param("business_context", json.dumps(business_context))
            mlflow.log_metric("n_features", dataset.shape[1])
            mlflow.log_metric("n_samples", dataset.shape[0])
            
            # Basic statistics with pandas
            try:
                eda_results['statistics'] = dataset.describe(include='all').to_dict()
                
                # Save statistics to artifacts
                stats_path = artifacts_dir / "statistics.json"
                with open(stats_path, "w") as f:
                    json.dump(eda_results['statistics'], f, indent=2)
                mlflow.log_artifact(str(stats_path))
                
                # Log key statistics to MLflow
                for col in dataset.select_dtypes(include=[np.number]).columns[:10]:
                    mlflow.log_metric(f"mean_{col}", dataset[col].mean())
                    mlflow.log_metric(f"std_{col}", dataset[col].std())
                    mlflow.log_metric(f"missing_rate_{col}", dataset[col].isna().sum() / len(dataset))
                
                # Generate and save pandas-profiling report
                profile = ProfileReport(dataset, minimal=False)
                profile_path = artifacts_dir / "eda_profile_report.html"
                profile.to_file(str(profile_path))
                mlflow.log_artifact(str(profile_path))
                eda_results['profile_report'] = str(profile_path)
                
            except Exception as e:
                mlflow.log_param("statistics_error", str(e))
                eda_results['statistics'] = self.calculate_robust_statistics_pandas(dataset)
                self.log_warning(f"Using robust statistics due to: {e}")
            
            # Generate visualizations with Python libraries
            viz_dir = artifacts_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            viz_configs = [
                ('distribution', px.histogram, {'nbins': 30}),
                ('correlation', sns.heatmap, {'annot': True}),
                ('pairplot', sns.pairplot, {'diag_kind': 'kde'}),
                ('outliers', px.box, {'points': 'outliers'})
            ]
            
            for viz_name, viz_func, viz_params in viz_configs:
                try:
                    viz = self.create_python_visualization(
                        dataset, viz_func, viz_params
                    )
                    eda_results['visualizations'].append(viz)
                    
                    # Save visualization as artifact
                    viz_path = viz_dir / f"{viz_name}_plot.png"
                    plt.savefig(viz_path)
                    mlflow.log_artifact(str(viz_path))
                    plt.close()
                    
                except Exception as e:
                    mlflow.log_param(f"{viz_name}_viz_error", str(e))
                    simple_viz = self.create_matplotlib_fallback(dataset, viz_name)
                    eda_results['visualizations'].append(simple_viz)
            
            # Generate insights using Python statistical libraries
            from scipy import stats
            from sklearn.preprocessing import StandardScaler
            
            insights_methods = [
                self.statistical_insights_scipy,
                self.pattern_detection_sklearn,
                self.anomaly_identification_pyod
            ]
            
            insights_path = artifacts_dir / "insights.json"
            for method in insights_methods:
                try:
                    insights = method(dataset, business_context)
                    eda_results['insights'].extend(insights)
                    mlflow.log_param(f"{method.__name__}_insights", len(insights))
                except Exception as e:
                    self.log_info(f"Insight method {method.__name__} skipped: {e}")
            
            # Save insights
            with open(insights_path, "w") as f:
                json.dump(eda_results['insights'], f, indent=2)
            mlflow.log_artifact(str(insights_path))
            
            # Log insights summary
            mlflow.log_metric("total_insights", len(eda_results['insights']))
            
            # Create test JSON for downstream tasks
            test_sample_path = artifacts_dir / "eda_test_sample.json"
            test_sample = dataset.head(10).to_dict(orient='records')
            with open(test_sample_path, "w") as f:
                json.dump(test_sample, f, indent=2)
            mlflow.log_artifact(str(test_sample_path))
            
            # Create interactive Streamlit dashboard code
            try:
                dashboard_code = self.generate_streamlit_dashboard(eda_results)
                eda_results['dashboard_code'] = dashboard_code
                
                # Save dashboard code to scripts directory
                dashboard_path = scripts_dir / "eda_dashboard.py"
                with open(dashboard_path, "w") as f:
                    f.write(dashboard_code)
                mlflow.log_artifact(str(dashboard_path))
                
            except Exception as e:
                # Fallback to static Jupyter notebook
                notebook_path = notebooks_dir / "eda_report.ipynb"
                notebook = self.create_jupyter_report(eda_results, notebook_path)
                eda_results['notebook_path'] = str(notebook_path)
                mlflow.log_artifact(str(notebook_path))
            
            # Create requirements.txt for this stage
            requirements_path = code_dir / "requirements.txt"
            with open(requirements_path, "w") as f:
                f.write("pandas>=2.0.0\nnumpy>=1.24.0\nmlflow>=2.9.0\n")
                f.write("plotly>=5.17.0\nseaborn>=0.12.0\nmatplotlib>=3.7.0\n")
                f.write("pandas-profiling>=3.6.0\nscipy>=1.10.0\n")
            mlflow.log_artifact(str(requirements_path))
            
            # Save complete EDA results
            results_path = artifacts_dir / "eda_results.json"
            with open(results_path, "w") as f:
                json.dump({k: v for k, v in eda_results.items() 
                          if k != 'visualizations'}, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))
            
            mlflow.set_tag("eda_status", "success")
                
        except Exception as e:
            mlflow.log_param("eda_critical_error", str(e))
            mlflow.set_tag("eda_status", "failed")
            self.log_error(f"EDA failed catastrophically: {e}")
            eda_results = self.minimal_pandas_eda(dataset)
    
    return eda_results
```