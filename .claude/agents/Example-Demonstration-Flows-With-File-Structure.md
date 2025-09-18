---
name: Example-Demonstration-Flows
description: Reference documentation for demonstration workflows - not an executable agent
model: none
color: gray
---

### Quick Proof of Concept
```python
# Requirements favor speed
requirements = {
    "project": "quick_demo",
    "deployment_urgency": "urgent",
    "ui_complexity": "low"
}
# Front-end agent recommends: Gradio
# Files organized as:
# - Code/Notebooks: /mnt/code/{stage}/notebooks/
# - Scripts: /mnt/code/{stage}/scripts/
# - Artifacts: /mnt/artifacts/{stage}/
# - Data: /mnt/data/quick_demo/{stage}/

# All agents automatically:
# 1. Create their stage directories with subdirectories
# 2. Save notebooks/scripts to /mnt/code/{stage}/
# 3. Save artifacts to /mnt/artifacts/{stage}/
# 4. Save data to /mnt/data/{project}/{stage}/
# 5. Create requirements.txt in each stage directory
# 6. Register everything with MLflow
```

### Enterprise Dashboard with Full Pipeline
```python
# Complete pipeline example showing file organization
requirements = {
    "project": "customer_churn",
    "target_metric": "f1_score",
    "deployment_strategy": "canary",
    "expected_users": 100,
    "ui_complexity": "medium"
}

# Directory structure created:
# /mnt/code/
#   ├── data_acquisition/
#   │   ├── notebooks/         # Data exploration notebooks
#   │   ├── scripts/           # Data wrangling scripts
#   │   └── requirements.txt   # pandas, numpy, etc.
#   ├── eda/
#   │   ├── notebooks/         # EDA notebooks
#   │   ├── scripts/           # Dashboard scripts
#   │   └── requirements.txt   # plotly, seaborn, etc.
#   ├── model_development/
#   │   ├── notebooks/         # Experimentation notebooks
#   │   ├── scripts/           # Training scripts
#   │   └── requirements.txt   # sklearn, xgboost, etc.
#   ├── model_validation/
#   │   ├── notebooks/         # Validation notebooks
#   │   ├── scripts/           # Test scripts
#   │   └── requirements.txt   # fairlearn, evidently, etc.
#   ├── deployment/
#   │   ├── scripts/           # API serving scripts
#   │   ├── configs/           # Deployment configs
#   │   └── requirements.txt   # fastapi, mlflow, etc.
#   └── frontend_development/
#       ├── scripts/           # UI applications
#       └── requirements.txt   # streamlit, dash, etc.
#
# /mnt/artifacts/
#   ├── data_acquisition/      # Data profiles, test JSONs
#   ├── eda/                   # Reports, visualizations
#   │   └── visualizations/    # Plot images
#   ├── model_development/     # Models, metrics, test data
#   │   └── models/           # Saved model files
#   ├── model_validation/      # Validation reports
#   └── deployment/            # Configs, monitoring specs
#
# /mnt/data/customer_churn/     # Mounted Domino dataset
#   ├── data_acquisition/      # Raw and synthetic data
#   ├── eda/                   # Processed datasets
#   ├── model_development/     # Train/val/test splits
#   └── features/              # Feature store

# Each agent logs artifacts both locally and to MLflow
with mlflow.start_run(run_name="customer_churn_pipeline"):
    
    # Data Wrangler
    data = data_wrangler.acquire_data(specs)
    # Saves notebooks: /mnt/code/data_acquisition/notebooks/
    # Saves scripts: /mnt/code/data_acquisition/scripts/
    # Saves artifacts: /mnt/artifacts/data_acquisition/
    # Saves data: /mnt/data/customer_churn/data_acquisition/
    # Creates: /mnt/code/data_acquisition/requirements.txt
    # Logs to MLflow: data samples, profile reports
    
    # Data Scientist
    eda_results = data_scientist.perform_eda(data)
    # Saves notebooks: /mnt/code/eda/notebooks/
    # Saves scripts: /mnt/code/eda/scripts/
    # Saves artifacts: /mnt/artifacts/eda/
    # Saves data: /mnt/data/customer_churn/eda/
    # Creates: /mnt/code/eda/requirements.txt
    # Logs to MLflow: profile report, plots, insights
    
    # Model Developer
    model = model_developer.develop_models(data)
    # Saves notebooks: /mnt/code/model_development/notebooks/
    # Saves scripts: /mnt/code/model_development/scripts/
    # Saves artifacts: /mnt/artifacts/model_development/models/
    # Saves data: /mnt/data/customer_churn/model_development/
    # Creates: /mnt/code/model_development/requirements.txt
    # Logs to MLflow: registered models with signatures
    
    # MLOps Engineer
    deployment = mlops_engineer.deploy(model)
    # Saves scripts: /mnt/code/deployment/scripts/
    # Saves configs: /mnt/code/deployment/configs/
    # Saves artifacts: /mnt/artifacts/deployment/
    # Creates: /mnt/code/deployment/requirements.txt
    # Logs to MLflow: deployment specs, monitoring config
    
    # Front-End Developer (recommends Streamlit)
    frontend = frontend_developer.create_app(model, requirements)
    # Saves scripts: /mnt/code/frontend_development/scripts/
    # Saves artifacts: /mnt/artifacts/frontend_development/
    # Creates: /mnt/code/frontend_development/requirements.txt
    # Logs to MLflow: app code, Docker configs
    
    # Model Validator
    validation = model_validator.validate(model)
    # Saves notebooks: /mnt/code/model_validation/notebooks/
    # Saves scripts: /mnt/code/model_validation/scripts/
    # Saves artifacts: /mnt/artifacts/model_validation/
    # Creates: /mnt/code/model_validation/requirements.txt
    # Logs to MLflow: validation reports, test results
```