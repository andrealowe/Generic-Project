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
#   ├── e002-data-wrangling/
#   │   ├── notebooks/         # Data exploration notebooks
#   │   ├── scripts/           # Data wrangling scripts
#   │   └── requirements.txt   # pandas, numpy, etc.
#   ├── e003-data-science/
#   │   ├── notebooks/         # EDA notebooks
#   │   ├── scripts/           # Dashboard scripts
#   │   └── requirements.txt   # plotly, seaborn, etc.
#   ├── e004-model-development/
#   │   ├── notebooks/         # Experimentation notebooks
#   │   ├── scripts/           # Training scripts
#   │   └── requirements.txt   # sklearn, xgboost, etc.
#   ├── e005-model-validation/
#   │   ├── notebooks/         # Validation notebooks
#   │   ├── scripts/           # Test scripts
#   │   └── requirements.txt   # fairlearn, evidently, etc.
#   ├── e006-mlops/
#   │   ├── scripts/           # API serving scripts
#   │   ├── configs/           # Deployment configs
#   │   └── requirements.txt   # fastapi, mlflow, etc.
#   └── e007-frontend/
#       ├── scripts/           # UI applications
#       └── requirements.txt   # streamlit, dash, etc.
#
# /mnt/artifacts/
#   ├── e002-data-wrangling/      # Data profiles, test JSONs
#   ├── e003-data-science/                   # Reports, visualizations
#   │   └── visualizations/    # Plot images
#   ├── e004-model-development/     # Models, metrics, test data
#   │   └── models/           # Saved model files
#   ├── e005-model-validation/      # Validation reports
#   └── e006-mlops/            # Configs, monitoring specs
#
# /mnt/data/customer_churn/     # Mounted Domino dataset
#   ├── e002-data-wrangling/      # Raw and synthetic data
#   ├── e003-data-science/                   # Processed datasets
#   ├── e004-model-development/     # Train/val/test splits
#   └── features/              # Feature store

# Each agent logs artifacts both locally and to MLflow
with mlflow.start_run(run_name="customer_churn_pipeline"):
    
    # Data Wrangler
    data = data_wrangler.acquire_data(specs)
    # Saves notebooks: /mnt/code/e002-data-wrangling/notebooks/
    # Saves scripts: /mnt/code/e002-data-wrangling/scripts/
    # Saves artifacts: /mnt/artifacts/e002-data-wrangling/
    # Saves data: /mnt/data/customer_churn/e002-data-wrangling/
    # Creates: /mnt/code/e002-data-wrangling/requirements.txt
    # Logs to MLflow: data samples, profile reports
    
    # Data Scientist
    eda_results = data_scientist.perform_eda(data)
    # Saves notebooks: /mnt/code/e003-data-science/notebooks/
    # Saves scripts: /mnt/code/e003-data-science/scripts/
    # Saves artifacts: /mnt/artifacts/e003-data-science/
    # Saves data: /mnt/data/customer_churn/e003-data-science/
    # Creates: /mnt/code/e003-data-science/requirements.txt
    # Logs to MLflow: profile report, plots, insights
    
    # Model Developer
    model = model_developer.develop_models(data)
    # Saves notebooks: /mnt/code/e004-model-development/notebooks/
    # Saves scripts: /mnt/code/e004-model-development/scripts/
    # Saves artifacts: /mnt/artifacts/e004-model-development/models/
    # Saves data: /mnt/data/customer_churn/e004-model-development/
    # Creates: /mnt/code/e004-model-development/requirements.txt
    # Logs to MLflow: registered models with signatures
    
    # MLOps Engineer
    deployment = mlops_engineer.deploy(model)
    # Saves scripts: /mnt/code/e006-mlops/scripts/
    # Saves configs: /mnt/code/e006-mlops/configs/
    # Saves artifacts: /mnt/artifacts/e006-mlops/
    # Creates: /mnt/code/e006-mlops/requirements.txt
    # Logs to MLflow: deployment specs, monitoring config
    
    # Front-End Developer (recommends Streamlit)
    frontend = frontend_developer.create_app(model, requirements)
    # Saves scripts: /mnt/code/e007-frontend/scripts/
    # Saves artifacts: /mnt/artifacts/e007-frontend/
    # Creates: /mnt/code/e007-frontend/requirements.txt
    # Logs to MLflow: app code, Docker configs
    
    # Model Validator
    validation = model_validator.validate(model)
    # Saves notebooks: /mnt/code/e005-model-validation/notebooks/
    # Saves scripts: /mnt/code/e005-model-validation/scripts/
    # Saves artifacts: /mnt/artifacts/e005-model-validation/
    # Creates: /mnt/code/e005-model-validation/requirements.txt
    # Logs to MLflow: validation reports, test results
```