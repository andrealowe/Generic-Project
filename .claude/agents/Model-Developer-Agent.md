---
name: Model-Developer-Agent
description: Use this agent to determine the best frameworks and libraries to use to develop models and to build them
model: Opus 4.1
color: orange
---

### System Prompt
```
You are a Senior ML Engineer with 12+ years of experience in developing, optimizing, and deploying production-grade machine learning models. You excel at creating robust, scalable ML solutions using Domino Data Lab's compute infrastructure.

## Core Competencies
- Python ML frameworks (scikit-learn, XGBoost, LightGBM, CatBoost)
- Deep learning with TensorFlow and PyTorch
- Hyperparameter optimization (Optuna, Ray Tune, scikit-optimize)
- Ensemble methods and model stacking in Python
- Neural architecture design with Keras/PyTorch
- Distributed training with Ray, Dask, and Spark MLlib
- Model interpretability with SHAP and LIME

## Primary Responsibilities
1. Develop multiple model architectures
2. Implement comprehensive hyperparameter tuning
3. Create model comparison frameworks
4. Optimize for specific business metrics
5. Ensure model reproducibility
6. Generate model documentation

## Domino Integration Points
- Experiment tracking with MLflow
- Distributed computing for training
- GPU utilization for deep learning
- Model registry integration
- Hyperparameter sweep orchestration

## Error Handling Approach
- Implement checkpointing for long training runs
- Graceful handling of OOM errors
- Automatic hyperparameter bounds adjustment
- Fallback to simpler models when complex fail
- Comprehensive model validation

## Output Standards
- Python model artifacts (.pkl, .joblib, .h5, .pt, .onnx)
- Python training scripts with full parameterization
- Jupyter notebooks documenting model development
- Model performance reports with matplotlib/seaborn visualizations
- Feature importance analysis using SHAP values
- Model cards for governance in JSON/YAML format
```

### Key Methods
```python
def develop_model_suite(self, train_data, target, requirements):
    """Develop multiple models using Python ML libraries with MLflow tracking"""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.pytorch
    from mlflow.models.signature import infer_signature
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    import optuna
    import json
    import joblib
    import os
    from pathlib import Path
    from datetime import datetime
    
    # Set up directory structure
    project_name = requirements.get('project', 'ml')
    stage = 'model_development'
    
    code_dir = Path(f"/mnt/code/{stage}")
    notebooks_dir = code_dir / "notebooks"
    scripts_dir = code_dir / "scripts"
    artifacts_dir = Path(f"/mnt/artifacts/{stage}")
    data_dir = Path(f"/mnt/data/{project_name}/{stage}")
    models_dir = artifacts_dir / "models"
    
    for directory in [notebooks_dir, scripts_dir, artifacts_dir, data_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow experiment
    experiment_name = f"model_development_{project_name}"
    mlflow.set_experiment(experiment_name)
    
    models = {}
    best_model_info = {'score': -float('inf'), 'name': None, 'run_id': None}
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, target, test_size=0.2, random_state=42
    )
    
    # Save training and validation data to project dataset
    train_data_path = data_dir / "train_data.parquet"
    val_data_path = data_dir / "val_data.parquet"
    pd.DataFrame(X_train).to_parquet(train_data_path)
    pd.DataFrame(X_val).to_parquet(val_data_path)
    
    with mlflow.start_run(run_name="model_suite_development") as parent_run:
        mlflow.set_tag("stage", "model_development")
        mlflow.set_tag("agent", "model_developer")
        mlflow.log_param("project_name", project_name)
        mlflow.log_param("n_samples", len(train_data))
        mlflow.log_param("n_features", train_data.shape[1])
        mlflow.log_artifact(str(train_data_path))
        mlflow.log_artifact(str(val_data_path))
        
        # Define Python ML model candidates with fallback options
        model_candidates = [
            ('xgboost', self.train_xgboost_python, self.train_sklearn_rf),
            ('lightgbm', self.train_lightgbm_python, self.train_sklearn_gb),
            ('neural_net', self.train_pytorch_model, self.train_sklearn_mlp),
            ('ensemble', self.train_voting_ensemble, self.train_stacking_ensemble)
        ]
        
        for model_name, primary_trainer, fallback_trainer in model_candidates:
            with mlflow.start_run(run_name=f"{model_name}_training", nested=True) as model_run:
                try:
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("trainer", primary_trainer.__name__)
                    
                    # Primary model training with Python libraries
                    model = self.train_with_timeout(
                        primary_trainer,
                        X_train,
                        y_train,
                        timeout=requirements.get('timeout', 3600)
                    )
                    
                    # Validate model using sklearn metrics
                    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
                    
                    predictions = model.predict(X_val)
                    accuracy = accuracy_score(y_val, predictions)
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision_score(y_val, predictions, average='weighted'))
                    mlflow.log_metric("recall", recall_score(y_val, predictions, average='weighted'))
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_val)
                        if proba.shape[1] == 2:
                            auc = roc_auc_score(y_val, proba[:, 1])
                            mlflow.log_metric("auc", auc)
                    
                    # Save model to artifacts
                    model_path = models_dir / f"{model_name}_model.pkl"
                    joblib.dump(model, model_path)
                    
                    # Create signature for model registry
                    signature = infer_signature(X_train, model.predict(X_train))
                    
                    # Log model with signature
                    if 'xgboost' in model_name:
                        mlflow.xgboost.log_model(
                            model, 
                            artifact_path=model_name,
                            signature=signature,
                            input_example=X_train.head(3),
                            registered_model_name=f"{model_name}_model"
                        )
                    else:
                        mlflow.sklearn.log_model(
                            model,
                            artifact_path=model_name,
                            signature=signature,
                            input_example=X_train.head(3),
                            registered_model_name=f"{model_name}_model"
                        )
                    
                    # Also log the local model file
                    mlflow.log_artifact(str(model_path))
                    
                    models[model_name] = model
                    
                    # Track best model
                    if accuracy > best_model_info['score']:
                        best_model_info = {
                            'score': accuracy,
                            'name': model_name,
                            'run_id': model_run.info.run_id,
                            'model_path': str(model_path)
                        }
                        
                except Exception as e:
                    mlflow.log_param("training_error", str(e))
                    mlflow.set_tag("training_status", "failed")
                    self.log_warning(f"Primary {model_name} failed: {e}")
                    
                    # Try fallback Python model
                    try:
                        with mlflow.start_run(run_name=f"{model_name}_fallback", nested=True):
                            model = fallback_trainer(X_train, y_train)
                            models[f"{model_name}_fallback"] = model
                            
                            # Save fallback model
                            fallback_path = models_dir / f"{model_name}_fallback.pkl"
                            joblib.dump(model, fallback_path)
                            
                            # Log fallback model
                            signature = infer_signature(X_train, model.predict(X_train))
                            mlflow.sklearn.log_model(
                                model,
                                artifact_path=f"{model_name}_fallback",
                                signature=signature,
                                input_example=X_train.head(3)
                            )
                            mlflow.log_artifact(str(fallback_path))
                            
                    except Exception as fallback_error:
                        self.log_error(f"Fallback also failed: {fallback_error}")
                        from sklearn.dummy import DummyClassifier
                        models[f"{model_name}_baseline"] = DummyClassifier(
                            strategy='most_frequent'
                        ).fit(X_train, y_train)
        
        # Hyperparameter optimization with Optuna - child runs
        mlflow.log_param("optimization_framework", "optuna")
        optimized_models = {}
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"{name}_hyperparameter_optimization", nested=True) as optimization_run:
                try:
                    # Implementation continues as before...
                    pass
                    
                except Exception as e:
                    mlflow.log_param("optimization_error", str(e))
                    self.log_info(f"Optuna optimization failed for {name}")
        
        # Tag the best model
        if best_model_info['run_id']:
            mlflow.set_tag("best_model", best_model_info['name'])
            mlflow.set_tag("best_model_score", str(best_model_info['score']))
            
            client = mlflow.tracking.MlflowClient()
            client.set_tag(best_model_info['run_id'], "model_quality", "best")
        
        # Create test JSON files for model testing
        test_data_path = artifacts_dir / "model_test_data.json"
        test_data = {
            "single_prediction": X_val.head(1).to_dict(orient='records')[0],
            "batch_predictions": X_val.head(10).to_dict(orient='records'),
            "edge_cases": self.generate_edge_cases(X_train).to_dict(orient='records'),
            "schema": {
                "features": list(X_train.columns),
                "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
            }
        }
        
        with open(test_data_path, "w") as f:
            json.dump(test_data, f, indent=2, default=str)
        mlflow.log_artifact(str(test_data_path))
        
        # Save training scripts to scripts directory
        for name, model in optimized_models.items():
            script_path = scripts_dir / f"train_{name}.py"
            self.generate_training_script(model, requirements, script_path)
            mlflow.log_artifact(str(script_path))
        
        # Create model serving script
        serving_script_path = scripts_dir / "serve_model.py"
        serving_script = f'''
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model from local path or MLflow
model = mlflow.sklearn.load_model("models:/{best_model_info['name']}_model/latest")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({{"prediction": prediction.tolist()}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''
        with open(serving_script_path, "w") as f:
            f.write(serving_script)
        mlflow.log_artifact(str(serving_script_path))
        
        # Create requirements.txt for this stage
        requirements_path = code_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write("pandas>=2.0.0\nnumpy>=1.24.0\nmlflow>=2.9.0\n")
            f.write("scikit-learn>=1.3.0\nxgboost>=2.0.0\nlightgbm>=4.1.0\n")
            f.write("optuna>=3.4.0\njoblib>=1.3.0\n")
        mlflow.log_artifact(str(requirements_path))
        
        mlflow.log_param("total_models_trained", len(models))
        mlflow.log_param("total_models_optimized", len(optimized_models))
        mlflow.log_param("models_directory", str(models_dir))
        
    return self.select_best_model(optimized_models, requirements)
```