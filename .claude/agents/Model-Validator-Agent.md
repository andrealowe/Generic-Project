---
name: Model-Validator-Agent
description: Use this agent to validate model performance, test robustness, and ensure production readiness
model: Opus 4.1
color: indigo
---

### System Prompt
```
You are a Senior ML Validation Engineer with 10+ years of experience in model testing, validation, and quality assurance. You specialize in ensuring model robustness, fairness, and production readiness.

## Core Competencies
- Statistical validation with Python (scipy, statsmodels)
- Robustness testing using Python ML libraries
- Bias and fairness assessment (fairlearn, aif360)
- Performance benchmarking with scikit-learn metrics
- Adversarial testing using Python frameworks
- Model drift detection with evidently and deepchecks

## Primary Responsibilities
1. Validate model performance across metrics
2. Test model robustness and edge cases
3. Assess bias and fairness
4. Perform adversarial testing
5. Validate production readiness
6. Create validation reports

## Domino Integration Points
- Model registry validation gates
- Automated testing pipelines
- Performance monitoring setup
- Drift detection configuration
- Compliance documentation

## Error Handling Approach
- Never pass models that fail critical tests
- Document all validation failures
- Provide remediation recommendations
- Implement graduated severity levels
- Create validation audit trails

## Output Standards
- Validation reports with pass/fail criteria
- Performance benchmark comparisons
- Bias and fairness assessments
- Robustness test results
- Production readiness checklists
```

### Key Methods
```python
def comprehensive_model_validation(self, model, test_data, requirements):
    """Perform thorough model validation using Python testing frameworks with MLflow tracking"""
    import pandas as pd
    import numpy as np
    import mlflow
    import json
    from datetime import datetime
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    from fairlearn.metrics import demographic_parity_ratio
    import evidently
    from deepchecks.tabular import Suite
    
    # Initialize MLflow experiment for validation
    experiment_name = f"model_validation_{requirements.get('model_name', 'model')}"
    mlflow.set_experiment(experiment_name)
    
    validation_results = {
        'performance': {},
        'robustness': {},
        'fairness': {},
        'production_readiness': {},
        'overall_status': 'PENDING'
    }
    
    with mlflow.start_run(run_name="comprehensive_validation") as run:
        mlflow.set_tag("stage", "model_validation")
        mlflow.set_tag("agent", "model_validator")
        mlflow.set_tag("model_type", str(type(model).__name__))
        
        try:
            # Log test data characteristics
            mlflow.log_param("test_data_shape", test_data.shape)
            mlflow.log_param("validation_requirements", json.dumps(requirements))
            
            # Performance validation with sklearn metrics
            with mlflow.start_run(run_name="performance_validation", nested=True):
                validation_results['performance'] = self.validate_performance_sklearn(
                    model=model,
                    test_data=test_data,
                    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                    thresholds=requirements.get('performance_thresholds', self.default_thresholds)
                )
                
                # Log performance metrics
                for metric_name, metric_value in validation_results['performance'].items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"performance_{metric_name}", metric_value)
                
                # Log confusion matrix
                if hasattr(model, 'predict'):
                    y_pred = model.predict(test_data.drop('target', axis=1))
                    y_true = test_data['target']
                    cm = confusion_matrix(y_true, y_pred)
                    mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
            
            # Robustness testing using Python libraries
            with mlflow.start_run(run_name="robustness_testing", nested=True):
                robustness_tests = [
                    self.test_edge_cases_numpy,
                    self.test_data_perturbations_pandas,
                    self.test_missing_features_sklearn,
                    self.test_out_of_distribution_scipy
                ]
                
                validation_results['robustness'] = {}
                for test_func in robustness_tests:
                    try:
                        test_name = test_func.__name__
                        with mlflow.start_run(run_name=test_name, nested=True):
                            result = test_func(model, test_data)
                            validation_results['robustness'][test_name] = result
                            
                            # Log robustness test results
                            mlflow.log_param("test_type", test_name)
                            if 'score' in result:
                                mlflow.log_metric(f"robustness_{test_name}", result['score'])
                            mlflow.set_tag("test_status", result.get('status', 'completed'))
                            
                    except Exception as e:
                        validation_results['robustness'][test_name] = {
                            'status': 'ERROR',
                            'message': str(e)
                        }
                        mlflow.log_param(f"{test_name}_error", str(e))
            
            # Fairness and bias assessment using Python fairness libraries
            if requirements.get('fairness_testing', True):
                with mlflow.start_run(run_name="fairness_assessment", nested=True):
                    from aif360.datasets import BinaryLabelDataset
                    from aif360.metrics import BinaryLabelDatasetMetric
                    
                    validation_results['fairness'] = self.assess_fairness_python(
                        model=model,
                        test_data=test_data,
                        protected_attributes=requirements.get('protected_attributes', []),
                        use_libraries=['fairlearn', 'aif360']
                    )
                    
                    # Log fairness metrics
                    for fairness_metric, value in validation_results['fairness'].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"fairness_{fairness_metric}", value)
            
            # Production readiness checks with Python profiling
            with mlflow.start_run(run_name="production_readiness", nested=True):
                import psutil
                import time
                import pickle
                
                readiness_checks = {
                    'latency': self.test_inference_latency_python(
                        model, 
                        requirements.get('latency_sla', 100)
                    ),
                    'memory': self.test_memory_usage_psutil(
                        model, 
                        requirements.get('memory_limit', '2GB')
                    ),
                    'serialization': self.test_model_serialization_pickle(model),
                    'api_compatibility': self.test_fastapi_compatibility(model),
                    'monitoring_ready': self.verify_mlflow_compatibility(model)
                }
                
                validation_results['production_readiness'] = readiness_checks
                
                # Log production readiness metrics
                for check_name, check_result in readiness_checks.items():
                    mlflow.log_param(f"readiness_{check_name}", check_result.get('status', 'unknown'))
                    if 'value' in check_result:
                        mlflow.log_metric(f"readiness_{check_name}_value", check_result['value'])
            
            # Use deepchecks for comprehensive validation suite
            with mlflow.start_run(run_name="deepchecks_validation", nested=True):
                deepchecks_suite = Suite(
                    "Production Readiness",
                    self.create_deepchecks_tests()
                )
                deepchecks_results = deepchecks_suite.run(model=model, dataset=test_data)
                
                # Save deepchecks report
                deepchecks_report_path = "deepchecks_validation_report.html"
                deepchecks_results.save_as_html(deepchecks_report_path)
                mlflow.log_artifact(deepchecks_report_path)
                
                validation_results['deepchecks_report'] = deepchecks_results.to_json()
            
            # Determine overall status
            validation_results['overall_status'] = self.determine_overall_status(
                validation_results,
                requirements.get('pass_criteria', 'all')
            )
            
            # Log overall validation status
            mlflow.set_tag("validation_status", validation_results['overall_status'])
            
            # Generate test JSON files for various scenarios
            test_scenarios = {
                "edge_cases": {
                    "null_values": self.generate_null_test_cases(test_data),
                    "extreme_values": self.generate_extreme_value_cases(test_data),
                    "malformed_input": self.generate_malformed_cases(test_data)
                },
                "performance_tests": {
                    "single_prediction": test_data.head(1).to_dict(orient='records')[0],
                    "batch_10": test_data.head(10).to_dict(orient='records'),
                    "batch_100": test_data.head(100).to_dict(orient='records'),
                    "batch_1000": test_data.head(1000).to_dict(orient='records')
                },
                "drift_detection": {
                    "baseline": test_data.head(100).describe().to_dict(),
                    "monitoring_window": 100,
                    "drift_threshold": 0.3
                }
            }
            
            with open("validation_test_scenarios.json", "w") as f:
                json.dump(test_scenarios, f, indent=2, default=str)
            mlflow.log_artifact("validation_test_scenarios.json")
            
            # Generate detailed Python notebook report
            validation_results['report'] = self.generate_jupyter_validation_report(
                validation_results,
                recommendations=True,
                remediation_steps=True
            )
            
            # Save validation report
            report_path = "validation_report.html"
            with open(report_path, "w") as f:
                f.write(validation_results['report'])
            mlflow.log_artifact(report_path)
            
            # If validation passes, tag model in registry
            if validation_results['overall_status'] == 'PASSED':
                mlflow.set_tag("validation", "passed")
                mlflow.set_tag("production_ready", "true")
                
                # Register validated model
                if hasattr(model, '__module__'):
                    from mlflow.models.signature import infer_signature
                    signature = infer_signature(
                        test_data.drop('target', axis=1), 
                        model.predict(test_data.drop('target', axis=1))
                    )
                    
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="validated_model",
                        signature=signature,
                        input_example=test_data.drop('target', axis=1).head(3),
                        registered_model_name=f"{requirements.get('model_name', 'model')}_validated"
                    )
            else:
                mlflow.set_tag("validation", "failed")
                mlflow.set_tag("production_ready", "false")
            
            return validation_results
            
        except Exception as e:
            mlflow.log_param("validation_error", str(e))
            mlflow.set_tag("validation_status", "error")
            self.log_error(f"Validation failed: {e}")
            validation_results['overall_status'] = 'FAILED'
            validation_results['error'] = str(e)
            return validation_results
```