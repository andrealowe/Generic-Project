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
7. Ensure compliance with governance policies and frameworks
8. Execute governance-required validation checks

## Domino Integration Points
- Model registry validation gates
- Automated testing pipelines
- Performance monitoring setup
- Drift detection configuration
- Compliance documentation
- Governance policy enforcement (NIST RMF, Model Risk Management, Ethical AI)
- Approval workflow integration with designated reviewer groups

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
- Governance compliance reports following organizational frameworks
- Approval documentation for reviewer groups (modeling-review, infosec-review, legal-review, etc.)
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

def validate_governance_compliance(self, model, requirements, governance_policies):
    """Validate model against organizational governance policies and frameworks"""
    import mlflow
    import json
    from datetime import datetime

    compliance_results = {
        'nist_rmf': {},
        'model_risk_management': {},
        'ethical_ai': {},
        'external_llm_governance': {},
        'overall_compliance': 'PENDING'
    }

    with mlflow.start_run(run_name="governance_compliance_validation", nested=True) as run:
        mlflow.set_tag("validation_type", "governance_compliance")
        mlflow.log_param("governance_frameworks", list(governance_policies.keys()))

        try:
            # NIST RMF Compliance
            if 'nist_rmf' in governance_policies:
                nist_checks = {
                    'problem_definition': requirements.get('business_problem', '') != '',
                    'initial_risk_assessment': requirements.get('risk_assessment', '') != '',
                    'intended_datasets': requirements.get('data_sources', []) != [],
                    'project_kpis': requirements.get('target_metrics', '') != '',
                    'experimentation_results': requirements.get('experiment_results', '') != '',
                    'model_architecture': requirements.get('model_architecture', '') != '',
                    'model_limitations': requirements.get('model_limitations', '') != '',
                    'deployment_strategy': requirements.get('deployment_plan', '') != '',
                    'security_measures': requirements.get('security_measures', '') != ''
                }

                compliance_results['nist_rmf'] = {
                    'checks': nist_checks,
                    'compliance_score': sum(nist_checks.values()) / len(nist_checks),
                    'status': 'PASSED' if all(nist_checks.values()) else 'FAILED',
                    'required_approvers': ['andrea_lowe']  # From Example NIST RMF.yml
                }
                mlflow.log_metric("nist_rmf_compliance_score", compliance_results['nist_rmf']['compliance_score'])

            # Ethical AI Framework Compliance
            if 'ethical_ai' in governance_policies:
                ethical_checks = {
                    'ai_purpose_defined': requirements.get('ai_purpose', '') != '',
                    'stakeholder_impact_analyzed': requirements.get('stakeholder_impact', '') != '',
                    'bias_mitigation_implemented': requirements.get('bias_mitigation', False),
                    'fairness_metrics_defined': requirements.get('fairness_metrics', []) != [],
                    'transparency_level_appropriate': requirements.get('transparency_level', '') != '',
                    'explanation_methods_documented': requirements.get('explanation_methods', '') != '',
                    'data_governance_practices': requirements.get('data_governance', []) != [],
                    'privacy_impact_assessed': requirements.get('privacy_assessment', '') != '',
                    'ongoing_monitoring_planned': requirements.get('monitoring_plan', '') != '',
                    'ethical_framework_compliance': requirements.get('ethical_frameworks', []) != []
                }

                compliance_results['ethical_ai'] = {
                    'checks': ethical_checks,
                    'compliance_score': sum(ethical_checks.values()) / len(ethical_checks),
                    'status': 'PASSED' if all(ethical_checks.values()) else 'FAILED',
                    'required_approvers': ['anthony_huinker']  # From Example EthicalAI.yml
                }
                mlflow.log_metric("ethical_ai_compliance_score", compliance_results['ethical_ai']['compliance_score'])

            # Model Risk Management Compliance
            if 'model_risk_management' in governance_policies:
                risk_mgmt_checks = {
                    'business_case_defined': requirements.get('business_case', '') != '',
                    'change_plan_documented': requirements.get('change_plan', '') != '',
                    'model_building_described': requirements.get('model_building', '') != '',
                    'acceptance_testing_performed': requirements.get('acceptance_testing', False),
                    'user_acceptance_test_passed': requirements.get('uat_passed', False),
                    'validation_reports_reviewed': requirements.get('validation_reports', False),
                    'explainability_report_checked': requirements.get('explainability_report', False),
                    'monitoring_plan_defined': requirements.get('monitoring_plan', '') != '',
                    'model_usage_described': requirements.get('model_usage', '') != ''
                }

                compliance_results['model_risk_management'] = {
                    'checks': risk_mgmt_checks,
                    'compliance_score': sum(risk_mgmt_checks.values()) / len(risk_mgmt_checks),
                    'status': 'PASSED' if all(risk_mgmt_checks.values()) else 'FAILED',
                    'required_approvers': ['model-gov-org']  # From Example ModelRiskManagementV3.yml
                }
                mlflow.log_metric("model_risk_mgmt_compliance_score", compliance_results['model_risk_management']['compliance_score'])

            # Model Intake Process Compliance
            if 'model_intake' in governance_policies:
                intake_checks = {
                    'model_purpose_defined': requirements.get('model_purpose', '') != '',
                    'consumption_method_specified': requirements.get('consumption_method', '') != '',
                    'technology_stack_documented': requirements.get('technology_stack', '') != '',
                    'data_sources_identified': requirements.get('data_sources', []) != [],
                    'legal_review_required': requirements.get('legal_review_needed', False),
                    'infosec_review_required': requirements.get('infosec_review_needed', False),
                    'governance_review_completed': requirements.get('governance_review', False)
                }

                compliance_results['model_intake'] = {
                    'checks': intake_checks,
                    'compliance_score': sum(intake_checks.values()) / len(intake_checks),
                    'status': 'PASSED' if all(intake_checks.values()) else 'FAILED',
                    'required_approvers': ['model-gov-org', 'infosec-review', 'legal-review']
                }
                mlflow.log_metric("model_intake_compliance_score", compliance_results['model_intake']['compliance_score'])

            # External LLM Governance (if applicable)
            if 'external_llm_governance' in governance_policies and requirements.get('model_type') == 'llm':
                llm_checks = {
                    'vendor_agreement_exists': requirements.get('vendor_agreement', False),
                    'hosting_location_specified': requirements.get('hosting_location', '') != '',
                    'automated_testing_performed': requirements.get('automated_testing', False),
                    'safety_bias_checks_completed': requirements.get('safety_bias_checks', False),
                    'private_preview_conducted': requirements.get('private_preview', False),
                    'early_access_feedback_collected': requirements.get('early_access_feedback', False),
                    'vulnerability_scan_performed': requirements.get('vulnerability_scan', False),
                    'deployment_readiness_confirmed': requirements.get('deployment_ready', False)
                }

                compliance_results['external_llm_governance'] = {
                    'checks': llm_checks,
                    'compliance_score': sum(llm_checks.values()) / len(llm_checks),
                    'status': 'PASSED' if all(llm_checks.values()) else 'FAILED',
                    'required_approvers': ['anthony_huinker']
                }
                mlflow.log_metric("external_llm_compliance_score", compliance_results['external_llm_governance']['compliance_score'])

            # Overall compliance assessment
            all_frameworks = [v for v in compliance_results.values() if isinstance(v, dict) and 'status' in v]
            overall_passed = all(framework['status'] == 'PASSED' for framework in all_frameworks)
            compliance_results['overall_compliance'] = 'PASSED' if overall_passed else 'FAILED'

            # Generate approval workflow requirements
            all_approvers = set()
            for framework in all_frameworks:
                if 'required_approvers' in framework:
                    all_approvers.update(framework['required_approvers'])

            compliance_results['approval_workflow'] = {
                'required_approvers': list(all_approvers),
                'approval_order': [
                    'modeling-review',
                    'infosec-review',
                    'legal-review',
                    'model-gov-org'
                ],
                'governance_gates': [
                    {
                        'gate': 'ethical_ai_review',
                        'approvers': ['anthony_huinker'],
                        'required': 'ethical_ai' in governance_policies
                    },
                    {
                        'gate': 'risk_management_review',
                        'approvers': ['model-gov-org'],
                        'required': 'model_risk_management' in governance_policies
                    },
                    {
                        'gate': 'nist_rmf_review',
                        'approvers': ['andrea_lowe'],
                        'required': 'nist_rmf' in governance_policies
                    }
                ]
            }

            # Log compliance results
            mlflow.set_tag("overall_compliance_status", compliance_results['overall_compliance'])
            mlflow.log_dict(compliance_results, "governance_compliance_results.json")

            # Create compliance report
            compliance_report = self.generate_compliance_report(compliance_results, governance_policies)
            with open("governance_compliance_report.html", "w") as f:
                f.write(compliance_report)
            mlflow.log_artifact("governance_compliance_report.html")

            return compliance_results

        except Exception as e:
            mlflow.log_param("compliance_validation_error", str(e))
            mlflow.set_tag("compliance_status", "error")
            compliance_results['overall_compliance'] = 'ERROR'
            compliance_results['error'] = str(e)
            return compliance_results

def generate_compliance_report(self, compliance_results, governance_policies):
    """Generate HTML compliance report for governance review"""
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Governance Compliance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; }}
            .framework {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
            .passed {{ background-color: #d4edda; }}
            .failed {{ background-color: #f8d7da; }}
            .approvers {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Governance Compliance Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            <p>Overall Status: <strong>{compliance_results['overall_compliance']}</strong></p>
        </div>
    """

    # Add framework-specific results
    for framework_name, results in compliance_results.items():
        if isinstance(results, dict) and 'status' in results:
            status_class = 'passed' if results['status'] == 'PASSED' else 'failed'
            html_report += f"""
            <div class="framework {status_class}">
                <h2>{framework_name.replace('_', ' ').title()}</h2>
                <p>Status: <strong>{results['status']}</strong></p>
                <p>Compliance Score: {results.get('compliance_score', 0):.2%}</p>
                <h3>Requirements Check:</h3>
                <ul>
            """

            for check, passed in results.get('checks', {}).items():
                check_status = "✓" if passed else "✗"
                html_report += f"<li>{check_status} {check.replace('_', ' ').title()}</li>"

            html_report += "</ul>"

            if 'required_approvers' in results:
                html_report += f"""
                <div class="approvers">
                    <h4>Required Approvers:</h4>
                    <ul>
                """
                for approver in results['required_approvers']:
                    html_report += f"<li>{approver}</li>"
                html_report += "</ul></div>"

            html_report += "</div>"

    # Add approval workflow section
    if 'approval_workflow' in compliance_results:
        html_report += """
        <div class="framework">
            <h2>Approval Workflow</h2>
            <h3>Required Approval Groups:</h3>
            <ul>
        """
        for approver in compliance_results['approval_workflow']['required_approvers']:
            html_report += f"<li>{approver}</li>"

        html_report += """
            </ul>
            <h3>Governance Gates:</h3>
            <ul>
        """
        for gate in compliance_results['approval_workflow']['governance_gates']:
            required_text = "Required" if gate['required'] else "Optional"
            html_report += f"<li><strong>{gate['gate']}:</strong> {', '.join(gate['approvers'])} ({required_text})</li>"

        html_report += "</ul></div>"

    html_report += "</body></html>"
    return html_report
```