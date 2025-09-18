---
name: Master-Project-Manager-Agent
description: Master orchestrator for end-to-end ML pipelines - coordinates all agents and manages complete project workflows
model: Opus 4.1
color: purple
---

### System Prompt
```
You are a Senior ML Project Manager and Solutions Architect with 15+ years of experience managing enterprise ML initiatives. You specialize in orchestrating end-to-end ML pipelines using Domino Data Lab's platform capabilities.

## Core Competencies
- Strategic ML project planning and execution
- Cross-functional team orchestration
- Risk management and mitigation
- Stakeholder communication and alignment
- Domino platform architecture expertise
- Agile and MLOps methodology integration

## Primary Responsibilities
1. Decompose business requirements into actionable ML tasks
2. Orchestrate sub-agents for optimal workflow execution
3. Manage project state and dependencies
4. Monitor progress and handle exceptions
5. Ensure governance and compliance requirements across all frameworks
6. Coordinate approval workflows with designated reviewer groups
7. Synthesize results into executive-ready insights

## Domino Integration Points
- Project creation and configuration
- Workflow orchestration via Domino Flows
- Resource allocation across hardware tiers
- Experiment tracking and comparison
- Model registry management
- Stakeholder access control
- Governance policy enforcement and compliance tracking
- Approval workflow coordination across reviewer groups

## Communication Protocol
When managing a project:
1. First, clarify business objectives and constraints
2. Identify applicable governance policies and frameworks
3. Create project plan with clear milestones and approval gates
4. Delegate tasks to appropriate sub-agents with governance context
5. Monitor execution and handle exceptions
6. Coordinate governance compliance validation and approvals
7. Validate deliverables against requirements and compliance standards
8. Present synthesized results with business impact and governance status

## Error Handling Strategy
- Always implement fallback plans for critical paths
- Log all decisions and state changes
- Gracefully degrade functionality when resources limited
- Provide clear status updates even during failures
- Maintain project continuity through checkpointing

## Output Standards
- Executive summaries with ROI metrics
- Technical documentation for reproducibility
- Risk assessments and mitigation plans
- Resource utilization reports
- Success criteria validation
```

### Key Methods
```python
def orchestrate_ml_pipeline(self, requirements):
    """Master orchestration with comprehensive error handling and MLflow tracking"""
    import mlflow
    import json
    from datetime import datetime
    
    # Initialize master MLflow experiment
    master_experiment = f"ml_pipeline_{requirements.get('project', 'demo')}"
    mlflow.set_experiment(master_experiment)
    
    try:
        with mlflow.start_run(run_name="master_pipeline_orchestration") as master_run:
            mlflow.set_tag("orchestrator", "project_manager")
            mlflow.set_tag("pipeline_type", requirements.get('pipeline_type', 'end_to_end'))
            mlflow.log_params({
                "project_name": requirements.get('project', 'demo'),
                "pipeline_stages": len(requirements.get('stages', [])),
                "target_metric": requirements.get('target_metric', 'accuracy')
            })
            
            # Initialize project in Domino
            project = self.initialize_domino_project(requirements)
            mlflow.log_param("domino_project_id", project.id)
            
            # Create execution plan
            execution_plan = self.create_execution_plan(requirements)
            mlflow.log_dict(execution_plan.to_dict(), "execution_plan.json")
            
            # Track pipeline metadata
            pipeline_metadata = {
                "start_time": datetime.now().isoformat(),
                "requirements": requirements,
                "stages": []
            }
            
            # Delegate to sub-agents with monitoring
            results = {}
            for stage in execution_plan.stages:
                with mlflow.start_run(run_name=f"stage_{stage.name}", nested=True) as stage_run:
                    try:
                        mlflow.set_tag("stage_name", stage.name)
                        mlflow.set_tag("agent_type", stage.agent_type)
                        
                        agent = self.select_agent(stage.type)
                        
                        # Pass MLflow run context to agent
                        stage.task['mlflow_run_id'] = stage_run.info.run_id
                        stage.task['mlflow_experiment'] = master_experiment
                        
                        # Execute stage
                        result = agent.execute(stage.task, timeout=stage.timeout)
                        results[stage.name] = result
                        
                        # Log stage results
                        self.log_progress(project, stage, result)
                        mlflow.log_metric(f"{stage.name}_duration", result.get('duration', 0))
                        mlflow.set_tag(f"{stage.name}_status", "success")
                        
                        # Track stage metadata
                        pipeline_metadata["stages"].append({
                            "name": stage.name,
                            "status": "success",
                            "mlflow_run_id": stage_run.info.run_id,
                            "metrics": result.get('metrics', {})
                        })
                        
                        # Special handling for model development stage
                        if stage.name == "model_development" and 'best_model_run_id' in result:
                            mlflow.set_tag("best_model_run_id", result['best_model_run_id'])
                            mlflow.log_param("best_model_name", result.get('best_model_name', 'unknown'))
                            mlflow.log_metric("best_model_score", result.get('best_model_score', 0))
                        
                    except Exception as e:
                        mlflow.set_tag(f"{stage.name}_status", "failed")
                        mlflow.log_param(f"{stage.name}_error", str(e))
                        
                        self.handle_stage_failure(stage, e)
                        results[stage.name] = self.execute_fallback(stage)
                        
                        pipeline_metadata["stages"].append({
                            "name": stage.name,
                            "status": "failed_with_fallback",
                            "error": str(e)
                        })
            
            # Synthesize and validate results
            final_output = self.synthesize_results(results)
            validation_passed = self.validate_against_requirements(final_output, requirements)
            
            # Log final pipeline metrics
            mlflow.log_metric("pipeline_stages_completed", len(results))
            mlflow.log_metric("pipeline_validation_passed", int(validation_passed))
            mlflow.set_tag("pipeline_status", "success" if validation_passed else "partial_success")
            
            # Save pipeline metadata
            pipeline_metadata["end_time"] = datetime.now().isoformat()
            pipeline_metadata["final_status"] = "success" if validation_passed else "partial_success"
            
            with open("pipeline_metadata.json", "w") as f:
                json.dump(pipeline_metadata, f, indent=2, default=str)
            mlflow.log_artifact("pipeline_metadata.json")
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(
                requirements, results, final_output
            )
            mlflow.log_text(executive_summary, "executive_summary.md")
            
            # Register the complete pipeline in MLflow
            if validation_passed and 'model' in final_output:
                from mlflow.models.signature import infer_signature
                
                # Register the pipeline as a complete solution
                mlflow.pyfunc.log_model(
                    artifact_path="complete_pipeline",
                    python_model=self.create_pipeline_wrapper(final_output),
                    registered_model_name=f"pipeline_{requirements.get('project', 'demo')}",
                    signature=infer_signature(
                        final_output.get('sample_input'),
                        final_output.get('sample_output')
                    ),
                    input_example=final_output.get('sample_input'),
                    pip_requirements=self.generate_requirements_txt(results)
                )
            
            # Create comprehensive test suite
            test_suite = self.generate_comprehensive_test_suite(
                results, final_output, requirements
            )
            with open("pipeline_test_suite.json", "w") as f:
                json.dump(test_suite, f, indent=2, default=str)
            mlflow.log_artifact("pipeline_test_suite.json")
            
            return final_output
        
    except Exception as e:
        mlflow.set_tag("pipeline_status", "critical_failure")
        mlflow.log_param("critical_error", str(e))
        self.log_critical_failure(e)
        return self.graceful_degradation(requirements)

def create_pipeline_wrapper(self, pipeline_output):
    """Create a PythonModel wrapper for the complete pipeline"""
    import mlflow.pyfunc
    
    class PipelineModel(mlflow.pyfunc.PythonModel):
        def __init__(self, components):
            self.components = components
            
        def predict(self, context, model_input):
            # Data preprocessing
            processed = self.components['preprocessor'].transform(model_input)
            # Feature engineering
            features = self.components['feature_engineer'].transform(processed)
            # Model prediction
            predictions = self.components['model'].predict(features)
            # Post-processing
            return self.components['postprocessor'].transform(predictions)
    
    return PipelineModel(pipeline_output['components'])

def orchestrate_governance_compliance(self, requirements, project_context):
    """Orchestrate governance compliance across all applicable frameworks"""
    import mlflow
    import json
    from datetime import datetime

    # Load governance policies from reference directory
    governance_policies = self.load_governance_policies()

    # Determine applicable frameworks based on project context
    applicable_frameworks = self.determine_applicable_frameworks(
        requirements, governance_policies
    )

    with mlflow.start_run(run_name="governance_orchestration", nested=True) as run:
        mlflow.set_tag("orchestration_type", "governance_compliance")
        mlflow.log_param("applicable_frameworks", applicable_frameworks)

        governance_results = {
            'frameworks': applicable_frameworks,
            'compliance_status': {},
            'approval_workflow': {},
            'overall_status': 'PENDING'
        }

        try:
            # Execute compliance validation for each framework
            for framework in applicable_frameworks:
                framework_result = self.validate_framework_compliance(
                    framework, requirements, governance_policies[framework]
                )
                governance_results['compliance_status'][framework] = framework_result

                # Log framework compliance
                mlflow.log_metric(f"{framework}_compliance_score",
                                framework_result.get('compliance_score', 0))

            # Generate unified approval workflow
            governance_results['approval_workflow'] = self.generate_approval_workflow(
                governance_results['compliance_status']
            )

            # Determine overall governance status
            all_passed = all(
                result.get('status') == 'PASSED'
                for result in governance_results['compliance_status'].values()
            )
            governance_results['overall_status'] = 'PASSED' if all_passed else 'REQUIRES_REVIEW'

            # Create governance dashboard
            governance_dashboard = self.create_governance_dashboard(governance_results)
            with open("governance_dashboard.html", "w") as f:
                f.write(governance_dashboard)
            mlflow.log_artifact("governance_dashboard.html")

            # Log governance orchestration results
            mlflow.log_dict(governance_results, "governance_orchestration_results.json")
            mlflow.set_tag("governance_status", governance_results['overall_status'])

            return governance_results

        except Exception as e:
            mlflow.log_param("governance_orchestration_error", str(e))
            governance_results['overall_status'] = 'ERROR'
            governance_results['error'] = str(e)
            return governance_results

def load_governance_policies(self):
    """Load governance policies from reference directory"""
    import yaml
    import os

    policies = {}
    governance_dir = "/mnt/code/reference/governance"

    if os.path.exists(governance_dir):
        policy_files = {
            'nist_rmf': 'Example NIST RMF.yml',
            'model_risk_management': 'Example ModelRiskManagementV3.yml',
            'ethical_ai': 'Example EthicalAI.yml',
            'model_intake': 'Example ModelIntake.yml',
            'external_llm_governance': 'Example External LLM Governance Policy.yml'
        }

        for policy_name, filename in policy_files.items():
            file_path = os.path.join(governance_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        policies[policy_name] = yaml.safe_load(f)
                except Exception as e:
                    self.log_error(f"Failed to load {policy_name}: {e}")

        # Load approver groups
        approvers_file = os.path.join(governance_dir, 'Approvers.md')
        if os.path.exists(approvers_file):
            policies['approver_groups'] = self.parse_approver_groups(approvers_file)

    return policies

def determine_applicable_frameworks(self, requirements, governance_policies):
    """Determine which governance frameworks apply to this project"""
    applicable = []

    # Always apply model intake if it's a model project
    if 'model' in requirements.get('project_type', '').lower():
        applicable.append('model_intake')

    # Apply NIST RMF for all ML projects
    if 'ml' in requirements.get('project_type', '').lower() or 'model' in requirements.get('project_type', '').lower():
        applicable.append('nist_rmf')

    # Apply ethical AI if specified or if dealing with sensitive data
    if (requirements.get('ethical_ai_required', False) or
        any(sensitive in str(requirements.get('data_sources', [])).lower()
            for sensitive in ['personal', 'pii', 'protected', 'sensitive'])):
        applicable.append('ethical_ai')

    # Apply model risk management for high-risk models
    if (requirements.get('risk_level', '').lower() == 'high' or
        requirements.get('regulatory_impact', False)):
        applicable.append('model_risk_management')

    # Apply external LLM governance for LLM projects
    if 'llm' in requirements.get('model_type', '').lower():
        applicable.append('external_llm_governance')

    return applicable

def generate_approval_workflow(self, compliance_status):
    """Generate unified approval workflow across all frameworks"""
    approval_workflow = {
        'required_approvers': set(),
        'approval_stages': [],
        'governance_gates': []
    }

    # Standard approver groups from Approvers.md
    approver_groups = {
        'modeling': ['modeling-review', 'modeling-practitioners', 'modeling-leadership'],
        'it': ['it-review', 'it-leadership'],
        'infosec': ['infosec-review', 'infosec-leadership'],
        'legal': ['legal-review', 'legal-ledarship'],  # Note: typo in original file
        'lob': ['lob-leadership', 'lob-review'],
        'marketing': ['marketing-review', 'marketing-leadership']
    }

    # Collect all required approvers from each framework
    for framework, result in compliance_status.items():
        if 'required_approvers' in result:
            approval_workflow['required_approvers'].update(result['required_approvers'])

    # Define approval order based on organizational hierarchy
    approval_order = [
        'modeling-review',
        'infosec-review',
        'legal-review',
        'model-gov-org',
        'andrea_lowe',  # NIST RMF approver
        'anthony_huinker'  # Ethical AI approver
    ]

    # Create approval stages
    for approver in approval_order:
        if approver in approval_workflow['required_approvers']:
            approval_workflow['approval_stages'].append({
                'approver': approver,
                'required_frameworks': [
                    framework for framework, result in compliance_status.items()
                    if approver in result.get('required_approvers', [])
                ],
                'can_approve_parallel': approver not in ['model-gov-org']  # Final approvals are sequential
            })

    # Convert set to list for JSON serialization
    approval_workflow['required_approvers'] = list(approval_workflow['required_approvers'])

    return approval_workflow

def create_governance_dashboard(self, governance_results):
    """Create comprehensive governance dashboard"""
    from datetime import datetime

    html_dashboard = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Project Governance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
            .status-overview {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .status-card {{ background: white; padding: 20px; border-radius: 10px;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); flex: 1; }}
            .framework-section {{ background: white; padding: 20px; border-radius: 10px;
                                margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .passed {{ border-left: 5px solid #28a745; }}
            .failed {{ border-left: 5px solid #dc3545; }}
            .pending {{ border-left: 5px solid #ffc107; }}
            .approval-workflow {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 15px; }}
            .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ ML Project Governance Dashboard</h1>
            <p>Comprehensive compliance tracking across organizational frameworks</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="status-overview">
            <div class="status-card">
                <h3>📊 Overall Status</h3>
                <h2 style="color: {'#28a745' if governance_results['overall_status'] == 'PASSED' else '#ffc107'};">
                    {governance_results['overall_status']}
                </h2>
            </div>
            <div class="status-card">
                <h3>📋 Frameworks</h3>
                <h2>{len(governance_results['frameworks'])}</h2>
                <p>Applicable governance frameworks</p>
            </div>
            <div class="status-card">
                <h3>👥 Approvers</h3>
                <h2>{len(governance_results['approval_workflow'].get('required_approvers', []))}</h2>
                <p>Required approval groups</p>
            </div>
        </div>
    """

    # Add framework compliance details
    for framework, result in governance_results['compliance_status'].items():
        status_class = result.get('status', 'PENDING').lower()
        if status_class == 'passed':
            status_class = 'passed'
        elif status_class == 'failed':
            status_class = 'failed'
        else:
            status_class = 'pending'

        compliance_score = result.get('compliance_score', 0)
        progress_width = compliance_score * 100

        html_dashboard += f"""
        <div class="framework-section {status_class}">
            <h2>{framework.replace('_', ' ').title()}</h2>
            <p>Status: <strong>{result.get('status', 'PENDING')}</strong></p>

            <div style="margin: 15px 0;">
                <p>Compliance Score: {compliance_score:.1%}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%;"></div>
                </div>
            </div>

            <h4>Requirements Checklist:</h4>
            <ul>
        """

        for check, passed in result.get('checks', {}).items():
            check_icon = "✅" if passed else "❌"
            html_dashboard += f"<li>{check_icon} {check.replace('_', ' ').title()}</li>"

        html_dashboard += "</ul>"

        if 'required_approvers' in result:
            html_dashboard += """
            <div class="approval-workflow">
                <h4>Required Approvers:</h4>
                <ul>
            """
            for approver in result['required_approvers']:
                html_dashboard += f"<li>👤 {approver}</li>"
            html_dashboard += "</ul></div>"

        html_dashboard += "</div>"

    # Add approval workflow section
    html_dashboard += """
    <div class="framework-section">
        <h2>🔄 Approval Workflow</h2>
        <p>Coordinated approval process across all applicable frameworks</p>

        <h3>Approval Stages:</h3>
        <ol>
    """

    for stage in governance_results['approval_workflow'].get('approval_stages', []):
        parallel_text = "(Parallel)" if stage.get('can_approve_parallel', False) else "(Sequential)"
        html_dashboard += f"""
        <li>
            <strong>{stage['approver']}</strong> {parallel_text}
            <br><small>Frameworks: {', '.join(stage['required_frameworks'])}</small>
        </li>
        """

    html_dashboard += """
        </ol>
    </div>

    <div class="framework-section">
        <h2>📞 Contact Information</h2>
        <h3>Approval Organizations:</h3>
        <ul>
            <li><strong>Modeling:</strong> modeling-review, modeling-practitioners, modeling-leadership</li>
            <li><strong>IT:</strong> it-review, it-leadership</li>
            <li><strong>Information Security:</strong> infosec-review, infosec-leadership</li>
            <li><strong>Legal:</strong> legal-review, legal-leadership</li>
            <li><strong>Line of Business:</strong> lob-leadership, lob-review</li>
            <li><strong>Marketing:</strong> marketing-review, marketing-leadership</li>
        </ul>
    </div>

    </body>
    </html>
    """

    return html_dashboard
```