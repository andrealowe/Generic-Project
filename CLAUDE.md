# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a collection of specialized Claude Code agents designed for building end-to-end machine learning demonstrations on the Domino Data Lab platform. The agents work together to create production-ready ML solutions across the entire lifecycle.

## Directory Structure

The repository follows a standardized structure:

```
/mnt/
├── code/{stage}/           # Scripts, notebooks, requirements.txt
│   ├── notebooks/          # Jupyter notebooks for exploration
│   ├── scripts/           # Production Python scripts
│   └── requirements.txt   # Stage-specific dependencies
├── artifacts/{stage}/      # Models, reports, visualizations
│   ├── models/            # Saved model files
│   └── visualizations/    # Generated plots and reports
└── data/{project}/{stage}/ # Project-specific datasets
```

## Available Agents

### Core Agents
- **Master-Project-Manager-Agent**: Orchestrates complete ML pipelines with governance compliance
- **Data-Wrangler-Agent**: Data acquisition, generation, and pipeline management
- **Data-Scientist-Agent**: EDA, visualization, and insight generation
- **Model-Developer-Agent**: Model training and optimization
- **Model-Validator-Agent**: Performance validation, robustness testing, and governance compliance
- **Business-Analyst-Agent**: Requirements translation, success metrics, and governance assessment
- **MLOps-Engineer-Agent**: Deployment pipelines, monitoring, and compliance-aware CI/CD
- **Front-End-Developer-Agent**: UI development with technology recommendations

### Reference Documentation
- **Agent-Interaction-Protocol**: Communication patterns between agents
- **Example-Demonstration-Flows**: Workflow examples and file organization

### Governance Integration
- **Governance Policies**: Located in `/mnt/code/reference/governance/`
  - NIST Risk Management Framework (RMF)
  - Model Risk Management V3
  - Ethical AI Framework
  - Model Intake Process
  - External LLM Governance Policy
- **Approver Groups**: Defined in `/mnt/code/reference/governance/Approvers.md`
  - Modeling teams (modeling-review, modeling-practitioners, modeling-leadership)
  - IT teams (it-review, it-leadership)
  - Information Security (infosec-review, infosec-leadership)
  - Legal teams (legal-review, legal-leadership)
  - Line of Business (lob-leadership, lob-review)
  - Marketing teams (marketing-review, marketing-leadership)

## Technology Stack

- **Primary Language**: Python for all ML operations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Frameworks**: Streamlit (quick demos), Dash, Gradio, Panel, React/FastAPI
- **Experiment Tracking**: MLflow with parent-child run hierarchy
- **Deployment**: FastAPI, Flask, Docker, Domino Flows

## Key Patterns

### Agent Coordination
- Use Master-Project-Manager-Agent for complete end-to-end workflows with governance orchestration
- Individual agents can work independently for specific tasks
- All agents automatically create standardized directory structures
- Each stage produces requirements.txt for dependency management
- Governance compliance is automatically assessed and integrated into workflows

### Governance Workflow Patterns
- **Intake Phase**: Business-Analyst-Agent identifies applicable governance frameworks
- **Development Phase**: Model-Validator-Agent ensures compliance with all frameworks
- **Deployment Phase**: MLOps-Engineer-Agent implements governance-compliant pipelines
- **Approval Workflow**: Master-Project-Manager-Agent coordinates multi-stage approvals
- **Continuous Compliance**: Ongoing monitoring and governance validation

### MLflow Integration
- All experiments, metrics, and artifacts are logged to MLflow
- Models are registered with signatures and input examples
- Parent-child run relationships track complex pipelines
- Comprehensive artifact tracking at each stage

### File Organization
- Each agent creates its own stage directory under `/mnt/code/{stage}/`
- Notebooks go in `{stage}/notebooks/`, scripts in `{stage}/scripts/`
- Artifacts saved to `/mnt/artifacts/{stage}/`
- Data organized by project: `/mnt/data/{project}/{stage}/`

## Common Usage Patterns

```python
# Quick demonstration with governance
"Create a credit risk model demo with synthetic data following ethical AI guidelines"

# Full pipeline with compliance
"Build an end-to-end customer churn prediction system with dashboard meeting NIST RMF requirements"

# Governance-specific tasks
"Validate this model for compliance with model risk management framework"
"Generate governance compliance report for deployment approval"
"Assess this project for applicable governance frameworks and approval requirements"

# Specific tasks
"Generate synthetic financial data for fraud detection"
"Perform EDA on this dataset and create visualizations"
```

## Development Guidelines

- Always specify project names for proper organization
- Include business context for better agent recommendations
- Identify governance requirements early in project planning
- Test with small datasets before scaling
- Leverage Front-End-Developer-Agent's technology selection
- Use Model-Validator-Agent for governance and compliance validation
- Coordinate with appropriate approval groups based on project requirements
- Document compliance throughout the ML lifecycle