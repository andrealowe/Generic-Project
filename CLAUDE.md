# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a collection of specialized Claude Code agents designed for building end-to-end machine learning demonstrations on the Domino Data Lab platform. The agents work together to create production-ready ML solutions across the entire lifecycle.

## Directory Structure

The repository follows a standardized structure:

```
/mnt/code/
├── src/                    # Production Python code
│   ├── api/               # API endpoints and serving
│   ├── models/            # Model training and preprocessing
│   ├── monitoring/        # Model monitoring and dashboards
│   └── data/              # Data generation and processing
├── scripts/                # Utility scripts and applications
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
│   └── business-analysis/ # Business requirements and analysis
└── data/                   # Data files (not committed to git)
    └── {project}/         # Project-specific datasets

/mnt/artifacts/
├── models/                 # Saved model files
├── reports/                # Analysis reports
└── visualizations/         # Generated plots and charts
```

**Key directories:**
- `src/` - All production Python code organized by function
- `scripts/` - Standalone scripts and frontend applications
- `notebooks/` - Exploratory Jupyter notebooks
- `config/` - Configuration and deployment specs
- `tests/` - All test files
- `docs/` - Documentation including business analysis
- `data/{project}/` - Project-specific datasets

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
- **Governance Policies**: Located in `/mnt/code/.reference/governance/`
  - NIST Risk Management Framework (RMF)
  - Model Risk Management V3
  - Ethical AI Framework
  - Model Intake Process
  - External LLM Governance Policy
- **Approver Groups**: Defined in `/mnt/code/.reference/governance/Approvers.md`
  - Modeling teams (modeling-review, modeling-practitioners, modeling-leadership)
  - IT teams (it-review, it-leadership)
  - Information Security (infosec-review, infosec-leadership)
  - Legal teams (legal-review, legal-leadership)
  - Line of Business (lob-leadership, lob-review)
  - Marketing teams (marketing-review, marketing-leadership)

## Domino Data Lab Platform

This project is designed for deployment on **Domino Data Lab**, an enterprise MLOps platform.

### Domino Resources
- **Official Documentation**: https://docs.dominodatalab.com
- **Platform Access**: Workspaces, Datasets, Jobs, Apps, Model APIs, Flows

### Key Domino Features Used
- **Domino Workspaces** - Interactive development environments
- **Domino Datasets** - Centralized data storage and versioning
- **Domino Apps** - Deploy Streamlit/Dash applications with `app.sh` launcher
- **Domino Model APIs** - Scalable model serving with monitoring
- **Domino Flows** - Orchestrate multi-step ML pipelines
- **Domino Jobs** - Scheduled and on-demand execution
- **MLflow Integration** - Built-in experiment tracking at http://localhost:8768
- **Git Integration** - Automatic version control

### Agent Instructions for Domino Deployment

When working with Domino-specific features, agents should:

1. **Reference Latest Documentation**
   - Use WebFetch to retrieve current docs from https://docs.dominodatalab.com
   - Follow Domino best practices for deployment patterns
   - Check for version-specific features and compatibility

2. **File System Structure**
   - Use `/mnt/code/` for all code (Git-synced)
   - Use `/mnt/data/` for datasets (not in Git)
   - Use `/mnt/artifacts/` for models and outputs (persisted)
   - All paths must be absolute, not relative

3. **App Deployment**
   - Create `app.sh` as the launcher script for Domino Apps
   - Configure proper ports (8050 for Dash, 8501 for Streamlit)
   - Include health checks and startup validation
   - Document hardware requirements (compute tier, GPU needs)

4. **API Deployment**
   - Use FastAPI or Flask for Model APIs
   - Implement `/health` endpoint for monitoring
   - Log predictions for drift detection
   - Configure authentication if required

5. **Environment Configuration**
   - Document required environment variables
   - Specify Python version and dependencies in requirements.txt
   - Note any system packages needed
   - Include Dockerfile if custom environment needed

6. **Best Practices**
   - Use Domino Datasets for large data files (>100MB)
   - Leverage Domino's built-in MLflow (no separate setup needed)
   - Use Domino Jobs for scheduled training/retraining
   - Deploy monitoring dashboards as Domino Apps
   - Use Domino Flows for production pipelines

## Technology Stack

- **Platform**: Domino Data Lab (MLOps orchestration)
- **Primary Language**: Python for all ML operations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Frameworks**: Streamlit (quick demos), Dash, Gradio, Panel, React/FastAPI
- **Experiment Tracking**: MLflow (built-in at http://localhost:8768)
- **Deployment**: FastAPI, Flask, Docker, Domino Flows, Domino Apps, Domino Model APIs

## Key Patterns

### Agent Coordination
- Use Master-Project-Manager-Agent for complete end-to-end workflows with governance orchestration
- Individual agents can work independently for specific tasks
- All agents use the standardized directory structure
- Production code goes in `src/`, scripts in `scripts/`, notebooks in `notebooks/`
- Dependencies managed in project-level `requirements.txt`
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
- **Production code** → `/mnt/code/src/` (organized by function: api, models, monitoring, data)
- **Scripts** → `/mnt/code/scripts/` (utility scripts, frontend apps)
- **Notebooks** → `/mnt/code/notebooks/` (exploratory analysis)
- **Tests** → `/mnt/code/tests/` (all test files)
- **Configuration** → `/mnt/code/config/` (deployment configs, settings)
- **Documentation** → `/mnt/code/docs/` (including business analysis)
- **Artifacts** → `/mnt/artifacts/` (models, reports, visualizations)
- **Data** → `/mnt/data/{project}/` (project-specific datasets)

## Project Development Workflow

### Starting a New Project

When you request ML project development, the **Master-Project-Manager-Agent** will ask you to select which Domino deployment features you want:

**Available Domino Features:**
1. **Domino Flows** - Automated ML pipeline orchestration
2. **Domino Launchers** - Self-service parameter-driven execution
3. **Domino Model APIs (Endpoints)** - REST API for real-time predictions
4. **Domino Apps** - Interactive web applications (Streamlit/Dash/Gradio)

You can choose:
- **All** - Full deployment stack (Flows + Launchers + Endpoints + Apps)
- **Specific features** - e.g., "Just endpoints and apps"
- **None** - Model training and testing only

The agent will invoke only the relevant sub-agents based on your selection.

## Common Usage Patterns

```python
# Full project with all Domino features
"Build a customer churn prediction model"
→ Agent asks: Which Domino features?
→ You respond: "All features"
→ Creates: Flows, Launchers, Endpoint, and Dashboard

# Specific deployment features
"Create a credit risk model with API endpoint and dashboard"
→ Agent asks: Which Domino features?
→ You respond: "Endpoints and Apps"
→ Creates: Model API endpoint + Streamlit dashboard (skips Flows and Launchers)

# Training only
"Train a fraud detection model and test it"
→ Agent asks: Which Domino features?
→ You respond: "None, just training and testing"
→ Creates: Model training + validation (skips all deployment)

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