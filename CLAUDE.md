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
- **Master-Project-Manager-Agent**: Orchestrates complete ML pipelines
- **Data-Wrangler-Agent**: Data acquisition, generation, and pipeline management
- **Data-Scientist-Agent**: EDA, visualization, and insight generation
- **Model-Developer-Agent**: Model training and optimization
- **Model-Validator-Agent**: Performance validation and robustness testing
- **Business-Analyst-Agent**: Requirements translation and success metrics
- **MLOps-Engineer-Agent**: Deployment pipelines and monitoring
- **Front-End-Developer-Agent**: UI development with technology recommendations

### Reference Documentation
- **Agent-Interaction-Protocol**: Communication patterns between agents
- **Example-Demonstration-Flows**: Workflow examples and file organization

## Technology Stack

- **Primary Language**: Python for all ML operations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **UI Frameworks**: Streamlit (quick demos), Dash, Gradio, Panel, React/FastAPI
- **Experiment Tracking**: MLflow with parent-child run hierarchy
- **Deployment**: FastAPI, Flask, Docker, Domino Flows

## Key Patterns

### Agent Coordination
- Use Master-Project-Manager-Agent for complete end-to-end workflows
- Individual agents can work independently for specific tasks
- All agents automatically create standardized directory structures
- Each stage produces requirements.txt for dependency management

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
# Quick demonstration
"Create a credit risk model demo with synthetic data"

# Full pipeline
"Build an end-to-end customer churn prediction system with dashboard"

# Specific tasks
"Generate synthetic financial data for fraud detection"
"Perform EDA on this dataset and create visualizations"
```

## Development Guidelines

- Always specify project names for proper organization
- Include business context for better agent recommendations
- Test with small datasets before scaling
- Leverage Front-End-Developer-Agent's technology selection
- Use Model-Validator-Agent for governance and compliance demos