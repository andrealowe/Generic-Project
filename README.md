# Domino ML Agents Template

> A comprehensive collection of specialized Claude Code agents for building end-to-end machine learning demonstrations on the Domino Data Lab platform.

![Domino ML Agents Overview](./temp/domino-ml-agents-overview.png)

## Quick Start

1. **Create a new Domino project** using this template
2. **Launch Claude Code** in your Domino workspace
3. **Start building** with one simple command:

```python
# For a complete ML pipeline
"Build an end-to-end customer churn prediction system with dashboard"

# For quick prototyping
"Create a fraud detection model demo with synthetic data"
```

## What's Included

This template provides **8 specialized agents** that work together to create production-ready ML solutions:

| Agent | Purpose | Key Capabilities |
|-------|---------|-----------------|
| **Master Project Manager** | Orchestration | End-to-end pipeline coordination |
| **Data Wrangler** | Data Management | Data acquisition, quality, pipelines |
| **Data Scientist** | Analysis & Insights | EDA, visualizations, feature engineering |
| **Model Developer** | ML Development | Training, optimization, experimentation |
| **Model Validator** | Quality Assurance | Performance validation, robustness testing |
| **Business Analyst** | Requirements | Business-to-technical translation |
| **MLOps Engineer** | Deployment | Production pipelines, monitoring |
| **Frontend Developer** | UI/UX | Dashboards, apps, visualizations |

![Agent Workflow](./temp/agent-workflow-diagram.png)

## Architecture

### Directory Structure
```
/mnt/
├── code/{stage}/           # Scripts, notebooks, requirements
│   ├── notebooks/          # Jupyter notebooks for exploration
│   ├── scripts/           # Production Python scripts
│   └── requirements.txt   # Stage-specific dependencies
├── artifacts/{stage}/      # Models, reports, visualizations
│   ├── models/            # Saved model files
│   └── visualizations/    # Generated plots and reports
└── data/{project}/{stage}/ # Project-specific datasets
```

### Technology Stack
- **Python**: Primary language for all ML operations
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **Experiment Tracking**: MLflow with comprehensive logging
- **UI Frameworks**: Streamlit, Dash, Gradio, Panel, React/FastAPI
- **Deployment**: FastAPI, Flask, Docker, Domino Flows

![Technology Stack](./temp/tech-stack-visualization.png)

## Example Use Cases

### Financial Services
```python
"Create a credit risk assessment model with regulatory compliance reporting"
```
- Synthetic financial data generation
- Model interpretability and fairness validation
- Regulatory compliance dashboards

### E-commerce
```python
"Build a customer lifetime value prediction system with A/B testing"
```
- Customer segmentation analysis
- Recommendation engine development
- Real-time prediction API

### Healthcare
```python
"Develop a patient readmission prediction model with privacy protection"
```
- Synthetic patient data generation
- Privacy-preserving ML techniques
- Clinical decision support interface

![Use Cases](./temp/use-cases-grid.png)

## Getting Started

### Option 1: Complete Pipeline
Use the **Master Project Manager** for end-to-end automation:

```python
"Build a complete fraud detection system for credit card transactions"
```

This will:
1. Generate synthetic transaction data
2. Perform exploratory data analysis
3. Train and optimize ML models
4. Validate model performance and fairness
5. Create deployment pipeline
6. Build interactive dashboard

### Option 2: Individual Agents
Work with specific agents for targeted tasks:

```python
# Data acquisition
"Generate synthetic customer data for churn prediction"

# Model development
"Train an XGBoost model for this dataset with hyperparameter tuning"

# Dashboard creation
"Create a Streamlit app for model predictions with real-time updates"
```

![Getting Started Flow](./temp/getting-started-flow.png)

## MLflow Integration

All agents automatically integrate with MLflow for comprehensive experiment tracking:

- **Parent-child run hierarchy** for complex workflows
- **Automatic metric logging** (accuracy, precision, recall, etc.)
- **Model registry** with signatures and input examples
- **Artifact tracking** (models, plots, reports, data)

![MLflow Dashboard](./temp/mlflow-dashboard-example.png)

## Configuration

### Project Settings
Configure your project by updating the settings in any agent interaction:

```python
settings = {
    "project": "customer_analytics",
    "target_metric": "f1_score",
    "deployment_strategy": "canary",
    "ui_complexity": "medium",
    "compliance_requirements": ["GDPR", "SOX"]
}
```

### Agent Customization
Each agent can be customized for specific requirements:
- **Data Wrangler**: Custom data sources, quality rules
- **Model Developer**: Specific algorithms, evaluation metrics
- **Frontend Developer**: UI framework preferences, styling
- **MLOps Engineer**: Deployment platforms, monitoring tools

## Documentation

- **[Agent Guide](./docs/agent-guide.md)**: Detailed agent capabilities and usage
- **[Workflow Examples](./docs/workflow-examples.md)**: Step-by-step tutorials
- **[API Reference](./docs/api-reference.md)**: Technical specifications
- **[Best Practices](./docs/best-practices.md)**: Tips for optimal results
- **[Troubleshooting](./docs/troubleshooting.md)**: Common issues and solutions

## Demo Gallery

### Customer Churn Prediction
![Churn Demo](./temp/churn-prediction-demo.png)
- Real-time prediction API
- Interactive feature importance
- Business impact calculator

### Fraud Detection System
![Fraud Demo](./temp/fraud-detection-demo.png)
- Anomaly detection algorithms
- Real-time alerting system
- Investigation dashboard

### Recommendation Engine
![Recommendation Demo](./temp/recommendation-engine-demo.png)
- Collaborative filtering
- Content-based recommendations
- A/B testing framework

## Advanced Features

### Custom Data Sources
```python
# Connect to your data
"Use the sales data from our Snowflake warehouse for demand forecasting"
```

### Multi-Model Ensembles
```python
# Advanced modeling
"Create an ensemble of XGBoost, LightGBM, and neural networks"
```

### Automated Monitoring
```python
# Production monitoring
"Set up data drift detection and model performance monitoring"
```

![Advanced Features](./temp/advanced-features-overview.png)

## Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for:
- Bug reports and feature requests
- Agent improvements and new capabilities
- Documentation updates
- Example workflows and use cases

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Support

- **Documentation**: [docs.domino.ai](https://docs.domino.ai)
- **Community**: [Domino Community Forum](https://community.domino.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/domino-ml-agents/issues)
- **Contact**: support@domino.ai

---

<div align="center">
  <img src="./temp/domino-logo.png" alt="Domino Data Lab" width="200">
  <br>
  <em>Built with love for the Domino Data Lab community</em>
</div>