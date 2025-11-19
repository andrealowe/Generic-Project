# Diesel Fuel Quality Prediction Demo - Implementation Specification

## Executive Summary
This document outlines the technical requirements for building a data science demo that predicts diesel fuel quality from refinery process inputs. The demo will showcase ML/MLOps capabilities by combining realistic refinery data with AutoML frameworks and an interactive dashboard for quality monitoring.

---

## 1. Data Requirements

### 1.1 Target Quality Parameters
Based on industry standards (ASTM D975), the demo should predict the following diesel fuel quality indicators:

**Primary Quality Metrics:**
- **Cetane Number** (40-60): Indicates ignition quality and combustion characteristics
- **Sulfur Content** (ppm): Critical for emissions compliance (Ultra-Low Sulfur Diesel: ≤15 ppm)
- **Viscosity** (cSt @ 40°C): Typically 1.9-4.1 mm²/s for No. 2 diesel
- **Flash Point** (°C): Minimum 52°C for safety compliance
- **Density** (g/cm³): Typically 0.82-0.86 for diesel fuel

**Secondary Quality Metrics:**
- Cloud Point (°C): Temperature at which wax crystals form
- Distillation Temperature (T90): Temperature at which 90% of fuel evaporates
- Pour Point (°C): Lowest temperature fuel remains fluid
- Aromatic Content (%): Affects emissions and combustion

### 1.2 Process Input Features
Model inputs should include refinery distillation and hydrotreating parameters:

**Atmospheric Distillation Inputs:**
- Crude oil feed temperature (350-400°C)
- Column pressure (slightly above atmospheric)
- Feed rate (barrels per day)
- Distillation cut points (temperature ranges)
- Top, middle, and bottom tray temperatures

**Hydrotreating Process Inputs:**
- Hydrogen pressure (high pressure, 40-140 bar)
- Reactor temperature (300-380°C)
- Liquid hourly space velocity (LHSV)
- Hydrogen to oil ratio
- Catalyst age/activity indicator

**Crude Oil Properties:**
- API gravity (density indicator)
- Initial sulfur content
- Total acid number (TAN)
- Crude oil source/type

### 1.3 Data Sources

#### Option A: Synthetic Data Generation (Recommended)
Since no comprehensive open-source diesel quality dataset was found, **generate realistic synthetic data** using:

**Generation Strategy:**
1. Create 5,000-10,000 samples representing various operating conditions
2. Use domain knowledge to establish realistic relationships:
   - Higher hydrotreating temperature → lower sulfur content
   - Higher cetane index correlates with paraffinic content
   - Distillation temperature ranges determine viscosity
3. Add realistic noise (±2-5% for most parameters)
4. Include some anomalous samples (5-10%) representing off-spec production

**Reference Ranges (from literature):**
- Use ASTM D975 specifications as targets
- Model correlations from published refinery studies
- Incorporate seasonal variations (winter/summer diesel specs)

#### Option B: Proxy Dataset
As an alternative, the **Southwest Research Institute (SWRI) diesel NIR spectroscopy dataset** (784 samples) could be adapted:
- Available at: http://www.eigenvector.com/data/swri/
- Contains Near-Infrared Spectroscopy (NIRS) data for diesel samples
- Can be reverse-engineered to include process conditions

**Dataset Structure:**
```
Columns:
- timestamp
- crude_api_gravity
- crude_sulfur_ppm
- distillation_temp_top
- distillation_temp_middle
- distillation_temp_bottom
- distillation_pressure
- feed_rate_bpd
- hydrotreat_temp
- hydrotreat_pressure
- hydrogen_ratio
- catalyst_age_days
- [Target] cetane_number
- [Target] sulfur_ppm
- [Target] viscosity_cst
- [Target] flash_point_c
- [Target] density_gcc
- [Target] cloud_point_c
- quality_flag (pass/fail)
```

---

## 2. Machine Learning Models

### 2.1 Recommended Model Types

Based on the literature review of fuel quality prediction:

**Primary Models:**
1. **Gradient Boosting Machines (GBM/XGBoost)**
   - Excellent for tabular data with complex non-linear relationships
   - Handles feature interactions well (e.g., temperature × pressure effects)
   - Cited in multiple fuel quality studies with R² > 0.95

2. **Random Forest Regression**
   - Robust to outliers and noise
   - Provides feature importance rankings
   - Good baseline performance

3. **Partial Least Squares Regression (PLS/PLS2)**
   - Standard method in petrochemical industry
   - Handles multicollinearity in process variables
   - Particularly effective for spectroscopy-based applications
   - PLS2 can predict multiple quality parameters simultaneously

4. **Support Vector Regression (SVR)**
   - Effective for high-dimensional process data
   - Good generalization with limited samples

5. **Neural Networks / Deep Learning**
   - Gaussian Process Regression (GPR) coupled with neural networks
   - Feedforward neural networks for process modeling
   - Consider for complex, non-linear relationships

### 2.2 Model Architecture Strategy

**Multi-Output Approach:**
- Train separate models for each quality parameter OR
- Use multi-output models (e.g., PLS2, multi-output Random Forest) to predict all parameters simultaneously

**Feature Engineering:**
- Interaction terms (temperature × pressure)
- Polynomial features for non-linear relationships
- Time-based features (catalyst degradation over time)
- Rolling averages for process stability indicators

**Model Validation:**
- Use time-series cross-validation (not random splits) to respect temporal dependencies
- 70/15/15 split for train/validation/test
- Track both interpolation and extrapolation performance

---

## 3. AutoML Frameworks

Based on comprehensive benchmarking studies, here are 4 recommended AutoML frameworks:

### 3.1 H2O AutoML
**Best for: Production-ready models with minimal configuration**

**Pros:**
- Enterprise-grade scalability (handles large datasets efficiently)
- Multi-threaded processing and distributed computing
- Built-in model explainability (SHAP, feature importance)
- Web UI for visualization and monitoring
- Strong ensemble methods (stacked models)
- R² > 0.95 achievable with default settings

**Cons:**
- Requires Java runtime (JDK 8-13)
- Less flexible for custom preprocessing
- Uses random search (simpler than Bayesian optimization)

**Use Case:**
Ideal for rapid prototyping and getting production-ready models quickly with minimal tuning.

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
aml = H2OAutoML(max_models=20, max_runtime_secs=3600, seed=1)
aml.train(x=features, y='cetane_number', training_frame=train)
```

### 3.2 TPOT (Tree-based Pipeline Optimization Tool)
**Best for: Research and discovering optimal pipelines**

**Pros:**
- Uses genetic programming to find optimal pipelines
- Automatically explores feature preprocessing + model combinations
- Built on scikit-learn (familiar ecosystem)
- Exports Python code for transparency
- Excellent for academic/research contexts

**Cons:**
- Can be slow (genetic algorithms take time)
- Limited scalability compared to H2O
- Requires more computational resources

**Use Case:**
Best when you want to discover the optimal preprocessing + model combination and understand WHY it works.

```python
from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=10, population_size=50, 
                     cv=5, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
tpot.export('best_pipeline.py')
```

### 3.3 Auto-sklearn
**Best for: Automated pipeline with meta-learning**

**Pros:**
- Bayesian optimization for hyperparameter tuning
- Meta-learning: learns from past experiments
- Automatic ensemble construction
- Strong academic foundation
- Excellent for medium-sized datasets

**Cons:**
- Linux only (not available for Windows)
- Can be memory-intensive
- Slower than H2O for large datasets

**Use Case:**
Ideal for Linux environments where you want intelligent hyperparameter optimization with ensemble models.

```python
import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=3600,
    per_run_time_limit=300
)
automl.fit(X_train, y_train)
```

### 3.4 PyCaret
**Best for: Rapid experimentation and comparison**

**Pros:**
- Extremely simple API (low-code approach)
- Great for quick model comparison
- Integrates with MLflow for experiment tracking
- Excellent visualization capabilities
- Fast to get started

**Cons:**
- Less customizable than other frameworks
- Better for prototyping than production
- Limited advanced features

**Use Case:**
Perfect for demos and rapid proof-of-concept development with built-in visualization.

```python
from pycaret.regression import *

exp = setup(data=df, target='cetane_number', session_id=123)
best_model = compare_models()
tuned_model = tune_model(best_model)
```

### 3.5 Framework Comparison Summary

| Framework | Speed | Scalability | Customization | Best Use Case | R² Typical |
|-----------|-------|-------------|---------------|---------------|------------|
| H2O | Fast | Excellent | Medium | Production | 0.95+ |
| TPOT | Slow | Limited | High | Research | 0.90-0.95 |
| Auto-sklearn | Medium | Good | High | Linux/Academic | 0.92-0.96 |
| PyCaret | Fast | Good | Low | Demo/PoC | 0.90-0.94 |

**Recommendation for Demo:**
- **Primary**: H2O AutoML (for speed and production readiness)
- **Secondary**: PyCaret (for rapid visualization and comparison)
- **Optional**: TPOT (to show pipeline optimization and export code)

---

## 4. Dashboard Design

### 4.1 Dashboard Layout & Structure

**Dashboard Type:** Operational/Analytical Hybrid Dashboard
- Real-time monitoring of fuel quality predictions
- Historical trend analysis
- Model performance tracking

### 4.2 Key Dashboard Sections

#### Section 1: Executive Summary (Top Panel)
**Purpose:** At-a-glance status of fuel production quality

**Components:**
- **Overall Quality Score** (0-100 scale)
  - Composite score based on all quality parameters
  - Color-coded: Green (90-100), Yellow (75-89), Red (<75)
  
- **Batch Success Rate** (%)
  - Percentage of batches meeting all specifications
  - Last 24 hours, 7 days, 30 days views

- **Current Production Status**
  - Real-time indicator: "On Spec" / "Off Spec" / "Warning"
  - Time since last quality issue

- **Daily Production Volume**
  - Barrels produced today
  - Comparison to target

#### Section 2: Quality Parameter Monitoring (Main Panel - Left Side)
**Purpose:** Track individual quality metrics in real-time

**Key Visualizations:**

1. **Cetane Number Gauge**
   - Dial gauge showing current value
   - Colored zones: Red (<40), Yellow (40-48), Green (48-60), Red (>60)
   - Trend arrow (↑↓→) indicating direction

2. **Sulfur Content Chart**
   - Line chart showing sulfur PPM over time (last 8 hours)
   - Horizontal line at 15 PPM threshold (ULSD compliance)
   - Color gradient: Green below threshold, red above

3. **Viscosity Monitor**
   - Bar chart showing current viscosity
   - Acceptable range band (1.9-4.1 cSt)
   - Historical comparison (today vs. 7-day average)

4. **Multi-Parameter Heatmap**
   - Rows: Different quality parameters
   - Columns: Time intervals (hourly for last 24h)
   - Colors: Green (in spec), Yellow (marginal), Red (out of spec)

#### Section 3: Process Input Dashboard (Main Panel - Right Side)
**Purpose:** Monitor refinery operating conditions

**Key Visualizations:**

1. **Distillation Column Status**
   - Schematic diagram of distillation column
   - Real-time temperature readings at key trays
   - Color-coded by operating zone

2. **Hydrotreating Process Metrics**
   - Temperature gauge (300-380°C range)
   - Pressure gauge (40-140 bar range)
   - Hydrogen ratio indicator
   - Catalyst age alert (visual indicator when replacement due)

3. **Process Stability Indicator**
   - Control chart showing process variance over time
   - Upper/Lower control limits
   - Alert when approaching control limits

#### Section 4: Predictive Analytics (Bottom Panel)
**Purpose:** Show ML model predictions and insights

**Components:**

1. **Predicted vs. Actual Quality**
   - Scatter plot comparing model predictions to lab measurements
   - R² score displayed
   - Regression line

2. **Feature Importance Chart**
   - Horizontal bar chart showing top 10 influential variables
   - Updated based on recent predictions
   - Color-coded by variable type (crude properties, distillation, hydrotreating)

3. **Quality Forecast**
   - Time series line chart showing:
     - Historical quality (solid line)
     - Predicted quality for next 4 hours (dashed line)
     - Confidence interval (shaded region)

4. **Anomaly Detection Alert**
   - List of recent anomalies or out-of-spec predictions
   - Timestamp, parameter, deviation from spec
   - Suggested corrective actions

#### Section 5: Model Performance Metrics (Collapsible Panel)
**Purpose:** Track ML model health and performance

**Metrics:**
- **Model Accuracy Metrics**
  - R² score for each quality parameter
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Last updated timestamp

- **Prediction Confidence**
  - Distribution of prediction confidence scores
  - Percentage of high-confidence predictions (>90%)

- **Model Drift Detection**
  - Chart showing model performance over time
  - Alert if significant drift detected (>5% accuracy drop)

- **Training Data Coverage**
  - Heatmap showing which operating regions have sufficient training data
  - Alert for predictions in low-coverage regions

### 4.3 Dashboard Design Principles

**Visual Hierarchy:**
- Most critical info (overall status) at top
- Detailed metrics in middle tier
- Advanced analytics at bottom
- Limit to 8-10 key visualizations on main screen

**Color Scheme:**
- **Green**: In specification, normal operation
- **Yellow**: Marginal/warning, approaching limits
- **Red**: Out of specification, requires attention
- **Blue**: Informational, neutral status
- **Gray**: Inactive or unavailable data

**Interactivity:**
- Click on any chart to drill down into details
- Hover tooltips with precise values
- Date/time range selector (last hour, 8 hours, 24 hours, 7 days)
- Filter by operating unit or fuel grade
- Export data to CSV functionality

**Refresh Rate:**
- Real-time data: Update every 1-2 minutes
- Historical trends: Update every 15 minutes
- Model performance: Update daily

### 4.4 Technology Stack Recommendations

**Dashboard Framework Options:**

1. **Plotly Dash** (Python)
   - Best for Python ML workflows
   - Interactive visualizations with Plotly
   - Easy deployment
   ```python
   import dash
   from dash import dcc, html
   import plotly.graph_objs as go
   ```

2. **Streamlit** (Python)
   - Fastest development time
   - Great for prototypes and demos
   - Limited layout flexibility
   ```python
   import streamlit as st
   st.title("Diesel Fuel Quality Dashboard")
   ```

3. **Tableau / Power BI**
   - Professional-grade dashboards
   - Better for business users
   - Less integration with ML pipelines

4. **Grafana**
   - Excellent for time-series data
   - Real-time monitoring focus
   - Requires time-series database (InfluxDB, Prometheus)

**Recommended for Demo:** Plotly Dash or Streamlit
- Native Python integration
- Easy to connect with ML models
- Professional appearance
- Can be deployed to cloud platforms (Heroku, AWS, Azure)

### 4.5 Dashboard Example Wireframe

```
┌─────────────────────────────────────────────────────────────┐
│  DIESEL FUEL QUALITY MONITORING DASHBOARD                   │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│ │Quality  │ │Batch    │ │Status   │ │Daily    │          │
│ │Score    │ │Success  │ │ON SPEC  │ │Volume   │          │
│ │  94     │ │  96%    │ │         │ │ 45,230  │          │
│ │  ●●●●●  │ │ ●●●●○   │ │    ✓    │ │  BBL    │          │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
├───────────────────────────┬─────────────────────────────────┤
│ QUALITY PARAMETERS        │ PROCESS INPUTS                  │
│                           │                                 │
│ [Cetane Gauge]   50.2    │ [Distillation Column Diagram]  │
│                           │  Top: 85°C                     │
│ [Sulfur PPM Chart]        │  Mid: 245°C                    │
│  ████████░░░░ 12 ppm     │  Bot: 380°C                    │
│                           │                                 │
│ [Viscosity Bar] 2.8 cSt  │ [Hydrotreat Conditions]        │
│                           │  Temp: 355°C                   │
│ [Parameter Heatmap]       │  Press: 95 bar                 │
│  ■■■■■■■■■■■■           │  H2 Ratio: 300 SCF/BBL         │
│  ■■■■□■■■■■■■           │                                 │
├───────────────────────────┴─────────────────────────────────┤
│ PREDICTIVE ANALYTICS                                         │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐│
│ │Pred vs Act │ │Feature Imp │ │Forecast    │ │Anomalies ││
│ │  R²=0.96   │ │[Bar Chart] │ │[Line+Conf] │ │  None    ││
│ │[Scatter]   │ │            │ │            │ │ Detected ││
│ └────────────┘ └────────────┘ └────────────┘ └──────────┘│
├─────────────────────────────────────────────────────────────┤
│ Model Performance: R²=0.96 | RMSE=1.2 | Last Updated: 2min │
└─────────────────────────────────────────────────────────────┘
```

### 4.6 Additional Dashboard Features

**Alert System:**
- Configurable thresholds for each quality parameter
- Email/SMS notifications for critical deviations
- Alert history log

**Reporting:**
- Automated daily/weekly quality reports
- Compliance documentation (EPA, ASTM standards)
- Batch certification reports

**User Roles:**
- Operator view: Simplified, real-time only
- Engineer view: Full access with historical analysis
- Executive view: KPIs and trends only

---

## 5. Implementation Roadmap

### Phase 1: Data Preparation (Week 1)
- Generate or acquire synthetic diesel quality dataset
- Validate data ranges against industry standards
- Create train/validation/test splits
- Document data dictionary

### Phase 2: Model Development (Week 1-2)
- Implement baseline models (Linear Regression, Random Forest)
- Run AutoML experiments (H2O, PyCaret)
- Compare model performance
- Select best models for each quality parameter
- Save trained models and metadata

### Phase 3: Dashboard Development (Week 2-3)
- Design dashboard layout and wireframes
- Implement dashboard using Plotly Dash or Streamlit
- Connect models to dashboard backend
- Add interactivity and real-time updates
- Create visualization components

### Phase 4: Integration & Testing (Week 3)
- Integrate ML models with dashboard
- Test end-to-end workflow
- Validate predictions against test data
- Add error handling and edge cases
- Performance optimization

### Phase 5: Documentation & Deployment (Week 4)
- Create user documentation
- Deploy dashboard to cloud platform
- Set up monitoring and logging
- Prepare demo script and talking points

---

## 6. Key Demo Talking Points

**For Technical Audiences:**
- "We're using H2O AutoML to automatically train and tune multiple model types, achieving R² > 0.95"
- "The multi-output PLS2 model predicts all quality parameters simultaneously, capturing correlations"
- "Feature engineering includes interaction terms between temperature and pressure"
- "Time-series cross-validation ensures models generalize to future production runs"

**For Business Audiences:**
- "This system can predict fuel quality 1-4 hours in advance, allowing proactive adjustments"
- "Early detection of off-spec production can save $50K-$200K per incident"
- "The dashboard provides real-time visibility into quality metrics for all stakeholders"
- "Automated quality monitoring reduces lab testing costs by 30-40%"

**For Government/Compliance Audiences:**
- "All predictions reference ASTM D975 and EPA fuel standards"
- "The system maintains complete audit trails of quality predictions"
- "Automated alerts ensure immediate notification of compliance violations"
- "Historical data supports regulatory reporting and compliance documentation"

---

## 7. Success Metrics

**Model Performance:**
- R² > 0.93 for all quality parameters
- RMSE within ±2% of typical operating ranges
- Prediction confidence intervals < ±3%

**Dashboard Usability:**
- 5-second rule: Key insights visible within 5 seconds
- <1 minute to identify root cause of quality issue
- 95%+ user satisfaction in testing

**Business Impact:**
- Demonstrate 1-4 hour advance warning capability
- Show correlation between process changes and quality outcomes
- Identify optimization opportunities (cost savings scenarios)

---

## 8. References & Data Sources

**Industry Standards:**
- ASTM D975-24: Standard Specification for Diesel Fuel
- ISO 8217:2017: Petroleum products specifications
- EPA Ultra-Low Sulfur Diesel regulations

**Academic Research:**
- Fuel quality prediction using FTIR and chemometrics (PLS models, R² 0.90-0.95)
- Marine diesel engine ML modeling (R² > 0.95)
- XGBoost for diesel brand classification

**AutoML Benchmarking:**
- Georgian Impact Blog: AutoML framework comparison (87 datasets)
- OpenML AutoML Benchmark
- Comparative analysis: PyCaret, H2O, TPOT, Auto-sklearn, FLAML

**Process Data:**
- US EIA: Refining crude oil inputs and outputs
- API: Refinery process documentation
- Southwest Research Institute: NIRS diesel dataset (784 samples)

---

## Next Steps for Implementation

1. **Review this specification** with stakeholders
2. **Select AutoML framework(s)** based on environment (H2O recommended)
3. **Generate synthetic dataset** following the schema provided
4. **Begin with simple baseline models** before AutoML
5. **Iterate on dashboard design** based on user feedback
6. **Prepare demo narrative** tailored to audience

---

## Appendix: Code Snippets

### A1: Synthetic Data Generation Template
```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

# Generate process inputs
data = {
    'crude_api_gravity': np.random.uniform(25, 45, n_samples),
    'crude_sulfur_ppm': np.random.uniform(100, 5000, n_samples),
    'distillation_temp': np.random.uniform(350, 400, n_samples),
    'hydrotreat_temp': np.random.uniform(320, 370, n_samples),
    'hydrotreat_pressure': np.random.uniform(50, 120, n_samples),
    'hydrogen_ratio': np.random.uniform(200, 400, n_samples),
}

# Generate correlated targets with realistic relationships
df = pd.DataFrame(data)

# Cetane number: increases with paraffinic content (higher API gravity)
df['cetane_number'] = 35 + 0.3 * df['crude_api_gravity'] + \
                      np.random.normal(0, 2, n_samples)

# Sulfur: decreases with hydrotreating severity
df['sulfur_ppm'] = df['crude_sulfur_ppm'] * \
                   np.exp(-0.01 * df['hydrotreat_temp']) + \
                   np.random.normal(0, 2, n_samples)

# Add more target calculations...
```

### A2: H2O AutoML Quick Start
```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Load data
df = h2o.import_file("diesel_quality_data.csv")
train, valid, test = df.split_frame([0.7, 0.15])

# Define features and target
x = [col for col in df.columns if col not in ['cetane_number']]
y = 'cetane_number'

# Run AutoML
aml = H2OAutoML(max_models=20, max_runtime_secs=3600, seed=1)
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

# View leaderboard
lb = aml.leaderboard
print(lb)

# Get best model
best_model = aml.leader
```

### A3: Dashboard Skeleton (Plotly Dash)
```python
import dash
from dash import dcc, html
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diesel Fuel Quality Dashboard"),
    
    # Executive Summary
    html.Div([
        html.Div([
            html.H3("Quality Score"),
            html.H2("94", style={'color': 'green'})
        ], className="metric-box"),
    ], className="executive-row"),
    
    # Quality Parameters
    dcc.Graph(id='cetane-gauge'),
    dcc.Graph(id='sulfur-timeseries'),
    
    # Refresh interval
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

**Document Version:** 1.0  
**Last Updated:** November 18, 2025  
**Prepared for:** Domino Data Lab Product Marketing Demo
