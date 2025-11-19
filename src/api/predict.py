"""
Diesel Fuel Quality Prediction API

FastAPI endpoint for real-time diesel fuel quality predictions.
Designed for deployment as a Domino Model API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diesel Fuel Quality Prediction API",
    description="Real-time prediction of diesel fuel quality parameters from refinery process inputs",
    version="1.0.0"
)

# Global variables to store loaded models
models = {}
scalers = {}
feature_columns = []
paths = None

class ProcessInput(BaseModel):
    """Input schema for process parameters."""
    crude_api_gravity: float = Field(..., ge=25, le=45, description="Crude oil API gravity (°API)")
    crude_sulfur_ppm: float = Field(..., ge=100, le=8000, description="Crude oil sulfur content (ppm)")
    crude_nitrogen_ppm: float = Field(..., ge=50, le=2000, description="Crude oil nitrogen content (ppm)")
    crude_tan: float = Field(..., ge=0.1, le=3.0, description="Total Acid Number (mg KOH/g)")
    
    distillation_temp_top: float = Field(..., ge=80, le=120, description="Distillation top temperature (°C)")
    distillation_temp_middle: float = Field(..., ge=220, le=280, description="Distillation middle temperature (°C)")
    distillation_temp_bottom: float = Field(..., ge=350, le=420, description="Distillation bottom temperature (°C)")
    distillation_pressure: float = Field(..., ge=0.9, le=1.1, description="Distillation pressure (bar)")
    feed_rate_bpd: float = Field(..., ge=10000, le=100000, description="Feed rate (barrels/day)")
    reflux_ratio: float = Field(..., ge=1.5, le=4.0, description="Reflux ratio")
    
    hydrotreat_temp: float = Field(..., ge=300, le=380, description="Hydrotreating temperature (°C)")
    hydrotreat_pressure: float = Field(..., ge=40, le=140, description="Hydrotreating pressure (bar)")
    hydrogen_ratio: float = Field(..., ge=200, le=500, description="Hydrogen to oil ratio (SCF/BBL)")
    lhsv: float = Field(..., ge=0.8, le=3.0, description="Liquid Hourly Space Velocity (hr⁻¹)")
    catalyst_age_days: float = Field(..., ge=0, le=1095, description="Catalyst age (days)")

class QualityPrediction(BaseModel):
    """Output schema for quality predictions."""
    cetane_number: Optional[float] = Field(None, description="Cetane number (dimensionless)")
    sulfur_ppm: Optional[float] = Field(None, description="Sulfur content (ppm)")
    viscosity_cst: Optional[float] = Field(None, description="Kinematic viscosity (mm²/s)")
    flash_point_c: Optional[float] = Field(None, description="Flash point (°C)")
    density_gcc: Optional[float] = Field(None, description="Density (g/cm³)")
    cloud_point_c: Optional[float] = Field(None, description="Cloud point (°C)")
    aromatic_content: Optional[float] = Field(None, description="Aromatic content (% volume)")
    quality_score: Optional[float] = Field(None, description="Overall quality score (0-100)")

class PredictionResponse(BaseModel):
    """Response schema with metadata."""
    predictions: QualityPrediction
    confidence: Dict[str, float] = Field(default_factory=dict, description="Prediction confidence scores")
    compliance_status: Dict[str, bool] = Field(default_factory=dict, description="ASTM compliance flags")
    metadata: Dict = Field(default_factory=dict, description="Prediction metadata")

@app.on_event("startup")
async def load_models():
    """Load trained models and scalers on startup."""
    global models, scalers, feature_columns, paths
    
    try:
        logger.info("Loading models and scalers...")
        paths = get_data_paths()
        
        # Load feature columns
        feature_path = paths['models_path'] / 'feature_columns.json'
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                feature_columns = json.load(f)
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        else:
            raise FileNotFoundError("Feature columns file not found")
        
        # Load models and scalers
        target_columns = [
            'cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c',
            'density_gcc', 'cloud_point_c', 'aromatic_content', 'quality_score'
        ]
        
        models_dir = paths['models_path']
        models_loaded = 0
        
        for target in target_columns:
            model_path = models_dir / f"best_{target}_model.pkl"
            scaler_path = models_dir / f"scaler_{target}.pkl"
            
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    models[target] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scalers[target] = pickle.load(f)
                models_loaded += 1
                logger.info(f"Loaded model and scaler for {target}")
            else:
                logger.warning(f"Model or scaler not found for {target}")
        
        logger.info(f"Successfully loaded {models_loaded} models")
        
        if models_loaded == 0:
            raise RuntimeError("No models could be loaded")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")

def prepare_input_data(input_data: ProcessInput) -> List[float]:
    """Convert input data to feature vector."""
    try:
        # Create input vector matching feature_columns order
        input_dict = input_data.dict()
        input_vector = []
        
        for feature in feature_columns:
            if feature in input_dict:
                input_vector.append(input_dict[feature])
            else:
                raise ValueError(f"Feature {feature} not found in input data")
        
        return input_vector
    
    except Exception as e:
        logger.error(f"Error preparing input data: {e}")
        raise ValueError(f"Input data preparation failed: {e}")

def check_compliance(predictions: Dict[str, float]) -> Dict[str, bool]:
    """Check ASTM D975 compliance for predicted values."""
    compliance = {}
    
    # ASTM D975 specifications
    specs = {
        'cetane_number': {'min': 40, 'max': None},
        'sulfur_ppm': {'min': None, 'max': 15},  # ULSD
        'viscosity_cst': {'min': 1.9, 'max': 4.1},
        'flash_point_c': {'min': 52, 'max': None},
        'aromatic_content': {'min': None, 'max': 35}
    }
    
    for param, spec in specs.items():
        if param in predictions and predictions[param] is not None:
            value = predictions[param]
            is_compliant = True
            
            if spec['min'] is not None and value < spec['min']:
                is_compliant = False
            if spec['max'] is not None and value > spec['max']:
                is_compliant = False
            
            compliance[param] = is_compliant
    
    return compliance

def calculate_confidence(predictions: Dict[str, float]) -> Dict[str, float]:
    """Calculate prediction confidence scores (simplified)."""
    # This is a simplified confidence calculation
    # In production, you'd use proper uncertainty quantification
    confidence = {}
    
    for param, value in predictions.items():
        if value is not None:
            # Simple heuristic: confidence based on whether value is in normal range
            if param == 'cetane_number':
                confidence[param] = min(1.0, max(0.5, 1.0 - abs(value - 45) / 15))
            elif param == 'sulfur_ppm':
                confidence[param] = min(1.0, max(0.5, 1.0 - abs(value - 15) / 50))
            elif param == 'viscosity_cst':
                confidence[param] = min(1.0, max(0.5, 1.0 - abs(value - 3.0) / 1.5))
            else:
                confidence[param] = 0.8  # Default confidence
    
    return confidence

@app.post("/predict", response_model=PredictionResponse)
async def predict_quality(input_data: ProcessInput) -> PredictionResponse:
    """
    Predict diesel fuel quality parameters from process inputs.
    
    Returns predictions for all quality parameters along with compliance status.
    """
    try:
        logger.info("Received prediction request")
        
        # Prepare input data
        input_vector = prepare_input_data(input_data)
        logger.info(f"Input vector prepared: {len(input_vector)} features")
        
        # Make predictions
        predictions = {}
        
        for target in models.keys():
            try:
                # Scale input
                input_scaled = scalers[target].transform([input_vector])
                
                # Predict
                pred_value = models[target].predict(input_scaled)[0]
                predictions[target] = float(pred_value)
                
            except Exception as e:
                logger.warning(f"Failed to predict {target}: {e}")
                predictions[target] = None
        
        # Check compliance
        compliance_status = check_compliance(predictions)
        
        # Calculate confidence
        confidence = calculate_confidence(predictions)
        
        # Create response
        quality_pred = QualityPrediction(**predictions)
        
        response = PredictionResponse(
            predictions=quality_pred,
            confidence=confidence,
            compliance_status=compliance_status,
            metadata={
                "prediction_time": datetime.now().isoformat(),
                "model_version": "1.0.0",
                "features_used": len(input_vector)
            }
        )
        
        logger.info("Prediction completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_quality_batch(input_data: List[ProcessInput]) -> List[PredictionResponse]:
    """
    Predict diesel fuel quality for multiple sets of process inputs.
    """
    try:
        logger.info(f"Received batch prediction request for {len(input_data)} samples")
        
        results = []
        for i, single_input in enumerate(input_data):
            try:
                result = await predict_quality(single_input)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict sample {i}: {e}")
                # Return empty prediction for failed samples
                empty_pred = QualityPrediction()
                error_response = PredictionResponse(
                    predictions=empty_pred,
                    metadata={"error": str(e), "sample_index": i}
                )
                results.append(error_response)
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check if models are loaded
        model_count = len(models)
        scaler_count = len(scalers)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": model_count,
            "scalers_loaded": scaler_count,
            "features_configured": len(feature_columns)
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    try:
        # Load performance data if available
        perf_path = paths['models_path'] / 'model_performance.json'
        performance_data = {}
        
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                performance_data = json.load(f)
        
        return {
            "loaded_models": list(models.keys()),
            "feature_count": len(feature_columns),
            "performance_data": performance_data.get('best_models', {}),
            "training_date": performance_data.get('training_date', 'Unknown')
        }
    
    except Exception as e:
        logger.error(f"Model info request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Diesel Fuel Quality Prediction API",
        "version": "1.0.0",
        "description": "Real-time prediction of diesel fuel quality parameters",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)",
            "health": "/health (GET)",
            "model_info": "/models/info (GET)",
            "docs": "/docs (GET)"
        },
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)