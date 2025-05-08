import os
import time
import uuid
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn
import shutil
from datetime import datetime

# --- Configuration ---
MODELS_DIR = "models"
UPLOADS_DIR = "uploads"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Constants ---
class ModelType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"

VALID_EXTENSIONS = {
    ModelType.IMAGE: [".h5", ".hdf5", ".onnx", ".pkl"],
    ModelType.TEXT: [".pkl", ".joblib", ".onnx"],
    ModelType.TABULAR: [".pkl", ".joblib", ".h5"]
}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    type: ModelType
    formats: List[str] = []
    features: Optional[List[str]] = None
    is_active: bool
    created_at: str

class ModelUploadRequest(BaseModel):
    name: str
    description: str
    type: ModelType
    expected_features: Optional[str] = None
    supported_formats: Optional[str] = None

    @validator('expected_features')
    def validate_features(cls, v, values):
        if values.get('type') == ModelType.TABULAR and not v:
            raise ValueError("Tabular models require expected_features")
        return v

class PredictionRequest(BaseModel):
    data: Union[Dict[str, Any], str]  # For tabular/text inputs

class PredictionResponse(BaseModel):
    prediction: Union[str, float, int]
    confidence: Optional[float]
    latency_ms: float

# --- Model Loaders ---
class ModelLoader:
    @staticmethod
    def load_tensorflow(path: str):
        import tensorflow as tf
        return tf.keras.models.load_model(path)

    @staticmethod
    def load_sklearn(path: str):
        return joblib.load(path)

    @staticmethod
    def load_onnx(path: str):
        import onnxruntime as ort
        return ort.InferenceSession(path)

# --- Core Model Class ---
class AIModel:
    def __init__(self, model_id: str, config: dict):
        self.id = model_id
        self.name = config.get('name', 'Unnamed Model')
        self.description = config.get('description', '')
        self.type = ModelType(config.get('type'))
        self.model = None
        self.is_active = False
        self.metadata = {
            'features': config.get('features', []),
            'formats': config.get('formats', []),
            'created_at': datetime.now().isoformat()
        }

    def load(self, model_path: str) -> bool:
        """Load model from disk with format detection"""
        try:
            ext = os.path.splitext(model_path)[1].lower()
            
            if ext in ['.h5', '.hdf5']:
                self.model = ModelLoader.load_tensorflow(model_path)
            elif ext in ['.pkl', '.joblib']:
                self.model = ModelLoader.load_sklearn(model_path)
            elif ext == '.onnx':
                self.model = ModelLoader.load_onnx(model_path)
            else:
                raise ValueError(f"Unsupported format: {ext}")

            self.is_active = True
            return True

        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.is_active = False
            return False

    def predict(self, input_data: Any) -> dict:
        """Unified prediction interface"""
        if not self.is_active:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        
        try:
            # TensorFlow/Keras models
            if hasattr(self.model, 'predict'):
                result = self.model.predict(input_data)
                return {
                    'prediction': float(result[0][0]),
                    'confidence': None,
                    'latency_ms': (time.time() - start_time) * 1000
                }
            
            # scikit-learn models
            elif hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(input_data)
                return {
                    'prediction': int(self.model.predict(input_data)[0]),
                    'confidence': float(proba[0][1]),
                    'latency_ms': (time.time() - start_time) * 1000
                }
            
            # ONNX models
            elif hasattr(self.model, 'run'):
                # Implement ONNX-specific prediction logic
                pass

            raise RuntimeError("Unsupported model type")

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# --- Model Registry ---
class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        self.load_core_models()

    def load_core_models(self):
        """Preload essential models"""
        core_models = [
            {
                'id': 'asthma-predictor',
                'name': 'Asthma Predictor',
                'type': ModelType.TABULAR,
                'features': ['age', 'gender', 'height', ...],
                'path': os.path.join(MODELS_DIR, 'asthma.h5')
            },
            # Add other core models
        ]

        for config in core_models:
            model = AIModel(config['id'], config)
            if model.load(config['path']):
                self.models[model.id] = model

    def register_model(self, config: dict, model_path: str) -> bool:
        """Register new uploaded model"""
        model_id = str(uuid.uuid4())
        model = AIModel(model_id, config)
        
        if model.load(model_path):
            self.models[model_id] = model
            return True
        return False

# --- FastAPI Setup ---
app = FastAPI(
    title="AI Model Hub",
    version="2.0",
    docs_url="/api/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()

# --- API Endpoints ---
@app.post("/api/models", response_model=ModelInfo)
async def upload_model(
    metadata: ModelUploadRequest = Depends(),
    model_file: UploadFile = File(...)
):
    """Upload a model with validation"""
    # Validate file extension
    file_ext = os.path.splitext(model_file.filename)[1].lower()
    if file_ext not in VALID_EXTENSIONS[metadata.type]:
        raise HTTPException(400, f"Invalid extension for {metadata.type} model")

    # Save file
    model_id = str(uuid.uuid4())
    model_path = os.path.join(MODELS_DIR, f"{model_id}{file_ext}")
    
    try:
        with open(model_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
    except Exception as e:
        raise HTTPException(500, f"File save failed: {str(e)}")

    # Register model
    config = {
        'name': metadata.name,
        'description': metadata.description,
        'type': metadata.type,
        'features': metadata.expected_features.split(",") if metadata.expected_features else [],
        'formats': metadata.supported_formats.split(",") if metadata.supported_formats else []
    }

    if not registry.register_model(config, model_path):
        os.remove(model_path)
        raise HTTPException(500, "Model registration failed")

    return registry.models[model_id].__dict__

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    return [model.__dict__ for model in registry.models.values()]

@app.post("/api/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, request: PredictionRequest):
    """Generic prediction endpoint"""
    if model_id not in registry.models:
        raise HTTPException(404, "Model not found")
    
    model = registry.models[model_id]
    
    try:
        result = model.predict(request.data)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)