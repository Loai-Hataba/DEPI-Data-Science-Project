import os
import time
import uuid
import logging
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from keras.models import load_model as load_keras_model
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء المجلدات اللازمة
os.makedirs("saved_models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# نماذج Pydantic للطلب/الاستجابة
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float = 0.0

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    input_type: str  # مثل: "image", "text", "tabular"
    supported_formats: List[str] = []

class TextPredictionRequest(BaseModel):
    text: str

class TabularPredictionRequest(BaseModel):
    features: Dict[str, Any]

class ModelUploadResponse(BaseModel):
    model_id: str
    name: str
    status: str

# استيراد تعريفات الموديلات
from models.base_model import MLModel
from models.image_model import ImageClassificationModel
from models.text_model import TextAnalysisModel
from models.tabular_models import CardioModel, DiabetesModel, AsthmaModel,SchizophreniaModel

# مدير الموديلات
class ModelManager:
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """تحميل جميع النماذج عند البدء"""
        models_to_load = [
            ImageClassificationModel(
                model_id="image-classifier",
                name="Image Classifier",
                description="Classifies images"
            ),
            TextAnalysisModel(
                model_id="text-sentiment",
                name="Text Sentiment",
                description="Analyzes text sentiment"
            ),
            CardioModel(),
            DiabetesModel(),
            AsthmaModel(),
            SchizophreniaModel()
        ]
        
        for model in models_to_load:
            try:
                if model.load():
                    self.models[model.model_id] = model
                    logger.info(f"Successfully loaded: {model.name}")
                else:
                    logger.error(f"Failed to load: {model.name}")
            except Exception as e:
                logger.error(f"Error loading {model.name}: {str(e)}")
                raise
    
    def get_models(self) -> List[ModelInfo]:
        """الحصول على معلومات عن جميع الموديلات المتاحة"""
        return [model.get_info() for model in self.models.values()]
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """الحصول على موديل معين بالـ ID"""
        return self.models.get(model_id)
    
    def add_model(self, model: MLModel) -> bool:
        """إضافة موديل جديد للسجل"""
        if model.model_id in self.models:
            return False
        
        if model.load():
            self.models[model.model_id] = model
            return True
        return False
    
    def remove_model(self, model_id: str) -> bool:
        """إزالة موديل من السجل"""
        if model_id not in self.models:
            return False
        
        del self.models[model_id]
        return True
    
    def predict_image(self, model_id: str, image_path: str) -> Dict[str, Any]:
        """التنبؤ باستخدام موديل الصور"""
        model = self.get_model(model_id)
        if not model or model.input_type != "image":
            raise ValueError(f"Invalid model ID or model type: {model_id}")
        
        start_time = time.time()
        result = model.predict(image_path)
        processing_time = time.time() - start_time
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": processing_time
        }
    
    def predict_text(self, model_id: str, text: str) -> Dict[str, Any]:
        """التنبؤ باستخدام موديل النصوص"""
        model = self.get_model(model_id)
        if not model or model.input_type != "text":
            raise ValueError(f"Invalid model ID or model type: {model_id}")
        
        start_time = time.time()
        result = model.predict(text)
        processing_time = time.time() - start_time
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": processing_time
        }
    
    def predict_tabular(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ باستخدام موديل الجداول"""
        model = self.get_model(model_id)
        if not model or model.input_type != "tabular":
            raise ValueError(f"Invalid model ID or model type: {model_id}")
        
        start_time = time.time()
        result = model.predict(features)
        processing_time = time.time() - start_time
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": processing_time
        }

# إنشاء تطبيق FastAPI
app = FastAPI(
    title="ML Models API",
    description="API for machine learning models integration",
    version="0.1.0"
)

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# تهيئة مدير الموديلات
model_manager = ModelManager()

# نقاط النهاية (Endpoints)
@app.get("/")
async def root():
    return {"message": "Welcome to ML Model API", "version": "0.1.0"}

@app.get("/api/health")
async def health_check():
    """فحص صحة الخدمة"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """الحصول على قائمة الموديلات المتاحة"""
    return model_manager.get_models()

@app.get("/api/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """الحصول على معلومات عن موديل معين"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.get_info()

@app.post("/api/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, file: UploadFile = File(...)):
    """التنبؤ باستخدام موديل معين مع رفع ملف"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.input_type != "image":
        raise HTTPException(status_code=400, detail=f"This endpoint is for image models only. Model {model_id} expects {model.input_type} input.")
    
    file_location = f"uploads/{file.filename}"
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    
    try:
        result = model_manager.predict_image(model_id, file_location)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/models/{model_id}/predict/text", response_model=PredictionResponse)
async def predict_text(model_id: str, request: TextPredictionRequest):
    """التنبؤ باستخدام موديل النصوص"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.input_type != "text":
        raise HTTPException(status_code=400, detail=f"This endpoint is for text models only. Model {model_id} expects {model.input_type} input.")
    
    try:
        result = model_manager.predict_text(model_id, request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/models/{model_id}/predict/tabular", response_model=None)
async def predict_tabular(model_id: str, request: TabularPredictionRequest):
    """Make prediction using tabular models"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.input_type != "tabular":
        raise HTTPException(status_code=400, detail=f"This endpoint is for tabular models only. Model {model_id} expects {model.input_type} input.")
    
    try:
        result = model_manager.predict_tabular(model_id, request.features)
        
        # Check if result contains an error flag
        if result.get("error", False):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Missing required features: {result.get('missing_features', [])}",
                    "required_features": result.get("required_features", [])
                }
            )
        
        return PredictionResponse(**result)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/cardio/predict", response_model=PredictionResponse)
async def predict_cardio(features: Dict[str, Any]):
    """التنبؤ باستخدام موديل القلب"""
    try:
        model = model_manager.get_model("cardio-predictor")
        if not model:
            raise HTTPException(status_code=404, detail="Cardio model not found")
        
        start_time = time.time()
        result = model.predict(features)
        processing_time = time.time() - start_time
        
        result["processing_time"] = processing_time
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/diabetes/predict", response_model=PredictionResponse)
async def predict_diabetes(features: Dict[str, Any]):
    """التنبؤ باستخدام موديل السكري"""
    try:
        model = model_manager.get_model("diabetes-predictor")
        if not model:
            raise HTTPException(status_code=404, detail="Diabetes model not found")
        
        start_time = time.time()
        result = model.predict(features)
        processing_time = time.time() - start_time
        
        result["processing_time"] = processing_time
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/asthma/predict", response_model=PredictionResponse)
async def predict_asthma(features: Dict[str, Any]):
    """التنبؤ باستخدام موديل الربو"""
    try:
        model = model_manager.get_model("asthma-predictor")
        if not model:
            raise HTTPException(status_code=404, detail="Asthma model not found")
        
        start_time = time.time()
        result = model.predict(features)
        processing_time = time.time() - start_time
        
        result["processing_time"] = processing_time
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.post("/api/schizo/predict", response_model=PredictionResponse)
async def predict_schizo(features: Dict[str, Any]):
    """Make prediction using the schizophrenia risk model"""
    try:
        model = model_manager.get_model("schizo-predictor")
        if not model:
            raise HTTPException(status_code=404, detail="Schizophrenia model not found")
        
        start_time = time.time()
        result = model.predict(features)
        processing_time = time.time() - start_time
        
        result["processing_time"] = processing_time
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") 
    
@app.delete("/api/models/{model_id}")
async def delete_model_endpoint(model_id: str):
    """
    Deletes a model from the manager and its corresponding file from saved_models.
    Note: This is a basic implementation. For production, consider more robust
    file deletion, access control, and handling of models currently in use.
    """
    model_to_delete = model_manager.get_model(model_id)
    if not model_to_delete:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")

    # Attempt to remove from model manager (in-memory)
    if model_manager.remove_model(model_id):
        logger.info(f"Model '{model_id}' removed from manager.")
        
        # Attempt to delete the model file(s) from disk
        # This part needs to know the actual file name and path.
        # The MLModel base class or specific model classes should store their file paths.
        # For now, let's assume a convention or try to find it.
        
        deleted_files_count = 0
        # Example: if models store their file path in model.model_path
        if hasattr(model_to_delete, 'model_path') and model_to_delete.model_path:
            if os.path.exists(model_to_delete.model_path):
                try:
                    os.remove(model_to_delete.model_path)
                    logger.info(f"Successfully deleted model file: {model_to_delete.model_path}")
                    deleted_files_count += 1
                except OSError as e:
                    logger.error(f"Error deleting model file {model_to_delete.model_path}: {e}")
                    # Decide if this is a critical error. Maybe the model was removed from manager
                    # but file deletion failed. For now, we continue.
            else:
                logger.warning(f"Model file path for {model_id} exists in object but not on disk: {model_to_delete.model_path}")
        else:
            # Fallback: Try to guess file names based on model_id (less robust)
            # This requires knowing the extensions (.pkl, .h5, etc.)
            possible_extensions = ['.pkl', '.h5', '.pth', '.keras'] # Add more as needed
            model_base_name = model_id # Or some other convention if model_id isn't the file base name
            
            # You'd need to get the actual file name from somewhere, e.g., model.get_info()
            # or by modifying your model classes to store their file path.
            # For example, if `schizoModel.pkl` corresponds to `schizo-predictor`
            # This part is highly dependent on how your model files are named and stored.
            # A simple placeholder for now. Ideally, each model object knows its file path.
            
            # Let's assume your model file paths are stored in the model instances
            # e.g., self.model_file_path = 'saved_models/my_model.pkl'
            # This logic needs to be more robust based on your MLModel design.
            # For example, CardioModel uses 'saved_models/cardio_model.pkl' AND 'saved_models/cardio_scaler.pkl'

            # A better approach: MLModel base class should have a method like `get_file_paths()`
            # and `delete_files()`.
            
            # For now, just log that file deletion logic needs implementation here
            logger.warning(f"Physical file deletion for model '{model_id}' needs more specific implementation based on model type and file storage.")

        if deleted_files_count > 0:
            return {"message": f"Model '{model_id}' and its associated files successfully deleted."}
        else:
            # If only removed from manager but no files found/deleted
            return {"message": f"Model '{model_id}' removed from manager. Associated files might need manual cleanup or were not found."}
            
    else:
        # This case should ideally not happen if model_to_delete was found earlier
        # but model_manager.remove_model somehow failed (e.g., if model_id disappeared between checks)
        raise HTTPException(status_code=500, detail=f"Failed to remove model '{model_id}' from manager, though it was found initially.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
