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
# models/tabular_models.py
from models.schemas import ModelInfo
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
            CardioModel(),  # تم تعديل هذا السطر
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


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# لو الفرونت شغال على localhost:3000 مثلاً
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# استبدال تكرار CORS بهذا:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.post("/api/models/{model_id}/predict/tabular", response_model=PredictionResponse)
async def predict_tabular(model_id: str, request: TabularPredictionRequest):
    """التنبؤ باستخدام موديل الجداول"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.input_type != "tabular":
        raise HTTPException(status_code=400, detail=f"This endpoint is for tabular models only. Model {model_id} expects {model.input_type} input.")
    
    try:
        result = model_manager.predict_tabular(model_id, request.features)
        return PredictionResponse(**result)
    except Exception as e:
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

@app.post("/api/models/upload", response_model=ModelUploadResponse)
async def upload_model(
    name: str = Form(...),
    description: str = Form(...),
    input_type: str = Form(...),
    model_file: UploadFile = File(...)
):
    try:
        # حفظ الملف مؤقتًا
        file_location = f"uploads/{model_file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
        
        # هنا يمكنك معالجة النموذج وحفظه
        model_id = str(uuid.uuid4())
        return {
            "model_id": model_id,
            "name": name,
            "status": "uploaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    