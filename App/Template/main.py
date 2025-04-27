import os
import time
import uuid
from typing import Dict, List, Optional, Any, Union
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil
from datetime import datetime

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Model registry to store loaded models
model_registry = {}

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float = 0.0
    
class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    input_type: str  # e.g., "image", "text", "tabular"
    supported_formats: List[str] = []
    
class TextPredictionRequest(BaseModel):
    text: str

class TabularPredictionRequest(BaseModel):
    features: Dict[str, Any]

class ModelUploadResponse(BaseModel):
    model_id: str
    name: str
    status: str

# Abstract base class for ML models
class MLModel:
    def __init__(self, model_id: str, name: str, description: str, input_type: str):
        self.model_id = model_id
        self.name = name
        self.description = description
        self.input_type = input_type
        self.model = None
        
    def load(self):
        """Load the model into memory"""
        raise NotImplementedError("Subclasses must implement load()")
    
    def predict(self, data):
        """Make a prediction using the model"""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def get_info(self) -> ModelInfo:
        """Return model information"""
        return ModelInfo(
            id=self.model_id,
            name=self.name,
            description=self.description,
            input_type=self.input_type
        )

# Example implementation for image classification model  #####
class ImageClassificationModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str, supported_formats: List[str] = None):
        super().__init__(model_id, name, description, "image")
        self.supported_formats = supported_formats or ["jpg", "jpeg", "png"]
        
    def load(self):
        # In a real implementation, this would load the actual model
        # For example: self.model = tensorflow.keras.models.load_model(f"models/{self.model_id}.h5")
        self.model = "dummy_image_model"
        return True
    
    def predict(self, image_path: str) -> Dict[str, Union[str, float]]:
        # In a real implementation, this would preprocess the image and run inference
        # For example:
        # image = preprocess_image(image_path)
        # prediction = self.model.predict(image)
        # result = postprocess_prediction(prediction)
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Return dummy result
        return {
            "prediction": "cat",
            "confidence": 0.95
        }
    
    def get_info(self) -> ModelInfo:
        info = super().get_info()
        info.supported_formats = self.supported_formats
        return info

# Example implementation for text analysis model   #####
class TextAnalysisModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str):
        super().__init__(model_id, name, description, "text")
        
    def load(self):
        # In a real implementation, this would load the actual model
        # For example: self.model = transformers.pipeline("sentiment-analysis")
        self.model = "dummy_text_model"
        return True
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        # In a real implementation, this would process the text and run inference
        # For example:
        # result = self.model(text)
        
        # Simulate processing time
        time.sleep(0.2)
        
        # Return dummy result
        return {
            "prediction": "positive" if "good" in text.lower() else "negative",
            "confidence": 0.87
        }

# Example implementation for tabular data model 
class TabularModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str):
        super().__init__(model_id, name, description, "tabular")
        
    def load(self):
        # In a real implementation, this would load the actual model
        # For example: self.model = joblib.load(f"models/{self.model_id}.pkl")
        self.model = "dummy_tabular_model"
        return True
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        # In a real implementation, this would process the features and run inference
        # For example:
        # feature_array = preprocess_features(features)
        # prediction = self.model.predict(feature_array)
        # result = postprocess_prediction(prediction)
        
        # Simulate processing time
        time.sleep(0.3)
        
        # Return dummy result
        return {
            "prediction": "Class A" if sum(float(v) for v in features.values() if isinstance(v, (int, float))) > 10 else "Class B",
            "confidence": 0.78
        }

# Model manager for handling model operations ##
class ModelManager:
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.load_sample_models()
        
    def load_sample_models(self):
        """Load sample models for demonstration"""
        # Image classification model
        image_model = ImageClassificationModel(
            model_id="image-classifier",
            name="Image Classifier",
            description="Classifies images into categories",
            supported_formats=["jpg", "jpeg", "png"]
        )
        image_model.load()
        self.models[image_model.model_id] = image_model
        
        # Text analysis model
        text_model = TextAnalysisModel(
            model_id="text-sentiment",
            name="Text Sentiment Analyzer",
            description="Analyzes sentiment in text"
        )
        text_model.load()
        self.models[text_model.model_id] = text_model
        
        # Tabular data model
        tabular_model = TabularModel(
            model_id="tabular-predictor",
            name="Tabular Data Predictor",
            description="Makes predictions on tabular data"
        )
        tabular_model.load()
        self.models[tabular_model.model_id] = tabular_model
    
    def get_models(self) -> List[ModelInfo]:
        """Get information about all available models"""
        return [model.get_info() for model in self.models.values()]
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get a specific model by ID"""
        return self.models.get(model_id)
    
    def add_model(self, model: MLModel) -> bool:
        """Add a new model to the registry"""
        if model.model_id in self.models:
            return False
        
        model.load()
        self.models[model.model_id] = model
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry"""
        if model_id not in self.models:
            return False
        
        del self.models[model_id]
        return True
    
    def predict_image(self, model_id: str, image_path: str) -> Dict[str, Any]:  ####
        """Make a prediction using an image model"""
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
    
    def predict_text(self, model_id: str, text: str) -> Dict[str, Any]:  ####
        """Make a prediction using a text model"""
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
    
    def predict_tabular(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:  ####
        """Make a prediction using a tabular data model"""
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

# Create FastAPI app
app = FastAPI(
    title="ML Models API",
    description="API for machine learning models integration",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model manager
model_manager = ModelManager()

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to ML Model API", "version": "0.1.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of available ML models"""
    return model_manager.get_models()

@app.get("/api/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get information about a specific model"""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.get_info()

@app.post("/api/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, file: UploadFile = File(...)):
    """Make prediction using specified model with file upload"""
    # Check if model exists
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check input type
    if model.input_type != "image":
        raise HTTPException(status_code=400, detail=f"This endpoint is for image models only. Model {model_id} expects {model.input_type} input.")
    
    # Save uploaded file
    file_location = f"uploads/{file.filename}"
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    
    # Make prediction
    try:
        result = model_manager.predict_image(model_id, file_location)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/models/{model_id}/predict/text", response_model=PredictionResponse)  ####
async def predict_text(model_id: str, request: TextPredictionRequest):
    """Make prediction on text using specified model"""
    # Check if model exists
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check input type
    if model.input_type != "text":
        raise HTTPException(status_code=400, detail=f"This endpoint is for text models only. Model {model_id} expects {model.input_type} input.")
    
    # Make prediction
    try:
        result = model_manager.predict_text(model_id, request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/models/{model_id}/predict/tabular", response_model=PredictionResponse)  ####
async def predict_tabular(model_id: str, request: TabularPredictionRequest):
    """Make prediction on tabular data using specified model"""
    # Check if model exists
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check input type
    if model.input_type != "tabular":
        raise HTTPException(status_code=400, detail=f"This endpoint is for tabular models only. Model {model_id} expects {model.input_type} input.")
    
    # Make prediction
    try:
        result = model_manager.predict_tabular(model_id, request.features)
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
    """Upload a new ML model"""
    # Validate input type
    if input_type not in ["image", "text", "tabular"]:
        raise HTTPException(status_code=400, detail="Invalid input type. Must be one of: image, text, tabular")
    
    # Generate model ID
    model_id = f"{name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Save model file
    file_location = f"models/{model_id}.bin"
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
    finally:
        model_file.file.close()
    
    # Create and register model
    try:
        if input_type == "image":
            model = ImageClassificationModel(model_id, name, description)
        elif input_type == "text":
            model = TextAnalysisModel(model_id, name, description)
        else:  # tabular
            model = TabularModel(model_id, name, description)
        
        success = model_manager.add_model(model)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register model")
        
        return ModelUploadResponse(
            model_id=model_id,
            name=name,
            status="uploaded"
        )
    except Exception as e:
        # Clean up on failure
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    # Check if model exists
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Remove model
    success = model_manager.remove_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    
    # Clean up model file
    model_file = f"models/{model_id}.bin"
    if os.path.exists(model_file):
        os.remove(model_file)
    
    return JSONResponse(content={"status": "success", "message": f"Model {model_id} deleted"})

# Mount static files for serving frontend (will be used later)
# app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
