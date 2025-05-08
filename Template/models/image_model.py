from typing import List, Dict, Union
import time
from .base_model import MLModel, ModelInfo

class ImageClassificationModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str, supported_formats: List[str] = None):
        super().__init__(model_id, name, description, "image")
        self.supported_formats = supported_formats or ["jpg", "jpeg", "png"]
        
    def load(self):
        self.model = "dummy_image_model"
        return True
    
    def predict(self, image_path: str) -> Dict[str, Union[str, float]]:
        time.sleep(0.5)
        return {
            "prediction": "cat",
            "confidence": 0.95
        }
    
    def get_info(self) -> ModelInfo:
        info = super().get_info()
        info.supported_formats = self.supported_formats
        return info