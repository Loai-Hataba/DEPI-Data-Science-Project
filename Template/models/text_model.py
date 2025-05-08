import time
from typing import Dict, Union
from .base_model import MLModel

class TextAnalysisModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str):
        super().__init__(model_id, name, description, "text")
        
    def load(self):
        self.model = "dummy_text_model"
        return True
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        time.sleep(0.2)
        return {
            "prediction": "positive" if "good" in text.lower() else "negative",
            "confidence": 0.87
        }