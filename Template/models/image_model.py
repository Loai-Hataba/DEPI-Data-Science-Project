import os
from keras.models import load_model as load_keras_model 
from .base_model import MLModel, logger 
from typing import Dict,Union,Optional

class ImageClassificationModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str, model_file_path: Optional[str] = None):
        # If model_file_path is None, it means it's the pre-loaded model.
        # The actual path for pre-loaded model is set in ModelManager or here.
        # For uploaded models, model_file_path will be provided.
        super().__init__(model_id, name, description, "image", model_file_path)
        self.supported_formats = ["image/jpeg", "image/png"]
        # self.model_path = model_file_path # Already handled by super().__init__

    def load(self) -> bool:
        if not self.model_file_path: # If no path is set (e.g. for default built-in model)
            # This logic needs to align with how you define paths for built-in models
            # For a default model, you might set self.model_file_path in __init__
            # e.g. self.model_file_path = model_file_path or "models/default_image_model.h5"
            # For now, let's assume pre-loaded models have their paths set correctly by ModelManager.
             # For built-in, it might be like: self.model_file_path = "models/your_default_image_model.h5"
            logger.warning(f"No model file path specified for {self.name}. Using default or assuming pre-configured path.")
            # Fallback for existing pre-loaded model structure if it hardcodes path here
            if not self.model_file_path: self.model_file_path = "models/image_model.h5" # EXAMPLE default path

        if not os.path.exists(self.model_file_path):
            logger.error(f"Model file not found for {self.name} at {self.model_file_path}")
            return False
        try:
            self.model = load_keras_model(self.model_file_path) # Use self.model_file_path
            logger.info(f"{self.name} (Keras model) loaded successfully from {self.model_file_path}.")
            return True
        except Exception as e:
            logger.error(f"Error loading Keras model {self.name} from {self.model_file_path}: {str(e)}")
            return False

    def predict(self, image_path: str) -> Dict[str, Union[str, float]]:
        # ... (keep existing predict logic) ...
        # Ensure it uses self.model
        if not self.model:
            return {"prediction": "Error: Model not loaded", "confidence": 0.0}
        
        # Placeholder for actual prediction logic
        try:
            # Example: (This depends on your actual image model's input requirements)
            # from tensorflow.keras.preprocessing import image
            # import numpy as np
            # img = image.load_img(image_path, target_size=(224, 224)) # Adjust target_size
            # img_array = image.img_to_array(img)
            # img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            # processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims) # Example
            # predictions = self.model.predict(processed_image)
            # predicted_class_idx = np.argmax(predictions[0])
            # confidence = float(np.max(predictions[0]))
            # Define your class labels
            # class_labels = ["Cat", "Dog", "Other"] # Example
            # predicted_label = class_labels[predicted_class_idx]
            
            # Dummy prediction for now if logic is complex elsewhere
            logger.warning("ImageClassificationModel.predict is using placeholder logic.")
            return {"prediction": "dummy_image_prediction", "confidence": 0.95}

        except Exception as e:
            logger.error(f"Error during image prediction with {self.name}: {e}")
            return {"prediction": "Error during prediction", "confidence": 0.0}

# Apply similar changes to models/text_model.py TextAnalysisModel