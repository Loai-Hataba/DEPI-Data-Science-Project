from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional # Add Optional
import logging
import os

logger = logging.getLogger(__name__)

class MLModel(ABC):
    def __init__(self, model_id: str, name: str, description: str, input_type: str, model_file_path: Optional[str] = None):
        self.model_id = model_id
        self.name = name
        self.description = description
        self.input_type = input_type
        self.model: Any = None
        # Store the primary file path for the model
        self.model_file_path = model_file_path
        # Store other associated file paths if needed, e.g., for scalers
        self.associated_file_paths: Dict[str, str] = {} 
        self.supported_formats: List[str] = []

    @abstractmethod
    def load(self) -> bool:
        pass

    @abstractmethod
    def predict(self, data: Any) -> Dict[str, Union[str, float]]:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.name,
            "description": self.description,
            "input_type": self.input_type,
            "supported_formats": self.supported_formats,
            # You might want to add expected_features for tabular models here
        }

    # Optional: a method to get all file paths for deletion
    def get_all_file_paths(self) -> List[str]:
        paths = []
        if self.model_file_path and os.path.exists(self.model_file_path):
            paths.append(self.model_file_path)
        for path in self.associated_file_paths.values():
            if path and os.path.exists(path):
                paths.append(path)
        return paths