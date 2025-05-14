# models/schemas.py
from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    input_type: str
    supported_formats: list[str] = []
    expected_features: list[str] = []