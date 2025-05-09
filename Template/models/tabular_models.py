import logging
import traceback
import time
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model as load_keras_model
from typing import Dict, Any, Union
from .base_model import MLModel
import logging
from typing import Optional,List
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get a logger instance (usually at module level)
logger = logging.getLogger(__name__)

class CardioModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="cardio-predictor",
            name="Cardiovascular Disease Predictor",
            description="Predicts cardiovascular disease risk based on health metrics",
            input_type="tabular"
        )
        self.scaler = None
        self.feature_names = [
            'age', 'gender', 'height', 'weight', 
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
            'smoke', 'alco', 'active'
        ]

    def load(self):
        try:
            self.model = joblib.load('saved_models/cardio_model.pkl')
            self.scaler = joblib.load('saved_models/cardio_scaler.pkl')
            return True
        except Exception as e:
            logging.error(f"Error loading CardioModel: {str(e)}")
            return False

    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            # 1. تحويل البيانات إلى DataFrame
            import pandas as pd
            input_df = pd.DataFrame([features])
            
            # 2. معالجة العمود age
            #input_df['age'] = input_df['age'] / 1000

            # 3. التحقق من وجود كل الميزات المطلوبة
            required_features = [
                'age', 'gender', 'height', 'weight', 
                'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                'smoke', 'alco', 'active'
            ]
            
            missing = [f for f in required_features if f not in input_df.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")

            # 4. تطبيق المعايرة (Scaling)
            X = self.scaler.transform(input_df[required_features])
            
            # 5. التنبؤ (حل يعمل مع كل أنواع النماذج)
            if hasattr(self.model, 'predict_proba'):
                # للنماذج التي تدعم predict_proba
                proba = self.model.predict_proba(X)
                confidence = float(np.max(proba))
                prediction = int(np.argmax(proba))
            else:
                # للنماذج التي لا تدعمها (مثل Keras)
                pred = self.model.predict(X)
                confidence = float(pred[0][0])
                prediction = 1 if confidence > 0.5 else 0
                confidence = max(confidence, 1 - confidence)

            return {
                "prediction": "high risk" if prediction == 1 else "low risk",
                "confidence": confidence,
                "processing_time": 0.0
            }
            
        except Exception as e:
            import traceback
            logging.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            raise ValueError(f"Prediction failed: {str(e)}")

class DiabetesModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="diabetes-predictor",
            name="Diabetes Predictor",
            description="Predicts diabetes risk",
            input_type="tabular"
        )

    def load(self):
        try:
            self.model = joblib.load('saved_models/diabetes_pipeline.pkl')
            return True
        except Exception as e:
            logging.error(f"Error loading DiabetesModel: {str(e)}")
            return False

    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            input_df = pd.DataFrame([features])

            # ترتيب الأعمدة حسب المتوقع أثناء التدريب
            expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_df = input_df[expected_features]

            prediction = self.model.predict(input_df)
            proba = self.model.predict_proba(input_df)
            
            return {
                "prediction": "diabetic" if prediction[0] == 1 else "non-diabetic",
                "confidence": float(proba[0][1] if prediction[0] == 1 else proba[0][0])
            }
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise


class AsthmaModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="asthma-predictor",
            name="Asthma Predictor",
            description="Predicts asthma diagnosis based on health metrics",
            input_type="tabular"
        )
        
    def load(self):
        try:
            self.model = load_keras_model("saved_models/asthma_diagnosis_model.h5")
            return True
        except Exception as e:
            logging.error(f"Error loading AsthmaModel: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            input_df = pd.DataFrame([features])
            X = input_df.values.astype('float32')
            prediction = (self.model.predict(X) > 0.5).astype("int32")
            proba = self.model.predict(X)
            
            return {
                "prediction": "asthma" if prediction[0] == 1 else "no asthma",
                "confidence": float(proba[0][0] if prediction[0] == 1 else 1 - proba[0][0])
            }
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
class SchizophreniaModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="schizo-predictor",
            name="Schizophrenia Risk Predictor",
            description="Predicts schizophrenia risk level based on behavioral indicators",
            input_type="tabular"
        )
        self.feature_names = [
            'Age', 'Gender', 'Marital_Status', 'Fatigue', 
            'Slowing', 'Pain', 'Hygiene', 'Movement'
        ]
        self.class_labels = [
            'Low Proneness',
            'Moderate Proneness',
            'Elevated Proneness',
            'High Proneness',
            'Very High Proneness'
        ]
        # Define the mappings for your categorical features
        # IMPORTANT: These must match how the model was trained!
        self.gender_mapping = {'Male': 0, 'Female': 1} # Example: Male=0, Female=1
        self.marital_status_mapping = {
            'Single': 0, 
            'Married': 1, 
            'Divorced': 2, 
            'Widowed': 3
        } # Example mappings

    def load(self):
        try:
            self.model = joblib.load('saved_models/schizoModel.pkl')
            return True
        except Exception as e:
            logging.error(f"Error loading SchizophreniaModel: {str(e)}")
            return False

    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            # 1. Create a copy of features to avoid modifying the original dict
            processed_features = features.copy()

            # 2. Apply mappings for categorical features
            if 'Gender' in processed_features:
                gender_val = processed_features['Gender']
                if gender_val not in self.gender_mapping: # Check against keys of the mapping
                    raise ValueError(f"Invalid value for Gender: '{gender_val}'. Expected one of {list(self.gender_mapping.keys())}")
                processed_features['Gender'] = self.gender_mapping[gender_val]
            
            if 'Marital_Status' in processed_features:
                marital_val = processed_features['Marital_Status']
                if marital_val not in self.marital_status_mapping: # Check against keys
                    raise ValueError(f"Invalid value for Marital_Status: '{marital_val}'. Expected one of {list(self.marital_status_mapping.keys())}")
                processed_features['Marital_Status'] = self.marital_status_mapping[marital_val]

            # 3. Convert data to DataFrame using the processed features
            input_df = pd.DataFrame([processed_features])
            
            # 4. Ensure all self.feature_names are columns and numeric
            for feature_name in self.feature_names:
                if feature_name not in input_df.columns:
                    return {
                        "prediction": f"Error: Missing required feature '{feature_name}'",
                        "confidence": 0.0, "error": True,
                        "required_features": self.feature_names,
                        "missing_features": [feature_name],
                        "message": f"Feature '{feature_name}' is missing."
                    }
                try:
                    # Ensure values are numeric, converting if possible
                    input_df[feature_name] = pd.to_numeric(input_df[feature_name], errors='raise')
                except ValueError:
                    # If conversion fails, it's a bad input for a numeric field
                    return {
                        "prediction": f"Error: Invalid non-numeric value for feature '{feature_name}'",
                        "confidence": 0.0, "error": True,
                        "message": f"Feature '{feature_name}' expected a number but got '{input_df[feature_name].iloc[0]}'."
                    }
            
            # 5. Reorder columns to match training order
            X = input_df[self.feature_names]
            
            # 6. Make prediction
            #    self.model.predict(X) is returning the string label directly
            prediction_label_array = self.model.predict(X) 
            pred_label = str(prediction_label_array[0]) # Get the string label

            # Ensure the predicted label is one of the known class labels
            if pred_label not in self.class_labels:
                logging.error(f"Model predicted an unknown label: '{pred_label}'. Known labels: {self.class_labels}")
                return {
                    "prediction": "Error: Model returned an unexpected prediction label.",
                    "confidence": 0.0, "error": True,
                    "message": f"Model predicted '{pred_label}', which is not in the expected list of labels."
                }
            
            # 7. Calculate probabilities
            proba_array = self.model.predict_proba(X)[0] # Get probabilities for the first (and only) sample
            confidence = float(np.max(proba_array))
            
            # Map probabilities to class labels
            class_probabilities_dict = {
                label: float(prob) for label, prob in zip(self.model.classes_, proba_array)
            }
            # If self.model.classes_ is not available or doesn't match self.class_labels order,
            # you might need to be careful here. Assuming scikit-learn's predict_proba
            # returns probabilities in the order of self.model.classes_.
            # If schizoModel.pkl is a Pipeline, model.classes_ should refer to the final estimator's classes.

            return {
                "prediction": pred_label,
                "confidence": confidence,
                "class_probabilities": class_probabilities_dict,
                "processing_time": 0.0 # Should be handled by ModelManager or main.py
            }
        except ValueError as ve: # Catch specific ValueError for bad inputs during mapping or to_numeric
            logging.error(f"Input validation error for SchizophreniaModel: {str(ve)}\n{traceback.format_exc()}")
            return {
                "prediction": "Error: Invalid input value",
                "confidence": 0.0,
                "error": True,
                "message": str(ve) # This message will be shown to the user
            }
        except Exception as e:
            logging.error(f"General prediction error in SchizophreniaModel: {str(e)}\n{traceback.format_exc()}")
            return {
                "prediction": "Error: An unexpected error occurred during prediction.",
                "confidence": 0.0,
                "error": True,
                "message": "An internal server error occurred." # Generic message for unexpected errors
            }
        
class GenericTabularModel(MLModel):
    def __init__(self, model_id: str, name: str, description: str, model_file_path: str, feature_names: Optional[List[str]] = None):
        super().__init__(model_id, name, description, "tabular", model_file_path)
        self.user_provided_feature_names = feature_names
        self.loaded_feature_names_in_: Optional[List[str]] = None # From model if available

    def load(self) -> bool:
        if not self.model_file_path or not os.path.exists(self.model_file_path):
            logger.error(f"Model file not found for {self.name} at {self.model_file_path}")
            return False
        try:
            self.model = joblib.load(self.model_file_path)
            if hasattr(self.model, 'feature_names_in_'):
                self.loaded_feature_names_in_ = list(self.model.feature_names_in_)
            logger.info(f"{self.name} (Generic Tabular) loaded successfully from {self.model_file_path}.")
            return True
        except Exception as e:
            logger.error(f"Error loading generic tabular model {self.name} from {self.model_file_path}: {str(e)}")
            return False

    @property
    def active_feature_names(self) -> List[str]:
        if self.user_provided_feature_names:
            return self.user_provided_feature_names
        if self.loaded_feature_names_in_:
            return self.loaded_feature_names_in_
        logger.warning(f"No feature names available for {self.name}. Prediction might be unreliable or require specific input format.")
        return []

    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        if not self.model:
            return {"prediction": "Error: Model not loaded", "confidence": 0.0, "error": True, "message": "Model not loaded."}

        try:
            current_f_names = self.active_feature_names
            
            if not current_f_names:
                logger.warning(f"Attempting prediction for {self.name} without defined feature order. Input dict keys will be used in their current order.")
                # This is risky; order of dict items is not guaranteed for older Python versions
                # For robust behavior, feature_names should always be defined or input should be a list.
                input_values = [list(features.values())] # [[val1, val2, ...]]
                input_df_for_predict = pd.DataFrame(input_values)
            else:
                missing_keys = set(current_f_names) - set(features.keys())
                if missing_keys:
                    return {
                        "prediction": "Error: Missing features", "confidence": 0.0, "error": True,
                        "message": f"Missing input features: {missing_keys}",
                        "required_features": current_f_names
                    }
                # Ensure correct order and selection of features
                ordered_features = {fname: features[fname] for fname in current_f_names}
                input_df_for_predict = pd.DataFrame([ordered_features])[current_f_names]

            # Basic prediction
            raw_prediction = self.model.predict(input_df_for_predict)
            prediction_result = str(raw_prediction[0]) # Convert to string

            confidence_score = 1.0  # Default confidence
            class_probs = None

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_df_for_predict)
                confidence_score = float(np.max(probabilities[0]))
                if hasattr(self.model, 'classes_'):
                    class_probs = {str(cls): float(prob) for cls, prob in zip(self.model.classes_, probabilities[0])}

            response = {"prediction": prediction_result, "confidence": confidence_score}
            if class_probs:
                response["class_probabilities"] = class_probs
            return response

        except Exception as e:
            logger.error(f"Error during prediction with {self.name}: {str(e)}")
            return {"prediction": "Error", "confidence": 0.0, "error": True, "message": str(e)}