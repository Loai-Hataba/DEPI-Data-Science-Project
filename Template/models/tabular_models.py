import logging
import traceback
import time
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model as load_keras_model
from typing import Dict, Any, Union
from .base_model import MLModel
# main.py
from models.schemas import ModelInfo
# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tensorflow as tf
import logging

class CardioModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="cardiovascular-disease-predictor",
            name="Cardiovascular Disease Predictor",
            description="Predicts cardiovascular disease risk based on health metrics",
            input_type="tabular"
        )
        self.expected_features = [
            'age', 'gender', 'height', 'weight', 
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active'
        ]
    def get_info(self):
        base_info = super().get_info()  # جلب البيانات الأساسية
        return ModelInfo(              # إنشاء كائن جديد مع كل الحقول
            id=base_info.id,
            name=base_info.name,
            description=base_info.description,
            input_type=base_info.input_type,
            supported_formats=base_info.supported_formats,
            expected_features=self.expected_features  # إضافة الميزات
        )
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
        self.expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    def get_info(self):
        base_info = super().get_info()  # جلب البيانات الأساسية
        return ModelInfo(              # إنشاء كائن جديد مع كل الحقول
            id=base_info.id,
            name=base_info.name,
            description=base_info.description,
            input_type=base_info.input_type,
            supported_formats=base_info.supported_formats,
            expected_features=self.expected_features)  # إضافة الميزات
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
            description="Predicts asthma diagnosis based on 7 key metrics",
            input_type="tabular"
        )
        self.feature_names = [
            'Age', 'Gender', 'Symptoms', 'Allergy',
            'Cough', 'Wheezing', 'Breathlessness'
        ]
        self.expected_features = self.feature_names
    def get_info(self):
        base_info = super().get_info()  # جلب البيانات الأساسية
        return ModelInfo(              # إنشاء كائن جديد مع كل الحقول
            id=base_info.id,
            name=base_info.name,
            description=base_info.description,
            input_type=base_info.input_type,
            supported_formats=base_info.supported_formats,
            expected_features=self.expected_features)  # إضافة الميزات        
    def load(self):
        try:
            from .model_utils import adapt_keras_model
            self.model = adapt_keras_model(
                "saved_models/asthma_diagnosis_model.h5",
                new_input_shape=(7,)
            )
            return True
        except Exception as e:
            logging.error(f"Error loading AsthmaModel: {str(e)}")
            return False
    
    def predict(self, features):
        try:
            # ترتيب الميزات حسب self.feature_names
            input_data = np.array([
                features[name] for name in self.feature_names
            ]).reshape(1, -1).astype('float32')
            
            proba = self.model.predict(input_data)
            prediction = (proba > 0.5).astype(int)
            
            return {
                "prediction": "asthma" if prediction[0][0] else "no asthma",
                "confidence": float(proba[0][0] if prediction[0][0] else 1 - proba[0][0]),
                "processing_time": 0.0
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise
class SchizophreniaModel(MLModel):
    def __init__(self):
        super().__init__(
            model_id="schizophrenia-risk-predictor",
            name="Schizophrenia Risk Predictor",
            description="Predicts schizophrenia risk level",
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
        self.expected_features = self.feature_names
    def get_info(self):
        base_info = super().get_info()  # جلب البيانات الأساسية
        return ModelInfo(              # إنشاء كائن جديد مع كل الحقول
            id=base_info.id,
            name=base_info.name,
            description=base_info.description,
            input_type=base_info.input_type,
            supported_formats=base_info.supported_formats,
            expected_features=self.expected_features)  # إضافة الميزات  
    def load(self):
        try:
            self.model = joblib.load('saved_models/schizoModel.pkl')
            
            # تحقق من توافق النموذج
            if not hasattr(self.model, 'predict'):
                raise ValueError("النموذج المحمل لا يحتوي على دالة predict")
                
            return True
        except Exception as e:
            logging.error(f"Error loading SchizophreniaModel: {str(e)}")
            return False
    def predict(self, features: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        try:
            # 1. تحويل البيانات إلى DataFrame
            input_df = pd.DataFrame([features])
            
            # 2. التحقق من الميزات المطلوبة
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # 3. إعادة ترتيب الأعمدة
            X = input_df[self.feature_names]
            
            # 4. التنبؤ
            prediction = self.model.predict(X)
            
            # 5. معالجة النتيجة سواء كانت نصًا أو رقمًا
            if isinstance(prediction[0], str):
                pred_label = prediction[0]  # إذا كان النموذج يعيد النص مباشرة
            else:
                pred_label = self.class_labels[int(prediction[0])]  # إذا كان يعيد رقمًا
            
            # 6. حساب الثقة
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            
            return {
                "prediction": pred_label,
                "confidence": confidence,
                "processing_time": 0.0
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            raise ValueError(f"Prediction failed: {str(e)}")