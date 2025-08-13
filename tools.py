from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from PIL import Image
import os

class HeartPredictionInput(BaseModel):
    """Input schema for Heart Disease Predictor."""
    patient_data: str = Field(..., description="Patient data string containing symptoms and medical history")

class HeartPredictionTool(BaseTool):
    name: str = "Heart Disease Predictor"
    description: str = "Predicts heart condition based on patient medical data using a trained ML model."
    args_schema: Type[BaseModel] = HeartPredictionInput

    def _run(self, patient_data: str) -> str:
        """
        Predicts heart condition based on patient medical data using a trained ML model.
        
        Args:
            patient_data: A string containing patient medical information that will be parsed
                         or sample data will be used for demonstration
        
        Returns:
            str: Predicted heart condition diagnosis
        """
        try:
            # Load the trained model and columns
            model = joblib.load('Random_Forest_Heart_Model.joblib')
            model_columns = joblib.load('model_columns.joblib')
            
            # For now, using sample data - you can modify this to parse patient_data
            # In a real implementation, you'd extract medical values from patient_data string
            sample_data = {
                'Age': [45],
                'F.History': [0],
                'Diabetes': [0],
                'BP': [90.6],
                'Thrombolysis': [0],
                'BGR': [150],
                'B.Urea': [20],
                'S.Cr': [1.0],
                'S.Sodium': [140],
                'S.Potassium': [4.0],
                'S.Chloride': [100],
                'C.P.K': [200],
                'CK.MB': [50],
                'ESR': [10],
                'WBC': [8000],
                'RBC': [4.5],
                'Hemoglobin': [14],
                'P.C.V': [40],
                'M.C.V': [90],
                'M.C.H': [30],
                'M.C.H.C': [33],
                'PLATELET_COUNT': [250000],
                'NEUTROPHIL': [60],
                'LYMPHO': [30],
                'MONOCYTE': [5],
                'EOSINO': [2],
                'cp': [1],
                'trestbps': [120],
                'chol': [200],
                'fbs': [0],
                'restecg': [1],
                'thalach': [150],
                'exang': [0],
                'oldpeak': [1.0],
                'slope': [2],
                'ca': [0],
                'thal': [3],
            }
            
            input_df = pd.DataFrame(sample_data)
            
            # Ensure all model columns are present
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 2
            
            input_df = input_df[model_columns]
            
            # Make prediction
            pred_encoded = model.predict(input_df)
            
            # Label mapping
            label_mapping = {
                0: 'Anterior Wall MI',
                1: 'Cardiogenic Shock',
                2: 'Extensive MI',
                3: 'Inferior Wall MI',
                4: 'Lateral Wall MI',
                5: 'NSTEMI/ACS',
                6: 'Other',
                7: 'Other Cardiac Issue',
                8: 'Posterior Wall MI',
                9: 'Recurrent MI',
                10: 'Septal/Side Wall MI',
                11: 'STEMI',
                12: 'Inferoposterior MI'
            }
            
            pred_label = label_mapping.get(pred_encoded[0], "Unknown")
            
            return f"ML Model Prediction: {pred_label}"
            
        except Exception as e:
            return f"Error in heart disease prediction: {str(e)}"

class HeartImageInput(BaseModel):
    """Input schema for Heart Condition Image Display."""
    condition: str = Field(..., description="The heart condition diagnosis to display image for")

class HeartConditionImageTool(BaseTool):
    name: str = "Heart Condition Image Display"
    description: str = "Displays an image related to the diagnosed heart condition."
    args_schema: Type[BaseModel] = HeartImageInput

    def _run(self, condition: str) -> str:
        """
        Displays an image related to the diagnosed heart condition.
        
        Args:
            condition: The heart condition diagnosis
        
        Returns:
            str: Status message about image display
        """
        try:
            # Map predicted label to image file name
            image_filename_map = {
                'Anterior Wall MI': 'Gemini_Generated_Image_Anterior Wall.png',
                'Cardiogenic Shock': 'Gemini_Generated_Image_Cardiogenic Shock.png',
                'Extensive MI': 'Gemini_Generated_Image_Extensive MI.png',
                'Inferior Wall MI': 'Gemini_Generated_Image_Inferior Wall.png',
                'Lateral Wall MI': 'Gemini_Generated_Image_Lateral Wall.png',
                'NSTEMI/ACS': 'Gemini_Generated_Image_NSTEMI_ACS.png',
                'Other': None,
                'Other Cardiac Issue': None,
                'Posterior Wall MI': 'Gemini_Generated_Image_Posterior Wall.png',
                'Recurrent MI': 'Gemini_Generated_Image_Recurrent MI.png',
                'Septal/Side Wall MI': 'Gemini_Generated_Image_Septal Wall.png',
                'STEMI': 'Gemini_Generated_Image_STEMI.png',
                'Inferoposterior MI': 'Gemini_Generated_Image_Inferoposterior.png'
            }
            
            image_filename = image_filename_map.get(condition)
            
            if image_filename is not None:
                image_path = os.path.join('images', image_filename)
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    img.show()
                    return f"Successfully displayed image for {condition}"
                else:
                    return f"Image file not found: {image_path}"
            else:
                return f"No image available for condition: {condition}"
                
        except Exception as e:
            return f"Error displaying image: {str(e)}"

# Create instances of the tools to use
heart_predictor = HeartPredictionTool()
heart_image_display = HeartConditionImageTool()