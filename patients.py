import pandas as pd
import joblib
from PIL import Image
import os

def main():
    model = joblib.load('Random_Forest_Heart_Model.joblib')
    model_columns = joblib.load('model_columns.joblib')

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

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 2

    input_df = input_df[model_columns]

    pred_encoded = model.predict(input_df)

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

    print("Predicted Diagnosis:", pred_label)

    # Map predicted label to image file name (adjust the names exactly as your files)
    image_filename_map = {
        'Anterior Wall MI': 'Gemini_Generated_Image_Anterior Wall.png',
        'Cardiogenic Shock': 'Gemini_Generated_Image_Cardiogenic Shock.png',
        'Extensive MI': 'Gemini_Generated_Image_Extensive MI.png',
        'Inferior Wall MI': 'Gemini_Generated_Image_Inferior Wall.png',
        'Lateral Wall MI': 'Gemini_Generated_Image_Lateral Wall.png',
        'NSTEMI/ACS': 'Gemini_Generated_Image_NSTEMI_ACS.png',
        'Other': None,  # No image, or add if you have one
        'Other Cardiac Issue': None,
        'Posterior Wall MI': 'Gemini_Generated_Image_Posterior Wall.png',
        'Recurrent MI': 'Gemini_Generated_Image_Recurrent MI.png',
        'Septal/Side Wall MI': 'Gemini_Generated_Image_Septal Wall.png',
        'STEMI': 'Gemini_Generated_Image_STEMI.png',
        'Inferoposterior MI': 'Gemini_Generated_Image_Inferoposterior.png'
    }

    image_filename = image_filename_map.get(pred_label)

    if image_filename is not None:
        image_path = os.path.join('images', image_filename)
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.show()  # Opens default viewer with the image
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("No image available for this diagnosis.")

if __name__ == '__main__':
    main()
