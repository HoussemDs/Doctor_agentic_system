import pandas as pd
import joblib

def main():
    # Load the trained model
    model = joblib.load('Random_Forest_Heart_Model.joblib')
    # Load the list of columns used during training
    model_columns = joblib.load('model_columns.joblib')
    
    # Prepare new sample input data as a dict
    # Make sure to include all columns your model expects, 
    # missing columns will be added with zero below
    sample_data = {
        'Age': [65],
        'C.P.K': [300],
        'BGR': [150],
        'PLATELET_COUNT': [250000],
        'Hemoglobin': [14],
        'RBC': [4.5],
        'B.Urea': [20],
        'WBC': [8000],
        'CK.MB': [50],
        'NEUTROPHIL': [60],
        # Add other columns if you want or just rely on zeros added automatically
    }
    
    # Convert dict to DataFrame
    input_df = pd.DataFrame(sample_data)
    
    # Add missing columns with zeros
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[model_columns]
    
    # Predict using the model
    pred_encoded = model.predict(input_df)
    
    # Mapping from encoded labels to original diagnosis names (update if needed)
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
    
    # Decode prediction to original label
    pred_label = label_mapping.get(pred_encoded[0], "Unknown")
    
    print("Predicted Diagnosis:", pred_label)

if __name__ == '__main__':
    main()
