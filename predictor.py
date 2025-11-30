import pandas as pd
import streamlit as st
from feature_engineer import engineer_features

def preprocess_input(input_data, scaler=None):
    try:
        df_processed = engineer_features(input_data)
        expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal', 'age_to_thalach_ratio', 'age_to_ca_ratio',
            'sex_to_thal_ratio', 'row_mean', 'row_std', 'row_max_min_diff', 'age_binned',
            'trestbps_binned', 'chol_binned', 'thalach_binned', 'oldpeak_binned',
            'age_to_thalach_ratio_binned', 'age_to_ca_ratio_binned', 'row_mean_binned',
            'row_std_binned', 'row_max_min_diff_binned', 'age_squared', 'sex_squared',
            'cp_squared', 'trestbps_squared', 'chol_squared', 'age_x_sex', 'age_x_cp',
            'sex_x_cp', 'sex_x_trestbps', 'cp_x_trestbps', 'cp_x_chol', 'trestbps_x_chol'
        ]
        for feature in expected_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
        
        df_processed = df_processed[expected_features]
        
        if scaler:
            df_processed = pd.DataFrame(
                scaler.transform(df_processed),
                columns=df_processed.columns
            )
        
        return df_processed
    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def make_prediction(model, processed_data, threshold=0.6):
    try:
       
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)[0]
            
            if len(probabilities) == 2:
                prob_high_risk = probabilities[1]
                prob_low_risk = probabilities[0]
                prediction = 1 if prob_high_risk >= threshold else 0
                confidence = prob_high_risk * 100 if prediction == 1 else prob_low_risk * 100
                
                return prediction, confidence, prob_high_risk, prob_low_risk
            else:
                prediction = model.predict(processed_data)[0]
                confidence = max(probabilities) * 100
                return prediction, confidence, None, None
        else:
            prediction = model.predict(processed_data)[0]
            confidence = 85.0 if prediction == 1 else 15.0
            return prediction, confidence, None, None
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None
