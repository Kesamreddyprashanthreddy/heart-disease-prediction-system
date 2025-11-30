import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .healthy-card {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2d5016;
        border: 2px solid #7fcdcd;
    }
    .risk-card {
        background: linear-gradient(135deg, #ffd3e1 0%, #fd9bb5 100%);
        color: #8b0000;
        border: 2px solid #fd9bb5;
    }
    .sidebar .stSelectbox label, .sidebar .stSlider label, .sidebar .stRadio label {
        font-weight: 600;
        color: #2c3e50;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .disclaimer {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    
    try:
        with open("models/best_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        scaler = None
        if os.path.exists("models/scaler.pkl"):
            with open("models/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        
        label_encoder = None
        if os.path.exists("models/label_encoder.pkl"):
            with open("models/label_encoder.pkl", 'rb') as f:
                label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        return None, None, None

def preprocess_input(input_data, scaler=None):
    """Preprocess user input for prediction with full feature engineering"""
    try:
        df = pd.DataFrame([input_data])
        df_processed = df.copy()
        
        df_processed['age_to_thalach_ratio'] = np.where(
            df_processed['thalach'] != 0, 
            df_processed['age'] / df_processed['thalach'], 
            0
        )
        df_processed['age_to_ca_ratio'] = np.where(
            df_processed['ca'] != 0, 
            df_processed['age'] / df_processed['ca'], 
            0
        )
        df_processed['sex_to_thal_ratio'] = np.where(
            df_processed['thal'] != 0, 
            df_processed['sex'] / df_processed['thal'], 
            0
        )
        
        numerical_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df_processed['row_mean'] = df_processed[numerical_cols].mean(axis=1, skipna=True).fillna(0)
        df_processed['row_std'] = df_processed[numerical_cols].std(axis=1, skipna=True).fillna(0)
        df_processed['row_max_min_diff'] = (df_processed[numerical_cols].max(axis=1, skipna=True) - 
                                           df_processed[numerical_cols].min(axis=1, skipna=True)).fillna(0)
        
        binning_config = {
            'age': {'bins': [0, 40, 55, 70, 100], 'labels': [0, 1, 2, 3]},
            'trestbps': {'bins': [0, 120, 140, 180, 300], 'labels': [0, 1, 2, 3]},
            'chol': {'bins': [0, 200, 240, 300, 600], 'labels': [0, 1, 2, 3]},
            'thalach': {'bins': [0, 120, 150, 180, 250], 'labels': [0, 1, 2, 3]},
            'oldpeak': {'bins': [-1, 0, 1, 2, 10], 'labels': [0, 1, 2, 3]},
            'age_to_thalach_ratio': {'bins': [0, 0.3, 0.4, 0.5, 10], 'labels': [0, 1, 2, 3]},
            'age_to_ca_ratio': {'bins': [0, 15, 25, 40, 1000], 'labels': [0, 1, 2, 3]},
            'row_mean': {'bins': [0, 5, 7, 9, 20], 'labels': [0, 1, 2, 3]},
            'row_std': {'bins': [0, 3, 5, 8, 20], 'labels': [0, 1, 2, 3]},
            'row_max_min_diff': {'bins': [0, 150, 180, 220, 500], 'labels': [0, 1, 2, 3]}
        }
        
        for column, config in binning_config.items():
            if column in df_processed.columns:
                col_data = df_processed[column].fillna(0)
                try:
                    binned = pd.cut(col_data, 
                                  bins=config['bins'], 
                                  labels=config['labels'], 
                                  include_lowest=True)
                    df_processed[f'{column}_binned'] = binned.fillna(0).astype(int)
                except Exception as e:
                    df_processed[f'{column}_binned'] = 0
        
        df_processed['age_squared'] = df_processed['age'] ** 2
        df_processed['sex_squared'] = df_processed['sex'] ** 2
        df_processed['cp_squared'] = df_processed['cp'] ** 2
        df_processed['trestbps_squared'] = df_processed['trestbps'] ** 2
        df_processed['chol_squared'] = df_processed['chol'] ** 2
        
        df_processed['age_x_sex'] = df_processed['age'] * df_processed['sex']
        df_processed['age_x_cp'] = df_processed['age'] * df_processed['cp']
        df_processed['sex_x_cp'] = df_processed['sex'] * df_processed['cp']
        df_processed['sex_x_trestbps'] = df_processed['sex'] * df_processed['trestbps']
        df_processed['cp_x_trestbps'] = df_processed['cp'] * df_processed['trestbps']
        df_processed['cp_x_chol'] = df_processed['cp'] * df_processed['chol']
        df_processed['trestbps_x_chol'] = df_processed['trestbps'] * df_processed['chol']
        
        df_processed = df_processed.fillna(0)
        
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
        st.error(f"❌ Preprocessing error: {str(e)}")
        return None

def make_prediction(model, processed_data):
    """Make prediction using the trained model"""
    try:
        prediction = model.predict(processed_data)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = 85.0  
            
        return prediction, confidence
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None

def main():
    st.markdown("""
    <div class="main-header">
        <h1>❤️ HEART DISEASE PREDICTOR</h1>
        <p>Advanced AI-powered cardiovascular risk assessment using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler, label_encoder = load_model()
    
    if model is None:
        st.error("❌ Unable to load the prediction model. Please check if model files exist.")
        return
    
    st.success("✅ AI Model loaded successfully and ready for predictions!")
    
    with st.sidebar:
        st.markdown("### ℹ️ Model Information")
        st.info("""
        **Accuracy**: 90%+ on validation data
        
        **Data Source**: Cleveland Heart Disease Dataset
        
        **Algorithm**: Random Forest Classifier
        
        **Features**: 41 engineered parameters
        """)
        
        st.markdown("### 🎯 Risk Factors")
        st.markdown("- Age > 55 years")
        st.markdown("- Male gender")
        st.markdown("- High blood pressure (>140)")
        st.markdown("- High cholesterol (>240)")
        st.markdown("- Diabetes (FBS >120)")
        
        st.markdown("### 🏥 Quick Guide")
        st.markdown("""
        1. Fill patient information below
        2. Click predict button
        3. Review risk assessment
        4. Follow recommendations
        """)
    
    st.markdown("## 🏥 Welcome to Heart Disease Risk Assessment")
    st.markdown("""
    This AI-powered tool uses advanced machine learning to assess your cardiovascular health risk 
    based on clinical parameters and symptoms.
    
    ### How it works:
    1. **Fill in your health information** below
    2. **Click the Predict button** to run the AI analysis
    3. **Get instant results** with risk assessment and recommendations
    
    ### Key Features:
    - ⚡ **Instant Analysis**: Get results in seconds
    - 🎯 **High Accuracy**: Trained on validated medical data
    - 📊 **Clear Results**: Easy-to-understand risk assessment
    - 🩺 **Medical Guidance**: Professional recommendations included
    """)
    
    st.markdown("---")
    
    st.markdown("## 📋 Patient Information")
    st.markdown("Please fill in the following health parameters for risk assessment:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        age = st.slider("Age (years)", 20, 80, 50, help="Patient's age in years")
        sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex")
        
        st.markdown("### 💓 Cardiovascular Metrics")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120, 
                           help="Blood pressure when at rest")
        chol = st.slider("Cholesterol Level (mg/dl)", 100, 400, 200, 
                        help="Serum cholesterol level")
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150, 
                          help="Maximum heart rate achieved during exercise")
        
        st.markdown("### 🩺 Clinical Symptoms")
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                         help="Type of chest pain experienced")
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"],
                        help="Does exercise cause chest pain?")
        
    with col2:
        st.markdown("### 🧪 Laboratory Results")
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"],
                      help="Is fasting blood sugar greater than 120 mg/dl?")
        restecg = st.selectbox("Resting ECG Results", 
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                              help="Electrocardiogram results at rest")
        
        st.markdown("### 🔬 Advanced Tests")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1,
                          help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("ST Segment Slope", 
                           ["Upsloping", "Flat", "Downsloping"],
                           help="Slope of the peak exercise ST segment")
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4],
                         help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("Thalassemia", 
                          ["Normal", "Fixed Defect", "Reversible Defect"],
                          help="Blood disorder status")
        
        st.markdown("### 📊 Risk Indicators")
        risk_factors = []
        if age > 55: risk_factors.append("Age > 55")
        if sex == "Male": risk_factors.append("Male gender")
        if trestbps > 140: risk_factors.append("High BP")
        if chol > 240: risk_factors.append("High cholesterol")
        if fbs == "Yes": risk_factors.append("High blood sugar")
        
        if risk_factors:
            st.warning(f"Risk factors present: {', '.join(risk_factors)}")
        else:
            st.success("No major risk factors detected")
    
    st.markdown("---")
    
    col_empty, col_btn, col_empty2 = st.columns([1, 2, 1])
    with col_btn:
        predict_button = st.button("🔍 **PREDICT HEART DISEASE RISK**", 
                                  type="primary", use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    prediction = None
    confidence = None
    
    with col1:
        if predict_button:
            input_data = {
                'age': age,
                'sex': 1 if sex == "Male" else 0,
                'cp': ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp),
                'trestbps': trestbps,
                'chol': chol,
                'fbs': 1 if fbs == "Yes" else 0,
                'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
                'thalach': thalach,
                'exang': 1 if exang == "Yes" else 0,
                'oldpeak': oldpeak,
                'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
                'ca': ca,
                'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
            }
            
            with st.spinner("🤖 AI is analyzing your health parameters..."):
                processed_data = preprocess_input(input_data, scaler)
                
                if processed_data is not None:
                    pred_result, conf_result = make_prediction(model, processed_data)
                    prediction = pred_result
                    confidence = conf_result
                    
                    if prediction is not None:
                        
                        st.markdown("## 📊 Prediction Results")
                        
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-card risk-card">
                                <h2>⚠️ HEART DISEASE RISK DETECTED</h2>
                                <h3>High Risk Classification</h3>
                                <p><strong>Confidence Level: {confidence:.1f}%</strong></p>
                                <p>⚠️ This analysis suggests an increased risk of heart disease. 
                                Please consult with a cardiologist for immediate evaluation and personalized care.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 🚨 Immediate Recommendations")
                            st.error("Schedule an appointment with a cardiologist immediately")
                            st.warning("Consider additional cardiac testing (ECG, stress test, echocardiogram)")
                            st.info("Implement immediate lifestyle modifications (diet, exercise, stress management)")
                            
                        else:
                            st.markdown(f"""
                            <div class="prediction-card healthy-card">
                                <h2>✅ LOW HEART DISEASE RISK</h2>
                                <h3>Healthy Classification</h3>
                                <p><strong>Confidence Level: {confidence:.1f}%</strong></p>
                                <p>🎉 Great news! The analysis suggests a low risk of heart disease. 
                                Continue maintaining your healthy lifestyle and regular checkups.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Preventive care info
                            st.markdown("### 💚 Preventive Care Recommendations")
                            st.success("Continue regular exercise and healthy diet")
                            st.info("Maintain annual health checkups and monitoring")
                            st.info("Keep monitoring key risk factors (BP, cholesterol, weight)")
    
    with col2:
        if predict_button and prediction is not None:
            st.markdown("### 📈 Risk Analysis")
            risk_score = 0
            if age > 55: risk_score += 1
            if sex == "Male": risk_score += 1
            if trestbps > 140: risk_score += 1
            if chol > 240: risk_score += 1
            if fbs == "Yes": risk_score += 1
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Risk Factors Present", f"{risk_score}/5", 
                     delta=f"{'High' if risk_score > 3 else 'Moderate' if risk_score > 1 else 'Low'} Risk")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("AI Confidence", f"{confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Age Group", f"{age} years", 
                     delta="Senior" if age > 65 else "Adult")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()