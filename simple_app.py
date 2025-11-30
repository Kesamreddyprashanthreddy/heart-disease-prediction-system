import streamlit as st
from model_loader import load_models
from predictor import preprocess_input, make_prediction
from ui_components import (apply_custom_css, show_header, show_sidebar_info, 
                           show_introduction, show_risk_indicators)
from results_display import (display_risk_result, display_healthy_result, display_risk_analysis)

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

def main():
    show_header()
    model, scaler, label_encoder = load_models()
    
    if model is None:
        st.error("Unable to load the prediction model. Please check if model files exist.")
        return
    
    st.success("AI Model loaded successfully and ready for predictions!")
    
    show_sidebar_info()
    show_introduction()
    
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
        show_risk_indicators(age, sex, trestbps, chol, fbs)
    
    st.markdown("---")
    
    col_empty, col_btn, col_empty2 = st.columns([1, 2, 1])
    with col_btn:
        predict_button = st.button("🔍 **PREDICT HEART DISEASE RISK**", 
                                  type="primary", use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.prob_high = None
        st.session_state.prob_low = None
        st.session_state.last_prediction_input = None
    current_input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    current_input_hash = hash(str(current_input_data))
    if st.session_state.last_prediction_input is not None:
        if current_input_hash != st.session_state.last_prediction_input and not predict_button:
         
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.prob_high = None
            st.session_state.prob_low = None
            st.session_state.last_prediction_input = None
    
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
                    pred_result, conf_result, prob_high, prob_low = make_prediction(model, processed_data)
                    
                    # Store results in session state
                    st.session_state.prediction = pred_result
                    st.session_state.confidence = conf_result
                    st.session_state.prob_high = prob_high
                    st.session_state.prob_low = prob_low
                    st.session_state.last_prediction_input = current_input_hash
        
        if st.session_state.prediction is not None:
            if st.session_state.prob_high is not None and st.session_state.prob_low is not None:
                with st.expander("🔍 View Prediction Details", expanded=False):
                    st.markdown("### Probability Breakdown")
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Low Risk Probability", f"{st.session_state.prob_low:.1%}")
                    with col_prob2:
                        st.metric("High Risk Probability", f"{st.session_state.prob_high:.1%}")
                    
                    threshold = 0.6
                    st.info(f"**Threshold:** {threshold*100:.0f}% (High risk predicted if probability ≥ {threshold*100:.0f}%)")
                    
                    if 0.4 <= st.session_state.prob_high <= 0.6:
                        st.warning("⚠️ **Borderline Case:** The prediction is close to the threshold. Consider consulting a healthcare professional for a comprehensive evaluation.")
            
            if st.session_state.prediction == 1:
                display_risk_result(st.session_state.prediction, st.session_state.confidence)
            elif st.session_state.prediction == 0:
                display_healthy_result(st.session_state.prediction, st.session_state.confidence)
            else:
                st.warning(f"Unexpected prediction value: {st.session_state.prediction}")
                if st.session_state.confidence and st.session_state.confidence > 50:
                    display_risk_result(st.session_state.prediction, st.session_state.confidence)
                else:
                    display_healthy_result(st.session_state.prediction, st.session_state.confidence)
    
    with col2:
        if st.session_state.prediction is not None:
            display_risk_analysis(age, sex, trestbps, chol, fbs, st.session_state.confidence)

if __name__ == "__main__":
    main()