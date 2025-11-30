import streamlit as st

def apply_custom_css():
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
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>❤️ HEART DISEASE PREDICTOR</h1>
        <p>Advanced AI-powered cardiovascular risk assessment using machine learning</p>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar_info():
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

def show_introduction():
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

def show_risk_indicators(age, sex, trestbps, chol, fbs):
    risk_factors = []
    if age > 55: 
        risk_factors.append("Age > 55")
    if sex == "Male": 
        risk_factors.append("Male gender")
    if trestbps > 140: 
        risk_factors.append("High BP")
    if chol > 240: 
        risk_factors.append("High cholesterol")
    if fbs == "Yes": 
        risk_factors.append("High blood sugar")
    
    if risk_factors:
        st.warning(f"Risk factors present: {', '.join(risk_factors)}")
    else:
        st.success("No major risk factors detected")
