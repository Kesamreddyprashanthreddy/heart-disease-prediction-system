import streamlit as st

def display_risk_result(prediction, confidence):
    st.markdown("## 📊 Prediction Results")
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

def display_healthy_result(prediction, confidence):
    st.markdown("## 📊 Prediction Results")
    st.markdown(f"""
    <div class="prediction-card healthy-card">
        <h2>✅ LOW HEART DISEASE RISK</h2>
        <h3>Healthy Classification</h3>
        <p><strong>Confidence Level: {confidence:.1f}%</strong></p>
        <p>🎉 Great news! The analysis suggests a low risk of heart disease. 
        Continue maintaining your healthy lifestyle and regular checkups.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💚 Preventive Care Recommendations")
    st.success("Continue regular exercise and healthy diet")
    st.info("Maintain annual health checkups and monitoring")
    st.info("Keep monitoring key risk factors (BP, cholesterol, weight)")

def display_risk_analysis(age, sex, trestbps, chol, fbs, confidence):
    st.markdown("### 📈 Risk Analysis")
    
    risk_factors = []
    risk_score = 0
    
    if age > 55: 
        risk_score += 1
        risk_factors.append(f"⚠️ Age over 55 ({age} years)")
    if sex == "Male": 
        risk_score += 1
        risk_factors.append("⚠️ Male gender (higher risk)")
    if trestbps > 140: 
        risk_score += 1
        risk_factors.append(f"⚠️ High blood pressure ({trestbps} mmHg)")
    if chol > 240: 
        risk_score += 1
        risk_factors.append(f"⚠️ High cholesterol ({chol} mg/dl)")
    if fbs == "Yes": 
        risk_score += 1
        risk_factors.append("⚠️ High fasting blood sugar")
    
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
    
