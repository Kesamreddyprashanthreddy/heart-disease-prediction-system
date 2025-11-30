import pickle
import os
import streamlit as st

@st.cache_data
def load_models():
    model = None
    scaler = None
    label_encoder = None
    
    try:
        with open("models/best_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        if os.path.exists("models/scaler.pkl"):
            with open("models/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
        
        if os.path.exists("models/label_encoder.pkl"):
            with open("models/label_encoder.pkl", 'rb') as f:
                label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None
