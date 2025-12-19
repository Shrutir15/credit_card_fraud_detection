import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. --- Streamlit App Configuration (MUST BE THE FIRST ST COMMAND) ---
st.set_page_config(
    page_title="Real-Time Credit Card Fraud Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        # Load the saved model and scaler
        model = joblib.load('fraud_detection_rf_model.pkl')
        scaler = joblib.load('scaler_for_time_amount.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or Scaler file not found. Please ensure the .pkl files are in the same directory.")
        return None, None

model, scaler = load_assets()

# --- Main App Content ---
st.title(" Real-Time Credit Card Fraud Detector")
st.markdown("""
    This application simulates a real-time fraud detection system. The objective is to **implement a robust fraud detection system** using machine learning to identify and prevent fraudulent activities.
    
    **Model:** Random Forest Classifier (Trained on balanced data using Random Under-Sampling).
""")

# --- Sidebar for Input Features ---
st.sidebar.header("Transaction Features Input")
st.sidebar.markdown("Adjust the values to simulate a new transaction.")

# Input fields for Time and Amount (unscaled)
with st.sidebar.expander("Time and Amount", expanded=True):
    # Time remains a slider
    time_input = st.slider('Time (Seconds since first transaction)', 0.0, 172792.0, 10000.0)
    # FIX: Changed Amount to number_input for precision and set format to 2 decimal places
    amount_input = st.number_input('Amount (Transaction Value)', min_value=0.00, max_value=25691.16, value=100.00, step=1.00, format="%.2f")

# Input fields for V-features (V1-V28)
v_features = {}
st.sidebar.subheader("Anonymized Features (V1-V28)")

# Split V-features into columns for a cleaner UI
cols = st.sidebar.columns(2)
for i in range(1, 29):
    col_index = (i - 1) % 2
    with cols[col_index]:
        v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, step=0.01, format="%.4f")


# --- Function to make prediction ---
def make_prediction(time, amount, v_features):
    if model is None or scaler is None:
        return 0, 0.0

    # 1. Scale Time and Amount 
    time_amount_scaled = scaler.transform([[time, amount]])
    time_scaled = time_amount_scaled[0, 0]
    amount_scaled = time_amount_scaled[0, 1]

    # 2. Create the input DataFrame (MUST match the training column order)
    
    v_data = [v_features[f'V{i}'] for i in range(1, 29)]
    input_data = v_data + [time_scaled, amount_scaled]
    
    feature_columns = [f'V{i}' for i in range(1, 29)] + ['Time_Scaled', 'Amount_Scaled']
    
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # 3. Make Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] 

    return prediction, probability

# --- Main Prediction Display ---
col_button, col_spacer = st.columns([1, 3])
with col_button:
    if st.button("Analyze Transaction", type="primary"):
        prediction, probability = make_prediction(time_input, amount_input, v_features)

        st.header("🔍 Analysis Result")
        
        if prediction == 1:
            st.error(f" FRAUDULENT TRANSACTION DETECTED (High Risk)")
            st.markdown(f"**Probability of Fraud:** <span style='font-size: 28px; color: red;'>**{probability*100:.2f}%**</span>", unsafe_allow_html=True)
            st.markdown(
                "**Action:** This transaction must be blocked immediately to **minimize financial losses**."
            )
        else:
            st.success(f" LEGITIMATE TRANSACTION (Low Risk)")
            st.markdown(f"**Probability of Fraud:** <span style='font-size: 28px; color: green;'>**{probability*100:.2f}%**</span>", unsafe_allow_html=True)
            st.markdown(
                "**Status:** Transaction is safe and approved. **Continuous monitoring** is recommended."
            )