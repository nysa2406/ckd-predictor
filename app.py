import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('ckd_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 CKD Progression Predictor")

st.write("Enter patient details:")

# Inputs (IMPORTANT FEATURES ONLY)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
hemo = st.number_input("Hemoglobin", 0.0, 20.0)
pcv = st.number_input("Packed Cell Volume")
sc = st.number_input("Serum Creatinine")
al = st.slider("Albumin", 0, 5)
dm = st.selectbox("Diabetes", ["yes", "no"])
bgr = st.number_input("Blood Glucose Random")
rc = st.number_input("Red Blood Cell Count")
htn = st.selectbox("Hypertension", ["yes", "no"])
sod = st.number_input("Sodium")

# Encode manually (same as training)
dm = 1 if dm == "yes" else 0
htn = 1 if htn == "yes" else 0

# Prediction
if st.button("Predict"):

    input_data = np.array([[sg, hemo, pcv, sc, al, dm, bgr, rc, htn, sod]])

    # ⚠️ If scaler mismatch happens, remove scaling line
    try:
        input_scaled = scaler.transform(input_data)
    except:
        input_scaled = input_data

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: CKD Progression Likely")
    else:
        st.success("✅ Low Risk: Stable Condition")