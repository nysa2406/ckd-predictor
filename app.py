import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('ckd_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Chronic Kidney Disease Progression Predictor")
st.markdown("""
### 🏥 AI-Based Clinical Decision Support System

This tool predicts the progression of Chronic Kidney Disease (CKD)
into severe stages using Machine Learning.

⚠️ *For educational use only*
""")

st.write("Enter patient details:")

# Inputs (IMPORTANT FEATURES ONLY)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, value=12.0)
pcv = st.number_input("Packed Cell Volume (%)", 20.0, 55.0, value=40.0)
sc = st.number_input("Serum Creatinine (mg/dL)", 0.5, 15.0, value=1.2)
al = st.slider("Albumin", 0, 5)
dm = st.selectbox("Diabetes", ["yes", "no"])
bgr = st.number_input("Blood Glucose (mg/dL)", 70.0, 300.0, value=120.0)
rc = st.number_input("RBC Count (millions)", 2.0, 7.0, value=4.5)
htn = st.selectbox("Hypertension", ["yes", "no"])
sod = st.number_input("Sodium (mEq/L)", 120.0, 150.0, value=135.0)

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

    # Get probability
prediction = model.predict(input_scaled)

if prediction[0] == 1:
    st.error("⚠️ High Risk: CKD Progression Likely")
else:
    st.success("✅ Low Risk: Stable Condition")

# Show risk %
st.subheader(f"Risk Score: {prob*100:.2f}%")

# Show result
if prob > 0.5:
    st.error("⚠️ High Risk: CKD Progression Likely")
else:
    st.success("✅ Low Risk: Stable Condition")
    st.subheader("Top Risk Factors")
st.write(["Specific Gravity", "Hemoglobin", "Creatinine", "Albumin"])
