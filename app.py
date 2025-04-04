import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin_thickness = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success("Diabetic" if prediction[0] == 1 else "Not Diabetic")

