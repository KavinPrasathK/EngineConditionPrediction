# Script for the Streamlit UI app

# Importing the necessary libraries
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="KavinPrasathK/Engine_Condition_Prediction", filename="best_engine_condition_prediction_model_v1.joblib")
    model = joblib.load(model_path)
    return model

model = load_model()

# Streamlit UI for Engine Condition Prediction
st.title("Engine Condition Prediction")
st.write("""
This application predicts the condition of an engine (Normal or Faulty)
based on its sensor readings. Please input the engine parameters below.
""")

# User input fields
st.header("Engine Parameters")
engine_rpm = st.number_input("Engine RPM", min_value=50, max_value=2500, value=750)
lub_oil_pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.001, max_value=10.00, value=3.00)
fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.001, max_value=25.00, value=5.00)
coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.001, max_value=10.00, value=2.00)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=50.00, max_value=100.00, value=75.00)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=50.00, max_value=200.00, value=75.00)

# Assemble input into DataFrame
input_data = pd.DataFrame([
    {
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'Lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
    }
])

if st.button("Predict"):
    # Make prediction
    prediction_proba = model.predict_proba(input_data)[:, 1][0] # Get probability of class 1 (Faulty)

    # Classification threshold
    classification_threshold = 0.45   # Using the same threshold as in training

    # Converting probability to a binary prediction based on the threshold
    prediction = 1 if prediction_proba >= classification_threshold else 0

    status = "FAULTY (Maintenance Required)" if prediction == 1 else "NORMAL"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"The model predicts the engine condition is **{status}** with a probability of {prediction_proba:.2f}.")
    else:
        st.success(f"The model predicts the engine condition is **{status}** with a probability of {prediction_proba:.2f}.")
