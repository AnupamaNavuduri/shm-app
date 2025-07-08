# streamlit_app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Paths to saved model/scaler
MODEL_PATH = "model/lstm_autoencoder.h5"
SCALER_PATH = "model/scaler.joblib"
THRESHOLD_PATH = "model/anomaly_threshold.txt"

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read())

st.title("🔍 Soil Sensor Anomaly Detection")

st.markdown("Enter the most recent sensor values to detect anomalies:")

# Input form
with st.form("input_form"):
    AccX = st.number_input("AccX")
    AccY = st.number_input("AccY")
    AccZ = st.number_input("AccZ")
    GyroX = st.number_input("GyroX")
    GyroY = st.number_input("GyroY")
    GyroZ = st.number_input("GyroZ")
    OrganicMatter = st.number_input("Organic Matter")
    Porosity = st.number_input("Porosity")
    WaterHoldingCapacity = st.number_input("Water Holding Capacity")
    submitted = st.form_submit_button("Detect")

if submitted:
    try:
        # -------------------------------
        # 1. Collect and scale input
        # -------------------------------
        input_features = np.array([[AccX, AccY, AccZ, GyroX, GyroY, GyroZ,
                                    OrganicMatter, Porosity, WaterHoldingCapacity]])
        input_scaled = scaler.transform(input_features)

        # Create a sequence with dummy data for context (30 timesteps required)
        dummy_sequence = np.tile(input_scaled, (30, 1))  # shape: (30, num_features)
        sequence = np.expand_dims(dummy_sequence, axis=0)  # shape: (1, 30, num_features)

        # -------------------------------
        # 2. Predict & calculate reconstruction error
        # -------------------------------
        reconstructed = model.predict(sequence)
        reconstruction_error = np.mean(np.abs(sequence - reconstructed))
        is_anomaly = reconstruction_error > threshold

        # -------------------------------
        # 3. Display result
        # -------------------------------
        st.subheader("Result")
        st.write(f"Reconstruction Error: **{reconstruction_error:.5f}**")
        st.success("✅ No anomaly detected.") if not is_anomaly else st.error("⚠️ Anomaly detected!")

    except Exception as e:
        st.error(f"Error during processing: {e}")
