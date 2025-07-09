import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model, scaler, threshold
model = load_model("model/lstm_autoencoder.h5")
scaler = joblib.load("model/scaler.joblib")

with open("model/anomaly_threshold.txt", "r") as f:
    threshold = float(f.read())

st.title("🔍 Real-Time Anomaly Detection (LSTM Autoencoder)")

st.markdown("Enter the latest sensor values to detect anomalies:")

# Feature Inputs
features = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "OrganicMatter", "Porosity", "WaterHoldingCapacity"
]

user_input = []

for feature in features:
    value = st.number_input(f"{feature}", step=0.01, format="%.5f")
    user_input.append(value)

if st.button("🔎 Check for Anomaly"):
    # Prepare input: shape must be (1, 30, num_features)
    input_array = np.array([user_input] * 30)  # repeat same values for sequence
    input_scaled = scaler.transform(input_array)
    input_sequence = input_scaled.reshape(1, 30, len(features))

    # Predict & calculate reconstruction error
    prediction = model.predict(input_sequence)
    error = np.mean(np.abs(prediction - input_sequence))

    st.write(f"📉 Reconstruction Error: {error:.5f}")
    st.write(f"📊 Threshold: {threshold:.5f}")

    if error > threshold:
        st.error("🚨 Anomaly Detected!")
    else:
        st.success("✅ Normal Reading")

