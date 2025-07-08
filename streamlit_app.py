import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# -------------------------------
# Load Model and Scaler
# -------------------------------
MODEL_PATH = "model/lstm_autoencoder.h5"
SCALER_PATH = "model/scaler.pkl"  # if you saved it using pickle

# Load model without compiling (fixes 'mae' deserialization issue)
model = load_model(MODEL_PATH, compile=False)

# Load the same scaler used during training
import pickle
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# UI: Streamlit Input Form
# -------------------------------
st.title("🔍 SHM Anomaly Detection")

st.markdown("Enter values for the following 9 features:")

feature_names = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "OrganicMatter", "Porosity", "WaterHoldingCapacity"
]

# Collect user input for each feature
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

# -------------------------------
# Predict Anomaly
# -------------------------------
if st.button("Check for Anomaly"):
    # Convert input to 2D array
    input_array = np.array([user_input])

    # Scale input using saved scaler
    scaled_input = scaler.transform(input_array)

    # Create sequence (reshape to 3D for LSTM)
    TIME_STEPS = 30
    repeated_input = np.repeat(scaled_input, TIME_STEPS, axis=0)
    input_seq = np.reshape(repeated_input, (1, TIME_STEPS, len(feature_names)))

    # Get reconstruction
    reconstructed = model.predict(input_seq)
    error = np.mean(np.abs(reconstructed - input_seq), axis=(1, 2))

    # Set your previously computed threshold
    threshold = 0.015  # ⚠️ Replace with your actual threshold

    # Output result
    if error[0] > threshold:
        st.error(f"🚨 Anomaly detected! Reconstruction error = {error[0]:.5f}")
    else:
        st.success(f"✅ Normal behavior. Reconstruction error = {error[0]:.5f}")
