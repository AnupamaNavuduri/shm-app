import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# --------------------------
# 1. Load model and scaler
# --------------------------
MODEL_PATH = "model/lstm_autoencoder.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

TIME_STEPS = 30
FEATURES = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "OrganicMatter", "Porosity", "WaterHoldingCapacity"
]

# --------------------------
# 2. Streamlit UI
# --------------------------
st.title("🚨 Anomaly Detection in Soil Sensor Data")
st.subheader("Enter Sensor Feature Values")

user_input = {}
for feature in FEATURES:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Check for Anomaly"):
    # --------------------------
    # 3. Prepare input for LSTM
    # --------------------------
    input_df = pd.DataFrame([user_input])[FEATURES]
    scaled_input = scaler.transform(input_df)

    # Repeat to create a sequence
    input_seq = np.repeat(scaled_input[np.newaxis, :], TIME_STEPS, axis=1)

    # --------------------------
    # 4. Predict & Calculate Error
    # --------------------------
    reconstructed = model.predict(input_seq)
    error = np.mean(np.abs(reconstructed - input_seq), axis=(1, 2))

    # Load your threshold (you can tune this or compute dynamically)
    threshold = 0.05  # Set your actual threshold here

    # --------------------------
    # 5. Display Result
    # --------------------------
    is_anomaly = error[0] > threshold
    st.metric("Reconstruction Error", f"{error[0]:.5f}")
    if is_anomaly:
        st.error("❌ Anomaly Detected!")
    else:
        st.success("✅ Normal Data")
