import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import requests
import os
from io import BytesIO

# --- App Title ---
st.title("Crash Predictor App 🚀")

# --- Constants ---
MODEL_URL = "https://github.com/mustechza/mustech-predict/raw/main/trained_crash_predictor_model.pkl"
MODEL_PATH = "trained_model.pkl"

# --- Download model if not already present ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            st.error("Failed to download model from GitHub.")
            return None
    return joblib.load(MODEL_PATH)

# --- Load Model ---
model = download_model()
if model is None:
    st.stop()

# --- Helper Functions ---
def parse_input(text):
    try:
        raw = [float(x.strip().lower().replace('x', '')) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 7:
        return None
    last_vals = values[-7:]
    return np.array([[
        np.mean(last_vals),
        np.std(last_vals),
        last_vals[-1],
        max(last_vals),
        min(last_vals),
        last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0,
        np.percentile(last_vals, 75) - np.percentile(last_vals, 25),  # IQR
    ]])

# --- Input Section ---
st.header("🔢 Input & Feedback")
input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
user_feedback = st.text_input("Actual next multiplier (optional, for accuracy tracking):")

# --- Prediction & Display ---
crash_values = parse_input(input_text)

if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)
        st.subheader(f"📈 Predicted Next Crash: {prediction:.2f}")
        st.success(f"🎯 Safe Target (3% edge): {safe_target:.2f}")

        # Feedback Accuracy
        if user_feedback:
            try:
                actual_val = float(user_feedback.strip().lower().replace('x', ''))
                if actual_val > 10.99:
                    actual_val = 10.5
                accuracy = 100 - abs(prediction - actual_val) / actual_val * 100
                st.info(f"📊 Accuracy of Last Prediction: {accuracy:.2f}%")
            except:
                st.warning("⚠️ Invalid feedback format.")

        # Indicators
        st.subheader("📊 Indicators")
        st.text(f"Mean (7): {np.mean(crash_values[-7:]):.2f}")
        st.text(f"Std Dev (7): {np.std(crash_values[-7:]):.2f}")
        if len(crash_values) > 1:
            st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")
        st.text(f"IQR: {np.percentile(crash_values[-7:], 75) - np.percentile(crash_values[-7:], 25):.2f}")

        # Crash Chart
        st.subheader("📉 Crash History")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-20:], marker='o', label='Recent')
        ax.axhline(np.mean(crash_values[-7:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("❗ Need at least 7 values for prediction.")
else:
    st.info("Enter at least 7 recent crash values to generate prediction.")
