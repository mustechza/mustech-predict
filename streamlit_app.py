# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

from sklearn.ensemble import GradientBoostingRegressor

# === Load Pretrained Model ===
MODEL_FILE = "trained_crash_predictor_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("Trained model not found. Please upload or retrain the model.")
    st.stop()

model = joblib.load(MODEL_FILE)

# === Feature Extraction Function ===
def extract_features(values):
    if len(values) < 10:
        return None
    last_ten = values[-10:]
    return np.array([[
        np.mean(last_ten),
        np.std(last_ten),
        last_ten[-1],
        max(last_ten),
        min(last_ten),
        last_ten[-1] - last_ten[-2] if len(last_ten) > 1 else 0,
        np.percentile(last_ten, 25),
        np.percentile(last_ten, 75),
    ]])

# === Input Section ===
st.title("Crash Predictor App 🎯")
input_text = st.text_input("🔢 Enter recent crash multipliers (comma-separated):")
user_feedback = st.text_input("📥 Actual next multiplier (optional, for feedback training):")

def parse_input(text):
    try:
        raw = [float(x.strip().replace("x", "")) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

crash_values = parse_input(input_text)

# === Prediction Section ===
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)

        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-10:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
        st.text(f"Last Change: {crash_values[-1] - crash_values[-2]:.2f}" if len(crash_values) > 1 else "N/A")

        # Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent')
        ax.axhline(np.mean(crash_values[-10:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Need at least 10 crash multipliers for prediction.")
