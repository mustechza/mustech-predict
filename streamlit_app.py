import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = "trained_crash_predictor_model.pkl"

st.set_page_config(page_title="Crash Predictor App", layout="centered")
st.title("🚀 Crash Predictor App")

# --- Data Preprocessing & Training ---
@st.cache_data
def preprocess_csv(file):
    df = pd.read_csv(file)
    values = df[df.columns[0]].astype(str).str.replace("x", "", regex=False).astype(float)
    capped = np.where(values > 10.99, 10.5, values)
    return capped

def extract_features(values, window=7):
    features = []
    labels = []
    for i in range(window, len(values)):
        window_vals = values[i - window:i]
        feat = [
            np.mean(window_vals),
            np.std(window_vals),
            window_vals[-1],
            max(window_vals),
            min(window_vals),
            window_vals[-1] - window_vals[-2],
        ]
        features.append(feat)
        labels.append(values[i])
    return np.array(features), np.array(labels)

def train_and_save_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

# --- Upload & Train Section ---
with st.expander("📤 Upload CSV & Train Model"):
    uploaded_file = st.file_uploader("Upload crash data CSV", type=["csv"])
    if uploaded_file:
        crash_data = preprocess_csv(uploaded_file)
        X, y = extract_features(crash_data)
        model = train_and_save_model(X, y)
        st.success("✅ Model trained and saved locally!")

# --- Load Existing Model ---
model = load_model()

# --- Prediction Section ---
if model:
    with st.expander("🔢 Predict Next Crash"):
        input_text = st.text_input("Enter recent crash multipliers (comma-separated)", key="predict_input")
        if input_text:
            try:
                values = np.array([min(float(x.strip().replace("x", "")), 10.5) for x in input_text.split(",")])
                if len(values) >= 7:
                    features = extract_features(values)[0][-1].reshape(1, -1)
                    prediction = model.predict(features)[0]
                    safe_target = round(prediction * 0.97, 2)

                    st.subheader(f"📈 Predicted: {prediction:.2f}")
                    st.success(f"🎯 Safe Target (3% edge): {safe_target:.2f}")

                    st.text(f"Mean: {np.mean(values[-5:]):.2f}")
                    st.text(f"Std Dev: {np.std(values[-5:]):.2f}")
                    st.text(f"Last Change: {(values[-1] - values[-2]):.2f}")

                    fig, ax = plt.subplots()
                    ax.plot(values[-10:], marker='o')
                    ax.axhline(np.mean(values[-5:]), linestyle='--', color='r', label='Mean')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("Please enter at least 7 crash multipliers.")
            except:
                st.error("Invalid input. Make sure values are comma-separated numbers like 1.2, 1.3x, etc.")
else:
    st.warning("⚠️ No trained model found. Please upload and train using a CSV file.")
