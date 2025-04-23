# crash_predictor_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Crash Predictor App 🚀", layout="centered")
st.title("Crash Predictor App 🚀")

# Hardcoded training data
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1],
    [2.3, 0.4, 2.4, 2.7, 1.9, 0.3],
    [1.7, 0.2, 1.6, 2.0, 1.5, -0.1]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1, 2.2, 1.5])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_sample, y_sample)

def parse_input(text):
    try:
        raw = [float(x.strip().replace("x", "")) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 10:
        return None
    last = values[-10:]
    return np.array([[
        np.mean(last),
        np.std(last),
        last[-1],
        max(last),
        min(last),
        last[-1] - last[-2] if len(last) > 1 else 0
    ]])

# === 📥 Input Section with Training Feedback ===
with st.expander("🧠 Input & Feedback"):
    input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
    user_feedback = st.text_input("Actual next multiplier (optional, for training)")

    crash_values = parse_input(input_text)

    if st.button("Train with Feedback"):
        try:
            feedback = float(user_feedback.strip())
            if feedback > 10.99:
                feedback = 10.5
            new_features = extract_features(crash_values)
            if new_features is not None:
                X_sample = np.vstack([X_sample, new_features])
                y_sample = np.append(y_sample, feedback)
                model.fit(X_sample, y_sample)
                st.success("Model retrained with your feedback.")
        except:
            st.error("Feedback must be a numeric multiplier.")

# === 🔮 Prediction Section ===
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)
        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")
    else:
        st.warning("Please enter at least 10 recent crash multipliers.")

# === 📊 Indicators Section ===
if crash_values:
    st.subheader("📊 Indicators")
    st.text(f"Mean: {np.mean(crash_values[-10:]):.2f}")
    st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
    if len(crash_values) >= 2:
        st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")

# === 📉 Chart ===
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-20:], marker='o', label='Recent Multipliers')
    ax.axhline(np.mean(crash_values[-10:]), color='r', linestyle='--', label='10x Mean')
    ax.legend()
    st.pyplot(fig)

# === 📋 Accuracy Comparison Table & Trend ===
if len(y_sample) >= 5:
    recent_X = X_sample[-30:]
    recent_y = y_sample[-30:]
    predicted_y = model.predict(recent_X)

    st.subheader("📋 Predicted vs Actual (Last 30)")
    df = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": recent_y.round(2),
        "Error": np.abs(predicted_y - recent_y).round(2)
    })
    st.dataframe(df)

    st.subheader("📈 Accuracy Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(100 - (np.abs(predicted_y - recent_y) / recent_y * 100), label='Accuracy (%)', marker='o')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Recent Predictions")
    ax2.legend()
    st.pyplot(fig2)
