import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.title("Crash Predictor App 🚀")

# Hardcoded training data
X_sample = np.array([
    [2.25, 1.12, 1.63, 4.53, 1.02, -1.81],
    [2.36, 1.16, 1.02, 4.53, 1.01, -0.61],
    [1.84, 1.08, 1.63, 3.26, 1.01, 0.61],
    [2.84, 1.10, 2.26, 4.53, 1.01, 0.63],
    [2.38, 1.17, 2.26, 4.53, 1.01, 0.00],
    [2.70, 1.23, 2.89, 4.53, 1.01, 0.63],
    [2.32, 1.04, 1.01, 4.53, 1.01, -1.88],
    [2.61, 1.22, 3.26, 4.53, 1.01, 2.25],
    [2.89, 1.18, 3.01, 4.53, 1.01, -0.25],
    [2.90, 1.09, 2.26, 4.53, 1.01, -0.75],
])
y_sample = np.array([
    1.63, 1.02, 1.63, 2.26, 2.26, 2.89, 1.01, 3.26, 3.01, 2.26
])

# Train model
model = GradientBoostingRegressor()
model.fit(X_sample, y_sample)

# Input section
st.subheader("📥 Input")
input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
user_feedback = st.text_input("Actual next multiplier (optional, for training)")

# Parse input
def parse_input(text):
    try:
        raw = [float(x.strip().replace('x', '')) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

crash_values = parse_input(input_text)

# Extract features
def extract_features(values):
    if len(values) < 10:
        return None
    last_vals = values[-10:]
    return np.array([[
        np.mean(last_vals),
        np.std(last_vals),
        last_vals[-1],
        max(last_vals),
        min(last_vals),
        last_vals[-1] - last_vals[-2]
    ]])

# Prediction
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)
        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

# Feedback training
if st.button("Train with Feedback"):
    try:
        feedback = float(user_feedback.strip())
        if feedback > 10.99:
            feedback = 10.5
        new_features = extract_features(crash_values)
        if new_features is not None:
            global X_sample, y_sample
            X_sample = np.vstack([X_sample, new_features])
            y_sample = np.append(y_sample, feedback)
            model.fit(X_sample, y_sample)
            st.success("Model updated with your feedback!")
    except:
        st.error("Feedback must be a valid number")

# Indicators
if crash_values:
    st.subheader("📊 Indicators")
    st.text(f"Mean: {np.mean(crash_values[-10:]):.2f}")
    st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
    st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")

# Chart
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o', label='Recent')
    ax.axhline(np.mean(crash_values[-10:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# Accuracy trend
if len(y_sample) >= 10:
    preds = model.predict(X_sample[-30:])
    df_compare = pd.DataFrame({
        "Predicted": preds.round(2),
        "Actual": y_sample[-30:].round(2),
        "Absolute Error": np.abs(preds - y_sample[-30:]).round(2)
    })
    st.subheader("🧾 Prediction Accuracy (Last 30)")
    st.dataframe(df_compare)
