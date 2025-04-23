import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.title("Crash Predictor App 🚀")

# Hardcoded training data
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1],
    [2.14, 0.71, 1.56, 3.26, 1.00, 0.97],
    [2.11, 0.74, 1.49, 3.24, 1.00, 1.73],
    [2.18, 0.74, 1.46, 3.29, 1.00, -0.03],
    [2.32, 0.94, 2.07, 4.26, 1.00, 0.61],
    [2.64, 1.18, 3.08, 5.00, 1.00, 1.01],
    [2.61, 1.20, 3.11, 5.10, 1.00, 0.03],
    [2.56, 1.12, 2.81, 4.65, 1.00, -0.30],
    [2.46, 1.13, 3.00, 4.68, 1.00, 0.19],
    [2.27, 1.05, 2.56, 4.44, 1.00, -0.44],
    [2.31, 1.06, 2.58, 4.32, 1.00, 0.02],
    [2.31, 1.09, 2.60, 4.50, 1.00, 0.02],
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1, 1.1, 1.00, 1.00, 1.00, 1.00, 2.80, 2.30, 2.10, 1.40, 1.10, 1.20, 1.30])

# Build model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_sample, y_sample)

# Input form
input_text = st.text_input("Enter recent crash multipliers (comma-separated)")
user_feedback = st.text_input("Actual next multiplier (optional, for model tuning)")

def parse_input(text):
    try:
        raw = [float(x.strip().replace("x", "")) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 5:
        return None
    last_five = values[-5:]
    return np.array([[
        np.mean(last_five),
        np.std(last_five),
        last_five[-1],
        max(last_five),
        min(last_five),
        last_five[-1] - last_five[-2] if len(last_five) > 1 else 0
    ]])

# Parse user input
crash_values = parse_input(input_text)

# Prediction
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)

        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

        # Accuracy display if feedback given
        if user_feedback:
            try:
                actual = float(user_feedback.strip().replace("x", ""))
                error = abs(prediction - actual)
                accuracy = max(0, (1 - error / actual)) * 100
                st.info(f"🧠 Last Prediction Accuracy: {accuracy:.2f}%")
            except:
                st.warning("⚠️ Invalid feedback format.")

        # Indicators
        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-5:]):.2f}")
        st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")

        # Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

        # Prediction comparison table
        if len(X_sample) >= 5:
            last_X = X_sample[-30:]
            last_y = y_sample[-30:]
            predicted_y = model.predict(last_X)
            df_compare = pd.DataFrame({
                "Predicted": predicted_y.round(2),
                "Actual": last_y.round(2),
                "Absolute Error": np.abs(predicted_y - last_y).round(2)
            })
            st.subheader("📋 Predicted vs Actual (Last 30)")
            st.dataframe(df_compare)

# Optional tuning via feedback
if st.button("Train with Feedback") and crash_values and user_feedback:
    try:
        feedback = float(user_feedback.strip().replace("x", ""))
        if feedback > 10.99:
            feedback = 10.5
        new_features = extract_features(crash_values)
        if new_features is not None:
            X_sample = np.vstack([X_sample, new_features])
            y_sample = np.append(y_sample, feedback)
            model.fit(X_sample, y_sample)
            st.success("Model updated with your input!")
    except:
        st.error("Feedback must be a number")
