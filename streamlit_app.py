# streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.title("Crash Predictor App 🚀")

# Sample hardcoded training data (features and labels)
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1])

# Initialize model
model = GradientBoostingRegressor()
model.fit(X_sample, y_sample)

def parse_input(text):
    try:
        raw = [float(x.strip().replace('x', '')) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

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
        last_ten[-1] - last_ten[-2]
    ]])

# Combined input & training section
st.subheader("🔢 Input & Feedback")
input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
user_feedback = st.text_input("Actual next multiplier (optional, for training):")

crash_values = parse_input(input_text)

if st.button("Train with Feedback"):
    if crash_values and len(crash_values) >= 10:
        features = extract_features(crash_values)
        if features is not None:
            prediction = model.predict(features)[0]
            st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
            st.success(f"🎯 Safe target (3% edge): {round(prediction * 0.97, 2)}")

            # Train if user gave feedback
            if user_feedback:
                try:
                    actual = float(user_feedback.strip().replace('x', ''))
                    if actual > 10.99:
                        actual = 10.5
                    X_sample = np.vstack([X_sample, features])
                    y_sample = np.append(y_sample, actual)
                    model.fit(X_sample, y_sample)
                    st.success("✅ Model trained with your input!")
                except:
                    st.error("Feedback must be a numeric multiplier")
        else:
            st.warning("Need at least 10 crash multipliers to predict.")
    else:
        st.warning("Enter at least 10 crash multipliers.")

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

# Accuracy table
if len(y_sample) >= 5:
    last_X = X_sample[-30:]
    last_y = y_sample[-30:]
    predicted_y = model.predict(last_X)

    st.subheader("🧾 Predicted vs Actual (Last 30 Entries)")
    df_compare = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": last_y.round(2),
        "Absolute Error": np.abs(predicted_y - last_y).round(2)
    })
    st.dataframe(df_compare)
