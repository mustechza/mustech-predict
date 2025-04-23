# crash_predictor_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# App title
st.title("Crash Predictor App 🚀")

# User input
input_text = st.text_input("Enter recent crash multipliers (comma-separated)")
user_feedback = st.text_input("Actual next multiplier (optional, for training)")

# Cap values over 10.99 to 10.5
def parse_input(text):
    try:
        raw = [float(x.strip()) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

crash_values = parse_input(input_text)

# Feature extraction
def extract_features(values):
    if len(values) < 5:
        return None
    last_five = values[-5:]
    return np.array([
        [
            np.mean(last_five),
            np.std(last_five),
            last_five[-1],
            max(last_five),
            min(last_five),
            last_five[-1] - last_five[-2] if len(last_five) > 1 else 0,
        ]
    ])

# Sample training data
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1])

model = LinearRegression()
model.fit(X_sample, y_sample)

if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)
        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-5:]):.2f}")
        st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")

# Train model with feedback
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
            st.success("Model trained with your input (session only)")
    except:
        st.error("Feedback must be a number")

# Plot chart
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o', label='Recent')
    ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)
