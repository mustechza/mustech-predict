# crash_predictor_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# App title
st.title("Crash Predictor App (Multiplier Prediction)")

# User input for crash values
input_text = st.text_input("Enter recent crash values (comma-separated)")
user_feedback = st.text_input("Enter correct multiplier (optional, e.g. 2.35)")

# Parse input into float list
def parse_input(text):
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except:
        return []

crash_values = parse_input(input_text)

# Feature extraction for regression
def extract_features(values):
    if len(values) < 5:
        return None
    last_five = values[-5:]
    features = [
        np.mean(last_five),
        np.std(last_five),
        last_five[-1],
        max(last_five),
        min(last_five),
        (last_five[-1] - last_five[-2]) if len(last_five) > 1 else 0
    ]
    return np.array(features).reshape(1, -1)

# Sample training data (can be appended with user feedback)
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
])
y_sample = np.array([2.6, 1.1, 3.9, 1.0])  # Regression targets (multipliers)

model = LinearRegression()
model.fit(X_sample, y_sample)

if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        st.subheader(f"Predicted Next Multiplier: **{prediction:.2f}x**")

        # Show indicators
        st.subheader("Indicators")
        st.text(f"Mean: {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-5:]):.2f}")
        st.text(f"Change: {(crash_values[-1] - crash_values[-2]):.2f}" if len(crash_values) > 1 else "Change: N/A")

        # Model training based on user input
        if st.button("Train Model"):
            if user_feedback:
                try:
                    correct_output = float(user_feedback.strip())
                    new_features = extract_features(crash_values)
                    if new_features is not None:
                        X_sample = np.vstack([X_sample, new_features])
                        y_sample = np.append(y_sample, correct_output)
                        model.fit(X_sample, y_sample)
                        st.success("Model updated with your input!")
                except ValueError:
                    st.error("Feedback must be a decimal number (e.g. 2.5)")
            else:
                st.warning("Enter feedback before training.")

        # Now draw the chart at the end
        st.subheader("Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent Values')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)
