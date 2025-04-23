import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Crash Predictor App 🚀")

# JSON storage path
DATA_FILE = "training_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
    else:
        return (
            np.array([
                [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
                [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
                [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
                [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
            ]),
            np.array([2.5, 1.0, 4.0, 1.1])
        )

def save_data(features, labels):
    data = {
        "features": features.tolist(),
        "labels": labels.tolist()
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def parse_input(text):
    try:
        raw = [float(x.strip()) for x in text.split(",") if x.strip()]
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
        last_five[-1] - last_five[-2] if len(last_five) > 1 else 0,
    ]])

# Load data
X_sample, y_sample = load_data()

# Train model
model = LinearRegression()
model.fit(X_sample, y_sample)

# Inputs
input_text = st.text_input("Enter recent crash multipliers (comma-separated)")
user_feedback = st.text_input("Actual next multiplier (optional, for training)")

crash_values = parse_input(input_text)

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

# Feedback training
if st.button("Train with Feedback"):
    try:
        feedback = float(user_feedback.strip())
        if feedback > 10.99:
            feedback = 10.5
        new_features = extract_features(crash_values)
        if new_features is not None:
            X_sample = np.vstack([X_sample, new_features])
            y_sample = np.append(y_sample, feedback)
            save_data(X_sample, y_sample)
            model.fit(X_sample, y_sample)
            st.success("Model trained with your input and saved!")
    except:
        st.error("Feedback must be a number")

# Chart
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o', label='Recent')
    ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# Comparison Table
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

    # Accuracy Trend Chart
    abs_errors = np.abs(predicted_y - last_y)
    normalized_accuracy = 1 - (abs_errors / last_y.clip(min=0.1))
    accuracy_percent = (normalized_accuracy * 100).clip(min=0, max=100)

    st.subheader("📈 Accuracy Trend (Last 30 Predictions)")
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(accuracy_percent, marker='o', linestyle='-', color='green')
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Prediction Index")
    ax_acc.set_title("Prediction Accuracy Over Time")
    ax_acc.set_ylim(0, 100)
    st.pyplot(fig_acc)
