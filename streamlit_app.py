# crash_predictor_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

st.title("Crash Predictor App 🚀")

# File paths
DATA_FILE = "training_data.json"

# Load data
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
    else:
        # Initial training data
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
    with open(DATA_FILE, "w") as f:
        json.dump({"features": features.tolist(), "labels": labels.tolist()}, f)

def parse_input(text):
    try:
        raw = [float(x.strip().replace('x', '')) for x in text.split(",") if x.strip()]
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

# Load and train model
X_sample, y_sample = load_data()
model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=42)
model.fit(X_sample, y_sample)

# UI
input_text = st.text_input("Enter recent crash multipliers (e.g. 1.23, 2.15, 3.00)")
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

        # Accuracy of last prediction
        if user_feedback:
            try:
                actual = float(user_feedback.strip())
                last_accuracy = 100 - abs((actual - prediction) / actual) * 100
                st.info(f"🧮 Accuracy of last prediction: {last_accuracy:.2f}%")
            except:
                pass

        # Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

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
            st.success("Model trained and saved!")
    except:
        st.error("Feedback must be a number")

# Comparison Table
if len(y_sample) >= 5:
    last_X = X_sample[-30:]
    last_y = y_sample[-30:]
    predicted_y = model.predict(last_X)

    st.subheader("🧾 Predicted vs Actual (Last 30 Entries)")
    df_compare = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": last_y.round(2),
        "Error": np.abs(predicted_y - last_y).round(2)
    })
    st.dataframe(df_compare)

    # Accuracy trend chart
    accuracy_trend = 100 - np.abs((predicted_y - last_y) / last_y * 100)
    st.subheader("📈 Prediction Accuracy Trend (Last 30)")
    fig2, ax2 = plt.subplots()
    ax2.plot(accuracy_trend, marker='o', color='green')
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Recent Predictions")
    st.pyplot(fig2)
