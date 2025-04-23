import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Crash Predictor", layout="centered")
st.title("🎯 Crash Predictor App")

# JSON file for persistent storage
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

# Load and train
X_sample, y_sample = load_data()
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
        st.success(f"✅ Safe target (3% edge): {safe_target:.2f}")

        # Accuracy vs previous round
        if len(y_sample) >= 1:
            last_actual = y_sample[-1]
            last_pred = model.predict([X_sample[-1]])[0]
            acc = 100 - abs(last_pred - last_actual) / last_actual * 100
            st.write(f"🧮 Last Prediction Accuracy: **{acc:.2f}%**")

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
            st.success("✅ Model trained and saved with your input!")
    except:
        st.error("⚠️ Feedback must be a number")

# 📤 CSV Upload Section
st.subheader("📤 Upload CSV to Train")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Multiplier' column", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)
        if "Multiplier" not in df_csv.columns:
            st.error("CSV must contain a 'Multiplier' column.")
        else:
            multipliers = df_csv["Multiplier"].astype(float).tolist()
            multipliers = [min(x, 10.5) if x > 10.99 else x for x in multipliers]

            for i in range(5, len(multipliers)):
                segment = multipliers[i-5:i]
                features = extract_features(segment)
                label = multipliers[i]
                if features is not None:
                    X_sample = np.vstack([X_sample, features])
                    y_sample = np.append(y_sample, label)

            save_data(X_sample, y_sample)
            model.fit(X_sample, y_sample)
            st.success("📊 Model trained with uploaded CSV data!")
    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")

# 📉 Chart Section
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o', label='Recent')
    ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# 🧾 Predicted vs Actual
if len(y_sample) >= 5:
    st.subheader("📋 Prediction Accuracy Table (Last 30)")
    last_X = X_sample[-30:]
    last_y = y_sample[-30:]
    predicted_y = model.predict(last_X)
    df_compare = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": last_y.round(2),
        "Absolute Error": np.abs(predicted_y - last_y).round(2)
    })
    st.dataframe(df_compare)

    # 📈 Live accuracy trend chart
    st.subheader("📈 Prediction Accuracy Trend")
    accuracy_percent = 100 - np.abs(predicted_y - last_y) / last_y * 100
    fig2, ax2 = plt.subplots()
    ax2.plot(accuracy_percent, marker='o', color='green')
    ax2.set_title("Accuracy (%) Over Last 30 Predictions")
    ax2.set_ylabel("Accuracy %")
    ax2.set_xlabel("Round")
    st.pyplot(fig2)
