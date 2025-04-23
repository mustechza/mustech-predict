import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# File to store data
DATA_FILE = "crash_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return data.get("X", []), data.get("y", [])
    return [], []

def save_data(X, y):
    with open(DATA_FILE, "w") as f:
        json.dump({"X": X, "y": y}, f)

def parse_input(text):
    try:
        return [min(float(x.strip()), 10.5) for x in text.split(",") if x.strip()]
    except:
        return []

def extract_features(values):
    if len(values) < 5:
        return None
    last = values[-5:]
    return [
        round(np.mean(last), 3),
        round(np.std(last), 3),
        round(last[-1], 3),
        round(max(last), 3),
        round(min(last), 3),
        round(last[-1] - last[-2], 3) if len(last) > 1 else 0
    ]

model = LinearRegression()
X_stored, y_stored = load_data()
X = np.array(X_stored)
y = np.array(y_stored)

if len(X) > 0:
    model.fit(X, y)

# Streamlit UI
st.title("Crash Predictor App")

input_text = st.text_input("Enter recent crash multipliers (comma-separated)")
user_feedback = st.text_input("Actual multiplier for last prediction (optional)")

crash_values = parse_input(input_text)

if crash_values:
    features = extract_features(crash_values)
    if features:
        features_array = np.array(features).reshape(1, -1)
        raw_prediction = model.predict(features_array)[0]

        st.subheader(f"Predicted Next Crash: {raw_prediction:.2f}x")

        if len(y) > 0:
            last_actual = y[-1]
            last_pred = model.predict([X[-1]])[0]
            error = abs(last_pred - last_actual)
            accuracy = 100 - (error / max(last_actual, 1) * 100)
            accuracy = max(0, min(accuracy, 100))
            st.markdown(f"**Prediction Accuracy (Last Round):** {accuracy:.2f}%")

        if st.button("Train Model"):
            if user_feedback:
                try:
                    actual_value = min(float(user_feedback), 10.5)
                    X = np.vstack([X, features_array])
                    y = np.append(y, actual_value)
                    save_data(X.tolist(), y.tolist())
                    model.fit(X, y)
                    st.success("Model trained with new data.")
                except ValueError:
                    st.error("Invalid multiplier. Please enter a number.")
            else:
                st.warning("Please enter actual multiplier before training.")

        st.subheader("Indicators")
        st.text(f"Mean (Last 5): {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev (Last 5): {np.std(crash_values[-5:]):.2f}")
        st.text(f"Change (Last 2): {crash_values[-1] - crash_values[-2]:.2f}")

        if len(y) > 1:
            st.subheader("Prediction vs Actual (Last 30)")
            compare_count = min(30, len(y))
            actuals = y[-compare_count:]
            preds = model.predict(X[-compare_count:])

            fig, ax = plt.subplots()
            ax.plot(range(compare_count), actuals, label="Actual", marker='o')
            ax.plot(range(compare_count), preds, label="Predicted", marker='x')
            ax.set_title("Prediction Accuracy")
            ax.legend()
            st.pyplot(fig)
