import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.title("Crash Predictor App 🚀")

# Hardcoded training data (from previous sessions)
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1],
    [1.93, 0.74, 1.31, 3.04, 1.18, 0.22],
    [2.02, 0.87, 1.46, 3.55, 1.11, 0.15],
    [2.63, 0.97, 3.1, 3.74, 1.29, 1.64],
    [2.58, 0.76, 2.76, 3.32, 1.63, -0.34],
    [2.53, 0.75, 2.53, 3.36, 1.55, -0.23],
    [2.44, 0.72, 2.26, 3.3, 1.53, -0.27]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1, 1.31, 1.46, 3.1, 2.76, 2.53, 2.26])

# Model training
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_sample, y_sample)

# --- Input Section ---
input_text = st.text_input("Enter recent crash multipliers (comma-separated, e.g., 1.2, 2.1, 1.05x)")
user_feedback = st.text_input("Actual next multiplier (optional, for training)")

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

crash_values = parse_input(input_text)

# --- Prediction + Feedback ---
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

        # 🔼 Train with Feedback button (Moved up)
        if st.button("Train with Feedback"):
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

        # 📊 Indicators
        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-5:]):.2f}")
        st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}")

        # 📉 Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

        # 📋 Prediction vs Actual Table
        if len(y_sample) >= 5:
            last_X = X_sample[-30:]
            last_y = y_sample[-30:]
            predicted_y = model.predict(last_X)

            st.subheader("🧾 Predicted vs Actual (Last 30 Entries)")
            df_compare = pd.DataFrame({
                "Predicted": predicted_y.round(2),
                "Actual": last_y.round(2),
                "Abs Error": np.abs(predicted_y - last_y).round(2)
            })
            st.dataframe(df_compare)

        # 📈 Live accuracy chart
        if len(y_sample) > 1:
            errors = np.abs(model.predict(X_sample) - y_sample)
            accuracy = 100 * (1 - errors / y_sample)
            accuracy = np.clip(accuracy, 0, 100)
            st.subheader("📈 Live Prediction Accuracy Trend")
            fig2, ax2 = plt.subplots()
            ax2.plot(accuracy[-30:], marker='o', color='green')
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_xlabel("Recent Predictions")
            ax2.set_title("Model Accuracy Over Time")
            st.pyplot(fig2)
