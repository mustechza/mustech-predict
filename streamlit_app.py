import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Crash Predictor App 🚀")

# Hardcoded initial training data
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1],
    [2.3, 0.4, 2.2, 2.6, 1.9, -0.1],
    [3.0, 0.6, 3.5, 3.8, 2.7, 0.5]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1, 2.2, 3.4])

# Input section
with st.form("prediction_form"):
    input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
    user_feedback = st.text_input("Actual next multiplier (optional, for training):")
    submitted = st.form_submit_button("Train with Feedback & Predict")

# Parse crash input
def parse_input(text):
    try:
        raw = [float(x.strip().replace("x", "")) for x in text.split(",") if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 10:
        return None
    last = values[-10:]
    return np.array([[
        np.mean(last),
        np.std(last),
        last[-1],
        max(last),
        min(last),
        last[-1] - last[-2] if len(last) > 1 else 0
    ]])

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_sample, y_sample)

# On submit
crash_values = parse_input(input_text)
if submitted and crash_values:
    features = extract_features(crash_values)
    if features is not None:
        # Predict
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)

        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

        # Handle feedback
        try:
            if user_feedback:
                feedback = float(user_feedback.strip().replace("x", ""))
                if feedback > 10.99:
                    feedback = 10.5
                X_sample = np.vstack([X_sample, features])
                y_sample = np.append(y_sample, feedback)
                model.fit(X_sample, y_sample)
                st.success("✅ Model trained with your feedback!")
        except:
            st.error("⚠️ Feedback must be a number")

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

        # Compare last 30 predictions
        if len(y_sample) >= 5:
            last_X = X_sample[-30:]
            last_y = y_sample[-30:]
            predicted_y = model.predict(last_X)
            df_compare = pd.DataFrame({
                "Predicted": predicted_y.round(2),
                "Actual": last_y.round(2),
                "Abs Error": np.abs(predicted_y - last_y).round(2)
            })
            st.subheader("🧾 Predicted vs Actual (Last 30)")
            st.dataframe(df_compare)

            # Live Accuracy Chart
            st.subheader("📈 Accuracy Trend (Last 30)")
            fig2, ax2 = plt.subplots()
            ax2.plot(100 - np.abs((predicted_y - last_y) / last_y * 100), label='Accuracy (%)', color='green')
            ax2.set_ylim(0, 110)
            ax2.set_ylabel("Accuracy %")
            ax2.set_xlabel("Prediction #")
            ax2.axhline(97, linestyle="--", color="red", label="Target")
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.warning("Please enter at least 10 crash multipliers for prediction.")
