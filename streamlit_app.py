import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Crash Predictor App (Multiplier Regression)")

# --- Inputs ---
input_text = st.text_input("Enter recent crash multipliers (comma-separated)")
user_feedback = st.text_input("Enter actual multiplier (optional, e.g., 2.31)")

# Parse input
def parse_input(text):
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except:
        return []

crash_values = parse_input(input_text)

# --- Feature extraction ---
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

# --- Initialize model and training data ---
if "X_sample" not in st.session_state:
    st.session_state.X_sample = np.array([
        [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
        [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
        [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
        [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
    ])
    st.session_state.y_sample = np.array([2.5, 1.0, 4.0, 1.1])

model = LinearRegression()
model.fit(st.session_state.X_sample, st.session_state.y_sample)

# --- Make prediction ---
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        predicted_multiplier = model.predict(features)[0]
        st.subheader(f"📈 Predicted Multiplier: {predicted_multiplier:.2f}")

        # Plot
        st.subheader("Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent Values')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.axhline(predicted_multiplier, color='g', linestyle=':', label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Indicators
        st.subheader("Indicators")
        st.text(f"Mean: {np.mean(crash_values[-5:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-5:]):.2f}")
        st.text(f"Change: {(crash_values[-1] - crash_values[-2]):.2f}" if len(crash_values) > 1 else "Change: N/A")

# --- Train with feedback ---
if st.button("Train Model"):
    if user_feedback:
        try:
            actual_multiplier = float(user_feedback.strip())
            new_features = extract_features(crash_values)
            if new_features is not None:
                st.session_state.X_sample = np.vstack([st.session_state.X_sample, new_features])
                st.session_state.y_sample = np.append(st.session_state.y_sample, actual_multiplier)
                model.fit(st.session_state.X_sample, st.session_state.y_sample)
                st.success(f"Model updated with new data (Multiplier = {actual_multiplier})")
        except ValueError:
            st.error("⚠️ Please enter a valid number for feedback.")
    else:
        st.warning("Enter feedback before training.")
