import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Crash Predictor App 🚀")

# Hardcoded training data (previously in JSON)
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5, 1.5, 2.5, 3.5, 2.2],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2, 1.0, 1.1, 1.3, 1.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0, 3.2, 3.7, 4.2, 3.9],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1, 1.0, 1.1, 1.2, 1.0]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1])

model = RandomForestRegressor()
model.fit(X_sample, y_sample)

# Input Section (sync feedback + input)
st.subheader("🔧 Model Input & Training")
input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
user_feedback = st.text_input("Actual next multiplier (optional, for training):")

# Parse multipliers
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
        last_ten[-1] - last_ten[-2],
        *last_ten[-3:]
    ]])

crash_values = parse_input(input_text)

# Feedback Training + Prediction
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
                    global X_sample, y_sample
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

# Chart
if crash_values:
    st.subheader("📉 Recent Crash Chart")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o', label='Recent')
    ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# Accuracy Table
if len(y_sample) >= 5:
    last_X = X_sample[-30:]
    last_y = y_sample[-30:]
    predicted_y = model.predict(last_X)

    st.subheader("🧾 Predicted vs Actual (Last 30)")
    df_compare = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": last_y.round(2),
        "Error": np.abs(predicted_y - last_y).round(2)
    })
    st.dataframe(df_compare)
