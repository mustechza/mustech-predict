import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.title("Crash Predictor App 🚀")

# Helper to extract features
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
        last_ten[-1] - last_ten[-2] if len(last_ten) > 1 else 0,
        sum(np.diff(last_ten) > 0),
        sum(np.diff(last_ten) < 0)
    ]])

# Robust CSV parser with capping
@st.cache_data
def preprocess_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    raw_values = df[df.columns[0]].astype(str).str.replace("x", "", regex=False)
    cleaned_values = pd.to_numeric(raw_values, errors="coerce").dropna()
    capped_values = cleaned_values.apply(lambda x: min(x, 10.5) if x > 10.99 else x)
    return capped_values.tolist()

# File uploader
uploaded_file = st.file_uploader("📂 Upload crash data CSV", type=["csv"])
crash_values = []

if uploaded_file:
    crash_values = preprocess_csv(uploaded_file)
    st.success("Data loaded successfully!")

# Show preview of uploaded data
if crash_values:
    st.write("🔢 Preview of parsed values:", crash_values[-10:])

# Training logic
model = GradientBoostingRegressor()
X_data = []
y_data = []

if len(crash_values) > 10:
    for i in range(10, len(crash_values)):
        features = extract_features(crash_values[:i])
        if features is not None:
            X_data.append(features[0])
            y_data.append(crash_values[i])

if X_data and y_data:
    model.fit(np.array(X_data), np.array(y_data))
    last_features = extract_features(crash_values)
    if last_features is not None:
        prediction = model.predict(last_features)[0]
        safe_target = round(prediction * 0.97, 2)

        st.subheader(f"📈 Predicted Next Multiplier: `{prediction:.2f}`")
        st.success(f"🎯 Safe Target (3% margin): `{safe_target:.2f}`")

        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-10:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
        st.text(f"Change: {(crash_values[-1] - crash_values[-2]):.2f}")

        # Chart
        st.subheader("📉 Recent Multiplier Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-30:], marker='o', label='Crash History')
        ax.axhline(np.mean(crash_values[-10:]), color='red', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

        # Comparison
        if len(y_data) >= 5:
            predicted_vals = model.predict(np.array(X_data[-30:]))
            comparison_df = pd.DataFrame({
                "Predicted": predicted_vals.round(2),
                "Actual": np.array(y_data[-30:]).round(2),
                "Error": np.abs(predicted_vals - y_data[-30:]).round(2)
            })
            st.subheader("🧾 Predicted vs Actual (Last 30)")
            st.dataframe(comparison_df)
