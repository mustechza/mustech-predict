import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crash Predictor", layout="centered")
st.title("🚀 Crash Predictor App")

# Helper: Clean + cap values
def clean_and_cap_values(series):
    return series.astype(str).str.replace("x", "").astype(float).apply(lambda x: min(x, 10.5))

# Helper: Feature extraction
def extract_features(values):
    features = []
    labels = []
    for i in range(6, len(values)):
        window = values[i-5:i+1]
        X = [
            np.mean(window[:-1]),
            np.std(window[:-1]),
            window[-2],
            max(window[:-1]),
            min(window[:-1]),
            window[-2] - window[-3]
        ]
        features.append(X)
        labels.append(window[-1])
    return np.array(features), np.array(labels)

# Load CSV from GitHub raw URL
st.subheader("📂 Load Crash History CSV from URL")
csv_url = "https://raw.githubusercontent.com/mustechza/mustech-predict/main/training_data_mock.csv"

try:
    df = pd.read_csv(csv_url)
    crash_values = clean_and_cap_values(df.iloc[:, 0])
    
    # Train model live from CSV
    X, y = extract_features(crash_values)
    model = GradientBoostingRegressor()
    model.fit(X, y)

    # Predict next value from most recent 5
    if len(crash_values) >= 6:
        recent = crash_values[-6:]
        features = np.array([[
            np.mean(recent[:-1]),
            np.std(recent[:-1]),
            recent[-2],
            max(recent[:-1]),
            min(recent[:-1]),
            recent[-2] - recent[-3],
        ]])
        prediction = model.predict(features)[0]
        st.subheader(f"📈 Predicted next crash: {prediction:.2f}")
        st.success(f"🎯 Safe target (3% edge): {round(prediction * 0.97, 2)}")

        st.subheader("📊 Indicators")
        st.text(f"Mean of last 5: {np.mean(recent[:-1]):.2f}")
        st.text(f"Std Dev of last 5: {np.std(recent[:-1]):.2f}")
        st.text(f"Change: {(recent[-2] - recent[-3]):.2f}")

        # Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-30:], marker='o', label='Recent')
        ax.axhline(np.mean(recent[:-1]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

        # Accuracy Trend
        st.subheader("📊 Accuracy (Last 30 Predictions)")
        pred_last = model.predict(X[-30:])
        actual_last = y[-30:]
        df_accuracy = pd.DataFrame({
            "Predicted": pred_last,
            "Actual": actual_last,
            "Abs Error": np.abs(pred_last - actual_last)
        })
        st.dataframe(df_accuracy.style.format({"Predicted": "{:.2f}", "Actual": "{:.2f}", "Abs Error": "{:.2f}"}))
        st.line_chart(df_accuracy["Abs Error"])

        avg_mae = mean_absolute_error(actual_last, pred_last)
        st.info(f"📏 Mean Absolute Error: {avg_mae:.3f}")
    else:
        st.warning("Need at least 6 crash values to make a prediction.")
except Exception as e:
    st.error(f"Error loading or processing file: {e}")

