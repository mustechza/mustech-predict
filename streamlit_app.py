import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from io import StringIO

st.set_page_config(page_title="Crash Predictor", layout="centered")
st.title("🚀 Crash Predictor App")

# --- Helper Functions ---

@st.cache_data
def load_csv(file):
    try:
        df = pd.read_csv(file)
        values = df[df.columns[0]].astype(str).str.replace("x", "").astype(float)
        values = values.apply(lambda x: min(x, 10.5))  # Cap >10.99 to 10.5
        return values.to_list()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def extract_features(values):
    features = []
    targets = []
    for i in range(10, len(values) - 1):
        window = values[i-10:i]
        label = values[i]
        features.append([
            np.mean(window),
            np.std(window),
            window[-1],
            max(window),
            min(window),
            window[-1] - window[-2]
        ])
        targets.append(label)
    return np.array(features), np.array(targets)

def train_models(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    gb.fit(X, y)
    return rf, gb

def predict_and_compare(model, features, name):
    pred = model.predict(features)[0]
    safe_target = round(pred * 0.97, 2)
    return pred, safe_target

# --- Upload CSV ---

st.header("📤 Upload Historical Crash Data")
uploaded_file = st.file_uploader("Upload CSV file with crash multipliers", type=["csv"])

if uploaded_file:
    crash_data = load_csv(uploaded_file)
    
    if len(crash_data) > 50:
        st.success("✅ File loaded and processed!")

        # Train models
        X, y = extract_features(crash_data)
        rf_model, gb_model = train_models(X, y)

        # Live Prediction
        st.header("📈 Live Prediction")
        features = extract_features(crash_data)[0][-1].reshape(1, -1)

        pred_rf, safe_rf = predict_and_compare(rf_model, features, "Random Forest")
        pred_gb, safe_gb = predict_and_compare(gb_model, features, "Gradient Boosting")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌲 Random Forest")
            st.text(f"Predicted: {pred_rf:.2f}")
            st.success(f"Safe Target: {safe_rf:.2f}")
        with col2:
            st.subheader("🔥 Gradient Boosting")
            st.text(f"Predicted: {pred_gb:.2f}")
            st.success(f"Safe Target: {safe_gb:.2f}")

        # Accuracy Comparison Table
        st.header("📊 Accuracy Comparison (Last 30)")
        last_X = X[-30:]
        last_y = y[-30:]
        pred_rf_all = rf_model.predict(last_X)
        pred_gb_all = gb_model.predict(last_X)

        df_compare = pd.DataFrame({
            "Actual": last_y.round(2),
            "Random Forest": pred_rf_all.round(2),
            "GB Boosting": pred_gb_all.round(2),
            "RF Error": np.abs(pred_rf_all - last_y).round(2),
            "GB Error": np.abs(pred_gb_all - last_y).round(2),
        })
        st.dataframe(df_compare)

        # Live Trend Chart
        st.header("📉 Accuracy Trend (Last 30)")
        fig, ax = plt.subplots()
        ax.plot(df_compare["Actual"], label="Actual", marker='o')
        ax.plot(df_compare["Random Forest"], label="RF Prediction", marker='x')
        ax.plot(df_compare["GB Boosting"], label="GB Prediction", marker='s')
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Please upload at least 50+ data points for training.")
else:
    st.info("Upload a CSV to get started.")
