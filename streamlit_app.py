import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Crash Predictor App", layout="centered")
st.title("🚀 Crash Predictor App")

# File upload
uploaded_file = st.file_uploader("📂 Upload CSV file (e.g., 1.23x format)", type=["csv"])

def extract_features(values):
    if len(values) < 10:
        return None
    last = values[-10:]
    return np.array([[
        np.mean(last),
        np.std(last),
        max(last),
        min(last),
        last[-1] - last[-2],
        np.median(last),
        last[-1],
        last[-2],
        sum(last[-3:]) / 3,
        last[0]
    ]])

@st.cache_data
def preprocess_csv(file):
    df = pd.read_csv(file)
    values = df[df.columns[0]].astype(str).str.replace("x", "").astype(float)
    values = values[values < 100]  # Remove outliers
    return values.tolist()

def prepare_training_set(series):
    X, y = [], []
    for i in range(10, len(series)):
        feat = extract_features(series[i-10:i])
        if feat is not None:
            X.append(feat[0])
            y.append(series[i])
    return np.array(X), np.array(y)

model = None
X_sample, y_sample = None, None

if uploaded_file:
    crash_values = preprocess_csv(uploaded_file)
    X_sample, y_sample = prepare_training_set(crash_values)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    st.success("✅ Model trained from uploaded file!")

    st.subheader("🎯 Input & Feedback")
    input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
    user_feedback = st.text_input("Actual next multiplier (optional, for training)")
    submit_train = st.button("Train with Feedback")

    parsed = []
    if input_text:
        try:
            parsed = [float(x.replace("x", "").strip()) for x in input_text.split(",") if x.strip()]
        except:
            st.error("Invalid input format.")

    if parsed:
        features = extract_features(parsed)
        if features is not None:
            pred = model.predict(features)[0]
            safe_target = round(pred * 0.97, 2)
            st.subheader(f"📈 Predicted next crash: {pred:.2f}")
            st.success(f"🎯 Safe target multiplier (3% edge): {safe_target:.2f}")

    if submit_train and user_feedback:
        try:
            feedback_val = float(user_feedback)
            new_feat = extract_features(parsed)
            if new_feat is not None:
                X_sample = np.vstack([X_sample, new_feat])
                y_sample = np.append(y_sample, feedback_val)
                model.fit(X_sample, y_sample)
                st.success("✅ Model updated with your feedback.")
        except:
            st.error("Feedback must be a number.")

    if parsed and len(parsed) >= 10:
        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(parsed[-10:]):.2f}")
        st.text(f"Std Dev: {np.std(parsed[-10:]):.2f}")
        st.text(f"Last Change: {parsed[-1] - parsed[-2]:.2f}")

    if parsed:
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(parsed[-20:], marker='o', label='Recent')
        ax.axhline(np.mean(parsed[-10:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

    if len(X_sample) >= 30:
        st.subheader("📋 Predicted vs Actual (Last 30)")
        preds = model.predict(X_sample[-30:])
        comp_df = pd.DataFrame({
            "Predicted": preds.round(2),
            "Actual": y_sample[-30:].round(2),
            "Error": np.abs(preds - y_sample[-30:]).round(2)
        })
        st.dataframe(comp_df)

        st.subheader("📈 Accuracy Trend")
        fig2, ax2 = plt.subplots()
        ax2.plot(comp_df["Error"], label="Absolute Error", marker='o')
        ax2.set_ylabel("Prediction Error")
        ax2.set_xlabel("Last 30 Predictions")
        ax2.axhline(np.mean(comp_df["Error"]), color='red', linestyle='--', label='Mean Error')
        ax2.legend()
        st.pyplot(fig2)
else:
    st.info("Please upload a CSV file to begin training.")
