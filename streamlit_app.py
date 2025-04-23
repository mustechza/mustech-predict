import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# GitHub raw CSV URL
CSV_URL = "https://raw.githubusercontent.com/mustechza/mustech-predict/main/training_data_mock.csv"

st.title("🚀 Crash Predictor App")

# ==================
# 📊 Load and Train
# ==================
@st.cache_data
def load_training_data(url):
    df = pd.read_csv(url)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'target' not in df.columns:
        return None, None
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda x: min(x, 10.5))
    return X, y

X_train, y_train = load_training_data(CSV_URL)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# State management
if "stage" not in st.session_state:
    st.session_state.stage = "input_recent"
if "recent_values" not in st.session_state:
    st.session_state.recent_values = []
if "features" not in st.session_state:
    st.session_state.features = None

st.header("🧠 Prediction & Feedback")
input_label = "Enter recent crash multipliers (comma-separated):" if st.session_state.stage == "input_recent" else "Enter actual next multiplier:"
user_input = st.text_input(input_label)
submit = st.button("Submit")

# Utils
def parse_input(text):
    try:
        raw = [float(x.strip().lower().replace('x', '')) for x in text.split(',') if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

def extract_features(values):
    if len(values) < 10:
        return None
    last = values[-10:]
    return np.array([[np.mean(last), np.std(last), last[-1], max(last), min(last), last[-1] - last[-2]]])

# Logic
if submit:
    if st.session_state.stage == "input_recent":
        values = parse_input(user_input)
        if len(values) >= 10:
            st.session_state.recent_values = values
            st.session_state.features = extract_features(values)
            prediction = model.predict(st.session_state.features)[0]
            safe_target = round(prediction * 0.97, 2)
            st.success(f"🎯 Predicted next crash: {prediction:.2f}")
            st.info(f"🛡️ Safe multiplier target: {safe_target:.2f}")
            st.session_state.stage = "input_feedback"
        else:
            st.warning("Please enter at least 10 valid multipliers.")
    else:
        try:
            actual = float(user_input)
            actual = min(actual, 10.5)
            X_train.loc[len(X_train)] = st.session_state.features.flatten()
            y_train.loc[len(y_train)] = actual
            model.fit(X_train, y_train)
            st.success("✅ Model retrained with feedback!")
            st.session_state.stage = "input_recent"
        except:
            st.error("Invalid input for actual multiplier. Please enter a number.")

# Visual indicators
if st.session_state.recent_values:
    st.subheader("📉 Crash History (Last 10)")
    vals = st.session_state.recent_values[-10:]
    fig, ax = plt.subplots()
    ax.plot(vals, marker='o')
    ax.axhline(np.mean(vals), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# Accuracy Trend
if len(X_train) >= 30:
    st.subheader("📈 Accuracy Trend")
    preds = model.predict(X_train.tail(30))
    errors = np.abs(preds - y_train.tail(30))
    fig2, ax2 = plt.subplots()
    ax2.plot(errors, label='Absolute Error', marker='o')
    ax2.set_title("Prediction Error (Last 30)")
    ax2.legend()
    st.pyplot(fig2)
