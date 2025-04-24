import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# GitHub raw CSV URL
CSV_URL = "https://raw.githubusercontent.com/mustechza/mustech-predict/main/training_data_mock.csv"

st.title("🚀 Crash Predictor App")

# ==================
# 📥 Load Training Data
# ==================
@st.cache_data
def load_training_data(url):
    df = pd.read_csv(url)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'target' not in df.columns:
        st.error("CSV must contain a 'target' column.")
        return None, None
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda x: min(x, 10.5))
    return X, y

X_train, y_train = load_training_data(CSV_URL)

# ==================
# 🤖 Initialize Models
# ==================
gb_model = GradientBoostingRegressor()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

gb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ==================
# 📥 User Input Form
# ==================
st.header("📥 Input & Feedback")
with st.form("input_form"):
    recent_input = st.text_input("Enter recent crash multipliers (comma-separated)")
    feedback_value = st.text_input("Actual next multiplier (optional, for training)")
    submitted = st.form_submit_button("🔁 Train with Feedback")

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
    last_vals = values[-10:]
    return np.array([[np.mean(last_vals), np.std(last_vals), last_vals[-1],
                      max(last_vals), min(last_vals),
                      last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0]])

crash_values = parse_input(recent_input)
features = extract_features(crash_values) if crash_values else None

# ==================
# 🔮 Prediction
# ==================
if features is not None:
    gb_pred = gb_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    st.subheader("🎯 Predicted next crash")
    st.text(f"Gradient Boosting: {gb_pred:.2f}")
    st.text(f"Random Forest: {rf_pred:.2f}")
    st.success(f"🛡️ Suggested Safe Target: {round(min(gb_pred, rf_pred) * 0.97, 2)}")

# ==================
# 📊 Indicators
# ==================
if crash_values:
    st.header("📊 Indicators")
    st.text(f"Mean (last 10): {np.mean(crash_values[-10:]):.2f}")
    st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
    st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}" if len(crash_values) > 1 else "N/A")

    st.subheader("📉 Crash History (Last 10)")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o')
    ax.axhline(np.mean(crash_values[-10:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# ==================
# 📥 Feedback Retraining
# ==================
if submitted and features is not None:
    try:
        feedback = float(feedback_value)
        feedback = min(feedback, 10.5)
        X_train = pd.concat([X_train, pd.DataFrame(features, columns=X_train.columns)], ignore_index=True)
        y_train = pd.concat([y_train, pd.Series([feedback])], ignore_index=True)
        gb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        st.success("Model retrained with feedback!")
    except:
        st.error("Invalid feedback value. Please enter a number.")

# ==================
# 📈 Accuracy Comparison
# ==================
st.header("📈 Model Performance Comparison (Last 30 Predictions)")
if len(X_train) >= 30:
    sample_X = X_train.tail(30)
    sample_y = y_train.tail(30)
    gb_preds = gb_model.predict(sample_X)
    rf_preds = rf_model.predict(sample_X)

    df_perf = pd.DataFrame({
        "GradientBoosting Error": np.abs(gb_preds - sample_y),
        "RandomForest Error": np.abs(rf_preds - sample_y),
    })

    st.dataframe(df_perf.round(2))

    fig3, ax3 = plt.subplots()
    ax3.plot(df_perf["GradientBoosting Error"], label="Gradient Boosting", marker='o')
    ax3.plot(df_perf["RandomForest Error"], label="Random Forest", marker='x')
    ax3.set_title("📊 Prediction Error Comparison")
    ax3.set_ylabel("Absolute Error")
    ax3.legend()
    st.pyplot(fig3)
else:
    st.warning("Not enough data to compare model performance.")
