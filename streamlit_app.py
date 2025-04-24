import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

CSV_FILE = "training_data.csv"

st.set_page_config(page_title="Crash Predictor", layout="wide")
st.title("🚀 Crash Predictor App with Live Feedback & Alerts")

# ==================
# 🔁 Data Persistence
# ==================
def load_or_init_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=[
            'mean', 'std', 'last', 'max', 'min', 'last_change', 'target'
        ])
    return df

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

data_df = load_or_init_data()

# ==================
# 🔁 Model Training
# ==================
def train_model(df):
    X = df.drop(columns=['target'])
    y = df['target']
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

model = train_model(data_df) if not data_df.empty else None

# ==================
# 📥 Input & Feedback
# ==================
st.header("📥 Input & Feedback")
with st.form("input_form"):
    user_input = st.text_input("Enter recent crash multipliers (comma-separated)")
    next_actual = st.text_input("Enter the next actual crash multiplier (for training)", "")
    submitted = st.form_submit_button("Submit")

# ==================
# 📊 Feature Extraction
# ==================
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
    return pd.DataFrame([[
        np.mean(last_vals),
        np.std(last_vals),
        last_vals[-1],
        max(last_vals),
        min(last_vals),
        last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0,
    ]], columns=['mean', 'std', 'last', 'max', 'min', 'last_change'])

crash_values = parse_input(user_input)
features = extract_features(crash_values) if crash_values else None

# ==================
# 🔮 Prediction
# ==================
if features is not None and model:
    prediction = model.predict(features)[0]
    safe_target = round(prediction * 0.97, 2)
    st.subheader(f"🎯 Predicted next crash: {prediction:.2f}")
    st.success(f"🛡️ Safe multiplier target (3% edge): {safe_target:.2f}")

    # ✅ Threshold-based Alerts
    if prediction < 1.5:
        st.error("⚠️ High risk: Next multiplier might crash early!")
    elif prediction > 4.0:
        st.success("🚀 Potential high multiplier! Consider a larger multiplier target.")

    if np.std(crash_values[-10:]) > 2.0:
        st.warning("⚠️ High volatility detected. Recent crash values are unpredictable.")

# ==================
# 📉 Crash History Chart
# ==================
if crash_values:
    st.header("📉 Crash History (Last 10)")
    st.line_chart(pd.Series(crash_values[-10:], name="Crash Value"))

# ==================
# 🧠 Feedback Training
# ==================
if submitted and features is not None and next_actual:
    try:
        actual = min(float(next_actual), 10.5)
        features['target'] = actual
        data_df = pd.concat([data_df, features], ignore_index=True)
        save_data(data_df)
        model = train_model(data_df)
        st.success("✅ Model retrained with new feedback.")
    except:
        st.error("Invalid actual crash multiplier value.")

# ==================
# 📈 Accuracy Table
# ==================
st.header("📈 Model Accuracy (Last 30)")
if len(data_df) >= 30:
    recent = data_df.tail(30)
    model_eval = train_model(data_df)
    preds = model_eval.predict(recent.drop(columns=['target']))
    actuals = recent['target']
    errors = np.abs(preds - actuals)

    acc_df = pd.DataFrame({
        "Predicted": preds.round(2),
        "Actual": actuals.round(2),
        "Error": errors.round(2)
    })

    st.dataframe(acc_df)

    st.subheader("📊 Accuracy Trend")
    fig, ax = plt.subplots()
    ax.plot(errors, label='Absolute Error', marker='o')
    ax.set_title("Prediction Error (Last 30)")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)

    if errors.mean() < 0.2:
        st.info("✅ Model confidence is high. Predictions are very accurate recently.")
else:
    st.warning("Not enough data for accuracy analysis (need 30+ samples).")
