import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import os

st.set_page_config(layout="wide")

CSV_URL = "https://raw.githubusercontent.com/mustechza/mustech-predict/main/training_data_mock.csv"
PERSISTENCE_FILE = "crash_training_data.csv"
WIN_LOG_FILE = "win_loss_log.csv"

st.title("🚀 Crash Predictor App")

# Load and persist training data
def load_persistent_data():
    if os.path.exists(PERSISTENCE_FILE):
        df = pd.read_csv(PERSISTENCE_FILE)
    else:
        df = pd.read_csv(CSV_URL)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'target' not in df.columns:
        return pd.DataFrame(), pd.Series()
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda x: min(x, 10.5))
    return X, y

def save_persistent_data(X, y):
    df = X.copy()
    df['target'] = y
    df.to_csv(PERSISTENCE_FILE, index=False)

# Win/loss logging
def load_win_log():
    if os.path.exists(WIN_LOG_FILE):
        return pd.read_csv(WIN_LOG_FILE).tail(20)
    return pd.DataFrame(columns=['Predicted', 'Actual', 'Outcome'])

def save_to_win_log(pred, actual):
    outcome = 'Win' if actual >= pred * 0.97 else 'Loss'
    log = load_win_log()
    log = pd.concat([log, pd.DataFrame([{
        'Predicted': round(pred, 2),
        'Actual': round(actual, 2),
        'Outcome': outcome
    }])]).tail(20)
    log.to_csv(WIN_LOG_FILE, index=False)
    return log

# Feature & input handling
def extract_features(values):
    if len(values) < 10:
        return None
    last_vals = values[-10:]
    return np.array([[np.mean(last_vals), np.std(last_vals), last_vals[-1],
                      max(last_vals), min(last_vals),
                      last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0]])

def parse_input(text):
    try:
        raw = [float(x.strip().lower().replace('x', '')) for x in text.split(',') if x.strip()]
        capped = [min(x, 10.5) if x > 10.99 else x for x in raw]
        return capped
    except:
        return []

# Load and train
X_train, y_train = load_persistent_data()
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# ==================
# 📥 Input Section
# ==================
st.header("📥 Input & Feedback")
with st.form("input_form"):
    crash_text = st.text_input("Enter recent crash multipliers (comma-separated)")
    next_multiplier = st.text_input("Next actual multiplier (for feedback)")
    submitted = st.form_submit_button("🔁 Submit")

crash_values = parse_input(crash_text)
features = extract_features(crash_values) if crash_values else None

# ==================
# 🔮 Prediction
# ==================
if features is not None:
    prediction = model.predict(features)[0]
    safe_target = round(prediction * 0.97, 2)
    st.subheader(f"🎯 Predicted next crash: {prediction:.2f}")
    st.success(f"🛡️ Safe multiplier target (3% edge): {safe_target:.2f}")

    # Alerts
    if prediction >= 8.0:
        st.warning("⚠️ Very high prediction — rare event!")
    elif prediction <= 1.5:
        st.warning("🔻 Very low prediction — consider avoiding!")

# ==================
# 📥 Feedback + Retrain
# ==================
if submitted and features is not None:
    try:
        feedback = float(next_multiplier)
        feedback = min(feedback, 10.5)
        X_train = pd.concat([X_train, pd.DataFrame(features, columns=X_train.columns)], ignore_index=True)
        y_train = pd.concat([y_train, pd.Series([feedback])], ignore_index=True)
        save_persistent_data(X_train, y_train)
        model.fit(X_train, y_train)
        save_to_win_log(prediction, feedback)
        st.success("Model retrained and feedback logged!")
    except:
        st.error("Invalid feedback value.")

# ==================
# 📊 Indicators
# ==================
if crash_values:
    st.subheader("📊 Crash History (Last 10)")
    fig, ax = plt.subplots()
    ax.plot(crash_values[-10:], marker='o')
    ax.axhline(np.mean(crash_values[-10:]), color='r', linestyle='--', label='Mean')
    ax.legend()
    st.pyplot(fig)

# ==================
# 📈 Accuracy Trend
# ==================
if len(X_train) >= 30:
    st.subheader("📈 Recent Prediction Accuracy (Last 30)")
    sample_X = X_train.tail(30)
    sample_y = y_train.tail(30)
    preds = model.predict(sample_X)
    df_accuracy = pd.DataFrame({
        "Predicted": preds.round(2),
        "Actual": sample_y.round(2),
        "Error": np.abs(preds - sample_y).round(2)
    })
    st.dataframe(df_accuracy)

    st.subheader("📊 Accuracy Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(np.abs(preds - sample_y), label='Absolute Error', marker='o')
    ax2.set_ylabel("Error")
    ax2.set_title("Prediction Error Over Time")
    ax2.legend()
    st.pyplot(fig2)

# ==================
# ✅ Win/Loss Table
# ==================
st.subheader("🧾 Last 20 Wins/Losses")
log = load_win_log()
if not log.empty:
    def color_row(row):
        if row['Outcome'] == 'Win':
            return ['background-color: #d4edda']*3
        else:
            return ['background-color: #f8d7da']*3

    styled_log = log.style.apply(color_row, axis=1)
    st.dataframe(styled_log)

    # Summary
    win_count = (log['Outcome'] == 'Win').sum()
    loss_count = (log['Outcome'] == 'Loss').sum()
    total = len(log)
    win_rate = (win_count / total) * 100 if total > 0 else 0
    st.markdown(f"✅ **Wins:** {win_count} &nbsp;&nbsp;&nbsp; ❌ **Losses:** {loss_count} &nbsp;&nbsp;&nbsp; 🎯 **Win Rate:** {win_rate:.2f}%")
