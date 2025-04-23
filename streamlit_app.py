import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

st.title("🚀 Crash Predictor App with CSV Persistence")

CSV_FILE = "training_data.csv"

# ============================
# 📥 Load or Create Training Data
# ============================
def load_training_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        X = df.drop(columns=['target'])
        y = df['target']
        return X, y
    else:
        return pd.DataFrame(columns=['mean', 'std', 'last', 'max', 'min', 'delta']), pd.Series(dtype='float64')

def save_training_data(X, y):
    df = X.copy()
    df['target'] = y
    df.to_csv(CSV_FILE, index=False)

X_train, y_train = load_training_data()

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
if not X_train.empty:
    model.fit(X_train, y_train)

# ============================
# 📥 Input Section
# ============================
st.header("📥 Input & Feedback")
with st.form("input_form"):
    user_input = st.text_input("Enter crash multipliers (comma-separated):")
    feedback_value = st.text_input("Actual next multiplier (optional, for training):")
    submitted = st.form_submit_button("🔁 Submit")

def parse_input(text):
    try:
        values = [float(x.strip().lower().replace('x', '')) for x in text.split(',') if x.strip()]
        values = [min(x, 10.5) if x > 10.99 else x for x in values]
        return values
    except:
        return []

def extract_features(values):
    if len(values) < 10:
        return None
    last_vals = values[-10:]
    return np.array([[np.mean(last_vals),
                      np.std(last_vals),
                      last_vals[-1],
                      max(last_vals),
                      min(last_vals),
                      last_vals[-1] - last_vals[-2] if len(last_vals) > 1 else 0]])

parsed = parse_input(user_input)
features = extract_features(parsed) if parsed else None

# ============================
# 🔮 Prediction
# ============================
if features is not None and not X_train.empty:
    prediction = model.predict(features)[0]
    safe_target = round(prediction * 0.97, 2)
    st.subheader(f"🎯 Predicted crash: {prediction:.2f}")
    st.success(f"🛡️ Safe target: {safe_target:.2f}")

# ============================
# 🔁 Feedback Training
# ============================
if submitted and features is not None:
    try:
        feedback = float(feedback_value)
        feedback = min(feedback, 10.5)
        new_X = pd.DataFrame(features, columns=X_train.columns if not X_train.empty else ['mean', 'std', 'last', 'max', 'min', 'delta'])
        X_train = pd.concat([X_train, new_X], ignore_index=True)
        y_train = pd.concat([y_train, pd.Series([feedback])], ignore_index=True)
        model.fit(X_train, y_train)
        save_training_data(X_train, y_train)
        st.success("Model retrained and data saved.")
    except:
        st.error("Please enter a valid feedback number.")

# ============================
# 📈 Accuracy Trend
# ============================
if len(X_train) >= 30:
    st.header("📈 Recent Accuracy (Last 30)")
    recent_X = X_train.tail(30)
    recent_y = y_train.tail(30)
    preds = model.predict(recent_X)
    errors = np.abs(preds - recent_y)

    df = pd.DataFrame({
        'Predicted': preds.round(2),
        'Actual': recent_y.round(2),
        'Error': errors.round(2)
    })

    st.dataframe(df)

    st.subheader("📉 Accuracy Trend")
    fig, ax = plt.subplots()
    ax.plot(errors, marker='o', label='Absolute Error')
    ax.set_title("Prediction Error (Last 30)")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Not enough data to show accuracy trend.")
