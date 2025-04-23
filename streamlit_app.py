import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.title("Crash Predictor App 🚀")

# --- Preprocessing uploaded CSV ---
@st.cache_data
def preprocess_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ CSV Columns:", df.columns.tolist())  # Debug

        for col in df.columns:
            cleaned = df[col].astype(str).str.replace("x", "", regex=False)
            numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
            if not numeric.empty:
                st.write(f"📊 Using column: `{col}` with {len(numeric)} values.")
                capped = numeric.apply(lambda x: min(x, 10.5) if x > 10.99 else x)
                return capped.tolist()

        st.error("❌ No valid numeric column found in uploaded CSV.")
        return []

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return []

# --- Feature Extraction ---
def extract_features(values):
    if len(values) < 10:
        return None
    last_ten = values[-10:]
    return np.array([
        [
            np.mean(last_ten),
            np.std(last_ten),
            last_ten[-1],
            max(last_ten),
            min(last_ten),
            last_ten[-1] - last_ten[-2] if len(last_ten) > 1 else 0,
        ]
    ])

# --- Hardcoded Training Data ---
X_sample = np.array([
    [2.0, 0.5, 2.5, 2.8, 1.5, 0.5],
    [1.2, 0.3, 1.0, 1.5, 0.9, -0.2],
    [3.5, 0.7, 4.0, 4.5, 2.9, 1.0],
    [1.0, 0.1, 1.1, 1.3, 0.8, 0.1]
])
y_sample = np.array([2.5, 1.0, 4.0, 1.1])

# --- Upload Section ---
with st.expander("📂 Upload CSV to Train Model"):
    uploaded_file = st.file_uploader("Upload a CSV file containing crash multipliers (e.g. 1.05x, 2.0x)", type=["csv"])
    if uploaded_file:
        crash_values = preprocess_csv(uploaded_file)
    else:
        crash_values = []

# --- Model Training ---
model = GradientBoostingRegressor()
model.fit(X_sample, y_sample)

# --- Prediction Interface ---
with st.expander("🧠 Train / Predict"):
    input_text = st.text_input("Enter recent crash multipliers (comma-separated):")
    user_feedback = st.text_input("Actual next multiplier (optional, for training):")
    train_button = st.button("Train with Feedback")

    # Parse input manually
    if input_text:
        try:
            values = [float(x.strip().replace("x", "")) for x in input_text.split(",")]
            crash_values = [min(x, 10.5) if x > 10.99 else x for x in values]
        except:
            st.error("❌ Error parsing input values.")

    # Feedback training
    if train_button and crash_values:
        try:
            feedback = float(user_feedback.strip().replace("x", ""))
            feedback = min(feedback, 10.5)
            new_feat = extract_features(crash_values)
            if new_feat is not None:
                X_sample = np.vstack([X_sample, new_feat])
                y_sample = np.append(y_sample, feedback)
                model.fit(X_sample, y_sample)
                st.success("✅ Model trained with your feedback!")
        except:
            st.error("❌ Feedback must be a number")

# --- Prediction Output ---
if crash_values:
    features = extract_features(crash_values)
    if features is not None:
        prediction = model.predict(features)[0]
        safe_target = round(prediction * 0.97, 2)
        st.subheader(f"📈 Predicted next crash: `{prediction:.2f}`")
        st.success(f"🎯 Safe target multiplier (3% edge): `{safe_target:.2f}`")

        st.subheader("📊 Indicators")
        st.text(f"Mean: {np.mean(crash_values[-10:]):.2f}")
        st.text(f"Std Dev: {np.std(crash_values[-10:]):.2f}")
        st.text(f"Last Change: {(crash_values[-1] - crash_values[-2]):.2f}" if len(crash_values) > 1 else "N/A")

        # Chart
        st.subheader("📉 Recent Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-20:], marker='o', label='Recent Crashes')
        ax.axhline(np.mean(crash_values[-10:]), color='red', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

# --- Accuracy Comparison Table ---
if len(y_sample) >= 5:
    st.subheader("🧾 Predicted vs Actual (Last 30 Entries)")
    predicted_y = model.predict(X_sample[-30:])
    df_compare = pd.DataFrame({
        "Predicted": predicted_y.round(2),
        "Actual": y_sample[-30:].round(2),
        "Error": np.abs(predicted_y - y_sample[-30:]).round(2)
    })
    st.dataframe(df_compare)

    # Accuracy chart
    st.subheader("📈 Live Accuracy Trend (Last 30)")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_compare["Error"], marker='x', color='orange', label='Absolute Error')
    ax2.set_ylabel("Prediction Error")
    ax2.set_xlabel("Round")
    ax2.legend()
    st.pyplot(fig2)
