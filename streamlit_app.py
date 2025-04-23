import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# File paths
MODEL_FILE = "model.pkl"
DATA_FILE = "data.csv"

# Load or initialize model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = LinearRegression()

# Load or initialize data
if os.path.exists(DATA_FILE):
    data_df = pd.read_csv(DATA_FILE)
else:
    data_df = pd.DataFrame(columns=[
        'mean', 'std', 'last', 'max', 'min', 'change', 'target'
    ])

# App UI
st.title("Crash Predictor App with Memory")

# Input section
input_text = st.text_input("Enter recent crash values (comma-separated)")
user_feedback = st.text_input("Enter actual multiplier result (optional, e.g. 2.45)")
predict_button = st.button("Predict")
train_button = st.button("Train Model")

# Convert input
def parse_input(text):
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except:
        return []

crash_values = parse_input(input_text)

# Feature extraction
def extract_features(values):
    if len(values) < 5:
        return None
    last_five = values[-5:]
    return {
        'mean': np.mean(last_five),
        'std': np.std(last_five),
        'last': last_five[-1],
        'max': max(last_five),
        'min': min(last_five),
        'change': last_five[-1] - last_five[-2] if len(last_five) > 1 else 0
    }

# Make prediction
if predict_button and crash_values:
    features = extract_features(crash_values)
    if features:
        feature_df = pd.DataFrame([features])
        prediction = model.predict(feature_df)[0]
        target_with_margin = prediction * 0.97  # Adjusted for 3% house edge

        st.subheader("📊 Prediction")
        st.success(f"Predicted Multiplier: {prediction:.2f}")
        st.info(f"Safe Target (adjusted for 3% house edge): {target_with_margin:.2f}")

        # Display chart at the bottom
        st.subheader("📈 Crash Chart")
        fig, ax = plt.subplots()
        ax.plot(crash_values[-10:], marker='o', label='Recent Values')
        ax.axhline(np.mean(crash_values[-5:]), color='r', linestyle='--', label='Mean')
        ax.legend()
        st.pyplot(fig)

# Train model
if train_button:
    if user_feedback and crash_values:
        try:
            feedback_val = float(user_feedback)
            features = extract_features(crash_values)
            if features:
                features['target'] = feedback_val
                data_df = pd.concat([data_df, pd.DataFrame([features])], ignore_index=True)

                # Save data
                data_df.to_csv(DATA_FILE, index=False)

                # Train model
                X = data_df.drop(columns=['target'])
                y = data_df['target']
                model.fit(X, y)

                # Save model
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)

                st.success("Model trained and saved successfully!")
        except ValueError:
            st.error("Feedback must be a numeric value.")
    else:
        st.warning("Provide crash values and actual result to train the model.")

# Reminder
st.markdown("---")
st.caption("📌 This app stores training data and the model in local files (`data.csv` and `model.pkl`).")
