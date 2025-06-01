import streamlit as st
import pandas as pd
import random
import datetime
import time
from collections import Counter

# Title
st.set_page_config(page_title="UK49s Predictor Level 15", layout="centered")
st.title("🔮 UK49s Predictor Level 15")

# Load or fetch historical results
@st.cache_data

def load_data():
    try:
        return pd.read_csv("uk49s_results.csv")
    except:
        return pd.DataFrame(columns=["Date", "Numbers", "Bonus"])

df_results = load_data()

# Match score function
def match_score(prediction, actual=None):
    if actual is None:
        actual = df_results.iloc[-1]["Numbers"]
        actual = list(map(int, actual.strip("[]").split(", ")))
    score = len(set(prediction) & set(actual))
    matched = sorted(set(prediction) & set(actual))
    return score, matched

# Generate prediction
def generate_prediction(seed_offset=0):
    numbers = list(range(1, 50))
    random.seed(time.time() + seed_offset)
    return sorted(random.sample(numbers, 6))

# Unique prediction generator
used_predictions = set(tuple(sorted(eval(p))) for p in df_results.get('Prediction', []))
def generate_unique_prediction(seed_offset):
    attempts = 0
    while attempts < 50:
        pred = generate_prediction(seed_offset + attempts)
        if tuple(pred) not in used_predictions:
            return pred
        attempts += 1
    return pred

# Payout table
payout_table = {6: 1500, 5: 500, 4: 100, 3: 25, 2: 5, 1: 1}

# Sidebar inputs
stake = st.sidebar.number_input("💰 Stake per prediction", min_value=1, value=10)

# Show last result
st.markdown("### 🕒 Latest Draw Result")
draw_numbers = []  # Ensure draw_numbers is defined
df_results = df_results.copy()
latest_draw = df_results.iloc[-1] if not df_results.empty else None
if latest_draw is not None:
    draw_numbers = list(map(int, latest_draw["Numbers"].strip("[]").split(", ")))
    st.write("**Date:**", latest_draw["Date"])
    st.write("**Numbers:**", draw_numbers)
    st.write("**Bonus:**", latest_draw["Bonus"])

# Prediction section
st.markdown("### 📈 Next Prediction")
next_prediction = generate_unique_prediction(seed_offset=int(time.time()) % 1000)
st.write("**Prediction:**", next_prediction)

# Match and Profit calculation
if latest_draw is not None:
    score, matched = match_score(next_prediction, draw_numbers)
    profit = (payout_table.get(score, 0) * stake) - stake if score >= 1 else -stake
    st.metric("🎯 Matches", f"{score} ({', '.join(map(str, matched)) if matched else 'None'})")
    st.metric("💵 Profit", f"{'+' if profit > 0 else ''}{profit}")

# Match breakdown table
def generate_breakdown_row(pred, actual):
    return [{"Number": n, "Matched": "✅" if n in actual else "❌"} for n in pred]
if draw_numbers:
    st.markdown("### 🧩 Match Breakdown")
    st.table(pd.DataFrame(generate_breakdown_row(next_prediction, draw_numbers)))

# Append prediction to history (simulate)
def save_prediction(prediction, score):
    new_entry = {
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": str(prediction),
        "Matches": score,
    }
    return pd.concat([df_results, pd.DataFrame([new_entry])], ignore_index=True)

# History table filters
st.markdown("### 🗂️ Prediction History")
min_matches = st.slider("Minimum matches", 0, 6, 0)
date_range = st.date_input("Select date range", [])

# History filtering
if "Prediction" not in df_results.columns:
    df_results["Prediction"] = "[]"
    df_results["Matches"] = 0

history_df = df_results.copy()
history_df["Date"] = pd.to_datetime(history_df["Date"], errors='coerce')
if date_range and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    history_df = history_df[(history_df["Date"] >= start) & (history_df["Date"] <= end)]
history_df = history_df[history_df["Matches"] >= min_matches]
st.dataframe(history_df.sort_values("Date", ascending=False), use_container_width=True)
