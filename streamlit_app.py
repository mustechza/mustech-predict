import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os

st.set_page_config(page_title="UK49s Predictor Level 15 🚀", layout="wide")

st.markdown("<h1 style='color:purple;'>UK49s Predictor 🎯 Level 15 🚀</h1>", unsafe_allow_html=True)

draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"], horizontal=True)

# === Fetch Results ===
def fetch_latest_results():
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'past-results'})
    rows = table.select('tbody tr')
    past_results = []
    for row in rows[:10]:
        balls = row.select('ul.balls li.ball')
        numbers = [int(ball.text.strip()) for ball in balls]
        if len(numbers) >= 6:
            past_results.append(numbers[:6])
    draw_date = rows[0].select_one('td.date-row').text.strip()
    latest_draw = past_results[0]
    return past_results, draw_date, latest_draw

try:
    past_results, draw_date, latest_draw = fetch_latest_results()
    st.success(f"✅ Live results fetched! (Last Draw: {draw_date})")
except:
    draw_date = "Sample Data"
    past_results = [
        [5, 12, 23, 34, 45, 48],
        [1, 14, 22, 33, 39, 44],
        [7, 9, 16, 29, 36, 40],
        [3, 18, 21, 30, 42, 49],
        [6, 13, 27, 31, 38, 47],
        [2, 10, 20, 32, 41, 46],
        [4, 11, 19, 28, 35, 43],
        [8, 15, 17, 24, 25, 37],
    ]
    latest_draw = past_results[0]

st.markdown(f"<h3 style='color:green;'>📅 Last Draw Date: {draw_date}</h3>", unsafe_allow_html=True)

# === Color Number ===
def color_number(n):
    if n <= 9: c = 'red'
    elif n <= 19: c = 'blue'
    elif n <= 29: c = 'green'
    elif n <= 39: c = 'orange'
    else: c = 'purple'
    return f"<span style='color:{c};font-weight:bold;font-size:24px'>{n}</span>"

# === Frequency Count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)
hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = list(reversed(hot_numbers))

# === Lucky Number Generator ===
st.markdown("<h2 style='color:gold;'>🎲 Lucky Number Generator</h2>", unsafe_allow_html=True)
if st.button("🎯 Spin My Lucky Numbers!"):
    lucky_numbers = sorted(random.sample(range(1, 50), 6))
    lucky_colored = " ".join([color_number(n) for n in lucky_numbers])
    st.markdown(lucky_colored, unsafe_allow_html=True)
    st.balloons()

# === Hot-Cold Chart ===
st.markdown("<h2 style='color:orange;'>📊 Hot vs Cold Chart</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar([str(n) for n in hot_numbers[:10]], [number_counts[n] for n in hot_numbers[:10]], color='red', label='Hot')
ax.bar([str(n) for n in cold_numbers[:10]], [number_counts[n] for n in cold_numbers[:10]], color='blue', alpha=0.5, label='Cold')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)

# === Generate Prediction ===
def generate_prediction(seed_offset):
    random.seed(seed_offset)
    prediction = set()
    prediction.update(random.sample(hot_numbers[:15], 2))
    prediction.update(random.sample(cold_numbers[:20], 2))
    while len(prediction) < 6:
        candidate = random.randint(1, 49)
        temp = list(prediction) + [candidate]
        odd_count = sum(1 for n in temp if n % 2 != 0)
        even_count = sum(1 for n in temp if n % 2 == 0)
        low_count = sum(1 for n in temp if n <= 24)
        high_count = sum(1 for n in temp if n >= 25)
        if odd_count <= 4 and even_count <= 4 and low_count <= 4 and high_count <= 4:
            prediction.add(candidate)
    return sorted(prediction)

# === Match Score ===
def match_score(prediction):
    matches = set(prediction) & set(latest_draw)
    return len(matches), matches

# === Smart Filter Combos ===
st.markdown("<h2 style='color:purple;'>🔥 Smart Filter: Best Combos</h2>", unsafe_allow_html=True)
cols = st.columns(3)
shown = 0
for i in range(10):
    seed_offset = i if draw_type == "Lunch Time" else i + 100
    prediction = generate_prediction(seed_offset)
    score, matched = match_score(prediction)
    confidence = round((score / 6) * 100)
    if score >= 3:
        with cols[shown % 3]:
            st.markdown(f"<h4 style='color:green;'>Combo {i+1} (Confidence: {confidence}%)</h4>", unsafe_allow_html=True)
            colored_pred = " ".join([
                f"<span style='background-color:yellow'>{color_number(n)}</span>" if n in matched else color_number(n)
                for n in prediction
            ])
            st.markdown(colored_pred, unsafe_allow_html=True)
        shown += 1
if shown == 0:
    st.info("No combos with 3+ matches found this time. Spin again!")

# === Last Draw Numbers ===
st.markdown("<h2 style='color:teal;'>✅ Last Draw Numbers</h2>", unsafe_allow_html=True)
colored_last = " ".join([color_number(n) for n in latest_draw])
st.markdown(colored_last, unsafe_allow_html=True)

# === Profit Calculator ===
st.markdown("<h2 style='color:gold;'>💰 Profit Calculator</h2>", unsafe_allow_html=True)
stake = st.number_input("Enter your stake amount (e.g. 10):", min_value=1)
payout_table = {1: 7, 2: 57, 3: 401, 4: 2000, 5: 10000, 6: 125000}

score, matched = match_score(latest_draw)
profit = (payout_table.get(score, 0) * stake) - stake if score >= 1 else -stake
st.markdown(f"🎯 Matches: {score} numbers")
st.markdown(f"💰 Estimated Profit/Loss: **R{profit}**")

# === Frequency Chart ===
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)
fig2, ax2 = plt.subplots()
ax2.bar(number_counts.keys(), number_counts.values(), color='purple')
ax2.set_xlabel('Number')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)

# === Level 13-15: Prediction Tracker ===
history_file = "predictions_history.csv"
if not os.path.exists(history_file):
    df_init = pd.DataFrame(columns=["Date", "Prediction", "Matches", "Matched Numbers"])
    df_init.to_csv(history_file, index=False)
df_history = pd.read_csv(history_file)

st.markdown("<h2 style='color:darkblue;'>📅 Prediction History vs Actual Results</h2>", unsafe_allow_html=True)
st.dataframe(df_history.tail(10))

# === Next Prediction ===
st.markdown("<h2 style='color:green;'>🔮 Next Prediction (Auto-Saved)</h2>", unsafe_allow_html=True)
next_prediction = generate_prediction(seed_offset=int(time.time()) % 1000)
colored_next_pred = " ".join([color_number(n) for n in next_prediction])
st.markdown(colored_next_pred, unsafe_allow_html=True)

score, matched = match_score(next_prediction)
new_entry = {
    "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "Prediction": ", ".join(map(str, next_prediction)),
    "Matches": score,
    "Matched Numbers": ", ".join(map(str, matched)) if matched else "None"
}
df_history = pd.concat([df_history, pd.DataFrame([new_entry])], ignore_index=True)
df_history.to_csv(history_file, index=False)
st.success(f"✅ Saved prediction with {score} match(es)!")

# === Accuracy Chart ===
st.markdown("<h2 style='color:orange;'>📈 Prediction Accuracy Over Time</h2>", unsafe_allow_html=True)
df_history["Matches"] = pd.to_numeric(df_history["Matches"], errors='coerce')
fig3, ax3 = plt.subplots()
ax3.plot(df_history["Date"], df_history["Matches"], marker='o', color='purple', linestyle='-')
ax3.set_xlabel("Date")
ax3.set_ylabel("Matches (out of 6)")
ax3.set_title("Prediction Accuracy Over Time")
ax3.grid(True)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig3)

# === Download Button ===
st.markdown("<h2 style='color:teal;'>⬇️ Download Your Full Prediction History</h2>", unsafe_allow_html=True)
csv = df_history.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "prediction_history.csv", "text/csv", key='download-csv')

# === Clear History ===
st.markdown("<h2 style='color:red;'>🧹 Reset Prediction History</h2>", unsafe_allow_html=True)
if st.button("⚠️ Clear All History"):
    df_clear = pd.DataFrame(columns=["Date", "Prediction", "Matches", "Matched Numbers"])
    df_clear.to_csv(history_file, index=False)
    st.success("✅ History cleared! Please refresh app to see changes.")

# === Level 15: Win Rate + Best Prediction ===
st.markdown("<h2 style='color:gold;'>🏆 Tracker Stats</h2>", unsafe_allow_html=True)
total_preds = len(df_history)
total_wins = df_history[df_history["Matches"] >= 3].shape[0]
win_rate = round((total_wins / total_preds) * 100, 2) if total_preds > 0 else 0

best_row = df_history.sort_values("Matches", ascending=False).iloc[0] if total_preds > 0 else None

st.markdown(f"✅ Total Predictions: **{total_preds}**")
st.markdown(f"🎯 Predictions with 3+ matches: **{total_wins}**")
st.markdown(f"🔥 Win Rate (3+ matches): **{win_rate}%**")

if best_row is not None:
    st.markdown(f"🏅 Best Ever Prediction: {best_row['Prediction']} with {best_row['Matches']} matches on {best_row['Date']}")

# === Optional Jackpot Alert ===
if score >= 4:
    st.markdown(f"<h2 style='color:red;'>🎉 JACKPOT! You got {score} matches in this prediction! 🎉</h2>", unsafe_allow_html=True)
