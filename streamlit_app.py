# === UK49s Predictor App (Level 3 + Level 4 Enhanced) ===
import random
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
import time

# Streamlit config
st.set_page_config(page_title="UK49s Predictor 🎯", layout="wide")

# Page title
st.markdown("<h1 style='color:purple;'>UK49s Predictor Pro 🎯</h1>", unsafe_allow_html=True)

# Draw selection
draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"], horizontal=True)

# === Function to fetch latest UK49s results ===
def fetch_latest_results():
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'past-results'})
    if not table:
        raise Exception("Could not find results table.")

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

# === Fetch results ===
try:
    past_results, draw_date, latest_draw = fetch_latest_results()
    st.success(f"✅ Live results fetched! (Last Draw: {draw_date})")
except Exception as e:
    st.warning("⚠️ Failed to fetch live results. Using sample data.")
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

# === Color Function ===
def color_number(n):
    if n <= 9:
        return f"<span style='color:red;font-weight:bold;font-size:24px'>{n}</span>"
    elif n <= 19:
        return f"<span style='color:blue;font-weight:bold;font-size:24px'>{n}</span>"
    elif n <= 29:
        return f"<span style='color:green;font-weight:bold;font-size:24px'>{n}</span>"
    elif n <= 39:
        return f"<span style='color:orange;font-weight:bold;font-size:24px'>{n}</span>"
    else:
        return f"<span style='color:purple;font-weight:bold;font-size:24px'>{n}</span>"

# === Frequency Count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

# === Hot & Cold Numbers ===
st.markdown("<h2 style='color:red;'>🔥 Hot Numbers</h2>", unsafe_allow_html=True)
st.markdown(" ".join([f"<span style='font-size:28px;color:red;font-weight:bold'>{n}</span>" for n in hot_numbers[:6]]), unsafe_allow_html=True)

st.markdown("<h2 style='color:blue;'>❄️ Cold Numbers</h2>", unsafe_allow_html=True)
st.markdown(" ".join([f"<span style='font-size:28px;color:blue;font-weight:bold'>{n}</span>" for n in cold_numbers[:6]]), unsafe_allow_html=True)

# === Frequency Chart ===
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar(number_counts.keys(), number_counts.values(), color='purple')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# === Generate Prediction ===
def generate_prediction(seed_offset):
    random.seed(seed_offset)
    prediction = set()

    prediction.update(random.sample(hot_numbers[:15], 2))  # Hot numbers
    prediction.update(random.sample(cold_numbers[:20], 2))  # Cold numbers

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

# === Smart Filter (3+ matches) ===
def smart_filter(predictions, actual_draw):
    filtered = []
    for pred in predictions:
        matches = len(set(pred) & set(actual_draw))
        if matches >= 3:
            filtered.append((pred, matches))
    return filtered

# === Animate Lucky Numbers with Spin Button ===
st.markdown("<h2 style='color:orange;'>🎲 Lucky Number Generator</h2>", unsafe_allow_html=True)

spin_button = st.button("Spin the Wheel 🎉")

lucky_numbers = []
if spin_button:
    lucky_numbers = random.sample(range(1, 50), 6)
    lucky_numbers = sorted(lucky_numbers)

st.markdown(f"Your Lucky Numbers: {' '.join(map(str, lucky_numbers))}", unsafe_allow_html=True)

# === Display Predictions (Filtered & Matched with Last Draw) ===
st.markdown("<h2 style='color:green;'>🔮 Predictions & Matches</h2>", unsafe_allow_html=True)

predictions = []
for i in range(3):
    seed_offset = i if draw_type == "Lunch Time" else i + 100
    prediction = generate_prediction(seed_offset)
    predictions.append(prediction)

# Smart filter for predictions
filtered_predictions = smart_filter(predictions, latest_draw)

# Display the filtered predictions
for i, (prediction, matches) in enumerate(filtered_predictions):
    st.markdown(f"<h4 style='color:purple;'>Combo {i+1} - {matches} Match(es)</h4>", unsafe_allow_html=True)
    colored_prediction = " ".join([color_number(n) for n in prediction])
    st.markdown(colored_prediction, unsafe_allow_html=True)



# === 📈 Hot vs Cold Chart (Side by Side Bars) ===
st.markdown("<h2 style='color:purple;'>📊 Hot vs Cold Numbers</h2>", unsafe_allow_html=True)

hot_counts = [number_counts.get(n, 0) for n in range(1, 50)]
cold_counts = [10 - c for c in hot_counts]

fig2, ax2 = plt.subplots(figsize=(10, 5))
bar_width = 0.35
numbers = list(range(1, 50))

ax2.bar([n - bar_width/2 for n in numbers], hot_counts, width=bar_width, color='red', label='Hot')
ax2.bar([n + bar_width/2 for n in numbers], cold_counts, width=bar_width, color='blue', label='Cold')

ax2.set_xlabel('Number')
ax2.set_ylabel('Count (Last 10 Draws)')
ax2.legend()
st.pyplot(fig2)

# === 💰 Profit Calculator ===
st.markdown("<h2 style='color:green;'>💰 Profit Calculator (Based on Matched Numbers)</h2>", unsafe_allow_html=True)

# UK49s Lunchtime payout example odds
odds_per_match = {
    6: 70000,  # Match 6
    5: 1250,   # Match 5
    4: 100,    # Match 4
    3: 13,     # Match 3
    2: 2,      # Match 2
}

bet_amount = st.number_input("Enter your bet amount (£):", min_value=1, value=1, step=1)

# Calculate for each filtered prediction
for i, (prediction, matches) in enumerate(filtered_predictions):
    payout = odds_per_match.get(matches, 0) * bet_amount
    st.markdown(f"**Combo {i+1}: Matches {matches} - Potential Win: £{payout}**")


import numpy as np  # Add at top if missing

# === 📈 Trend Chart (Numbers appearance over draws) ===
st.markdown("<h2 style='color:orange;'>📈 Number Appearance Trend (Last 10 Draws)</h2>", unsafe_allow_html=True)

trend_counts = {n: [] for n in range(1, 50)}

# Count per draw
for draw in past_results:
    draw_counts = Counter(draw)
    for n in trend_counts:
        trend_counts[n].append(draw_counts.get(n, 0))

fig3, ax3 = plt.subplots(figsize=(12, 5))
for n in range(1, 50):
    counts = trend_counts[n]
    if sum(counts) > 0:
        ax3.plot(counts, label=str(n))

ax3.set_xlabel('Draw Number (Recent → Older)')
ax3.set_ylabel('Count')
ax3.set_title('Trend per Number')
ax3.legend(loc='upper right', bbox_to_anchor=(1.12, 1), fontsize='x-small', ncol=2)
st.pyplot(fig3)


# === 🔥 Number Heatmap ===
st.markdown("<h2 style='color:red;'>🔥 Number Heatmap</h2>", unsafe_allow_html=True)

heatmap_data = np.zeros((7, 7))

for n in range(1, 50):
    row = (n - 1) // 7
    col = (n - 1) % 7
    heatmap_data[row, col] = number_counts.get(n, 0)

fig4, ax4 = plt.subplots(figsize=(8, 6))
cax = ax4.matshow(heatmap_data, cmap='hot')
fig4.colorbar(cax)

for i in range(7):
    for j in range(7):
        num = i * 7 + j + 1
        ax4.text(j, i, str(num), va='center', ha='center', color='white', fontsize=12, fontweight='bold')

ax4.set_xticks([])
ax4.set_yticks([])
st.pyplot(fig4)
import time  # Add at top if missing
import pandas as pd  # Add at top if missing

# === 🎲 Lucky Number Spinner ===
st.markdown("<h2 style='color:purple;'>🎲 Lucky Number Spinner</h2>", unsafe_allow_html=True)

if st.button("🎯 Spin My Lucky Numbers!"):
    lucky_placeholder = st.empty()
    for _ in range(20):  # Spin 20 times
        spin_numbers = random.sample(range(1, 50), 6)
        colored_spin = " ".join([color_number(n) for n in spin_numbers])
        lucky_placeholder.markdown(colored_spin, unsafe_allow_html=True)
        time.sleep(0.1)
    st.success("✨ Here’s your Lucky Combo!")

# === 🏅 Best Performer Numbers ===
st.markdown("<h2 style='color:gold;'>🏅 Best Performer Numbers (Top 12)</h2>", unsafe_allow_html=True)

best_numbers = hot_numbers[:12]
best_colored = " ".join([color_number(n) for n in best_numbers])
st.markdown(best_colored, unsafe_allow_html=True)

# === 📥 Export Full Results ===
st.markdown("<h2 style='color:green;'>📥 Download Full Results (CSV)</h2>", unsafe_allow_html=True)

# Prepare DataFrame
draws_data = {
    "Draw #": list(range(1, len(past_results)+1)),
    "Numbers": [", ".join(map(str, draw)) for draw in past_results]
}

df_results = pd.DataFrame(draws_data)

csv = df_results.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Last 10 Draws as CSV",
    data=csv,
    file_name='uk49s_last10_draws.csv',
    mime='text/csv',
)


