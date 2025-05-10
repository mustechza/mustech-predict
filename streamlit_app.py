import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt

# === Streamlit Page Setup ===
st.set_page_config(page_title="UK49s Predictor 🎯", layout="wide")

st.markdown("<h1 style='color:purple; text-align:center;'>UK49s Predictor 🎯</h1>", unsafe_allow_html=True)

# === Draw Type Selector ===
draw_type = st.radio("🎲 Select Draw Type", ["Lunch Time", "Tea Time"], horizontal=True)

# === Fetch Latest UK49s Results ===
def fetch_latest_results():
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
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

# === Fetch results with fallback ===
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

# === Color function with highlight option ===
def color_number(n, highlight=False):
    size = "28px" if not highlight else "32px"
    border = "3px solid gold" if highlight else "none"
    bg = "yellow" if highlight else "none"
    color = "black" if highlight else (
        "red" if n <= 9 else
        "blue" if n <= 19 else
        "green" if n <= 29 else
        "orange" if n <= 39 else
        "purple"
    )
    return f"<span style='color:{color}; font-weight:bold; font-size:{size}; border:{border}; background:{bg}; border-radius:50%; padding:8px; margin:4px;'>{n}</span>"

# === Frequency count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)
hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

# === Layout: Hot & Cold numbers side by side ===
top_cols = st.columns(2)

with top_cols[0]:
    st.markdown("<h2 style='color:red;'>🔥 Hot Numbers</h2>", unsafe_allow_html=True)
    st.markdown(" ".join([color_number(n) for n in hot_numbers[:6]]), unsafe_allow_html=True)

with top_cols[1]:
    st.markdown("<h2 style='color:blue;'>❄️ Cold Numbers</h2>", unsafe_allow_html=True)
    st.markdown(" ".join([color_number(n) for n in cold_numbers[:6]]), unsafe_allow_html=True)

# === Generate prediction ===
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

# === Confidence score ===
def confidence_score(prediction, hot_nums, cold_nums):
    score = 0
    for n in prediction:
        if n in hot_nums[:10]:
            score += 2
        elif n in cold_nums[:10]:
            score += 1
    return score * 10  # out of 100

# === Profit calculator ===
def calculate_profit(matches, stake):
    odds = {6: 100000, 5: 12000, 4: 1800, 3: 200, 2: 20, 1: 0, 0: 0}
    return stake * odds.get(matches, 0)

# === Stake input ===
stake = st.number_input("💷 Enter your stake (£)", min_value=1, value=1)

# === Layout: Last Draw vs Predictions ===
st.markdown("<h2 style='color:orange;'>🎯 Last Draw vs Predictions</h2>", unsafe_allow_html=True)
cols = st.columns(2)

# === Left: Last Draw Numbers ===
with cols[0]:
    st.markdown(f"<h3 style='color:green;'>📅 Last Draw Date: {draw_date}</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:green;'>✅ Last Draw Numbers</h3>", unsafe_allow_html=True)
    colored_last_draw = " ".join([color_number(n) for n in latest_draw])
    st.markdown(colored_last_draw, unsafe_allow_html=True)

# === Right: Predictions + Profit ===
with cols[1]:
    st.markdown(f"<h3 style='color:blue;'>🔮 {draw_type} Predictions & Profits</h3>", unsafe_allow_html=True)
    table_html = "<table style='width:100%; border-collapse: collapse;'>"
    table_html += "<tr style='background-color:lightgrey;'><th>Combo</th><th>Numbers</th><th>Matches</th><th>Confidence</th><th>Profit (£)</th></tr>"

    for i in range(3):
        seed_offset = i if draw_type == "Lunch Time" else i + 100
        prediction = generate_prediction(seed_offset)
        matches = len(set(prediction) & set(latest_draw))
        profit = calculate_profit(matches, stake)
        confidence = confidence_score(prediction, hot_numbers, cold_numbers)

        # Highlight matches
        colored_pred = " ".join([color_number(n, highlight=(n in latest_draw)) for n in prediction])
        profit_color = "green" if profit > 0 else "red"
        table_html += f"<tr><td style='text-align:center;'>Combo {i+1}</td><td>{colored_pred}</td><td style='text-align:center;'>{matches}</td><td style='text-align:center;'>{confidence}%</td><td style='text-align:center; color:{profit_color}; font-weight:bold;'>£{profit}</td></tr>"

    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

# === Bottom: Frequency Chart ===
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar(number_counts.keys(), number_counts.values(), color='purple')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
ax.set_title('Frequency Chart')
st.pyplot(fig)
