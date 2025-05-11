import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="UK49s Predictor 🎯", layout="wide")

st.markdown("<h1 style='color:purple; font-size: 48px;'>UK49s Predictor 🎯</h1>", unsafe_allow_html=True)

# === Select draw type ===
draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"], horizontal=True)

# === Function to fetch latest UK49s results ===
def fetch_latest_results():
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {
        'User-Agent': 'Mozilla/5.0'
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

# === Fetch results with error handling ===
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

# === Show Draw Date ===
st.markdown(f"<h2 style='color:green;'>📅 Last Draw Date: {draw_date}</h2>", unsafe_allow_html=True)

# === Color function ===
def color_number(n, highlight=False):
    size = "28px" if not highlight else "32px"
    weight = "bold"
    border = "2px solid gold" if highlight else "none"
    border_radius = "50%"
    bgcolor = "#FFFF99" if highlight else "transparent"

    if n <= 9:
        color = "red"
    elif n <= 19:
        color = "blue"
    elif n <= 29:
        color = "green"
    elif n <= 39:
        color = "orange"
    else:
        color = "purple"

    return f"<span style='color:{color};font-weight:{weight};font-size:{size};border:{border};border-radius:{border_radius};padding:8px;background:{bgcolor};margin:4px;display:inline-block;width:40px;text-align:center;'>{n}</span>"

# === Frequency count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

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

# === Profit Calculator ===
def calculate_profit(matches, stake):
    payout_table = {
        6: 150000,
        5: 5000,
        4: 500,
        3: 50,
        2: 5,
        1: 0,
        0: -1
    }
    profit = payout_table.get(matches, -1) * stake
    return profit

# === Confidence Score ===
def confidence_score(prediction, hot_list, cold_list):
    hot_hits = sum(1 for n in prediction if n in hot_list[:10])
    cold_hits = sum(1 for n in prediction if n in cold_list[:10])
    balance = abs(sum(n for n in prediction if n <= 24) - sum(n for n in prediction if n >= 25))
    return min(100, 50 + hot_hits * 10 - cold_hits * 5 - balance)

# === Lucky Number Generator ===
def lucky_numbers():
    return sorted(random.sample(range(1, 50), 6))

# === Layout ===
top_cols = st.columns(3)

# Left: Last Draw Numbers
with top_cols[0]:
    st.markdown("<h2 style='color:green;'>✅ Last Draw Numbers</h2>", unsafe_allow_html=True)
    colored_last_draw = " ".join([color_number(n) for n in latest_draw])
    st.markdown(colored_last_draw, unsafe_allow_html=True)

# Center: Lucky Numbers
with top_cols[1]:
    st.markdown("<h2 style='color:magenta;'>🍀 Lucky Numbers</h2>", unsafe_allow_html=True)
    lucky = lucky_numbers()
    colored_lucky = " ".join([color_number(n) for n in lucky])
    st.markdown(colored_lucky, unsafe_allow_html=True)

# Right: Predictions
with top_cols[2]:
    st.markdown(f"<h2 style='color:blue;'>🔮 {draw_type} Predictions</h2>", unsafe_allow_html=True)

pred_row = st.columns(3)  # At root level

stake = 1  # Fixed stake for now

for i in range(3):
    seed_offset = i if draw_type == "Lunch Time" else i + 100
    prediction = generate_prediction(seed_offset)
    matches = len(set(prediction) & set(latest_draw))
    profit = calculate_profit(matches, stake)
    confidence = confidence_score(prediction, hot_numbers, cold_numbers)
    profit_color = "green" if profit > 0 else "red"

    with pred_row[i]:
        st.markdown(f"<h4 style='color:purple;'>Combo {i+1}</h4>", unsafe_allow_html=True)
        colored = " ".join([color_number(n, highlight=(n in latest_draw)) for n in prediction])
        st.markdown(colored, unsafe_allow_html=True)
        st.markdown(f"<b>Matches:</b> {matches} 🎯<br><b>Confidence:</b> {confidence}%<br><b style='color:{profit_color};'>Profit: £{profit}</b>", unsafe_allow_html=True)

# === Number Frequency Chart (At Bottom) ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(number_counts.keys(), number_counts.values(), color='purple')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
ax.set_title('Last 10 Draws Frequency')
st.pyplot(fig)
