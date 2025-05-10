import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="UK49s Predictor 🎯", layout="wide")

st.markdown("<h1 style='color:purple;'>UK49s Predictor 🎯</h1>", unsafe_allow_html=True)

# === Select draw type ===
draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"], horizontal=True)

# === Function to fetch latest UK49s results ===
def fetch_latest_results():
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
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
st.markdown(f"<h3 style='color:green;'>📅 Last Draw Date: {draw_date}</h3>", unsafe_allow_html=True)

# === Color function ===
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

# === Frequency count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

# === Hot & Cold Numbers ===
st.markdown("<h2 style='color:red;'>🔥 Hot Numbers</h2>", unsafe_allow_html=True)
st.markdown(" ".join([f"<span style='font-size:28px;color:red;font-weight:bold'>{n}</span>" for n in hot_numbers[:6]]), unsafe_allow_html=True)

st.markdown("<h2 style='color:blue;'>❄️ Cold Numbers</h2>", unsafe_allow_html=True)
st.markdown(" ".join([f"<span style='font-size:28px;color:blue;font-weight:bold'>{n}</span>" for n in cold_numbers[:6]]), unsafe_allow_html=True)

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

# === Side by Side ===
st.markdown("<h2 style='color:orange;'>🎯 Last Draw vs Predictions</h2>", unsafe_allow_html=True)

cols = st.columns(2)

# Left: Last Draw
with cols[0]:
    st.markdown("<h3 style='color:green;'>✅ Last Draw Numbers</h3>", unsafe_allow_html=True)
    colored_last_draw = " ".join([color_number(n) for n in latest_draw])
    st.markdown(colored_last_draw, unsafe_allow_html=True)

# Right: Predictions with matches highlighted and confidence score
with cols[1]:
    st.markdown(f"<h3 style='color:blue;'>🔮 {draw_type} Predictions</h3>", unsafe_allow_html=True)
    pred_cols = st.columns(3)

    for i in range(3):
        seed_offset = i if draw_type == "Lunch Time" else i + 100
        prediction = generate_prediction(seed_offset)

        # Count matches
        matches = set(prediction).intersection(set(latest_draw))
        match_count = len(matches)
        confidence = int((match_count / 6) * 100)

        def highlight_match(n):
            if n in matches:
                return f"<span style='background-color:gold;color:black;font-weight:bold;font-size:24px;border-radius:50%;padding:6px'>{n}</span>"
            else:
                return color_number(n)

        colored_pred = " ".join([highlight_match(n) for n in prediction])

        with pred_cols[i]:
            st.markdown(f"<h4 style='color:purple;'>Combo {i+1}</h4>", unsafe_allow_html=True)
            st.markdown(colored_pred, unsafe_allow_html=True)
            st.markdown(f"<p style='color:green;font-size:18px;'>Matches: {match_count} 🎯 — Confidence: {confidence}%</p>", unsafe_allow_html=True)

# === Number Frequency at the Bottom ===
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar(number_counts.keys(), number_counts.values(), color='purple')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
st.pyplot(fig)
