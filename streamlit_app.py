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
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
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
def color_number(n, highlight=False):
    base = ''
    if n <= 9: base = 'red'
    elif n <= 19: base = 'blue'
    elif n <= 29: base = 'green'
    elif n <= 39: base = 'orange'
    else: base = 'purple'
    style = f"color:{base}; font-weight:bold; font-size:24px"
    if highlight: style += "; background-color:yellow; border-radius:50%; padding:5px"
    return f"<span style='{style}'>{n}</span>"

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

# === Confidence score ===
def confidence_score(prediction, hot, cold):
    score = 0
    for n in prediction:
        if n in hot[:10]:
            score += 15
        elif n in cold[:10]:
            score += 10
        else:
            score += 5
    return min(100, score)

# === Profit calculator ===
def calculate_profit(matches, stake):
    payouts = {6: 100000, 5: 1000, 4: 100, 3: 10, 2: 2}
    profit = payouts.get(matches, 0) * stake - stake
    return profit

# === Top section: Side by Side Layout ===
top_cols = st.columns(2)

with top_cols[0]:
    # Hot Numbers
    st.markdown("<h2 style='color:red;'>🔥 Hot Numbers</h2>", unsafe_allow_html=True)
    st.markdown(" ".join([f"<span style='font-size:28px;color:red;font-weight:bold'>{n}</span>" for n in hot_numbers[:6]]), unsafe_allow_html=True)

    # Cold Numbers
    st.markdown("<h2 style='color:blue;'>❄️ Cold Numbers</h2>", unsafe_allow_html=True)
    st.markdown(" ".join([f"<span style='font-size:28px;color:blue;font-weight:bold'>{n}</span>" for n in cold_numbers[:6]]), unsafe_allow_html=True)

with top_cols[1]:
    # Last Draw vs Predictions
    st.markdown("<h2 style='color:orange;'>🎯 Last Draw vs Predictions</h2>", unsafe_allow_html=True)
    cols = st.columns(2)

    # Left: Last Draw
    with cols[0]:
        st.markdown("<h3 style='color:green;'>✅ Last Draw Numbers</h3>", unsafe_allow_html=True)
        colored_last_draw = " ".join([color_number(n) for n in latest_draw])
        st.markdown(colored_last_draw, unsafe_allow_html=True)

    # Right: Predictions
    with cols[1]:
        st.markdown(f"<h3 style='color:blue;'>🔮 {draw_type} Predictions</h3>", unsafe_allow_html=True)
        pred_cols = st.columns(3)
        stake = 1  # default stake

        for i in range(3):
            seed_offset = i if draw_type == "Lunch Time" else i + 100
            prediction = generate_prediction(seed_offset)
            matches = len(set(prediction) & set(latest_draw))
            profit = calculate_profit(matches, stake)
            confidence = confidence_score(prediction, hot_numbers, cold_numbers)
            profit_color = "green" if profit > 0 else "red"
            with pred_cols[i]:
                st.markdown(f"<h4 style='color:purple;'>Combo {i+1}</h4>", unsafe_allow_html=True)
                colored = " ".join([color_number(n, highlight=(n in latest_draw)) for n in prediction])
                st.markdown(colored, unsafe_allow_html=True)
                st.markdown(f"<b>Matches:</b> {matches} | <b>Confidence:</b> {confidence}%<br><b style='color:{profit_color};'>Profit: £{profit}</b>", unsafe_allow_html=True)

# === Lucky Number Generator ===
st.markdown("<h2 style='color:magenta;'>🍀 Lucky Number Generator</h2>", unsafe_allow_html=True)

def lucky_number_generator():
    lucky = set()
    lucky.update(random.sample(hot_numbers[:10], 3))
    lucky.update(random.sample(range(1, 49), 3))
    return sorted(lucky)

lucky_numbers = lucky_number_generator()
colored_lucky = " ".join([color_number(n) for n in lucky_numbers])
st.markdown(f"<h3>Your Lucky Numbers:</h3> {colored_lucky}", unsafe_allow_html=True)

# === Smart Filter (Best Combo Selector) ===
st.markdown("<h2 style='color:teal;'>🎯 Smart Filter: Best Combos</h2>", unsafe_allow_html=True)

best_table_html = "<table style='width:100%; border-collapse: collapse;'>"
best_table_html += "<tr style='background-color:lightgrey;'><th>Combo</th><th>Numbers</th><th>Matches</th><th>Confidence</th><th>Profit (£)</th></tr>"

best_combos = []

for i in range(10):  # Generate 10 candidate combos
    seed_offset = i + 1000
    prediction = generate_prediction(seed_offset)
    matches = len(set(prediction) & set(latest_draw))
    profit = calculate_profit(matches, stake)
    confidence = confidence_score(prediction, hot_numbers, cold_numbers)
    best_combos.append({
        'combo': f"Combo {i+1}",
        'numbers': prediction,
        'matches': matches,
        'confidence': confidence,
        'profit': profit
    })

# Filter: only combos with confidence >= 50%
filtered = [c for c in best_combos if c['confidence'] >= 50]
filtered = sorted(filtered, key=lambda x: (-x['matches'], -x['confidence']))

if filtered:
    for combo in filtered[:3]:  # Show best 3
        colored_pred = " ".join([color_number(n, highlight=(n in latest_draw)) for n in combo['numbers']])
        profit_color = "green" if combo['profit'] > 0 else "red"
        best_table_html += f"<tr><td style='text-align:center;'>{combo['combo']}</td><td>{colored_pred}</td><td style='text-align:center;'>{combo['matches']}</td><td style='text-align:center;'>{combo['confidence']}%</td><td style='text-align:center; color:{profit_color}; font-weight:bold;'>£{combo['profit']}</td></tr>"
else:
    best_table_html += "<tr><td colspan='5' style='text-align:center;'>No high-confidence combos found!</td></tr>"

best_table_html += "</table>"
st.markdown(best_table_html, unsafe_allow_html=True)

# === Frequency Chart at Bottom ===
st.markdown("<h2 style='color:purple;'>📊 Number Frequency (Last 10 Draws)</h2>", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.bar(number_counts.keys(), number_counts.values(), color='purple')
ax.set_xlabel('Number')
ax.set_ylabel('Frequency')
st.pyplot(fig)
