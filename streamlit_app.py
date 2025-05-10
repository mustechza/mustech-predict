import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="UK49s Predictor 🎯", layout="wide")

st.title("UK49s Predictor 🎯")

# === Select draw type ===
draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"], horizontal=True)

# === URLs ===
draw_urls = {
    "Lunch Time": 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results',
    "Tea Time": 'https://za.lottonumbers.com/uk-49s-teatime/past-results'
}

# === Refresh button ===
if st.button("🔄 Refresh Predictions"):
    st.experimental_rerun()

# === Function to fetch latest UK49s results ===
def fetch_latest_results(selected_url):
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(selected_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        past_results = []

        # Find rows in results table
        rows = soup.select('table.past-results tbody tr')

        for row in rows[:5]:  # Get latest 5 draws
            balls = row.select('ul.balls li.ball')

            # Extract 6 main balls
            numbers = []
            for li in balls:
                num_text = li.text.strip()
                if num_text.isdigit():
                    numbers.append(int(num_text))
                if len(numbers) == 6:
                    break

            if len(numbers) == 6:
                past_results.append(numbers)

        if not past_results:
            raise ValueError("No results found")

        return past_results

    except Exception as e:
        st.error(f"Error fetching results: {e}")
        return None

# === Fetch results ===
past_results = fetch_latest_results(draw_urls[draw_type])

if past_results:
    st.success("✅ Live results fetched successfully!")
    st.subheader("Latest Draws")
    for draw in past_results:
        st.write(draw)
else:
    st.warning("⚠️ Failed to fetch live results. Using sample data.")
    past_results = [
        [1, 7, 17, 27, 30, 33],
        [2, 13, 23, 44, 45, 47],
        [5, 12, 23, 34, 45, 48],
        [1, 14, 22, 33, 39, 44],
        [7, 9, 16, 29, 36, 40],
    ]

# === Frequency count ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

st.subheader("🔥 Hot Numbers")
st.write(hot_numbers[:6])

st.subheader("❄️ Cold Numbers")
st.write(cold_numbers[:6])

# === Chart ===
st.subheader("📊 Number Frequency Chart")
df_freq = pd.DataFrame.from_dict(number_counts, orient='index', columns=['Frequency']).sort_index()
fig, ax = plt.subplots(figsize=(10, 4))
df_freq.plot(kind='bar', ax=ax, color='skyblue', legend=False)
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.tight_layout()
st.pyplot(fig)

# === Prediction logic ===
def generate_prediction(seed_offset):
    random.seed(seed_offset)
    prediction = set()

    prediction.update(random.sample(hot_numbers[:10], 2))
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

# === Color code ===
def color_number(n):
    if n <= 9:
        return f"<span style='color:red;font-weight:bold;font-size:20px;'>{n}</span>"
    elif n <= 19:
        return f"<span style='color:blue;font-weight:bold;font-size:20px;'>{n}</span>"
    elif n <= 29:
        return f"<span style='color:green;font-weight:bold;font-size:20px;'>{n}</span>"
    elif n <= 39:
        return f"<span style='color:orange;font-weight:bold;font-size:20px;'>{n}</span>"
    else:
        return f"<span style='color:purple;font-weight:bold;font-size:20px;'>{n}</span>"

# === Display Predictions ===
st.subheader(f"🎲 {draw_type} Predicted Combinations")

prediction_col, img_col = st.columns([2, 1])

with prediction_col:
    combos = []
    for i in range(3):
        seed_offset = i if draw_type == "Lunch Time" else i + 100
        prediction = generate_prediction(seed_offset)
        combos.append(prediction)

        st.markdown(f"### 🎯 Combo {i+1}")
        colored = " ".join([color_number(n) for n in prediction])
        st.markdown(colored, unsafe_allow_html=True)

# === Shareable Image ===
with img_col:
    st.markdown("### 📱 Shareable Image")
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.axis('off')
    ax2.set_title(f"{draw_type} Predictions", fontsize=14)
    
    y = 0.8
    for i, combo in enumerate(combos):
        combo_str = "-".join(map(str, combo))
        ax2.text(0.5, y - i*0.3, f"Combo {i+1}: {combo_str}", fontsize=12, ha='center')

    st.pyplot(fig2)
    st.info("Long press to save & share on WhatsApp or Facebook!")
