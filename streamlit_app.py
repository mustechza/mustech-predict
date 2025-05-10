import random
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st

st.title("UK49s Predictor 🎯")

# === Select draw type ===
draw_type = st.radio("Select Draw:", ["Lunch Time", "Tea Time"])

# === Refresh button ===
if st.button("🔄 Refresh Predictions"):
    st.experimental_rerun()

# === Function to fetch latest UK49s results ===
def fetch_latest_results():
    urls = [
        'https://www.lottery.co.uk/49s/results',  # More stable source
        'https://www.lottodraw.com/uk49s',
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            past_results = []
            draws = soup.select('.balls, .drawn')[:10]

            for draw in draws:
                numbers = [int(n.text.strip()) for n in draw.select('li, span') if n.text.strip().isdigit()]
                if len(numbers) >= 6:
                    past_results.append(numbers[:6])

            if past_results:
                return past_results

        except Exception as e:
            continue

    raise Exception("All sources failed")


# === Fetch results with error handling ===
try:
    past_results = fetch_latest_results()
    st.success("✅ Live results fetched successfully!")
except Exception as e:
    st.warning(f"⚠️ Failed to fetch live results. Using sample data.")
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


# === Count frequency ===
all_numbers = [num for draw in past_results for num in draw]
number_counts = Counter(all_numbers)

hot_numbers = [num for num, count in number_counts.most_common()]
cold_numbers = [num for num, count in number_counts.most_common()][::-1]

st.subheader("🔥 Hot Numbers")
st.write(hot_numbers[:6])

st.subheader("❄️ Cold Numbers")
st.write(cold_numbers[:6])


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


# === Display predictions ===
st.subheader(f"🎲 {draw_type} Predicted Combinations")

cols = st.columns(3)

for i in range(3):
    seed_offset = i if draw_type == "Lunch Time" else i + 100
    prediction = generate_prediction(seed_offset)
    with cols[i]:
        st.markdown(f"### 🎯 Combo {i+1}")
        st.write(prediction)
