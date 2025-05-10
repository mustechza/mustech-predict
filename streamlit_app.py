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
    url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        past_results = []

        # Find all rows in the results table
        rows = soup.select('table.past-results tbody tr')

        for row in rows[:5]:  # Get latest 5 draws
            date_td = row.select_one('td.date-row')
            balls = row.select('ul.balls li.ball')

            # Extract date
            draw_date = date_td.text.strip() if date_td else "Unknown Date"

            # Extract 6 main balls (skip the bonus)
            numbers = []
            for li in balls:
                num_text = li.text.strip()
                if num_text.isdigit():
                    numbers.append(int(num_text))
                if len(numbers) == 6:  # Only take first 6 balls
                    break

            if len(numbers) == 6:
                past_results.append(numbers)

        if not past_results:
            raise ValueError("No results found")

        return past_results

    except Exception as e:
        st.error(f"Error fetching results: {e}")
        return None

# === Fetch results with error handling ===
past_results = fetch_latest_results()

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

# === Color code function ===
def color_number(n):
    if n <= 9:
        return f"<span style='color:red;font-weight:bold;'>{n}</span>"
    elif n <= 19:
        return f"<span style='color:blue;font-weight:bold;'>{n}</span>"
    elif n <= 29:
        return f"<span style='color:green;font-weight:bold;'>{n}</span>"
    elif n <= 39:
        return f"<span style='color:orange;font-weight:bold;'>{n}</span>"
    else:
        return f"<span style='color:purple;font-weight:bold;'>{n}</span>"

# === Display predictions ===
st.subheader(f"🎲 {draw_type} Predicted Combinations")

cols = st.columns(3)

for i in range(3):
    seed_offset = i if draw_type == "Lunch Time" else i + 100
    prediction = generate_prediction(seed_offset)
    with cols[i]:
        st.markdown(f"### 🎯 Combo {i+1}")
        colored = " ".join([color_number(n) for n in prediction])
        st.markdown(colored, unsafe_allow_html=True)
