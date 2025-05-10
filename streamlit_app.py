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

        # Debugging: Print the first 500 characters of the HTML to check structure
        st.write(soup.prettify()[:500])

        # Extract the latest draw date and numbers
        draw_date_element = soup.find('h2')
        if draw_date_element:
            draw_date = draw_date_element.text.strip()
        else:
            raise ValueError("Could not find draw date.")

        # Extract numbers from the page
        numbers_element = soup.find('ul', class_='numbers')
        if numbers_element:
            numbers = [int(num) for num in numbers_element.text.split() if num.isdigit()]
        else:
            raise ValueError("Could not find numbers.")

        return draw_date, numbers

    except Exception as e:
        st.error(f"Error fetching results: {e}")
        return None, None

# === Fetch results with error handling ===
draw_date, past_results = fetch_latest_results()

if draw_date and past_results:
    st.success("✅ Live results fetched successfully!")
    st.subheader(f"Latest Lunchtime Draw: {draw_date}")
    st.write("Winning Numbers:", past_results)
else:
    st.warning("⚠️ Failed to fetch live results. Using sample data.")
    # Sample data for testing
    draw_date = "Saturday 10th May 2025"
    past_results = [1, 7, 17, 27, 30, 33, 38]
    st.write("Winning Numbers:", past_results)

# === Count frequency ===
all_numbers = [num for num in past_results]
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

