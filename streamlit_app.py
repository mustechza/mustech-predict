import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import itertools

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="UK49s Real-Time Dashboard")
st.title("🔢 UK49s Real-Time Results Dashboard")

HEADERS = {'User-Agent': 'Mozilla/5.0'}
ALL_NUMBERS = set(range(1, 50))

# ----------------- Fetch Results -----------------
def fetch_latest_results(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'past-results'})
        rows = table.select('tbody tr')
        past_results = []

        for i, row in enumerate(rows[:7]):  # Last 7 draws
            balls = row.select('ul.balls li.ball')
            numbers = [int(ball.text.strip()) for ball in balls]
            if len(numbers) >= 6:
                main = numbers[:6]
                bonus = numbers[6] if len(numbers) > 6 else None
                date = row.select_one('td.date-row').text.strip() if row.select_one('td.date-row') else 'Unknown'
                past_results.append({
                    'main': main,
                    'bonus': bonus,
                    'date': date
                })

        return past_results

    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return []

# ----------------- Data Processing -----------------
def results_to_dataframe(results, draw_type):
    records = []
    for entry in results:
        records.append({
            "Draw Type": draw_type,
            "Date": entry["date"],
            "Main Numbers": entry["main"],
            "Bonus": entry["bonus"]
        })
    return pd.DataFrame(records)

def get_number_frequency(results):
    all_numbers = []
    for draw in results:
        all_numbers.extend(draw['main'])
    return Counter(all_numbers)

def get_missing_numbers(results):
    appeared = set()
    for draw in results:
        appeared.update(draw['main'])
    return sorted(ALL_NUMBERS - appeared)

def hot_and_cold(counter):
    series = pd.Series(counter)
    hot = series.sort_values(ascending=False).head(5)
    cold = series.sort_values().head(5)
    return hot, cold

# ----------------- Abbreviated Wheel Generator -----------------
def generate_abbreviated_wheel(numbers, pick_count=10):
    if len(numbers) < pick_count:
        st.warning(f"Not enough missing numbers to pick {pick_count} numbers.")
        return [], []

    chosen_numbers = sorted(numbers)[:pick_count]
    combos = list(itertools.combinations(chosen_numbers, 6))
    return combos, chosen_numbers

# ----------------- Fetch Both Draws -----------------
lunch_url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
tea_url = 'https://za.lottonumbers.com/uk-49s-teatime/past-results'

lunch_data = fetch_latest_results(lunch_url)
tea_data = fetch_latest_results(tea_url)

if not lunch_data and not tea_data:
    st.stop()

lunch_df = results_to_dataframe(lunch_data, "Lunchtime")
tea_df = results_to_dataframe(tea_data, "Teatime")
df = pd.concat([lunch_df, tea_df], ignore_index=True)

# ----------------- Display Draw Table -----------------
st.subheader("📅 Last 7 UK49s Results (Lunchtime & Teatime)")
st.dataframe(df, use_container_width=True)

# ----------------- Frequency Charts -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Lunchtime Frequency")
    lunch_freq = get_number_frequency(lunch_data)
    if lunch_freq:
        st.bar_chart(pd.Series(lunch_freq).sort_values(ascending=False))

with col2:
    st.subheader("🔥 Teatime Frequency")
    tea_freq = get_number_frequency(tea_data)
    if tea_freq:
        st.bar_chart(pd.Series(tea_freq).sort_values(ascending=False))

# ----------------- Hot and Cold Numbers -----------------
st.subheader("📊 Hot & Cold Numbers (Top & Bottom 5)")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🔴 Lunchtime")
    hot_l, cold_l = hot_and_cold(lunch_freq)
    st.write("🔥 Hot:", hot_l.to_dict())
    st.write("❄️ Cold:", cold_l.to_dict())

with col4:
    st.markdown("### 🔵 Teatime")
    hot_t, cold_t = hot_and_cold(tea_freq)
    st.write("🔥 Hot:", hot_t.to_dict())
    st.write("❄️ Cold:", cold_t.to_dict())

# ----------------- Missing Numbers -----------------
st.subheader("🚫 Missing Numbers in Last 7 Draws")

col5, col6 = st.columns(2)

with col5:
    st.markdown("### 🟡 Lunchtime Missing Numbers")
    missing_lunch = get_missing_numbers(lunch_data)
    st.write(missing_lunch)

with col6:
    st.markdown("### 🟢 Teatime Missing Numbers")
    missing_tea = get_missing_numbers(tea_data)
    st.write(missing_tea)

# ----------------- Abbreviated Wheel Generator UI -----------------
st.subheader("🎯 Abbreviated Wheel Generator from Missing Numbers")

with st.form("wheel_form"):
    wheel_pick_count = st.slider(
        "Select how many missing numbers to use for the wheel",
        min_value=6, max_value=15, value=10)
    wheel_draw_type = st.radio(
        "Select Draw Type",
        options=["Lunchtime", "Teatime"],
        index=0)
    submitted = st.form_submit_button("Generate Wheel")

if submitted:
    missing = missing_lunch if wheel_draw_type == "Lunchtime" else missing_tea
    combos, chosen = generate_abbreviated_wheel(missing, wheel_pick_count)
    if combos:
        st.write(f"Using numbers: {chosen}")
        st.write(f"Total combinations (bets): {len(combos)}")
        st.write("First 10 combinations:")
        for combo in combos[:10]:
            st.write(combo)
