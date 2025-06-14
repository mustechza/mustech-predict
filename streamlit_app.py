import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="UK49s Real-Time Dashboard")
st.title("🔢 UK49s Real-Time Results Dashboard")

HEADERS = {'User-Agent': 'Mozilla/5.0'}

# ----------------- Scraper -----------------
@st.cache_data(ttl=3600)
def get_results(url: str, draw_type: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        result_cards = soup.find_all('div', class_='result')

        data = []
        for card in result_cards[:7]:  # Last 7 draws
            date_elem = card.find('span', class_='date')
            date = date_elem.text.strip() if date_elem else "Unknown Date"

            balls = [int(b.text.strip()) for b in card.find_all('li', class_='ball')]
            main = balls[:-1] if len(balls) == 7 else balls
            bonus = balls[-1] if len(balls) == 7 else None

            data.append({
                "Draw Type": draw_type,
                "Date": date,
                "Main Numbers": main,
                "Bonus": bonus
            })

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error fetching {draw_type} results: {e}")
        return pd.DataFrame()

# ----------------- Fetch Data -----------------
lt_url = 'https://za.lottonumbers.com/uk-49s-lunchtime/past-results'
tt_url = 'https://za.lottonumbers.com/uk-49s-teatime/past-results'

lt_df = get_results(lt_url, 'Lunchtime')
tt_df = get_results(tt_url, 'Teatime')
df = pd.concat([lt_df, tt_df], ignore_index=True)

# ----------------- Display Table -----------------
st.subheader("📅 Last 7 Draw Results (Lunchtime & Teatime)")
st.dataframe(df, use_container_width=True)

# ----------------- Frequency Analysis -----------------
def get_number_frequency(df, draw_type):
    filtered = df[df['Draw Type'] == draw_type]
    all_numbers = sum(filtered['Main Numbers'].tolist(), [])
    return Counter(all_numbers)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Lunchtime Frequency")
    lunch_freq = get_number_frequency(df, "Lunchtime")
    lunch_series = pd.Series(lunch_freq).sort_values(ascending=False)
    st.bar_chart(lunch_series)

with col2:
    st.subheader("🔥 Teatime Frequency")
    tea_freq = get_number_frequency(df, "Teatime")
    tea_series = pd.Series(tea_freq).sort_values(ascending=False)
    st.bar_chart(tea_series)

# ----------------- Hot & Cold -----------------
st.subheader("📊 Hot & Cold Numbers (Top & Bottom 5)")
def hot_and_cold(counter):
    series = pd.Series(counter)
    hot = series.sort_values(ascending=False).head(5)
    cold = series.sort_values().head(5)
    return hot, cold

hot_lunch, cold_lunch = hot_and_cold(lunch_freq)
hot_tea, cold_tea = hot_and_cold(tea_freq)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🔴 Lunchtime")
    st.write("🔥 Hot Numbers:", hot_lunch.to_dict())
    st.write("❄️ Cold Numbers:", cold_lunch.to_dict())

with col4:
    st.markdown("### 🔵 Teatime")
    st.write("🔥 Hot Numbers:", hot_tea.to_dict())
    st.write("❄️ Cold Numbers:", cold_tea.to_dict())

# ----------------- Missing Numbers -----------------
st.subheader("🚫 Numbers Missing in Last 7 Draws")

ALL_NUMBERS = set(range(1, 50))

def get_missing_numbers(df, draw_type):
    appeared = sum(df[df['Draw Type'] == draw_type]['Main Numbers'].tolist(), [])
    appeared_set = set(appeared)
    missing = sorted(ALL_NUMBERS - appeared_set)
    return missing

col5, col6 = st.columns(2)

with col5:
    st.markdown("### 🟡 Lunchtime Missing Numbers")
    missing_lunch = get_missing_numbers(df, 'Lunchtime')
    st.write(missing_lunch)

with col6:
    st.markdown("### 🟢 Teatime Missing Numbers")
    missing_tea = get_missing_numbers(df, 'Teatime')
    st.write(missing_tea)
