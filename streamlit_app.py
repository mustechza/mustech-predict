import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import itertools
from collections import Counter
import random
import unimatrix_zero as uz

# ------------------------ Config ------------------------
st.set_page_config(page_title="UK49s Wheel Predictor", layout="wide")
st.title("🎯 UK49s Predictor & Wheel Generator")

# ------------------------ Fetch Results ------------------------
def fetch_results(draw_type="lunchtime"):
    base_url = f"https://za.lottonumbers.com/uk-49s-{draw_type}/past-results"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(base_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'past-results'})
    rows = table.select('tbody tr')
    results = []
    for row in rows[:7]:  # Last 7 results
        balls = row.select('ul.balls li.ball')
        numbers = [int(ball.text.strip()) for ball in balls if ball.text.strip().isdigit()]
        if len(numbers) >= 6:
            results.append(numbers[:6])
    return results

# ------------------------ Frequency & Missing ------------------------
def get_frequency(results):
    flat = list(itertools.chain.from_iterable(results))
    return Counter(flat)

def get_missing_numbers(results):
    appeared = set(itertools.chain.from_iterable(results))
    return sorted(set(range(1, 50)) - appeared)

# ------------------------ Display Past Results ------------------------
lunch_results = fetch_results("lunchtime")
tea_results = fetch_results("teatime")

st.header("🕒 Past Draws")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Lunchtime (Last 7)")
    st.table(pd.DataFrame(lunch_results, columns=["1","2","3","4","5","6"]))
with col2:
    st.subheader("Teatime (Last 7)")
    st.table(pd.DataFrame(tea_results, columns=["1","2","3","4","5","6"]))

# ------------------------ Frequency & Missing Table ------------------------
lunch_freq = get_frequency(lunch_results)
tea_freq = get_frequency(tea_results)

missing_lunch = get_missing_numbers(lunch_results)
missing_tea = get_missing_numbers(tea_results)

st.header("📊 Analysis")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Missing Lunchtime Numbers")
    st.write(missing_lunch)

with col4:
    st.subheader("Missing Teatime Numbers")
    st.write(missing_tea)

# ------------------------ Random Wheel Generator ------------------------
def generate_random_wheel(numbers, pick_count=10):
    chosen = sorted(random.sample(numbers, pick_count))
    return list(itertools.combinations(chosen, 6)), chosen

st.header("🎲 Random Wheel Generator")
with st.form("random_wheel_form"):
    draw_type = st.radio("Draw type", ["Lunchtime", "Teatime"], key="r_draw")
    pool = missing_lunch if draw_type == "Lunchtime" else missing_tea
    pick_count = st.slider("Pick how many numbers to use", 6, min(len(pool), 15), 10)
    submit_random = st.form_submit_button("Generate Random Wheel")

if submit_random:
    combos, selected = generate_random_wheel(pool, pick_count)
    st.success(f"Generated {len(combos)} combinations from: {selected}")
    st.dataframe(pd.DataFrame(combos[:10]))
    csv = pd.DataFrame(combos).to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="random_wheel.csv")

# ------------------------ Guaranteed Wheel via unimatrix-zero ------------------------
st.header("🧩 Guaranteed Wheel Generator")
with st.form("guaranteed_form"):
    draw_type_g = st.radio("Draw type", ["Lunchtime", "Teatime"], key="g_draw")
    pool_g = missing_lunch if draw_type_g == "Lunchtime" else missing_tea
    pick_size = st.slider("Pick size (k)", 6, 8, 6)
    guarantee = st.slider("Guarantee size (t)", 2, pick_size - 1, 3)
    num_pool = st.slider("Number of missing numbers to use", pick_size + 1, min(20, len(pool_g)), pick_size + 4)
    submitted_g = st.form_submit_button("Generate Guaranteed Wheel")

if submitted_g:
    if len(pool_g) < num_pool:
        st.error("Not enough numbers in pool.")
    else:
        chosen_pool = sorted(pool_g[:num_pool])
        design = uz.generate_design(n=num_pool, k=pick_size, t=guarantee)
        wheel = [[chosen_pool[i] for i in combo] for combo in design]
        st.success(f"Generated {len(wheel)} guaranteed combinations from {num_pool} numbers")
        st.dataframe(pd.DataFrame(wheel[:10]))
        csv_g = pd.DataFrame(wheel).to_csv(index=False).encode()
        st.download_button("Download Guaranteed Wheel CSV", data=csv_g, file_name="guaranteed_wheel.csv")
