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

# ----------------- Unique Combinations Generator -----------------
def generate_unique_combinations(numbers, pick_count=10, combo_size=6, priority_order=None, exclude_combos=None):
    if len(numbers) < pick_count:
        st.warning(f"Not enough numbers to pick {pick_count}.")
        return [], []

    if priority_order:
        numbers = sorted(numbers, key=lambda x: -priority_order.get(x, 0))

    selected = sorted(numbers[:pick_count])
    combo_set = set(itertools.combinations(selected, combo_size))

    if exclude_combos:
        combo_set = combo_set - exclude_combos

    unique_combos = sorted(combo_set)
    return unique_combos, selected

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
st.subheader("🎯 Abbreviated Wheel Generator")

with st.form("wheel_form"):
    wheel_pick_count = st.slider("Select how many numbers to use for the wheel", min_value=6, max_value=15, value=10)
    wheel_draw_type = st.radio("Select Draw Type", options=["Lunchtime", "Teatime"], index=0)
    priority_mode = st.checkbox("🔥 Prioritize Hot Numbers")
    exclude_recent = st.checkbox("🚫 Exclude Recent Combos")
    submitted = st.form_submit_button("Generate Wheel")

if submitted:
    data = lunch_data if wheel_draw_type == "Lunchtime" else tea_data
    freq = lunch_freq if wheel_draw_type == "Lunchtime" else tea_freq
    recent_combos = set(tuple(sorted(draw['main'])) for draw in data)

    priority = freq if priority_mode else None
    exclude = recent_combos if exclude_recent else None

    combos, chosen = generate_unique_combinations(list(ALL_NUMBERS), pick_count=wheel_pick_count, priority_order=priority, exclude_combos=exclude)

    if combos:
        st.success(f"✅ Generated {len(combos)} unique combinations from {chosen}")
        preview_df = pd.DataFrame(combos[:10], columns=[f"Num {i+1}" for i in range(6)])
        st.dataframe(preview_df)

        csv = pd.DataFrame(combos, columns=[f"Num {i+1}" for i in range(6)]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Wheel as CSV", data=csv, file_name=f"{wheel_draw_type.lower()}_unique_combos.csv", mime="text/csv")

# ----------------- Simple Betting Simulator -----------------
st.subheader("💸 Bet Simulator")

with st.form("sim_form"):
    strategy = st.selectbox("Choose a staking strategy", ["Flat", "Martingale", "Percentage"])
    base_bet = st.number_input("Base Bet Amount", min_value=1.0, value=10.0)
    win_probability = st.slider("Estimated Win Probability (%)", min_value=1, max_value=100, value=10)
    total_rounds = st.slider("Number of Rounds", min_value=1, max_value=100, value=10)
    submit_sim = st.form_submit_button("Simulate")

if submit_sim:
    balance = 1000.0
    history = []
    last_bet = base_bet

    for round in range(total_rounds):
        win = random.randint(1, 100) <= win_probability
        profit = last_bet * 6 if win else -last_bet
        balance += profit
        history.append((round + 1, last_bet, profit, balance))

        if strategy == "Martingale":
            last_bet = base_bet if win else last_bet * 2
        elif strategy == "Percentage":
            last_bet = balance * 0.1
        else:  # Flat
            last_bet = base_bet

    sim_df = pd.DataFrame(history, columns=["Round", "Bet", "Profit", "Balance"])
    st.line_chart(sim_df.set_index("Round")["Balance"])
    st.dataframe(sim_df)
    st.success(f"Final Balance: {balance:.2f}")
