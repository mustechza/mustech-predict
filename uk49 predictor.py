import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from itertools import combinations
from collections import Counter
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔁 UK49s Wheeling & Backtesting App")

# ------------------ Fetch Live Data ------------------ #
@st.cache_data
def fetch_latest_results(draw_type="Lunchtime", limit=50):
    base_url = 'https://za.lottonumbers.com/uk-49s-{}time/past-results'.format(draw_type.lower())
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(base_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'past-results'})
    rows = table.select('tbody tr')
    past_results = []
    for row in rows[:limit]:
        balls = row.select('ul.balls li.ball')
        numbers = [int(ball.text.strip()) for ball in balls]
        if len(numbers) >= 6:
            past_results.append(numbers[:6])
    draw_date = rows[0].select_one('td.date-row').text.strip()
    return past_results, draw_date

# ------------------ Sidebar Controls ------------------ #
st.sidebar.header("⚙️ Settings")
draw_type = st.sidebar.radio("Select Draw Type", ["Lunchtime", "Teatime"])
user_numbers = st.sidebar.multiselect("🎯 Select 6–15 favorite numbers", list(range(1, 50)), default=[1, 5, 9, 12, 23, 31, 36, 42, 45, 49])
wheel_limit = st.sidebar.slider("🔢 Max Wheels to Test", min_value=10, max_value=500, value=100, step=10)
draw_limit = st.sidebar.selectbox("📅 Number of Past Draws to Analyze", [7, 14, 30, 50, 90, 120], index=3)

# ------------------ Load Draw Data ------------------ #
with st.spinner(f"📥 Fetching UK49s {draw_type} Results..."):
    results, draw_date = fetch_latest_results(draw_type=draw_type, limit=draw_limit)
    df = pd.DataFrame(results, columns=["N1", "N2", "N3", "N4", "N5", "N6"])
    df['Numbers'] = df.values.tolist()
    st.success(f"Loaded {len(df)} draws (Latest: {draw_date})")

# ------------------ Generate Wheels ------------------ #
if len(user_numbers) < 6:
    st.warning("Please select at least 6 numbers.")
    st.stop()

all_wheels = list(combinations(user_numbers, 6))
if wheel_limit:
    all_wheels = all_wheels[:wheel_limit]

st.success(f"🧩 Generated {len(all_wheels)} wheels from your selected pool.")

# ------------------ Backtest ------------------ #
with st.spinner("🔍 Backtesting wheels..."):
    results = []
    def count_hits(wheel, draw):
        return len(set(wheel) & set(draw))

    for wheel in all_wheels:
        hit_counts = Counter()
        earnings = 0
        for draw in df['Numbers']:
            hits = count_hits(wheel, draw)
            if hits >= 3:
                hit_counts[hits] += 1
                if hits == 3:
                    earnings += 85
                elif hits == 4:
                    earnings += 600
                elif hits == 5:
                    earnings += 4000
                elif hits == 6:
                    earnings += 125000
        results.append({
            "Wheel": wheel,
            "3 Hits": hit_counts[3],
            "4 Hits": hit_counts[4],
            "5 Hits": hit_counts[5],
            "6 Hits": hit_counts[6],
            "Total Success (3+)": sum(hit_counts.values()),
            "Total Earnings (R)": earnings,
            "Bets Placed": len(df),
            "Profit/Loss (R)": earnings - len(df)
        })

    results_df = pd.DataFrame(results)
    top_wheels = results_df.sort_values(by='Total Success (3+)', ascending=False).reset_index(drop=True)

# ------------------ Display Results ------------------ #
st.subheader("📊 Top Performing Wheels (with Earnings)")
st.dataframe(
    top_wheels.head(20)[['Wheel', '3 Hits', '4 Hits', '5 Hits', '6 Hits', 'Total Earnings (R)', 'Profit/Loss (R)']],
    use_container_width=True
)

# ------------------ Bar Chart Summary ------------------ #
st.subheader("📈 Hit Frequency Summary (Top Wheels)")
top_hit_stats = top_wheels[['3 Hits', '4 Hits', '5 Hits', '6 Hits']].sum()
hit_chart = pd.DataFrame({
    "Hits": top_hit_stats.index,
    "Count": top_hit_stats.values
})

fig = px.bar(hit_chart, x="Hits", y="Count", text="Count", color="Hits", title="Hit Frequencies (All Wheels)")
fig.update_layout(xaxis_title="Hit Count", yaxis_title="Occurrences", title_x=0.3)
st.plotly_chart(fig, use_container_width=True)

# ------------------ Profit Chart ------------------ #
st.subheader("💸 Profit Distribution (Top Wheels)")
profit_df = top_wheels.head(20)
fig2 = px.bar(profit_df, x=profit_df.index + 1, y="Profit/Loss (R)", text="Profit/Loss (R)",
              title="Profit/Loss per Wheel")
fig2.update_layout(xaxis_title="Wheel Rank", yaxis_title="Profit/Loss (R)", title_x=0.3)
st.plotly_chart(fig2, use_container_width=True)

# ------------------ Prediction Mode ------------------ #
st.subheader("🔮 Prediction Mode: Tomorrow's Top Wheels")
prediction_count = st.slider("How many wheels to suggest?", min_value=1, max_value=10, value=5)
st.write(f"🎯 Based on top {prediction_count} past performers:")

prediction_numbers = []
for i in range(prediction_count):
    wheel = top_wheels.iloc[i]["Wheel"]
    prediction_numbers.extend(wheel)
    st.markdown(f"✅ **Wheel {i+1}:** {wheel}")

# ------------------ Most Frequent Prediction Numbers ------------------ #
st.subheader("📌 Most Frequent Numbers in Predicted Wheels")
prediction_freq = Counter(prediction_numbers)
freq_df = pd.DataFrame(prediction_freq.items(), columns=["Number", "Frequency"]).sort_values(by="Frequency", ascending=False)
st.dataframe(freq_df.reset_index(drop=True), use_container_width=True)

# ------------------ Download Option ------------------ #
csv = top_wheels.to_csv(index=False)
st.download_button("📥 Download Full Results", data=csv, file_name=f"UK49s_{draw_type}_Backtest.csv")
