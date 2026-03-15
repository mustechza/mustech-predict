# crash_csv_analyzer_v2.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Crash CSV Analyzer", layout="wide")
st.title("💥 Crash Game CSV Analyzer & Strategy Simulator (CSV with 'x')")

# --- CSV Loader ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- CLEAN DATA ---
    # Remove 'x' and convert to float
    df['result'] = df['result'].str.replace('x','').astype(float)
    df['game_id'] = df['game_id'].astype(int)

    st.subheader("📊 Basic Stats")
    st.write(f"Total Rounds: {len(df)}")
    st.write(f"Average Multiplier: {round(df['result'].mean(),2)}")
    st.write(f"Chance >2x: {round((df['result']>2).mean()*100,2)}%")
    st.write(f"Chance >5x: {round((df['result']>5).mean()*100,2)}%")

    # --- Low Streak Detection ---
    def low_streak(data, threshold=1.3):
        streak = 0
        max_streak = 0
        for m in data:
            if m < threshold:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    streak_50 = low_streak(df['result'].tail(50).tolist())
    streak_100 = low_streak(df['result'].tail(100).tolist())

    st.subheader("⚡ Low Multiplier Streaks")
    st.write(f"Longest low streak (last 50 rounds, <1.3x): {streak_50}")
    st.write(f"Longest low streak (last 100 rounds, <1.3x): {streak_100}")

    # --- Recent Multipliers Chart ---
    st.subheader("📈 Recent Multipliers")
    st.line_chart(df['result'].tail(100).reset_index(drop=True))

    # --- Multiplier Distribution ---
    st.subheader("Histogram: Multiplier Distribution")
    st.bar_chart(df['result'].value_counts().sort_index())

    # --- Signal Suggestion ---
    recent = df['result'].tail(5)
    if all(m < 1.3 for m in recent):
        st.success("⚡ Low streak detected → High multiplier may appear soon! Consider betting cautiously.")
    else:
        st.info("No immediate signal. Wait for low streaks.")

    # --- Strategy Simulator ---
    st.subheader("🎯 Simulate R100 → 1.5× Target Strategy")
    if st.button("Run Simulator"):
        balance = 100
        base_bet = 2
        max_recovery = 4  # max Martingale steps

        for m in df['result']:
            bet = base_bet
            steps = 0
            while steps < max_recovery:
                if balance < bet:
                    break
                if m >= 1.5:
                    balance += bet * 0.5
                    break
                else:
                    balance -= bet
                    bet *= 1.5
                    steps += 1

        st.write(f"Simulated Balance after all rounds: R{round(balance,2)}")

else:
    st.info("Please upload a CSV file with 'game_id' and 'result' columns (like 4905002,1.05x).")
