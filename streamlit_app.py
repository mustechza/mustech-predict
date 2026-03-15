# app.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import random
import time
import threading

# --- Database Setup ---
conn = sqlite3.connect("crash_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS crashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    multiplier REAL,
    timestamp TEXT
)
""")
conn.commit()

# --- Data Collector (Simulation for now) ---
def simulate_crash():
    """Simulate a new multiplier every 10 seconds."""
    while True:
        multiplier = round(random.expovariate(1/1.5), 2)  # simulate crash multipliers
        multiplier = max(1.01, min(multiplier, 20))        # cap between 1.01 and 20
        cursor.execute(
            "INSERT INTO crashes(multiplier,timestamp) VALUES (?,?)",
            (multiplier, datetime.now())
        )
        conn.commit()
        time.sleep(10)

# Run simulation in a background thread
if st.button("Start Simulation"):
    threading.Thread(target=simulate_crash, daemon=True).start()
    st.success("Crash simulation started!")

# --- Load Data ---
df = pd.read_sql("SELECT * FROM crashes ORDER BY id DESC LIMIT 1000", conn)

# --- Analyzer ---
st.title("Crash Game Analyzer")

if not df.empty:
    st.subheader("Stats (last 1000 rounds)")
    st.metric("Total Rounds", len(df))
    st.metric("Average Multiplier", round(df["multiplier"].mean(), 2))
    st.metric("Chance >2x", f"{round((df['multiplier'] > 2).mean()*100, 2)}%")
    st.metric("Chance >5x", f"{round((df['multiplier'] > 5).mean()*100, 2)}%")
    
    # Detect low streaks
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

    streak = low_streak(df["multiplier"].tail(50).tolist())
    st.write(f"Longest low multiplier streak (last 50 rounds): {streak}")

    # Recent multipliers chart
    st.subheader("Recent Multipliers")
    st.line_chart(df["multiplier"].iloc[::-1])

    # Betting Signal Suggestion
    recent = df["multiplier"].tail(5)
    if all(m < 1.3 for m in recent):
        st.success("⚡ Low streak detected → High multiplier may appear soon! Consider betting cautiously.")
    else:
        st.info("No immediate signal. Wait for low streaks.")

else:
    st.info("No crash data collected yet. Start the simulation or connect WebSocket data.")

# --- Footer ---
st.caption("Simulated data for testing. Replace simulate_crash() with real WebSocket or API feed.")
