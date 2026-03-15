# crash_analyzer_app.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import threading
import json
import time

# Optional: uncomment if using real WebSocket feed
# import websocket

st.set_page_config(page_title="Crash Game Analyzer", layout="wide")

# --- DATABASE SETUP ---
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

# --- DATA COLLECTION ---

def store_multiplier(multiplier):
    """Store multiplier in DB"""
    cursor.execute(
        "INSERT INTO crashes(multiplier,timestamp) VALUES (?,?)",
        (multiplier, datetime.now())
    )
    conn.commit()

# --- SIMULATION MODE ---
def simulate_crash():
    """Simulate new multipliers every 10 seconds"""
    while True:
        multiplier = round(random.expovariate(1/1.5),2)
        multiplier = max(1.01, min(multiplier, 20))
        store_multiplier(multiplier)
        time.sleep(10)

# --- WEBSOCKET MODE (REAL GAME) ---
# Replace "wss://GAME_SERVER_URL" with real WebSocket URL
# def on_message(ws, message):
#     data = json.loads(message)
#     if "multiplier" in data:
#         store_multiplier(float(data["multiplier"]))
#         print("Crash:", data["multiplier"])

# def start_ws():
#     ws = websocket.WebSocketApp("wss://GAME_SERVER_URL", on_message=on_message)
#     ws.run_forever()

# --- BACKGROUND THREAD ---
def start_background(mode="simulation"):
    if mode=="simulation":
        threading.Thread(target=simulate_crash, daemon=True).start()
    elif mode=="ws":
        threading.Thread(target=start_ws, daemon=True).start()

# --- STREAMLIT INTERFACE ---
st.title("💥 Crash Game Analyzer & Signal Dashboard")

mode = st.radio("Data Collection Mode:", ["Simulation", "WebSocket (Real Game)"])

if st.button("Start Collector"):
    start_background("simulation" if mode=="Simulation" else "ws")
    st.success(f"{mode} started!")

# --- LOAD DATA ---
df = pd.read_sql("SELECT * FROM crashes ORDER BY id DESC LIMIT 5000", conn)

if not df.empty:
    st.subheader("📊 Stats (last 5000 rounds)")
    st.metric("Total Rounds", len(df))
    st.metric("Average Multiplier", round(df["multiplier"].mean(),2))
    st.metric("Chance >2x", f"{round((df['multiplier']>2).mean()*100,2)}%")
    st.metric("Chance >5x", f"{round((df['multiplier']>5).mean()*100,2)}%")
    
    # --- STREAK ANALYSIS ---
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
    
    streak_50 = low_streak(df["multiplier"].head(50).tolist())
    streak_100 = low_streak(df["multiplier"].head(100).tolist())
    st.write(f"Longest low multiplier streak (last 50 rounds): {streak_50}")
    st.write(f"Longest low multiplier streak (last 100 rounds): {streak_100}")

    # --- RECENT MULTIPLIER CHART ---
    st.subheader("📈 Recent Multipliers")
    st.line_chart(df["multiplier"].iloc[::-1])

    # --- SIGNAL LOGIC ---
    recent = df["multiplier"].head(5)
    if all(m < 1.3 for m in recent):
        st.success("⚡ Low streak detected → High multiplier may appear soon! Consider betting cautiously.")
    else:
        st.info("No immediate signal. Wait for low streaks.")

    # --- HISTOGRAM ---
    st.subheader("Multiplier Distribution")
    st.bar_chart(df["multiplier"].value_counts().sort_index())

    # --- STRATEGY SIMULATOR (Optional) ---
    st.subheader("Simulate R100 → Target 1.5x Strategy")
    if st.button("Run Simulator"):
        balance = 100
        base_bet = 2
        rounds_played = 0
        for m in df["multiplier"].iloc[::-1]:
            rounds_played += 1
            if balance < base_bet:
                break
            if m >= 1.5:
                balance += base_bet * 0.5  # cashout at 1.5x
            else:
                balance -= base_bet
        st.write(f"Simulated Balance: R{round(balance,2)} after {rounds_played} rounds")

else:
    st.info("No crash data collected yet. Start the collector to see stats and signals.")
