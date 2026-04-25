import streamlit as st
import os
import sys

# -------------------------------
# FIX STREAMLIT IMPORT PATH ISSUE
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# -------------------------------
# IMPORTS (SAFE)
# -------------------------------
from data.loader import load_json
from models.lstm import LSTMModel
from models.agent import Agent
from training.pretrain import pretrain
from training.utils import build_state

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="Hybrid Crash AI", layout="wide")
st.title("🚀 Hybrid LSTM + RL Crash AI")

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "lstm" not in st.session_state:
    st.session_state.lstm = LSTMModel()

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

history = st.session_state.history
lstm = st.session_state.lstm
agent = st.session_state.agent

# -------------------------------
# SIDEBAR: JSON PRETRAIN
# -------------------------------
st.sidebar.header("📂 Pretraining")

uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])

if uploaded_file is not None:
    data = load_json(uploaded_file)
    st.sidebar.success(f"Loaded {len(data)} rounds")

    if st.sidebar.button("🚀 Pretrain Model"):
        with st.spinner("Training AI on historical data..."):
            pretrain(data, lstm, agent)

        st.session_state.history = data[-50:]
        st.sidebar.success("✅ Pretraining complete")

# -------------------------------
# LIVE INPUT SECTION
# -------------------------------
st.subheader("🎮 Live Market Input")

col1, col2 = st.columns(2)

with col1:
    mult = st.number_input("Enter latest multiplier", min_value=1.0, step=0.01)

with col2:
    if st.button("➕ Add Round"):
        history.append(mult)

# -------------------------------
# BUILD STATE FOR AI
# -------------------------------
state = build_state(history, lstm)

# -------------------------------
# AI PREDICTION
# -------------------------------
if state is not None:
    action = agent.act(state)
    cashout = agent.actions[action]

    st.subheader("🧠 AI Recommendation")
    st.metric("Suggested Cashout", f"{cashout}x")

    # -------------------------------
    # RL TRAINING STEP
    # -------------------------------
    if len(history) > 10:
        next_state = build_state(history[:-1], lstm)

        if next_state is not None:
            reward = (cashout - 1) if history[-1] >= cashout else -1

            agent.remember(state, action, reward, next_state, False)
            agent.train()

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📊 Market Chart")
st.line_chart(history)

# -------------------------------
# DEBUG INFO (OPTIONAL)
# -------------------------------
st.sidebar.write("Rounds:", len(history))
st.sidebar.write("Exploration (epsilon):", round(agent.epsilon, 3))
