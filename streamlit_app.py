import streamlit as st
import os
import sys

# ===============================
# FIX STREAMLIT IMPORT PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ===============================
# IMPORTS
# ===============================
from data.loader import load_json
from models.lstm import LSTMModel
from models.agent import Agent
from training.pretrain import pretrain
from training.utils import build_state

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Hybrid Crash AI",
    layout="wide"
)

st.title("🚀 Hybrid LSTM + RL Crash AI")
st.caption("Live crash multiplier prediction with LSTM + Reinforcement Learning")

# ===============================
# SESSION STATE INIT
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "lstm" not in st.session_state:
    st.session_state.lstm = LSTMModel()

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

history = st.session_state.history
lstm = st.session_state.lstm
agent = st.session_state.agent

# ===============================
# SIDEBAR PRETRAINING
# ===============================
st.sidebar.header("📂 Model Pretraining")

uploaded_file = st.sidebar.file_uploader(
    "Upload crash JSON file",
    type=["json"]
)

if uploaded_file is not None:
    try:
        data = load_json(uploaded_file)
        st.sidebar.success(f"Loaded {len(data)} rounds")

        if st.sidebar.button("🚀 Start Pretraining"):
            with st.spinner("Training LSTM + RL models..."):
                pretrain(data, lstm, agent)

            # Keep recent rounds only
            st.session_state.history = data[-50:]

            st.sidebar.success("✅ Pretraining completed")

    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# ===============================
# LIVE INPUT SECTION
# ===============================
st.subheader("🎮 Live Market Input")

col1, col2 = st.columns([2, 1])

with col1:
    mult = st.number_input(
        "Enter latest crash multiplier",
        min_value=1.00,
        step=0.01,
        format="%.2f"
    )

with col2:
    st.write("")
    st.write("")
    add_round = st.button("➕ Add Round")

if add_round:
    history.append(float(mult))
    st.success(f"Added round: {mult}x")

# ===============================
# BUILD CURRENT STATE
# ===============================
state = build_state(history, lstm)

# ===============================
# AI PREDICTION ENGINE
# ===============================
if state is not None:
    action = agent.act(state)
    cashout = agent.actions[action]

    st.subheader("🧠 AI Prediction")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Suggested Cashout",
            f"{cashout}x"
        )

    with colB:
        st.metric(
            "Exploration Rate",
            f"{round(agent.epsilon, 3)}"
        )

    # ===============================
    # REINFORCEMENT LEARNING UPDATE
    # ===============================
    if len(history) > 10:
        next_state = build_state(history, lstm)

        if next_state is not None:
            latest_result = history[-1]

            reward = (
                (cashout - 1)
                if latest_result >= cashout
                else -1
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                False
            )

            agent.train()

# ===============================
# CHART DISPLAY
# ===============================
st.subheader("📊 Crash History")

if len(history) > 0:
    st.line_chart(history)
else:
    st.info("No rounds added yet")

# ===============================
# DEBUG PANEL
# ===============================
st.sidebar.subheader("📊 System Stats")
st.sidebar.write(f"Rounds Stored: {len(history)}")
st.sidebar.write(f"Agent Epsilon: {round(agent.epsilon, 3)}")

# ===============================
# RESET BUTTON
# ===============================
if st.sidebar.button("🗑 Reset History"):
    st.session_state.history = []
    st.success("History cleared successfully")
