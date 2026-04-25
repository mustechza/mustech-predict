import streamlit as st
import os
import importlib.util

# =====================================================
# SAFE MODULE LOADER (FIXES ALL STREAMLIT IMPORT ISSUES)
# =====================================================
def load_module(path, name):
    full_path = os.path.join(os.path.dirname(__file__), path)

    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =====================================================
# LOAD MODELS (NO IMPORT ERRORS EVER)
# =====================================================
agent_mod = load_module("models/agent.py", "agent")
lstm_mod = load_module("models/lstm.py", "lstm")

Agent = agent_mod.Agent
LSTMModel = lstm_mod.LSTMModel

# =====================================================
# LOAD TRAINING + DATA UTILITIES
# =====================================================
utils_mod = load_module("training/utils.py", "utils")
pretrain_mod = load_module("training/pretrain.py", "pretrain")
loader_mod = load_module("data/loader.py", "loader")

build_state = utils_mod.build_state
pretrain = pretrain_mod.pretrain
load_json = loader_mod.load_json


# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Hybrid Crash AI", layout="wide")
st.title("🚀 Hybrid LSTM + RL Crash AI (FIXED)")

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "lstm" not in st.session_state:
    st.session_state.lstm = LSTMModel()

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

history = st.session_state.history
lstm = st.session_state.lstm
agent = st.session_state.agent


# =====================================================
# SIDEBAR: JSON PRETRAIN
# =====================================================
st.sidebar.header("📂 Pretraining")

uploaded_file = st.sidebar.file_uploader("Upload crash JSON", type=["json"])

if uploaded_file is not None:
    data = load_json(uploaded_file)

    st.sidebar.success(f"Loaded {len(data)} rounds")

    if st.sidebar.button("🚀 Pretrain AI"):
        with st.spinner("Training LSTM + RL..."):
            pretrain(data, lstm, agent)

        st.session_state.history = data[-50:]
        st.sidebar.success("✅ Pretraining complete")


# =====================================================
# LIVE INPUT SYSTEM
# =====================================================
st.subheader("🎮 Live Market Input")

col1, col2 = st.columns(2)

with col1:
    mult = st.number_input("Enter multiplier", min_value=1.0, step=0.01)

with col2:
    if st.button("➕ Add Round"):
        history.append(float(mult))


# =====================================================
# AI STATE + PREDICTION
# =====================================================
state = build_state(history, lstm)

if state is not None:
    action = agent.act(state)
    cashout = agent.actions[action]

    st.subheader("🧠 AI Signal")
    st.metric("Suggested Cashout", f"{cashout}x")

    # =================================================
    # ONLINE RL TRAINING
    # =================================================
    if len(history) > 10:
        next_state = build_state(history[:-1], lstm)

        if next_state is not None:
            reward = (cashout - 1) if history[-1] >= cashout else -1

            agent.remember(state, action, reward, next_state, False)
            agent.train()


# =====================================================
# VISUALIZATION
# =====================================================
st.subheader("📊 Crash History")
st.line_chart(history)


# =====================================================
# DEBUG PANEL
# =====================================================
st.sidebar.subheader("📊 System Status")
st.sidebar.write("Rounds:", len(history))
st.sidebar.write("Exploration (epsilon):", round(agent.epsilon, 3))
