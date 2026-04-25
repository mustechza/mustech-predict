
import streamlit as st
import os
st.write(os.listdir())
st.write(os.listdir("models"))
from data.loader import load_json
from models.lstm import LSTMModel
from models.agent import Agent
from training.pretrain import pretrain
from training.utils import build_state

st.title("🚀 Hybrid Crash AI Pro")

# Init
if "history" not in st.session_state:
    st.session_state.history = []

if "lstm" not in st.session_state:
    st.session_state.lstm = LSTMModel()

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

history = st.session_state.history
lstm = st.session_state.lstm
agent = st.session_state.agent

# Upload
file = st.file_uploader("Upload JSON dataset")

if file:
    data = load_json(file)

    if st.button("Pretrain AI"):
        pretrain(data, lstm, agent)
        st.session_state.history = data[-50:]
        st.success("Model pre-trained")

# Live input
mult = st.number_input("Enter multiplier")

if st.button("Add"):
    history.append(mult)

# Prediction
state = build_state(history, lstm)

if state is not None:
    action = agent.act(state)
    cashout = agent.actions[action]

    st.subheader(f"💡 Cashout Suggestion: {cashout}x")

    if len(history) > 10:
        next_state = build_state(history[:-1], lstm)

        reward = (cashout - 1) if history[-1] >= cashout else -1

        agent.remember(state, action, reward, next_state, False)
        agent.train()

st.line_chart(history)
