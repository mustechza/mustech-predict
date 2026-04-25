import streamlit as st
from lstm_model import LSTMModel
from agent import Agent
from utils import train_lstm, build_state

st.set_page_config(layout="wide")
st.title("🤖 Hybrid LSTM + RL Crash AI")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "lstm" not in st.session_state:
    st.session_state.lstm = LSTMModel()

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

history = st.session_state.history
lstm = st.session_state.lstm
agent = st.session_state.agent

# Input
col1, col2 = st.columns(2)

with col1:
    new_mult = st.number_input("Enter multiplier", min_value=1.0, step=0.01)

with col2:
    if st.button("Add Round"):
        history.append(new_mult)

        if len(history) > 20:
            train_lstm(lstm, history)

# Build state
state = build_state(history, lstm)

if state is not None:
    action = agent.act(state)
    cashout = agent.actions[action]

    st.subheader(f"💡 Suggested Cashout: {cashout}x")

    # Train RL
    if len(history) > 10:
        prev_state = build_state(history[:-1], lstm)
        actual = history[-1]

        reward = (cashout - 1) if actual >= cashout else -1

        agent.remember(prev_state, action, reward, state, False)
        agent.train()

# Display chart
st.subheader("📊 Multiplier History")
st.line_chart(history)

# Debug info
st.sidebar.write(f"Epsilon: {round(agent.epsilon, 3)}")
st.sidebar.write(f"Rounds: {len(history)}")
