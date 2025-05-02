import streamlit as st
import pandas as pd

st.title("🧪 Streamlit File Upload Test")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    st.success("✅ File received.")
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        st.write("✅ File successfully read into DataFrame.")
        st.write("🔍 Preview of the data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to parse CSV: {e}")
else:
    st.info("📂 Please upload a CSV file to begin.")
