# Right: Predictions with matches highlighted and confidence score
with cols[1]:
    st.markdown(f"<h3 style='color:blue;'>🔮 {draw_type} Predictions</h3>", unsafe_allow_html=True)
    pred_cols = st.columns(3)

    for i in range(3):
        seed_offset = i if draw_type == "Lunch Time" else i + 100
        prediction = generate_prediction(seed_offset)

        # Count matches
        matches = set(prediction).intersection(set(latest_draw))
        match_count = len(matches)
        confidence = int((match_count / 6) * 100)

        def highlight_match(n):
            if n in matches:
                return f"<span style='background-color:gold;color:black;font-weight:bold;font-size:24px;border-radius:50%;padding:6px'>{n}</span>"
            else:
                return color_number(n)

        colored_pred = " ".join([highlight_match(n) for n in prediction])

        with pred_cols[i]:
            st.markdown(f"<h4 style='color:purple;'>Combo {i+1}</h4>", unsafe_allow_html=True)
            st.markdown(colored_pred, unsafe_allow_html=True)
            st.markdown(f"<p style='color:green;font-size:18px;'>Matches: {match_count} 🎯 — Confidence: {confidence}%</p>", unsafe_allow_html=True)
