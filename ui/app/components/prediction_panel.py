import streamlit as st

def render_prediction_panel(result: dict | None, mock_mode: bool) -> None:
    with st.container(border=True):
        st.write("**Resultado de la predicción**")

        if result is None:
            st.info("Completa los campos y presiona **Predecir** para ver el resultado.")
            if mock_mode:
                st.caption("Modo actual: Mock (simulado).")
            return

        predicted = result.get("crimen_predicho", "N/A")
        prob = float(result.get("probabilidad", 0.0))
        latency = result.get("latency_ms", None)

        st.caption("Crimen predictivo")
        st.markdown(f"### {predicted}")

        st.markdown("---")
        st.caption("Probabilidad estimada")
        st.markdown(f"## {round(prob*100)}%")
        st.progress(min(max(prob, 0.0), 1.0))

        note = result.get("note", "Resultados basados en datos históricos.")
        st.caption(note)

        if latency is not None:
            st.caption(f"Latencia estimada: {latency:.0f} ms")

        if mock_mode:
            st.caption("⚙️ Modo actual: Mock (simulado).")
