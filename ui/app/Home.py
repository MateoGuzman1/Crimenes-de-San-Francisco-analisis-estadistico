import streamlit as st
from datetime import date

from app.components.input_form import render_input_form
from app.components.prediction_panel import render_prediction_panel
from app.settings import MOCK_MODE, API_URL

from app.predictor.mock_predictor import mock_predict
from app.predictor.api_client import api_predict


def predict(payload: dict) -> dict:
    if MOCK_MODE:
        return mock_predict(payload)
    return api_predict(API_URL, payload)


def main():
    st.set_page_config(page_title="Clasificador de crimen - SF", layout="wide")

    st.title("Clasificador de crimen - SF")
    st.write("Predicción del tipo de crimen en San Francisco")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        payload = render_input_form(
            defaults={
                "fecha": date.today(),
                "hora": 12,
                "latitud": 37.775421,
                "longitud": -122.403405,
                "distrito": "SOUTHERN",
            }
        )

        do_predict = st.button("Predecir", use_container_width=True)

    with col_right:
        if "last_result" not in st.session_state:
            st.session_state["last_result"] = None

        if do_predict:
            with st.spinner("Generando predicción..."):
                st.session_state["last_result"] = predict(payload)

        render_prediction_panel(st.session_state["last_result"], mock_mode=MOCK_MODE)


if __name__ == "__main__":
    main()
