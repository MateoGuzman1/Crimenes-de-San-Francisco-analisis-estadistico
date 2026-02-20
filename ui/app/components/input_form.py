import streamlit as st
from datetime import date

DISTRICTS = [
    "BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN",
    "PARK", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN"
]

def render_input_form(defaults: dict) -> dict:
    """
    Renderiza el panel izquierdo de inputs según la UI:
    Fecha, Hora, Latitud, Longitud, Distrito
    """
    with st.container(border=True):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fecha = st.date_input("Fecha", value=defaults.get("fecha", date.today()))
        with c2:
            hora = st.number_input("Hora", min_value=0, max_value=23, value=int(defaults.get("hora", 12)), step=1)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            latitud = st.number_input("Latitud", value=float(defaults.get("latitud", 37.775421)), format="%.6f")
        with c4:
            longitud = st.number_input("Longitud", value=float(defaults.get("longitud", -122.403405)), format="%.6f")

        distrito = st.selectbox(
            "Distrito",
            options=DISTRICTS,
            index=DISTRICTS.index(defaults.get("distrito", "SOUTHERN")) if defaults.get("distrito", "SOUTHERN") in DISTRICTS else 0
        )

    return {
        "fecha": fecha.isoformat(),
        "hora": int(hora),
        "latitud": float(latitud),
        "longitud": float(longitud),
        "distrito": str(distrito),
    }
