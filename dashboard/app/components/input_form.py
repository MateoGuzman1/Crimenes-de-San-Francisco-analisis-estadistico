import streamlit as st
from datetime import date

DISTRICTS = [
    "BAYVIEW", "CENTRAL", "INGLESIDE", "MISSION", "NORTHERN",
    "PARK", "RICHMOND", "SOUTHERN", "TARAVAL", "TENDERLOIN"
]


def _init_state(defaults: dict) -> None:
    if "fecha" not in st.session_state:
        st.session_state["fecha"] = defaults.get("fecha", date.today())
    if "hora" not in st.session_state:
        st.session_state["hora"] = int(defaults.get("hora", 12))
    if "latitud" not in st.session_state:
        st.session_state["latitud"] = float(defaults.get("latitud", 37.775421))
    if "longitud" not in st.session_state:
        st.session_state["longitud"] = float(defaults.get("longitud", -122.403405))
    if "distrito" not in st.session_state:
        st.session_state["distrito"] = str(defaults.get("distrito", "SOUTHERN"))


def render_input_form(defaults: dict) -> dict:
    _init_state(defaults)

    # Aplicar click del mapa ANTES de renderizar widgets
    if st.session_state.get("map_clicked"):
        st.session_state["latitud"] = float(
            st.session_state.get("map_lat", st.session_state["latitud"])
        )
        st.session_state["longitud"] = float(
            st.session_state.get("map_lon", st.session_state["longitud"])
        )

        detected_district = st.session_state.get("map_distrito")
        if detected_district and detected_district in DISTRICTS:
            st.session_state["distrito"] = detected_district

        st.session_state["map_clicked"] = False

    with st.container(border=True):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.date_input("Fecha", key="fecha")
        with c2:
            st.number_input("Hora", min_value=0, max_value=23, step=1, key="hora")

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            st.number_input("Latitud", format="%.6f", key="latitud")
        with c4:
            st.number_input("Longitud", format="%.6f", key="longitud")

        current = st.session_state.get("distrito", "SOUTHERN")
        if current not in DISTRICTS:
            current = "SOUTHERN"
            st.session_state["distrito"] = current

        st.selectbox(
            "Distrito",
            options=DISTRICTS,
            index=DISTRICTS.index(current),
            key="distrito"
        )

    return {
        "fecha": st.session_state["fecha"].isoformat(),
        "hora": int(st.session_state["hora"]),
        "latitud": float(st.session_state["latitud"]),
        "longitud": float(st.session_state["longitud"]),
        "distrito": str(st.session_state["distrito"]),
    }
